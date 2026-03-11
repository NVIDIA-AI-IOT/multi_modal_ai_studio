"""
OpenAI-compatible LLM Backend

Supports OpenAI, Ollama, vLLM, SGLang, and other OpenAI-compatible APIs.
Provides streaming text generation for conversational AI.
Adapted from live-riva-webui with timeline event support.
"""

import json
import logging
import os
import re
import subprocess
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

from multi_modal_ai_studio.backends.base import (
    LLMBackend,
    LLMToken,
    ConnectionError,
    ConfigError,
)
from multi_modal_ai_studio.config.schema import LLMConfig

logger = logging.getLogger(__name__)


def _format_json(obj: Any) -> str:
    """Pretty-format JSON; use jq if available, else json.dumps with indent."""
    raw = json.dumps(obj, ensure_ascii=False)
    try:
        r = subprocess.run(
            ["jq", "-C", "."],
            input=raw,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Merge override into base in-place. Nested dicts are merged; other values overwrite."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _truncate_payload_for_log(obj: Any) -> Any:
    """Return a copy of obj with data URLs (base64 image/video) replaced by '<+ N characters>'.
    Normal text (system prompt, user message, etc.) is shown in full."""
    if isinstance(obj, str):
        if obj.startswith("data:image") or obj.startswith("data:video"):
            return f"<+ {len(obj)} characters>"
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_payload_for_log(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_payload_for_log(v) for v in obj]
    return obj


def _curl_for_post(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    """Build an equivalent curl command for a POST request (for debugging)."""
    payload_log = _truncate_payload_for_log(payload)
    body = json.dumps(payload_log, ensure_ascii=False)
    body_escaped = body.replace("'", "'\"'\"'")
    parts = [f"curl -X POST '{url}'"]
    for k, v in headers.items():
        v_escaped = str(v).replace("'", "'\"'\"'")
        parts.append(f"-H '{k}: {v_escaped}'")
    parts.append(f"-d '{body_escaped}'")
    return " \\\n  ".join(parts)


def _colorize_json(pretty_json: str) -> str:
    """Add ANSI colors to pretty-printed JSON. No-op if NO_COLOR is set."""
    if os.environ.get("NO_COLOR"):
        return pretty_json
    reset = "\033[0m"
    key_c = "\033[36m"      # cyan for keys
    str_c = "\033[32m"      # green for string values
    num_c = "\033[33m"      # yellow for numbers
    other_c = "\033[35m"    # magenta for true/false/null
    s = re.sub(r'(^\s*)"([^"]+)"(\s*:)', rf'\1{key_c}"\2"{reset}\3', pretty_json, flags=re.MULTILINE)
    s = re.sub(r'(: )"((?:[^"\\]|\\.)*)"([,\s\n\]}])', rf'\1{str_c}"\2"{reset}\3', s)
    s = re.sub(r'(: )([-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)([,}\s\n])', rf'\1{num_c}\2{reset}\3', s)
    s = re.sub(r'(: )(true|false|null)([,}\s\n])', rf'\1{other_c}\2{reset}\3', s)
    return s


def _format_payload_preview(payload: Dict[str, Any]) -> str:
    """Pretty-print truncated payload with optional color for debug log."""
    payload_log = _truncate_payload_for_log(payload)
    pretty = json.dumps(payload_log, indent=2, ensure_ascii=False)
    return _colorize_json(pretty)


def _build_request_debug_log(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    """Curl equivalent plus formatted, colorized payload preview."""
    curl = _curl_for_post(url, headers, payload)
    preview = _format_payload_preview(payload)
    return f"{curl}\n\nPayload (preview):\n{preview}"


def strip_markdown(text: str, preserve_spaces: bool = False) -> str:
    """Remove markdown formatting from text.

    Args:
        text: Text to process
        preserve_spaces: If True, preserve leading/trailing spaces (for streaming chunks)

    Returns:
        Text with markdown removed
    """
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove headers
    text = re.sub(r'#{1,6}\s+', '', text)

    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace (but preserve spaces in streaming mode)
    if not preserve_spaces:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

    return text


def _is_ollama_backend(config) -> bool:
    """Return True if the configured api_base points to an Ollama server."""
    api_base = getattr(config, "api_base", "") or ""
    return "11434" in api_base or "ollama" in api_base.lower()


def _is_tensorrt_edge_backend(config) -> bool:
    """Return True if api_base points to a TensorRT Edge LLM server (image payload format)."""
    api_base = getattr(config, "api_base", "") or ""
    base_lower = api_base.lower()
    return "58010" in api_base or "tensorrt" in base_lower or "edge-llm" in base_lower


def _should_use_video(config) -> bool:
    """Return True when frames should be encoded as MP4 video.

    Controlled by ``vision_video_encode`` config flag, but automatically
    disabled for backends that don't support video_url (e.g. Ollama, TensorRT Edge).
    """
    if not bool(getattr(config, "vision_video_encode", False)):
        return False
    if _is_ollama_backend(config):
        return False
    if _is_tensorrt_edge_backend(config):
        return False
    return True


def _strip_data_url_prefix(data_url: str) -> str:
    """Extract raw base64 payload from a data URL.

    "data:image/jpeg;base64,/9j/4AAQ..." → "/9j/4AAQ..."
    If the string has no prefix, return it unchanged.
    """
    if "," in data_url:
        return data_url.split(",", 1)[1]
    return data_url


# Cosmos-Reason models always use video encoding (even for few frames)
# for temporal understanding.  The browser capture FPS is raised
# dynamically for Cosmos so both browser and USB cameras provide a
# similar number of frames.
_MIN_FRAMES_FOR_VIDEO = 2

# When sending individual images (non-Cosmos VLMs), cap the count to
# limit token usage.  Each image ≈ 1000 tokens; 6 images ≈ 6000 tokens
# is a good trade-off between visual context and prefill latency.
_MAX_VIDEO_FALLBACK_IMAGES = 10
_MAX_TENSORRT_EDGE_IMAGES = 3
# Qwen3-VL uses ~170 tokens per video frame.  With max_model_len=8192,
# system prompt ≈ 300 tokens, user text ≈ 50, max_tokens ≈ 512,
# budget ≈ 8192 - 300 - 50 - 512 = 7330 → ~43 frames.
# Use 35 as a safe cap to leave headroom.
_MAX_COSMOS_VIDEO_FRAMES = 35


def _encode_images_to_video_base64(
    image_data_urls: List[str],
    speech_duration_secs: float = 0.0,
) -> Optional[str]:
    """
    Convert base64 JPEG images to MP4 video base64.

    Cosmos-Reason models support video input with temporal token compression,
    providing ~3x faster inference compared to multi-image input.

    The fps is calculated from the number of frames and speech duration so the
    resulting video length matches the original speech window.  This preserves
    temporal information (motion, actions, state changes) that the Cosmos
    temporal encoder relies on.

    Args:
        image_data_urls: List of base64 image data URLs (data:image/jpeg;base64,...)
        speech_duration_secs: Duration of the speech window these frames span.
            Used to calculate fps so video duration ≈ speech duration.
            If 0 or negative a sensible default (10 fps) is used.

    Returns:
        Video data URL (data:video/mp4;base64,...) or None if encoding fails
    """
    if not image_data_urls or len(image_data_urls) < 2:
        return None

    try:
        import av
        import io
        import base64
        import os
        import time as _time
        from PIL import Image
    except ImportError:
        logger.warning("[Video Encode] PyAV or PIL not available, falling back to multi-image")
        return None

    n_frames = len(image_data_urls)

    # Calculate fps so that video_duration ≈ speech_duration
    #   fps = n_frames / speech_duration
    # Clamp to [2, 30] to avoid degenerate cases.
    if speech_duration_secs > 0.1:
        fps = max(2, min(30, round(n_frames / speech_duration_secs)))
    else:
        fps = 10  # Sensible default when duration unknown

    t0 = _time.monotonic()

    try:
        # Decode first image to get dimensions
        first_b64 = image_data_urls[0].split(",", 1)[1]
        first_img = Image.open(io.BytesIO(base64.b64decode(first_b64)))
        width, height = first_img.size

        # Create video in memory
        output_buffer = io.BytesIO()
        container = av.open(output_buffer, mode='w', format='mp4')
        stream = container.add_stream('h264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '23'}

        # Encode each frame
        encoded_count = 0
        for img_url in image_data_urls:
            try:
                b64_data = img_url.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                av_frame = av.VideoFrame.from_image(img)
                av_frame = av_frame.reformat(format='yuv420p')
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                encoded_count += 1
            except Exception as e:
                logger.warning("[Video Encode] Failed to encode frame: %s", e)
                continue

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

        container.close()

        video_bytes = output_buffer.getvalue()
        video_b64 = base64.b64encode(video_bytes).decode('utf-8')
        encode_ms = (_time.monotonic() - t0) * 1000
        video_duration = encoded_count / fps if fps else 0

        logger.info(
            "[Video Encode] %d frames → MP4 %d KB | fps=%d video=%.1fs speech=%.1fs encode=%.0fms",
            encoded_count, len(video_bytes) // 1024, fps,
            video_duration, speech_duration_secs, encode_ms,
        )

        # Optional: save debug video when MMAS_DEBUG_VIDEOS=1
        if os.environ.get("MMAS_DEBUG_VIDEOS") == "1":
            try:
                from datetime import datetime
                debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "debug_videos")
                debug_dir = os.path.abspath(debug_dir)
                os.makedirs(debug_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"{debug_dir}/cosmos_{ts}_{encoded_count}f_{fps}fps.mp4"
                with open(path, 'wb') as f:
                    f.write(video_bytes)
                logger.info("[Video Encode] Debug video saved: %s", path)
            except Exception as e:
                logger.debug("[Video Encode] Debug save failed: %s", e)

        return f"data:video/mp4;base64,{video_b64}"

    except Exception as e:
        logger.warning("[Video Encode] Failed to create video: %s", e)
        return None


class OpenAILLMBackend(LLMBackend):
    """OpenAI-compatible LLM backend with streaming support.

    Supports:
    - OpenAI API
    - Ollama
    - vLLM
    - SGLang
    - Any OpenAI-compatible endpoint

    Features:
    - Streaming token generation
    - Conversation history management
    - Automatic model detection (for Ollama)
    - Markdown stripping for voice output
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI-compatible LLM backend.

        Args:
            config: LLMConfig instance

        Raises:
            ConfigError: If configuration is invalid
        """
        super().__init__(config)

        # Validate configuration
        if config.scheme not in ["openai", "anthropic"]:
            raise ConfigError(f"Unsupported LLM scheme: {config.scheme}")

        if not config.api_base:
            raise ConfigError("LLM API base URL is required")

        self.api_base = config.api_base.rstrip("/")
        self.api_key = config.api_key or ""

        self.logger.info(f"Initialized OpenAI-compatible LLM: {config.model} @ {self.api_base}")

    def _is_local_api_base(self, base: str) -> bool:
        """True if base URL is localhost or private IP (no auth sent to local vLLM/Ollama)."""
        try:
            parsed = urlparse(base if base.startswith("http") else f"http://{base}")
            host = (parsed.hostname or "").lower()
            if not host or host == "localhost":
                return True
            if host == "127.0.0.1":
                return True
            # Private IP ranges
            if host.startswith("10."):
                return True
            if host.startswith("192.168."):
                return True
            if host.startswith("172."):
                parts = host.split(".")
                if len(parts) == 4 and parts[1].isdigit():
                    b = int(parts[1])
                    if 16 <= b <= 31:
                        return True
            return False
        except Exception:
            return False

    def _should_send_auth(self) -> bool:
        """True if we should send Authorization header (e.g. for OpenAI). Skip for local vLLM/Ollama."""
        if self._is_local_api_base(self.api_base):
            return False
        key = (self.api_key or "").strip()
        return bool(key and key.upper() != "EMPTY")

    async def list_available_models(self) -> List[str]:
        """List available models from the LLM API.

        Attempts to detect models from Ollama's native API or OpenAI endpoint.

        Returns:
            List of model names, or empty list if detection fails
        """
        try:
            # Try Ollama's native API first (/api/tags)
            ollama_base = self.api_base.replace("/v1", "")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{ollama_base}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        model_names = [m["name"] for m in models]
                        self.logger.info(f"Detected {len(model_names)} Ollama models")
                        return model_names

        except Exception as e:
            self.logger.debug(f"Ollama model detection failed: {e}")

        # Try OpenAI /v1/models endpoint
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self._should_send_auth():
                    headers["Authorization"] = f"Bearer {self.api_key}"
                async with session.get(
                    f"{self.api_base}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_names = [m["id"] for m in data.get("data", [])]
                        self.logger.info(f"Detected {len(model_names)} OpenAI models")
                        return model_names

        except Exception as e:
            self.logger.debug(f"OpenAI model detection failed: {e}")

        self.logger.warning("Failed to detect models from API")
        return []

    # -----------------------------------------------------------------
    # Vision content formatting — one method per API format
    # -----------------------------------------------------------------

    def _build_vision_content(
        self,
        image_data_urls: List[str],
        prompt: str,
        speech_duration: Optional[float],
    ) -> list:
        """Build the multimodal ``content`` list for a user message.

        All backends (vLLM, Ollama, TensorRT Edge LLM, SGLang, OpenAI) use
        the standard OpenAI ``image_url`` format.  Cosmos-Reason models
        additionally support ``video_url`` when video encoding is enabled.

        Returns the ``content`` list ready to be placed inside a user message.
        """
        content: list = [{"type": "text", "text": prompt}]

        video_url = None
        if (
            _should_use_video(self.config)
            and len(image_data_urls) >= _MIN_FRAMES_FOR_VIDEO
        ):
            imgs_for_video = image_data_urls
            if len(imgs_for_video) > _MAX_COSMOS_VIDEO_FRAMES:
                step = len(imgs_for_video) / _MAX_COSMOS_VIDEO_FRAMES
                imgs_for_video = [imgs_for_video[int(i * step)] for i in range(_MAX_COSMOS_VIDEO_FRAMES)]
                self.logger.info("VLM: sub-sampled %d → %d frames for video (token budget)", len(image_data_urls), len(imgs_for_video))
            video_url = _encode_images_to_video_base64(
                imgs_for_video,
                speech_duration_secs=speech_duration or 0.0,
            )
            if video_url:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": video_url},
                })
                self.logger.info("VLM request (Cosmos video): %d frames encoded to MP4", len(image_data_urls))
            else:
                self.logger.warning("Video encoding failed, falling back to multi-image")
                video_url = None

        if not video_url:
            detail = getattr(self.config, "vision_detail", "auto")
            imgs = image_data_urls
            if _is_tensorrt_edge_backend(self.config) and len(imgs) > _MAX_TENSORRT_EDGE_IMAGES:
                step = len(imgs) / _MAX_TENSORRT_EDGE_IMAGES
                imgs = [imgs[int(i * step)] for i in range(_MAX_TENSORRT_EDGE_IMAGES)]
                self.logger.info("VLM: sub-sampled %d → %d images (Edge LLM limit)", len(image_data_urls), len(imgs))
            elif _should_use_video(self.config) and len(imgs) > _MAX_VIDEO_FALLBACK_IMAGES:
                step = len(imgs) / _MAX_VIDEO_FALLBACK_IMAGES
                imgs = [imgs[int(i * step)] for i in range(_MAX_VIDEO_FALLBACK_IMAGES)]
                self.logger.info("VLM: sub-sampled %d → %d images (Cosmos video fallback)", len(image_data_urls), len(imgs))
            for img_url in imgs:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url, "detail": detail},
                })
            self.logger.info("VLM request (multi-image): %d image(s) + text prompt", len(imgs))

        return content

    async def generate_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        image_data_urls: Optional[List[str]] = None,
        speech_duration: Optional[float] = None,
    ) -> AsyncIterator[LLMToken]:
        """Generate response tokens in streaming fashion.

        Args:
            prompt: User prompt/message
            history: Conversation history in format:
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            system_prompt: Optional system prompt (overrides config if provided)
            image_data_urls: Optional list of base64 image data URLs for VLM models.
                            Format: ["data:image/jpeg;base64,...", ...]
                            Images are sent in order (first = earliest frame).
            speech_duration: Duration in seconds of the speech window the frames
                            span.  Used by Cosmos video encoder to set fps so the
                            video duration matches the real speech duration.

        Yields:
            LLMToken: Generated tokens

        Raises:
            ConnectionError: If unable to connect to LLM service
        """
        # Build messages array
        messages = []
        # Add system prompt
        sys_prompt = system_prompt or self.config.system_prompt
        if self.config.minimal_output:
            suffix = " Answer with only a number or minimal tokens. No reasoning or explanation."
            sys_prompt = (sys_prompt or "") + suffix
        elif getattr(self.config, "enable_reasoning", False):
            reasoning_fmt = getattr(self.config, "reasoning_prompt", "") or ""
            if reasoning_fmt:
                sys_prompt = (sys_prompt or "") + reasoning_fmt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Add history
        if history:
            messages.extend(history)

        if image_data_urls and len(image_data_urls) > 0:
            user_content = self._build_vision_content(
                image_data_urls, prompt, speech_duration,
            )
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        # Prepare API request
        url = f"{self.api_base}/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._should_send_auth():
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_tokens = self.config.max_tokens
        if self.config.minimal_output:
            max_tokens = min(max_tokens, 16)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": True,
        }

        extra = getattr(self.config, "extra_request_body", None)
        if extra and isinstance(extra, str) and extra.strip():
            try:
                extra_obj = json.loads(extra)
                if isinstance(extra_obj, dict):
                    _deep_merge(payload, extra_obj)
            except json.JSONDecodeError as e:
                self.logger.warning("Invalid extra_request_body JSON: %s", e)

        self.logger.debug(f"LLM request: {prompt[:50]}...")
        self.logger.info("LLM request:\n%s", _build_request_debug_log(url, headers, payload))

        full_response = ""
        reasoning_response = ""
        token_count = 0
        reasoning_started = False
        in_think_block = False  # fallback tracker if server lacks reasoning parser
        _post_think_buf = ""  # buffer after </think> to detect <answer> tags
        _post_think_buffering = False
        _POST_THINK_BUF_MAX = 500  # flush if no <answer> found within this many chars
        _router_model = None  # track which model actually served the request (useful for routers)

        try:
            body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Length"] = str(len(body_bytes))
            self.logger.debug("LLM request body: %d bytes", len(body_bytes))
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=body_bytes) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"LLM API error: {resp.status} - {error_text}")
                        raise ConnectionError(f"LLM API error: {resp.status}")

                    # Stream response chunks
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()

                        # Skip empty lines and non-data lines
                        if not line or not line.startswith("data: "):
                            if line and ("error" in line.lower() or "500" in line):
                                self.logger.warning("SSE non-data line (possible error): %.300s", line)
                                if "500" in line or "Internal Server Error" in line:
                                    err_msg = "[LLM backend error — the model crashed. This often means the image payload is too large for Edge LLM.]"
                                    self.logger.error("Edge LLM 500: yielding error to user")
                                    yield LLMToken(token=err_msg, is_final=False)
                                    full_response += err_msg
                            continue

                        # Remove "data: " prefix
                        line = line[6:]

                        # Check for end of stream
                        if line == "[DONE]":
                            # Flush post-think buffer if stream ends while buffering
                            if _post_think_buffering and _post_think_buf:
                                leftover = _post_think_buf
                                if "<answer>" in leftover:
                                    leftover = leftover.split("<answer>", 1)[1]
                                leftover = leftover.replace("</answer>", "")
                                leftover = strip_markdown(leftover, preserve_spaces=True).strip()
                                if leftover:
                                    full_response += leftover
                                    token_count += 1
                                    yield LLMToken(token=leftover, is_final=False)
                                _post_think_buffering = False
                                _post_think_buf = ""

                            final_meta: Dict[str, Any] = {
                                "token_count": token_count,
                                "full_response": full_response,
                            }
                            if _router_model:
                                final_meta["router_model"] = _router_model
                            if reasoning_response:
                                final_meta["reasoning"] = reasoning_response
                            yield LLMToken(
                                token="",
                                is_final=True,
                                metadata=final_meta,
                            )
                            break

                        try:
                            chunk = json.loads(line)

                            if _router_model is None and chunk.get("model"):
                                _router_model = chunk["model"]
                                self.logger.info("[LLM] Router model: %s", _router_model)

                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta") or {}

                                # vLLM/cosmos-reason: thinking in reasoning_content or reasoning;
                                # answer in content.
                                rc = (
                                    delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                    or ""
                                )
                                if rc:
                                    if not reasoning_started:
                                        reasoning_started = True
                                        yield LLMToken(
                                            token="",
                                            is_final=False,
                                            metadata={"reasoning_start": True},
                                        )
                                    reasoning_response += rc
                                    continue

                                content = delta.get("content", "")
                                if not content:
                                    continue

                                # Fallback: strip <think>...</think> if the
                                # server doesn't have a reasoning parser.
                                if "<think>" in content:
                                    in_think_block = True
                                    content = content.split("<think>", 1)[0]
                                if in_think_block:
                                    if "</think>" in content:
                                        after = content.split("</think>", 1)[1]
                                        reasoning_response += content.split("</think>", 1)[0]
                                        in_think_block = False
                                        content = after
                                        _post_think_buffering = True
                                        _post_think_buf = ""
                                    else:
                                        reasoning_response += content
                                        continue

                                if not content and not _post_think_buffering:
                                    continue

                                # After </think>, buffer briefly to detect <answer> tags
                                # which indicate leaked reasoning before the real answer.
                                if _post_think_buffering:
                                    _post_think_buf += content
                                    if "<answer>" in _post_think_buf:
                                        content = _post_think_buf.split("<answer>", 1)[1]
                                        content = content.replace("</answer>", "")
                                        _post_think_buffering = False
                                        _post_think_buf = ""
                                    elif len(_post_think_buf) >= _POST_THINK_BUF_MAX:
                                        content = _post_think_buf
                                        content = content.replace("<answer>", "").replace("</answer>", "")
                                        _post_think_buffering = False
                                        _post_think_buf = ""
                                    else:
                                        continue

                                # Strip any remaining <answer>/<​/answer> tags
                                if "<answer>" in content:
                                    content = content.split("<answer>", 1)[1]
                                content = content.replace("</answer>", "")

                                if not content:
                                    continue

                                content = strip_markdown(content, preserve_spaces=True)

                                # Strip leading whitespace from first answer token after reasoning
                                if token_count == 0:
                                    content = content.lstrip()
                                    if not content:
                                        continue

                                full_response += content
                                token_count += 1

                                yield LLMToken(
                                    token=content,
                                    is_final=False,
                                    metadata={
                                        "token_count": token_count,
                                        "partial_response": full_response,
                                    }
                                )

                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse JSON: {e}")
                            continue

        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error during LLM streaming: {e}")
            raise ConnectionError(f"Failed to connect to LLM API: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error during LLM streaming: {e}", exc_info=True)
            raise

        finally:
            if full_response or token_count or reasoning_response:
                self.logger.info(f"LLM response: {len(full_response)} chars, {token_count} tokens")
                response_log: Dict[str, Any] = {
                    "response": full_response or "(empty)",
                    "chars": len(full_response),
                    "tokens": token_count,
                }
                if _router_model:
                    response_log["router_model"] = _router_model
                if reasoning_response:
                    response_log["reasoning"] = reasoning_response
                    self.logger.info(
                        "LLM reasoning (%d chars): %.200s%s",
                        len(reasoning_response), reasoning_response,
                        "..." if len(reasoning_response) > 200 else "",
                    )
                self.logger.info("LLM response (optional jq for pretty):\n%s", _format_json(response_log))
