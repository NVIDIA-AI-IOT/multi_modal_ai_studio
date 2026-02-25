"""
OpenAI-compatible LLM Backend

Supports OpenAI, Ollama, vLLM, SGLang, and other OpenAI-compatible APIs.
Provides streaming text generation for conversational AI.
Adapted from live-riva-webui with timeline event support.
"""

import json
import logging
import re
import subprocess
from typing import Any, AsyncIterator, Dict, List, Optional

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


def _curl_for_post(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    """Build an equivalent curl command for a POST request (for debugging)."""
    body = json.dumps(payload, ensure_ascii=False)
    body_escaped = body.replace("'", "'\"'\"'")
    parts = [f"curl -X POST '{url}'"]
    for k, v in headers.items():
        v_escaped = str(v).replace("'", "'\"'\"'")
        parts.append(f"-H '{k}: {v_escaped}'")
    parts.append(f"-d '{body_escaped}'")
    return " \\\n  ".join(parts)


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


def _is_cosmos_model(model_name: str) -> bool:
    """Check if model is a Cosmos-Reason model (supports video input)."""
    if not model_name:
        return False
    name_lower = model_name.lower()
    return "cosmos" in name_lower and "reason" in name_lower


# Cosmos-Reason models always use video encoding (even for few frames)
# for temporal understanding.  The browser capture FPS is raised
# dynamically for Cosmos so both browser and USB cameras provide a
# similar number of frames.
_MIN_FRAMES_FOR_VIDEO = 2

# When sending individual images (non-Cosmos VLMs), cap the count to
# limit token usage.  Each image ≈ 1000 tokens; 6 images ≈ 6000 tokens
# is a good trade-off between visual context and prefill latency.
_MAX_INDIVIDUAL_IMAGES = 6


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
        self.api_key = config.api_key or "EMPTY"

        self.logger.info(f"Initialized OpenAI-compatible LLM: {config.model} @ {self.api_base}")

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
                headers = {"Authorization": f"Bearer {self.api_key}"}
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
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Add history
        if history:
            messages.extend(history)

        # Add current prompt - with images/video if provided (VLM multi-modal format)
        if image_data_urls and len(image_data_urls) > 0:
            user_content = []
            
            # Cosmos-Reason models: always use video encoding for temporal
            # understanding.  Browser capture FPS is dynamically raised so
            # both browser and USB cameras provide similar frame counts.
            if (
                _is_cosmos_model(self.config.model)
                and len(image_data_urls) >= _MIN_FRAMES_FOR_VIDEO
            ):
                video_url = _encode_images_to_video_base64(
                    image_data_urls,
                    speech_duration_secs=speech_duration or 0.0,
                )
                if video_url:
                    user_content.append({
                        "type": "video_url",
                        "video_url": {"url": video_url}
                    })
                    self.logger.info("VLM request (Cosmos video): %d frames encoded to MP4", len(image_data_urls))
                else:
                    self.logger.warning("Video encoding failed, falling back to multi-image")
                    video_url = None
            else:
                video_url = None
            
            # Multi-image format: used for non-Cosmos models or when video
            # encoding failed.
            if not video_url:
                detail = getattr(self.config, "vision_detail", "auto")
                imgs = image_data_urls
                if len(imgs) > _MAX_INDIVIDUAL_IMAGES:
                    step = len(imgs) / _MAX_INDIVIDUAL_IMAGES
                    imgs = [imgs[int(i * step)] for i in range(_MAX_INDIVIDUAL_IMAGES)]
                    self.logger.info("VLM: sub-sampled %d → %d images", len(image_data_urls), len(imgs))
                for img_url in imgs:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                            "detail": detail,
                        }
                    })
                self.logger.info("VLM request (multi-image): %d image(s) + text prompt", len(imgs))
            
            # Add text prompt after images/video
            user_content.append({
                "type": "text",
                "text": prompt,
            })
            
            messages.append({"role": "user", "content": user_content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt})

        # Prepare API request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        max_tokens = self.config.max_tokens
        if self.config.minimal_output:
            max_tokens = min(max_tokens, 16)
        # Cosmos-Reason models can emit chain-of-thought reasoning.
        # Cap max_tokens for VLM turns to keep responses concise.
        if _is_cosmos_model(self.config.model) and image_data_urls:
            COSMOS_VLM_MAX_TOKENS = 256
            if max_tokens > COSMOS_VLM_MAX_TOKENS:
                self.logger.info(
                    "Cosmos VLM: capping max_tokens from %d to %d",
                    max_tokens, COSMOS_VLM_MAX_TOKENS,
                )
                max_tokens = COSMOS_VLM_MAX_TOKENS
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
        self.logger.info("LLM request (curl equivalent):\n%s", _curl_for_post(url, headers, payload))

        full_response = ""
        token_count = 0

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"LLM API error: {resp.status} - {error_text}")
                        raise ConnectionError(f"LLM API error: {resp.status}")

                    # Stream response chunks
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()

                        # Skip empty lines and non-data lines
                        if not line or not line.startswith("data: "):
                            continue

                        # Remove "data: " prefix
                        line = line[6:]

                        # Check for end of stream
                        if line == "[DONE]":
                            # Yield final token
                            yield LLMToken(
                                token="",
                                is_final=True,
                                metadata={
                                    "token_count": token_count,
                                    "full_response": full_response,
                                }
                            )
                            break

                        try:
                            # Parse JSON chunk
                            chunk = json.loads(line)

                            # Extract content from chunk
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    # Strip markdown for voice output
                                    content = strip_markdown(content, preserve_spaces=True)

                                    full_response += content
                                    token_count += 1

                                    # Yield token
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
            if full_response or token_count:
                self.logger.info(f"LLM response: {len(full_response)} chars, {token_count} tokens")
                response_log: Dict[str, Any] = {
                    "response": full_response or "(empty)",
                    "chars": len(full_response),
                    "tokens": token_count,
                }
                self.logger.info("LLM response (optional jq for pretty):\n%s", _format_json(response_log))