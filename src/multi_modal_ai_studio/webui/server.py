#!/usr/bin/env python3
"""
Simple web server for Multi-modal AI Studio WebUI.

Serves static HTML/CSS/JS and provides API endpoints for session data.
Supports HTTPS via self-signed certificates (same logic as Live RIVA WebUI).
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from aiohttp import web

# Load .env so OPENAI_API_KEY (and others) are available for prefills and backends
load_dotenv()
from aiohttp.abc import AbstractAccessLogger

from multi_modal_ai_studio.config.schema import LLMConfig
from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.backends.asr.riva import DEFAULT_ASR_MODEL, list_riva_asr_models_sync
from multi_modal_ai_studio.backends.tts.riva import list_riva_tts_voices_sync
from multi_modal_ai_studio.devices.local import (
    list_local_cameras,
    list_local_audio_inputs,
    list_local_audio_outputs,
)
from multi_modal_ai_studio.webui.voice_pipeline import handle_voice_ws, handle_mic_preview_ws
from multi_modal_ai_studio.webui.camera_webrtc import handle_camera_webrtc_ws

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _QuietAccessLogger(AbstractAccessLogger):
    """Access logger for HTTP requests (can skip high-frequency paths if needed)."""

    def log(self, request: web.BaseRequest, response: web.StreamResponse, time: float) -> None:
        try:
            size = getattr(response, "body_length", None) or getattr(response, "_body_length", None)
            if size is None:
                size = "-"
            from datetime import datetime
            req_time = datetime.utcnow().strftime("%d/%b/%Y:%H:%M:%S +0000")
            self.logger.info(
                '%s [%s] "%s %s HTTP/1.1" %s %s',
                request.remote,
                req_time,
                request.method,
                request.path,
                response.status,
                size,
            )
        except Exception:
            pass

def get_app_config_dir() -> Path:
    """Get the application config directory following OS conventions (same as Live RIVA WebUI)."""
    if os.name == "posix":
        if "darwin" in sys.platform.lower():
            config_dir = Path.home() / "Library" / "Application Support" / "multi-modal-ai-studio"
        else:
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "multi-modal-ai-studio"
            )
    else:
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "multi-modal-ai-studio"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def generate_self_signed_cert(cert_path: str = "cert.pem", key_path: str = "key.pem") -> bool:
    """Generate a self-signed SSL certificate if it doesn't exist (same logic as Live RIVA WebUI)."""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True

    logger.info("🔐 Generating self-signed SSL certificate...")
    logger.info(f"   Saving to: {os.path.dirname(os.path.abspath(cert_path)) or '.'}")
    logger.info("   (This may take 10-30 seconds on first run - openssl is generating the key)")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_path,
                "-keyout",
                key_path,
                "-days",
                "365",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        logger.info(f"✅ Generated {cert_path} and {key_path}")
        return True
    except FileNotFoundError:
        logger.warning("⚠️  openssl not found - cannot auto-generate certificates")
        logger.warning(
            "⚠️  Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  openssl timed out while generating certificates")
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to generate certificates: {e}")
        return False


class WebUIServer:
    """Web server for Multi-modal AI Studio UI."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        session_dir: Path = None,
        ssl_context: Optional[object] = None,
        initial_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize web server.

        Args:
            host: Host address to bind to
            port: Port to listen on
            session_dir: Directory containing session JSON files
            ssl_context: Optional ssl.SSLContext for HTTPS (e.g. self-signed cert)
            initial_config: Optional preset/config dict loaded from YAML (served to frontend)
        """
        self.host = host
        self.port = port
        self.session_dir = session_dir or Path("sessions")
        self._session_dir_override: Optional[str] = None  # "sessions" | "mock_sessions" | None
        self.ssl_context = ssl_context
        self.initial_config = initial_config
        self.app = web.Application()
        self.app["session_dir"] = self.session_dir
        self.app["_server"] = self  # so voice pipeline can read current effective session dir
        self.setup_routes()

    def _get_effective_session_dir(self) -> Path:
        """Return session directory to use (override if set, else startup default)."""
        if self._session_dir_override in ("sessions", "mock_sessions"):
            return (self.session_dir.parent / self._session_dir_override).resolve()
        return self.session_dir

    def setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get('/api/sessions', self.handle_list_sessions)
        self.app.router.add_get('/api/sessions/{session_id}', self.handle_get_session)
        self.app.router.add_patch('/api/sessions/{session_id}', self.handle_patch_session)
        self.app.router.add_delete('/api/sessions/{session_id}', self.handle_delete_session)
        self.app.router.add_get('/api/llm/models', self.handle_llm_models)
        self.app.router.add_post('/api/llm/warmup', self.handle_llm_warmup)
        self.app.router.add_get('/api/asr/models', self.handle_asr_models)
        self.app.router.add_get('/api/tts/voices', self.handle_tts_voices)
        self.app.router.add_get('/api/health/llm', self.handle_health_llm)
        self.app.router.add_get('/api/health/riva', self.handle_health_riva)
        self.app.router.add_get('/api/devices/cameras', self.handle_list_cameras)
        self.app.router.add_get('/api/devices/audio-inputs', self.handle_list_audio_inputs)
        self.app.router.add_get('/api/devices/audio-outputs', self.handle_list_audio_outputs)
        self.app.router.add_get('/api/camera/stream', self.handle_camera_stream)
        self.app.router.add_get('/api/config/initial', self.handle_initial_config)
        self.app.router.add_get('/api/videos/list', self.handle_list_videos)
        self.app.router.add_get('/api/videos/file', self.handle_serve_video)
        self.app.router.add_get('/api/app/session-dir', self.handle_get_session_dir)
        self.app.router.add_patch('/api/app/session-dir', self.handle_patch_session_dir)
        self.app.router.add_get('/api/config/prefills', self.handle_config_prefills)
        self.app.router.add_get('/ws/voice', handle_voice_ws)
        self.app.router.add_get('/ws/mic-preview', handle_mic_preview_ws)
        self.app.router.add_get('/ws/camera-webrtc', handle_camera_webrtc_ws)
        # Serve static files
        static_dir = Path(__file__).parent / 'static'
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/favicon.ico', self.handle_favicon)
        self.app.router.add_static('/static', static_dir, name='static')
        self.app.router.add_get('/{filename}', self.handle_static_file)

    async def handle_index(self, request):
        """Serve index.html."""
        static_dir = Path(__file__).parent / 'static'
        index_path = static_dir / 'index.html'

        if not index_path.exists():
            return web.Response(text="index.html not found", status=404)

        response = web.FileResponse(index_path)
        response.headers['Cache-Control'] = 'no-cache'
        return response

    async def handle_favicon(self, request):
        """Respond to /favicon.ico so the browser does not 404 (no icon file)."""
        return web.Response(status=204)

    async def handle_config_prefills(self, request: web.Request) -> web.Response:
        """GET /api/config/prefills: values from env (e.g. OPENAI_API_KEY) to prefill Configuration UI."""
        payload: Dict[str, Any] = {}
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if api_key:
            payload["openai_api_key"] = api_key
        return web.json_response(payload)

    async def handle_static_file(self, request):
        """Serve static files (CSS, JS)."""
        filename = request.match_info['filename']
        static_dir = Path(__file__).parent / 'static'
        file_path = static_dir / filename

        if not file_path.exists():
            return web.Response(text=f"{filename} not found", status=404)

        response = web.FileResponse(file_path)
        response.headers['Cache-Control'] = 'no-cache'
        return response

    async def handle_list_sessions(self, request):
        """API endpoint to list all sessions."""
        try:
            sessions = self.load_all_sessions()
            return web.json_response(sessions)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    async def handle_get_session(self, request):
        """API endpoint to get a specific session."""
        session_id = request.match_info['session_id']

        try:
            sessions = self.load_all_sessions()
            session = next((s for s in sessions if s['session_id'] == session_id), None)

            if session is None:
                return web.json_response(
                    {"error": "Session not found"},
                    status=404
                )

            return web.json_response(session)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    def _session_file_path(self, session_id: str) -> Optional[Path]:
        """Resolve session JSON file path; return None if session_id is invalid (e.g. path traversal)."""
        if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            return None
        base = self._get_effective_session_dir()
        path = base / f"{session_id}.json"
        return path if path.exists() else None

    async def handle_patch_session(self, request: web.Request) -> web.Response:
        """PATCH /api/sessions/{session_id}: update session (e.g. name). Body: { \"name\": \"...\" }."""
        session_id = request.match_info["session_id"]
        path = self._session_file_path(session_id)
        if path is None:
            return web.json_response({"error": "Session not found"}, status=404)
        try:
            body = await request.json()
            name = body.get("name")
            if name is None:
                return web.json_response({"error": "name is required"}, status=400)
            with open(path, "r") as f:
                data = json.load(f)
            data["name"] = str(name).strip() or data.get("name", "Untitled")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return web.json_response(data)
        except Exception as e:
            logger.error(f"Error patching session {session_id}: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_delete_session(self, request: web.Request) -> web.Response:
        """DELETE /api/sessions/{session_id}: delete the session file."""
        session_id = request.match_info["session_id"]
        path = self._session_file_path(session_id)
        if path is None:
            return web.json_response({"error": "Session not found"}, status=404)
        try:
            path.unlink()
            return web.Response(status=204)
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_session_dir(self, request: web.Request) -> web.Response:
        """GET /api/app/session-dir: current session directory and override."""
        effective = self._get_effective_session_dir()
        return web.json_response({
            "session_dir": effective.name,
            "override": self._session_dir_override,
        })

    async def handle_patch_session_dir(self, request: web.Request) -> web.Response:
        """PATCH /api/app/session-dir: set session directory override. Body: { \"session_dir\": \"sessions\" | \"mock_sessions\" | null }."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        val = body.get("session_dir")
        if val is None or val == "":
            self._session_dir_override = None
        elif val in ("sessions", "mock_sessions"):
            self._session_dir_override = val
        else:
            return web.json_response(
                {"error": "session_dir must be null, \"sessions\", or \"mock_sessions\""},
                status=400,
            )
        self.app["session_dir"] = self._get_effective_session_dir()
        effective = self._get_effective_session_dir()
        logger.info(f"Session directory override set to {self._session_dir_override!r}; effective dir: {effective}")
        return web.json_response({
            "session_dir": effective.name,
            "override": self._session_dir_override,
        })

    async def handle_initial_config(self, request: web.Request) -> web.Response:
        """GET /api/config/initial: return preset/config loaded from CLI --preset or --config."""
        if self.initial_config:
            return web.json_response(self.initial_config)
        return web.json_response({})

    async def handle_llm_models(self, request: web.Request) -> web.Response:
        """List available LLM models at the given API base URL (Ollama, vLLM, OpenAI, etc.)."""
        api_base = request.query.get("api_base", "").strip().rstrip("/")
        api_key = request.query.get("api_key") or None
        if not api_base:
            return web.json_response(
                {"error": "api_base query parameter is required"},
                status=400
            )
        try:
            config = LLMConfig(api_base=api_base, api_key=api_key, model="")
            backend = OpenAILLMBackend(config)
            models = await backend.list_available_models()
            return web.json_response({"models": models})
        except Exception as e:
            logger.exception("Failed to list LLM models")
            return web.json_response(
                {"error": str(e), "models": []},
                status=500
            )

    async def _detect_vlm_capability(self, api_base: str, api_key: Optional[str], model: str) -> dict:
        """
        Detect if a model supports vision (VLM) across different backends.
        
        Detection methods (in order of preference):
        1. Ollama API: Check /api/show for "projector" field
        2. Name pattern: Check for known VLM patterns (vl, vision, llava, etc.)
        3. Image probe: Try sending a tiny image (universal fallback)
        
        Returns: {"is_vlm": bool, "detection_method": str, "confidence": str}
        """
        import aiohttp
        import re
        
        result = {"is_vlm": False, "detection_method": "unknown", "confidence": "low"}
        
        # -------------------------------------------------------------------------
        # Method 1: Ollama-specific detection (fast, no inference)
        # -------------------------------------------------------------------------
        if "11434" in api_base or "ollama" in api_base.lower():
            try:
                # Ollama's /api/show returns model details including vision components
                ollama_base = api_base.replace("/v1", "").rstrip("/")
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{ollama_base}/api/show",
                        json={"name": model},
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("projector"):
                                result = {"is_vlm": True, "detection_method": "ollama_projector", "confidence": "high"}
                                logger.info("[VLM Detect] Ollama projector found → VLM")
                                return result
                            families = data.get("details", {}).get("families", [])
                            if any("vl" in f.lower() for f in families):
                                result = {"is_vlm": True, "detection_method": "ollama_family", "confidence": "high"}
                                logger.info("[VLM Detect] Ollama family contains 'vl' → VLM")
                                return result
                            model_info = data.get("model_info", {})
                            has_vision = any("vision" in k for k in model_info)
                            if has_vision:
                                result = {"is_vlm": True, "detection_method": "ollama_model_info", "confidence": "high"}
                                logger.info("[VLM Detect] Ollama model_info has vision keys → VLM")
                                return result
                            result = {"is_vlm": False, "detection_method": "ollama_api", "confidence": "high"}
                            logger.info("[VLM Detect] Ollama model has no vision components → LLM")
                            return result
            except Exception as e:
                logger.debug("[VLM Detect] Ollama API check failed: %s", e)
        
        # -------------------------------------------------------------------------
        # Method 3: Image probe (universal, works for any backend)
        # Send a tiny 1x1 image and see if model accepts it
        # -------------------------------------------------------------------------
        try:
            # 1x1 white PNG (smallest possible valid image)
            tiny_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
            
            config = LLMConfig(
                api_base=api_base,
                api_key=api_key,
                model=model,
                max_tokens=1,
                temperature=0.0,
            )
            backend = OpenAILLMBackend(config)
            
            # Create multimodal message with image
            test_messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": tiny_image}},
                    {"type": "text", "text": "1"}
                ]
            }]
            
            # Try to generate with image - VLMs will accept, LLMs will error
            async for token in backend.generate_stream("", test_messages):
                # If we get any response, model accepted the image → VLM
                result = {"is_vlm": True, "detection_method": "image_probe", "confidence": "high"}
                logger.info("[VLM Detect] Image probe succeeded → VLM")
                return result
                
        except Exception as e:
            error_str = str(e).lower()
            # Check for specific error messages that indicate image rejection
            if any(x in error_str for x in ["image", "vision", "multimodal", "unsupported"]):
                result = {"is_vlm": False, "detection_method": "image_probe_rejected", "confidence": "high"}
                logger.info("[VLM Detect] Image probe rejected → LLM")
                return result
            # Other errors might be network issues, etc.
            logger.debug("[VLM Detect] Image probe error: %s", e)
        
        # Default: unknown, assume LLM for safety
        result = {"is_vlm": False, "detection_method": "default", "confidence": "low"}
        logger.info("[VLM Detect] Could not determine, defaulting to LLM")
        return result

    async def handle_llm_warmup(self, request: web.Request) -> web.Response:
        """
        Warm up an LLM/VLM model by sending a minimal prompt.
        Also detects if the model supports vision (VLM).
        
        This loads the model into GPU memory before the user starts a session,
        reducing first-response latency. Works with any OpenAI-compatible backend
        (Ollama, vLLM, SGLang, OpenAI, etc.).
        
        POST body: {"api_base": "...", "api_key": "...", "model": "...", "detect_vlm": true}
        
        Returns: {
            "success": true,
            "model": "...",
            "warmup_time_seconds": X.XX,
            "is_vlm": true/false,
            "vlm_detection_method": "...",
            "vlm_confidence": "high/medium/low"
        }
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        
        api_base = (body.get("api_base") or "").strip().rstrip("/")
        api_key = body.get("api_key") or None
        model = (body.get("model") or "").strip()
        detect_vlm = body.get("detect_vlm", True)  # Default: detect VLM capability
        
        if not api_base or not model:
            return web.json_response(
                {"error": "api_base and model are required"},
                status=400
            )
        
        logger.info("[LLM Warmup] Warming up model: %s @ %s", model, api_base)
        
        try:
            import time
            start_time = time.time()
            
            # Detect VLM capability (if requested)
            vlm_info = {"is_vlm": False, "detection_method": "skipped", "confidence": "none"}
            if detect_vlm:
                vlm_info = await self._detect_vlm_capability(api_base, api_key, model)
            
            # Create config and backend for warmup
            config = LLMConfig(
                api_base=api_base,
                api_key=api_key,
                model=model,
                max_tokens=1,  # Minimal tokens to reduce overhead
                temperature=0.0,
            )
            backend = OpenAILLMBackend(config)
            
            # Send a minimal prompt to trigger model loading
            warmup_response = ""
            async for token in backend.generate_stream("Say OK", []):
                warmup_response += token.token
                break  # We only need the first token to confirm model is loaded
            
            elapsed = time.time() - start_time
            logger.info("[LLM Warmup] Model %s warmed up in %.2fs (VLM: %s)", 
                       model, elapsed, vlm_info["is_vlm"])
            
            return web.json_response({
                "success": True,
                "model": model,
                "warmup_time_seconds": round(elapsed, 2),
                "is_vlm": vlm_info["is_vlm"],
                "vlm_detection_method": vlm_info["detection_method"],
                "vlm_confidence": vlm_info["confidence"],
            })
        except Exception as e:
            logger.warning("[LLM Warmup] Failed to warm up %s: %s", model, e)
            return web.json_response({
                "success": False,
                "error": str(e),
            }, status=500)

    async def handle_asr_models(self, request: web.Request) -> web.Response:
        """List available Riva ASR models at the given server (query: server=host:port)."""
        server = (request.query.get("server") or "").strip()
        if not server:
            return web.json_response(
                {"error": "server query parameter is required (e.g. server=localhost:50051)"},
                status=400
            )
        use_ssl = server.startswith("https://") or ":443" in server
        try:
            loop = asyncio.get_running_loop()
            models = await loop.run_in_executor(
                None,
                lambda: list_riva_asr_models_sync(server, use_ssl=use_ssl)
            )
            # Prefer Silero VAD model for multi-utterance / second-turn; see docs/asr_model_for_multi_utterance.md
            if models and DEFAULT_ASR_MODEL in models:
                default_model = DEFAULT_ASR_MODEL
            else:
                default_model = models[0] if models else DEFAULT_ASR_MODEL
            return web.json_response({
                "models": models,
                "default_model": default_model,
            })
        except Exception as e:
            logger.exception("Failed to list Riva ASR models")
            return web.json_response(
                {"error": str(e), "models": []},
                status=500
            )

    async def handle_tts_voices(self, request: web.Request) -> web.Response:
        """List available TTS voices and model name(s) from Riva (query: server=host:port, optional language=en-US)."""
        server = (request.query.get("server") or "").strip()
        if not server:
            return web.json_response(
                {"error": "server query parameter is required (e.g. server=localhost:50051)"},
                status=400,
            )
        language = (request.query.get("language") or "en-US").strip()
        use_ssl = server.startswith("https://") or ":443" in server
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: list_riva_tts_voices_sync(server, use_ssl=use_ssl, language_code=language),
            )
            if isinstance(result, dict):
                return web.json_response({
                    "voices": result.get("voices", []),
                    "model_name": result.get("model_name"),
                    "model_names": result.get("model_names", []),
                })
            return web.json_response({"voices": result, "model_name": None, "model_names": []})
        except Exception as e:
            logger.exception("Failed to list Riva TTS voices")
            return web.json_response({"error": str(e), "voices": [], "model_name": None, "model_names": []}, status=500)

    async def handle_health_llm(self, request: web.Request) -> web.Response:
        """Check if the LLM server (Ollama, vLLM, etc.) at api_base is reachable. Returns { \"ok\": true } or { \"ok\": false, \"error\": \"...\" }."""
        api_base = (request.query.get("api_base") or "").strip().rstrip("/")
        if not api_base:
            return web.json_response({"ok": False, "error": "api_base query parameter is required"}, status=400)
        try:
            config = LLMConfig(api_base=api_base, api_key=None, model="")
            backend = OpenAILLMBackend(config)
            await asyncio.wait_for(backend.list_available_models(), timeout=5.0)
            return web.json_response({"ok": True})
        except asyncio.TimeoutError:
            return web.json_response({"ok": False, "error": "Connection timed out (5s)"})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)})

    async def handle_health_riva(self, request: web.Request) -> web.Response:
        """Check if the Riva server (ASR/TTS) at server=host:port is reachable. Returns { \"ok\": true } or { \"ok\": false, \"error\": \"...\" }."""
        server = (request.query.get("server") or "").strip()
        if not server:
            return web.json_response({"ok": False, "error": "server query parameter is required (e.g. server=localhost:50051)"}, status=400)
        use_ssl = server.startswith("https://") or ":443" in server
        try:
            loop = asyncio.get_running_loop()
            models = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: list_riva_asr_models_sync(server, use_ssl=use_ssl)),
                timeout=5.0,
            )
            return web.json_response({"ok": True})
        except asyncio.TimeoutError:
            return web.json_response({"ok": False, "error": "Connection timed out (5s)"})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)})

    async def handle_list_cameras(self, request: web.Request) -> web.Response:
        """List local USB video cameras (server). Used for Camera devices (Server USB) dropdown."""
        try:
            loop = asyncio.get_running_loop()
            cameras = await loop.run_in_executor(None, list_local_cameras)
            return web.json_response({"cameras": cameras})
        except Exception as e:
            logger.debug("List cameras error: %s", e)
            return web.json_response({"cameras": []})

    async def handle_list_audio_inputs(self, request: web.Request) -> web.Response:
        """List local audio input devices (microphones). Used for Microphone (Server USB) dropdown."""
        try:
            loop = asyncio.get_running_loop()
            devices = await loop.run_in_executor(None, list_local_audio_inputs)
            return web.json_response({"devices": devices})
        except Exception as e:
            logger.debug("List audio inputs error: %s", e)
            return web.json_response({"devices": []})

    async def handle_list_audio_outputs(self, request: web.Request) -> web.Response:
        """List local audio output devices (speakers). Used for Speaker (Server USB) dropdown."""
        try:
            loop = asyncio.get_running_loop()
            devices = await loop.run_in_executor(None, list_local_audio_outputs)
            return web.json_response({"devices": devices})
        except Exception as e:
            logger.debug("List audio outputs error: %s", e)
            return web.json_response({"devices": []})

    def _get_videos_dir(self) -> Path:
        """Return the videos/ folder (repo root or next to session_dir)."""
        candidates = [
            Path(__file__).resolve().parents[3] / "videos",  # repo root
            self.session_dir.parent / "videos",
        ]
        for d in candidates:
            if d.is_dir():
                return d
        return candidates[0]

    async def handle_list_videos(self, request: web.Request) -> web.Response:
        """GET /api/videos/list — list MP4 files in the videos/ folder."""
        videos_dir = self._get_videos_dir()
        if not videos_dir.is_dir():
            return web.json_response({"videos": [], "videos_dir": str(videos_dir)})
        result = []
        for f in sorted(videos_dir.iterdir()):
            if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".avi") and f.is_file():
                result.append({
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
        return web.json_response({"videos": result, "videos_dir": str(videos_dir)})

    async def handle_serve_video(self, request: web.Request) -> web.Response:
        """GET /api/videos/file?name=x.mp4 — serve an MP4 with range request support."""
        name = (request.query.get("name") or "").strip()
        if not name or "/" in name or "\\" in name or ".." in name:
            return web.Response(text="Invalid filename", status=400)
        videos_dir = self._get_videos_dir()
        file_path = videos_dir / name
        if not file_path.is_file():
            return web.Response(text="Video not found", status=404)

        file_size = file_path.stat().st_size
        suffix = file_path.suffix.lower()
        mime = {"mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska", "avi": "video/x-msvideo"}.get(
            suffix.lstrip("."), "video/mp4"
        )

        range_header = request.headers.get("Range")
        if range_header:
            try:
                range_spec = range_header.replace("bytes=", "")
                start_str, end_str = range_spec.split("-", 1)
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                with open(file_path, "rb") as f:
                    f.seek(start)
                    data = f.read(length)
                return web.Response(
                    body=data,
                    status=206,
                    headers={
                        "Content-Type": mime,
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Content-Length": str(length),
                        "Accept-Ranges": "bytes",
                    },
                )
            except (ValueError, IndexError):
                pass

        return web.FileResponse(file_path, headers={"Content-Type": mime, "Accept-Ranges": "bytes"})

    async def handle_camera_stream(self, request: web.Request) -> web.Response:
        """MJPEG stream from a server USB camera. device= query: /dev/video0 or empty for first camera.
        Requires opencv-python-headless and a camera attached to the server. When unavailable, returns 503."""
        try:
            import cv2
        except ImportError:
            logger.warning(
                "Camera stream unavailable: opencv-python-headless not installed. "
                "Install with: pip install opencv-python-headless (or pip install -e \".[camera]\")"
            )
            return web.Response(
                text="Camera stream requires opencv-python-headless. pip install opencv-python-headless",
                status=503,
                content_type="text/plain",
            )
        device = (request.query.get("device") or "").strip()
        if not device:
            cameras = list_local_cameras()
            if not cameras:
                logger.warning("Camera stream: no local cameras listed.")
                return web.Response(
                    text="No camera attached to server. Attach a USB camera and list devices in Configuration > Devices.",
                    status=503,
                    content_type="text/plain",
                )
            device = cameras[0]["id"]
        logger.info("[Camera MJPEG] Stream starting for device=%s", device)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            logger.warning("Camera stream: could not open device %s (in use or no permission?).", device)
            return web.Response(
                text="Could not open camera %s on server (is it in use or no permission?)." % device,
                status=503,
                content_type="text/plain",
            )
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=--frame"},
        )
        await response.prepare(request)

        # Get FrameBroker for VLM access
        frame_broker = None
        try:
            from multi_modal_ai_studio.backends.vision.frame_broker import get_frame_broker
            frame_broker = get_frame_broker()
            logger.info("[Camera MJPEG] FrameBroker connected for VLM frame storage")
        except ImportError:
            logger.debug("[Camera MJPEG] FrameBroker not available")
        
        frame_count = 0
        try:
            while True:
                ret, frame = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: cap.read()
                )
                if not ret or frame is None:
                    break
                
                # Store frame in FrameBroker for VLM (every 3rd frame = ~10fps)
                frame_count += 1
                if frame_broker and frame_count % 3 == 0:
                    try:
                        frame_copy = frame.copy()
                        await asyncio.get_running_loop().run_in_executor(
                            None, lambda f=frame_copy: frame_broker.store_frame(f, jpeg_quality=70)
                        )
                    except Exception as e:
                        logger.debug("[Camera MJPEG] FrameBroker store failed: %s", e)
                
                _, jpeg = await asyncio.get_running_loop().run_in_executor(
                    None, lambda f=frame: cv2.imencode(".jpg", f)
                )
                jpeg_bytes = jpeg.tobytes()
                await response.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg_bytes)
                    + jpeg_bytes
                    + b"\r\n"
                )
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            logger.info("[Camera MJPEG] Stream ended for device=%s", device)
            await asyncio.get_running_loop().run_in_executor(None, cap.release)
        return response

    def load_all_sessions(self) -> List[Dict[str, Any]]:
        """Load all session JSON files from session directory.

        Returns:
            List of session dictionaries
        """
        sessions = []
        base = self._get_effective_session_dir()

        if not base.exists():
            logger.warning(f"Session directory not found: {base}")
            return sessions

        for session_file in base.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                logger.error(f"Error loading {session_file}: {e}")

        # Sort by created_at (newest first)
        sessions.sort(key=lambda s: s.get('created_at', ''), reverse=True)

        logger.info(f"Loaded {len(sessions)} sessions from {base}")
        return sessions

    async def start(self):
        """Start the web server."""
        runner = web.AppRunner(
            self.app,
            access_log_class=_QuietAccessLogger,
            access_log_format='%a %t "%r" %s %b',
        )
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port, ssl_context=self.ssl_context)
        await site.start()

        protocol = "https" if self.ssl_context else "http"
        logger.info("🚀 Multi-modal AI Studio WebUI")
        logger.info(f"📡 Server running at {protocol}://{self.host}:{self.port}")
        logger.info(f"📂 Session directory: {self.session_dir.absolute()}")
        if self.ssl_context:
            logger.info("⚠️  Your browser will show a security warning (self-signed certificate)")
            logger.info("    Click 'Advanced' → 'Proceed to localhost' (or 'Accept Risk')")
        logger.info("Press Ctrl+C to stop")

        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await runner.cleanup()


def main():
    """Main entry point for development server (SSL logic same as Live RIVA WebUI)."""
    import argparse
    import ssl

    config_dir = get_app_config_dir()
    default_cert = str(config_dir / "cert.pem")
    default_key = str(config_dir / "key.pem")

    parser = argparse.ArgumentParser(description="Multi-modal AI Studio WebUI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("sessions"),
        help="Directory to load/save session JSON files (default: sessions). Use e.g. mock_sessions for sample data.",
    )
    parser.add_argument(
        "--ssl-cert",
        default=None,
        help=f"Path to SSL certificate (default: {default_cert}, auto-generated if missing)",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,
        help=f"Path to SSL private key (default: {default_key}, auto-generated if missing)",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable SSL/HTTPS (not recommended - getUserMedia/WebRTC require HTTPS)",
    )

    args = parser.parse_args()

    # Default cert/key to config dir
    if args.ssl_cert is None:
        args.ssl_cert = default_cert
    if args.ssl_key is None:
        args.ssl_key = default_key

    ssl_context = None
    if not args.no_ssl:
        if not os.path.exists(args.ssl_cert) or not os.path.exists(args.ssl_key):
            if not generate_self_signed_cert(args.ssl_cert, args.ssl_key):
                logger.error("❌ Cannot start server without SSL certificates")
                logger.error("🔧 Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)")
                logger.error("   Or run with --no-ssl if you don't need camera/mic (not recommended)")
                sys.exit(1)
        if os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
        else:
            logger.error("❌ SSL certificates missing after generation")
            sys.exit(1)
    else:
        logger.warning("⚠️  SSL disabled with --no-ssl (camera/mic may not work without HTTPS)")

    server = WebUIServer(
        host=args.host,
        port=args.port,
        session_dir=args.session_dir,
        ssl_context=ssl_context,
    )

    asyncio.run(server.start())


if __name__ == '__main__':
    main()
