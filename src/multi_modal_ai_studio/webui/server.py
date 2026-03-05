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

from aiohttp import web
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
    ):
        """Initialize web server.

        Args:
            host: Host address to bind to
            port: Port to listen on
            session_dir: Directory containing session JSON files
            ssl_context: Optional ssl.SSLContext for HTTPS (e.g. self-signed cert)
        """
        self.host = host
        self.port = port
        self.session_dir = session_dir or Path("sessions")
        self._session_dir_override: Optional[str] = None  # "sessions" | "mock_sessions" | None
        self.ssl_context = ssl_context
        self.app = web.Application()
        self.app["session_dir"] = self.session_dir
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
        self.app.router.add_get('/api/asr/models', self.handle_asr_models)
        self.app.router.add_get('/api/tts/voices', self.handle_tts_voices)
        self.app.router.add_get('/api/health/llm', self.handle_health_llm)
        self.app.router.add_get('/api/health/riva', self.handle_health_riva)
        self.app.router.add_get('/api/devices/cameras', self.handle_list_cameras)
        self.app.router.add_get('/api/devices/audio-inputs', self.handle_list_audio_inputs)
        self.app.router.add_get('/api/devices/audio-outputs', self.handle_list_audio_outputs)
        self.app.router.add_get('/api/camera/stream', self.handle_camera_stream)
        self.app.router.add_get('/api/app/session-dir', self.handle_get_session_dir)
        self.app.router.add_patch('/api/app/session-dir', self.handle_patch_session_dir)
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

        try:
            while True:
                ret, frame = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: cap.read()
                )
                if not ret or frame is None:
                    break
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
