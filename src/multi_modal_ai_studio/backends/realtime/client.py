"""
OpenAI Realtime API WebSocket client.

Connects to an OpenAI-compatible Realtime WebSocket (e.g. wss://api.openai.com/v1/realtime),
sends session config and input audio, and yields events: response audio chunks,
transcription (partial/final), response.done, and errors.

Audio format: API expects pcm16, 24 kHz, mono, little-endian. Caller must resample
if pipeline uses a different rate (e.g. 16 kHz).
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

import aiohttp

logger = logging.getLogger(__name__)

# Default sample rate for Realtime API (pcm16).
REALTIME_SAMPLE_RATE = 24000

# Pass as turn_detection= to send session turn_detection: null and disable server VAD
# (so only explicit input_audio_buffer.commit triggers processing).
DISABLE_TURN_DETECTION = object()


@dataclass
class RealtimeEvent:
    """One event from the Realtime stream for the pipeline to handle."""

    kind: str  # "audio" | "transcript_delta" | "transcript_completed" | "response_done" | "error"
    # For kind=="audio":
    audio: Optional[bytes] = None
    sample_rate: int = REALTIME_SAMPLE_RATE
    # For kind in ("transcript_delta", "transcript_completed"):
    text: Optional[str] = None
    is_final: bool = False
    # For kind=="error":
    message: Optional[str] = None
    # Optional raw payload for debugging
    raw: Optional[Dict[str, Any]] = None


class OpenAIRealtimeClient:
    """
    OpenAI-compatible Realtime WebSocket client.

    - connect(): open WebSocket, send session.update.
    - send_audio(pcm_bytes): append PCM to input buffer (base64).
    - events(): async iterator of RealtimeEvent (audio, transcript, response_done, error).
    - disconnect(): close WebSocket.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        model: str = "gpt-realtime",
        instructions: Optional[str] = None,
        voice: str = "alloy",
        input_audio_format: str = "pcm16",
        output_audio_format: str = "pcm16",
        input_audio_transcription: Optional[Dict[str, Any]] = None,
        turn_detection: Any = None,  # None=omit (server default), DISABLE_TURN_DETECTION=null, dict=config
        input_audio_sample_rate: int = REALTIME_SAMPLE_RATE,
        output_audio_sample_rate: int = REALTIME_SAMPLE_RATE,
        log_all_events: bool = False,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.instructions = instructions or ""
        self.voice = voice
        self.input_audio_format = input_audio_format
        self.output_audio_format = output_audio_format
        self.input_audio_transcription = input_audio_transcription
        self.turn_detection = turn_detection
        self.input_audio_sample_rate = input_audio_sample_rate
        self.output_audio_sample_rate = output_audio_sample_rate
        self._log_all_events = log_all_events

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False

    def _connect_url(self) -> str:
        """Build WebSocket URL with required model query parameter."""
        parsed = urlparse(self.url)
        q = parse_qs(parsed.query, keep_blank_values=True)
        q.setdefault("model", [self.model])
        new_query = urlencode(q, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    async def connect(self) -> None:
        """Connect to the Realtime WebSocket and send session.update."""
        if self._ws is not None:
            return
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._session = aiohttp.ClientSession()
        connect_url = self._connect_url()
        self._ws = await self._session.ws_connect(
            connect_url,
            headers=headers or None,
            heartbeat=30.0,
        )
        self._closed = False
        # Send session.update so the server configures model, voice, modalities, audio format.
        await self._send_session_update()
        # Start background receiver that pushes parsed events into _event_queue.
        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.info("Realtime WebSocket connected to %s", connect_url)

    async def _send_session_update(self) -> None:
        """Send session.update. Only include fields the API accepts."""
        # Many session fields are not accepted in session.update. Model is in URL.
        # Do not send turn_detection: production API returns "Unknown parameter: 'session.turn_detection'".
        session: Dict[str, Any] = {
            "type": "realtime",
            "instructions": self.instructions,
        }
        if self.input_audio_transcription is not None:
            session["input_audio_transcription"] = self.input_audio_transcription
        msg = {"type": "session.update", "session": session}
        await self._send_json(msg)
        logger.info("Sent session.update (session keys: %s)", list(session.keys()))

    async def _send_json(self, obj: Dict[str, Any]) -> None:
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Realtime WebSocket not connected")
        await self._ws.send_str(json.dumps(obj))

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Append PCM bytes to the input audio buffer. Must be in session input format (e.g. pcm16 24kHz)."""
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Realtime WebSocket not connected")
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        await self._ws.send_str(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))

    async def commit_audio(self) -> None:
        """Commit the input audio buffer so the server processes it (creates user message, triggers response)."""
        await self._send_json({"type": "input_audio_buffer.commit"})

    async def _receive_loop(self) -> None:
        """Read WebSocket messages and push RealtimeEvent into _event_queue."""
        ws = self._ws
        if ws is None:
            return
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        event_type = data.get("type") or ""
                        if self._log_all_events:
                            logger.info("Realtime server event: %s", event_type)
                        if event_type == "session.created":
                            logger.debug("Realtime session.created")
                        elif event_type == "session.updated":
                            logger.debug("Realtime session.updated")
                            await self._event_queue.put(
                                RealtimeEvent(kind="session_ready", raw=data)
                            )
                        elif event_type == "error":
                            err = data.get("error", {})
                            message = err.get("message", str(data))
                            await self._event_queue.put(
                                RealtimeEvent(kind="error", message=message, raw=data)
                            )
                        elif event_type == "response.done":
                            if self._log_all_events:
                                status = data.get("response", {}).get("status", data.get("status"))
                                logger.info("Realtime response.done: status=%s", status)
                            await self._event_queue.put(
                                RealtimeEvent(kind="response_done", raw=data)
                            )
                        elif event_type in ("response.output_audio.delta", "response.audio.delta"):
                            delta_b64 = data.get("delta")
                            if delta_b64:
                                try:
                                    audio_bytes = base64.b64decode(delta_b64)
                                    await self._event_queue.put(
                                        RealtimeEvent(
                                            kind="audio",
                                            audio=audio_bytes,
                                            sample_rate=self.output_audio_sample_rate,
                                            raw=data,
                                        )
                                    )
                                except Exception as e:
                                    await self._event_queue.put(
                                        RealtimeEvent(
                                            kind="error",
                                            message=f"Failed to decode output_audio.delta: {e}",
                                            raw=data,
                                        )
                                    )
                        elif event_type == "conversation.item.input_audio_transcription.delta":
                            raw_delta = data.get("delta")
                            delta = (raw_delta if isinstance(raw_delta, str) else str(raw_delta or "")).strip()
                            if delta:
                                await self._event_queue.put(
                                    RealtimeEvent(
                                        kind="transcript_delta",
                                        text=delta,
                                        is_final=False,
                                        raw=data,
                                    )
                                )
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            raw_transcript = data.get("transcript")
                            transcript = (raw_transcript if isinstance(raw_transcript, str) else str(raw_transcript or "")).strip()
                            if transcript:
                                await self._event_queue.put(
                                    RealtimeEvent(
                                        kind="transcript_completed",
                                        text=transcript,
                                        is_final=True,
                                        raw=data,
                                    )
                                )
                        elif event_type == "response.output_audio_transcript.delta":
                            raw_delta = data.get("delta")
                            delta = (raw_delta if isinstance(raw_delta, str) else str(raw_delta or "")).strip()
                            if delta:
                                await self._event_queue.put(
                                    RealtimeEvent(
                                        kind="output_transcript_delta",
                                        text=delta,
                                        is_final=False,
                                        raw=data,
                                    )
                                )
                        elif event_type == "conversation.item.done":
                            # Assistant reply transcript is in the item content; do NOT emit as
                            # transcript_completed (that is for user ASR only). Emit as
                            # output_transcript_completed so pipeline can use it for chat/display
                            # but not for the speech/ASR lane.
                            item = data.get("item") or {}
                            if item.get("role") == "assistant":
                                for part in (item.get("content") or []):
                                    if isinstance(part, dict) and part.get("type") == "output_audio":
                                        raw_t = part.get("transcript")
                                        transcript = (raw_t if isinstance(raw_t, str) else str(raw_t or "")).strip()
                                        if transcript:
                                            await self._event_queue.put(
                                                RealtimeEvent(
                                                    kind="output_transcript_completed",
                                                    text=transcript,
                                                    is_final=True,
                                                    raw=data,
                                                )
                                            )
                                        break
                        else:
                            # Skip logging structural/response events we don't need to handle
                            _skip_log = (
                                "conversation.item.added",
                                "conversation.item.done",
                                "response.created",
                                "response.output_item.added",
                                "response.output_item.done",
                                "response.content_part.added",
                                "response.content_part.done",
                                "response.output_audio.done",
                                "response.output_audio_transcript.done",
                                "response.output_text.delta",
                                "response.output_text.done",
                            )
                            if event_type and event_type not in _skip_log:
                                logger.info("Realtime unhandled server event: %s", event_type)
                    except json.JSONDecodeError as e:
                        await self._event_queue.put(
                            RealtimeEvent(kind="error", message=f"Invalid JSON: {e}")
                        )
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = getattr(ws, "exception", lambda: None)()
                    await self._event_queue.put(
                        RealtimeEvent(kind="error", message=str(exc) if exc else "WebSocket error")
                    )
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Realtime receive_loop error: %s", e)
            await self._event_queue.put(RealtimeEvent(kind="error", message=str(e)))
        finally:
            await self._event_queue.put(None)

    async def events(self) -> AsyncIterator[Optional[RealtimeEvent]]:
        """Async iterator of RealtimeEvent. Yields None when the stream ends."""
        while True:
            ev = await self._event_queue.get()
            yield ev
            if ev is None:
                break

    async def disconnect(self) -> None:
        """Close the WebSocket and stop the receive task."""
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            if not self._ws.closed:
                await self._ws.close()
            self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        logger.info("Realtime WebSocket disconnected")
