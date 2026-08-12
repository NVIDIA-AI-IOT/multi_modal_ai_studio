# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from typing import Any, AsyncIterator, Dict, List, Literal, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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

    # Includes session state, input speech/transcription, response audio,
    # response transcription, completion, warning, and error events.
    kind: str
    # For kind=="audio":
    audio: Optional[bytes] = None
    sample_rate: int = REALTIME_SAMPLE_RATE
    # For kind in ("transcript_delta", "transcript_completed"):
    text: Optional[str] = None
    is_final: bool = False
    # For kind=="error":
    message: Optional[str] = None
    # Correlates transcription events belonging to the same conversation item.
    item_id: Optional[str] = None
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
        # None=omit (server default), sentinel=null, dict=explicit config.
        turn_detection: Any = None,
        input_audio_sample_rate: int = REALTIME_SAMPLE_RATE,
        output_audio_sample_rate: int = REALTIME_SAMPLE_RATE,
        session_type: Literal["realtime", "transcription"] = "realtime",
        api_style: Literal["openai-ga", "openai-beta"] = "openai-ga",
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
        self.session_type = session_type
        self.api_style = api_style
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
        """Send an OpenAI GA or legacy beta-compatible ``session.update``."""
        if self.api_style == "openai-beta":
            # Speaches 0.8.x and older Realtime-compatible servers implement
            # the preview schema. Keep this isolated from the GA wire format.
            # PCM16 is the preview schema default. Omit its optional format
            # fields because several compatible providers expose them as
            # read-only session properties.
            session: Dict[str, Any] = {}
            if self.session_type == "realtime":
                session.update({
                    "modalities": ["text", "audio"],
                    "instructions": self.instructions,
                    "voice": self.voice,
                })
            if self.input_audio_transcription is not None:
                session["input_audio_transcription"] = self.input_audio_transcription
            if self.turn_detection is DISABLE_TURN_DETECTION:
                session["turn_detection"] = None
            elif isinstance(self.turn_detection, dict):
                session["turn_detection"] = self.turn_detection
        else:
            session = {"type": self.session_type}
            if self.session_type == "realtime":
                session["instructions"] = self.instructions
            audio_input: Dict[str, Any] = {
                "format": {
                    "type": "audio/pcm",
                    "rate": self.input_audio_sample_rate,
                }
            }
            if self.input_audio_transcription is not None:
                audio_input["transcription"] = self.input_audio_transcription
            if self.turn_detection is DISABLE_TURN_DETECTION:
                audio_input["turn_detection"] = None
            elif isinstance(self.turn_detection, dict):
                audio_input["turn_detection"] = self.turn_detection
            session["audio"] = {"input": audio_input}
            if self.session_type == "realtime":
                session["audio"]["output"] = {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.output_audio_sample_rate,
                    },
                    "voice": self.voice,
                }
        msg = {"type": "session.update", "session": session}
        await self._send_json(msg)
        logger.info("Sent session.update (session keys: %s)", list(session.keys()))

    async def _send_json(self, obj: Dict[str, Any]) -> None:
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Realtime WebSocket not connected")
        await self._ws.send_str(json.dumps(obj))

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Append PCM bytes in the configured session input format."""
        if self._ws is None or self._ws.closed:
            raise RuntimeError("Realtime WebSocket not connected")
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        await self._ws.send_str(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))

    async def commit_audio(self) -> None:
        """Commit the input audio buffer for provider-side processing."""
        await self._send_json({"type": "input_audio_buffer.commit"})

    async def send_text(self, text: str, *, role: str = "user") -> None:
        """Add a text conversation item to a full Realtime session."""
        await self._send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": text}],
            },
        })

    async def create_response(
        self,
        *,
        modalities: Optional[List[str]] = None,
        instructions: Optional[str] = None,
    ) -> None:
        """Request a response, including streaming audio when requested.

        This is a provider-neutral response-audio boundary. It does not imply
        exact-text TTS semantics; exact synthesis remains ``/v1/audio/speech``.
        """
        response: Dict[str, Any] = {}
        if modalities:
            response[
                "modalities" if self.api_style == "openai-beta" else "output_modalities"
            ] = modalities
        if instructions:
            response["instructions"] = instructions
        await self._send_json({"type": "response.create", "response": response})

    async def cancel_response(self, response_id: Optional[str] = None) -> None:
        """Cancel the active response when supported by the provider."""
        message: Dict[str, Any] = {"type": "response.cancel"}
        if response_id:
            message["response_id"] = response_id
        await self._send_json(message)

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
                            # Speaches 0.8.x requires the preview VAD object to
                            # contain prefix_padding_ms, then reports that same
                            # property as read-only while applying the rest of
                            # the update. Treat that self-contradictory notice
                            # as a session warning, not an ASR failure.
                            if (
                                self.api_style == "openai-beta"
                                and "session.turn_detection.prefix_padding_ms"
                                in message
                                and "not supported" in message
                            ):
                                logger.warning("Realtime session warning: %s", message)
                                await self._event_queue.put(
                                    RealtimeEvent(
                                        kind="session_warning",
                                        message=message,
                                        raw=data,
                                    )
                                )
                            else:
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
                        elif event_type == "input_audio_buffer.speech_started":
                            await self._event_queue.put(
                                RealtimeEvent(
                                    kind="speech_started",
                                    item_id=data.get("item_id"),
                                    raw=data,
                                )
                            )
                        elif event_type == "input_audio_buffer.speech_stopped":
                            await self._event_queue.put(
                                RealtimeEvent(
                                    kind="speech_stopped",
                                    item_id=data.get("item_id"),
                                    raw=data,
                                )
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
                                            item_id=data.get("item_id"),
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
                        elif event_type in ("response.output_audio.done", "response.audio.done"):
                            await self._event_queue.put(
                                RealtimeEvent(
                                    kind="audio_done",
                                    item_id=data.get("item_id"),
                                    raw=data,
                                )
                            )
                        elif event_type == "conversation.item.input_audio_transcription.delta":
                            raw_delta = data.get("delta")
                            delta = (
                                raw_delta
                                if isinstance(raw_delta, str)
                                else str(raw_delta or "")
                            )
                            if delta.strip():
                                await self._event_queue.put(
                                    RealtimeEvent(
                                        kind="transcript_delta",
                                        text=delta,
                                        is_final=False,
                                        item_id=data.get("item_id"),
                                        raw=data,
                                    )
                                )
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            raw_transcript = data.get("transcript")
                            transcript = (
                                raw_transcript
                                if isinstance(raw_transcript, str)
                                else str(raw_transcript or "")
                            ).strip()
                            if transcript:
                                await self._event_queue.put(
                                    RealtimeEvent(
                                        kind="transcript_completed",
                                        text=transcript,
                                        is_final=True,
                                        item_id=data.get("item_id"),
                                        raw=data,
                                    )
                                )
                        elif event_type == "response.output_audio_transcript.delta":
                            raw_delta = data.get("delta")
                            delta = (
                                raw_delta
                                if isinstance(raw_delta, str)
                                else str(raw_delta or "")
                            )
                            if delta.strip():
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
                                    if (
                                        isinstance(part, dict)
                                        and part.get("type") == "output_audio"
                                    ):
                                        raw_t = part.get("transcript")
                                        transcript = (
                                            raw_t
                                            if isinstance(raw_t, str)
                                            else str(raw_t or "")
                                        ).strip()
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
