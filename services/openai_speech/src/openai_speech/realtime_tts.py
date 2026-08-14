# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime-compatible exact-text TTS for NVIDIA Magpie."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from openai_speech.engines import MAGPIE_SAMPLE_RATE, TTSEngine

_AUDIO_FRAME_MS = 20


@dataclass
class RealtimeTTSSessionConfig:
    """Mutable settings for one exact-text TTS session."""

    model: str
    voice: str = "Sofia"
    language: str = "en"
    output_rate: int = MAGPIE_SAMPLE_RATE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "realtime",
            "model": self.model,
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.output_rate,
                    },
                    "voice": self.voice,
                }
            },
        }


@dataclass
class _ResponseState:
    response_id: str
    item_id: str
    text: str
    language: str
    cancelled: asyncio.Event
    done_sent: bool = False


class RealtimeTTSConnection:
    """Serve exact input text as Magpie audio over Realtime wire events.

    The protocol surface follows the OpenAI Realtime GA event names. Exact-text
    synthesis is an NVIDIA service semantic: the most recent user ``input_text``
    item is spoken verbatim when ``response.create`` arrives.
    """

    def __init__(self, websocket: WebSocket, engine: TTSEngine, model: str):
        self.websocket = websocket
        self.engine = engine
        self.session_id = f"sess_{secrets.token_urlsafe(12)}"
        self.config = RealtimeTTSSessionConfig(model=model)
        self._pending_text: Optional[str] = None
        self._pending_language: Optional[str] = None
        self._pending_item_id: Optional[str] = None
        self._active_response_id: Optional[str] = None
        self._responses: Dict[str, _ResponseState] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_text(json.dumps(payload))

    async def error(self, message: str, *, code: str = "invalid_request_error") -> None:
        await self.send(
            {
                "type": "error",
                "error": {"type": code, "message": message},
            }
        )

    async def run(self) -> None:
        await self.websocket.accept()
        await self.send(
            {
                "type": "session.created",
                "session": {"id": self.session_id, **self.config.as_dict()},
            }
        )
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    await self.error("Realtime TTS accepts JSON text events only")
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await self.error("Invalid JSON")
                    continue
                await self._handle(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await self.close()

    async def _handle(self, payload: Dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "session.update":
            await self._update_session(payload.get("session") or {})
        elif event_type == "conversation.item.create":
            await self._create_item(payload.get("item") or {})
        elif event_type == "response.create":
            await self._create_response(payload.get("response") or {})
        elif event_type == "response.cancel":
            await self._cancel_response(payload.get("response_id"))
        elif event_type == "conversation.item.truncate":
            await self.send(
                {
                    "type": "conversation.item.truncated",
                    "item_id": payload.get("item_id"),
                    "content_index": payload.get("content_index", 0),
                    "audio_end_ms": payload.get("audio_end_ms", 0),
                }
            )
        else:
            await self.error(f"Unsupported client event '{event_type}'")

    async def _update_session(self, session: Dict[str, Any]) -> None:
        if session.get("type", "realtime") != "realtime":
            await self.error("NVIDIA TTS service supports realtime sessions only")
            return
        requested_model = session.get("model", self.config.model)
        if requested_model != self.config.model:
            await self.error(f"Unknown model '{requested_model}'")
            return

        audio = session.get("audio") or {}
        output = audio.get("output") or {}
        audio_format = output.get("format") or {}
        if audio_format:
            if audio_format.get("type") != "audio/pcm":
                await self.error("Only audio/pcm output is supported")
                return
            rate = int(audio_format.get("rate", MAGPIE_SAMPLE_RATE))
            if rate != MAGPIE_SAMPLE_RATE:
                await self.error(f"Magpie output rate must be {MAGPIE_SAMPLE_RATE} Hz")
                return
            self.config.output_rate = rate
        voice = output.get("voice") or session.get("voice")
        if voice:
            self.config.voice = str(voice)
        await self.send(
            {
                "type": "session.updated",
                "session": {"id": self.session_id, **self.config.as_dict()},
            }
        )

    async def _create_item(self, item: Dict[str, Any]) -> None:
        if item.get("type", "message") != "message" or item.get("role") != "user":
            await self.error("Realtime TTS requires a user message item")
            return
        parts = item.get("content") or []
        text = "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "input_text"
        ).strip()
        if not text:
            await self.error("Realtime TTS message requires non-empty input_text")
            return
        metadata = item.get("metadata") or {}
        self._pending_text = text
        self._pending_language = str(metadata.get("language") or self.config.language)
        self._pending_item_id = str(item.get("id") or f"item_{secrets.token_urlsafe(12)}")
        await self.send(
            {
                "type": "conversation.item.created",
                "previous_item_id": None,
                "item": {
                    "id": self._pending_item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def _create_response(self, response: Dict[str, Any]) -> None:
        if self._pending_text is None or self._pending_item_id is None:
            await self.error("response.create requires a preceding input_text item")
            return
        modalities = response.get("output_modalities") or response.get("modalities") or ["audio"]
        if "audio" not in modalities:
            await self.error("NVIDIA TTS responses require the audio modality")
            return

        response_id = f"resp_{secrets.token_urlsafe(12)}"
        output_item_id = f"item_{secrets.token_urlsafe(12)}"
        state = _ResponseState(
            response_id=response_id,
            item_id=output_item_id,
            text=self._pending_text,
            language=self._pending_language or self.config.language,
            cancelled=asyncio.Event(),
        )
        self._pending_text = None
        self._pending_language = None
        self._pending_item_id = None
        self._responses[response_id] = state
        self._active_response_id = response_id
        await self.send(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "status": "in_progress",
                    "output": [],
                },
            }
        )
        task = asyncio.create_task(self._synthesize(state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _synthesize(self, state: _ResponseState) -> None:
        try:
            audio, _ = await self.engine.synthesize(
                state.text,
                self.config.voice,
                state.language,
                "pcm",
            )
            if state.cancelled.is_set():
                return
            await self.send(
                {
                    "type": "response.output_item.added",
                    "response_id": state.response_id,
                    "output_index": 0,
                    "item": {
                        "id": state.item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }
            )
            await self.send(
                {
                    "type": "response.content_part.added",
                    "response_id": state.response_id,
                    "item_id": state.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_audio", "transcript": state.text},
                }
            )
            frame_bytes = max(
                2,
                int(self.config.output_rate * 2 * _AUDIO_FRAME_MS / 1000) // 2 * 2,
            )
            for offset in range(0, len(audio), frame_bytes):
                if state.cancelled.is_set():
                    return
                await self.send(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": state.response_id,
                        "item_id": state.item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": base64.b64encode(audio[offset : offset + frame_bytes]).decode(
                            "ascii"
                        ),
                    }
                )
            await self.send(
                {
                    "type": "response.output_audio.done",
                    "response_id": state.response_id,
                    "item_id": state.item_id,
                    "output_index": 0,
                    "content_index": 0,
                }
            )
            await self.send(
                {
                    "type": "response.output_item.done",
                    "response_id": state.response_id,
                    "output_index": 0,
                    "item": {
                        "id": state.item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_audio", "transcript": state.text}],
                    },
                }
            )
            await self._send_response_done(state, "completed")
        except Exception as exc:
            if not state.cancelled.is_set():
                await self.error(f"TTS model inference failed: {exc}", code="server_error")
                await self._send_response_done(state, "failed")
        finally:
            if self._active_response_id == state.response_id:
                self._active_response_id = None
            self._responses.pop(state.response_id, None)

    async def _cancel_response(self, response_id: Optional[str]) -> None:
        target = response_id or self._active_response_id
        state = self._responses.get(target or "")
        if state is None:
            await self.error("No active response to cancel")
            return
        state.cancelled.set()
        await self._send_response_done(state, "cancelled")
        if self._active_response_id == state.response_id:
            self._active_response_id = None

    async def _send_response_done(self, state: _ResponseState, status: str) -> None:
        if state.done_sent:
            return
        state.done_sent = True
        await self.send(
            {
                "type": "response.done",
                "response": {
                    "id": state.response_id,
                    "status": status,
                    "output": [],
                },
            }
        )

    async def close(self) -> None:
        for state in self._responses.values():
            state.cancelled.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
