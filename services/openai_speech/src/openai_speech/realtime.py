# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime GA transcription protocol for NVIDIA ASR models."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import json
import math
import secrets
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from openai_speech.engines import ASREngine, ASRRealtimeStream


INPUT_SAMPLE_RATE = 24000
MODEL_SAMPLE_RATE = 16000


def _resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linearly resample one aligned mono PCM16 chunk."""
    pcm = pcm[: len(pcm) - (len(pcm) % 2)]
    if not pcm or from_rate == to_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2")
    count = int(round(len(samples) * to_rate / from_rate))
    output = np.interp(
        np.linspace(0, len(samples) - 1, count),
        np.arange(len(samples)),
        samples.astype(np.float64),
    ).astype("<i2")
    return output.tobytes()


def _normalized_rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2")
    if samples.size == 0:
        return 0.0
    values = samples.astype(np.float64)
    return math.sqrt(float(np.mean(values * values))) / 32768.0


def _rms_threshold(setting: float) -> float:
    return 0.005 * (10 ** max(0.0, min(float(setting), 1.0)))


@dataclass
class RealtimeSessionConfig:
    model: str
    language: str = "auto"
    input_rate: int = INPUT_SAMPLE_RATE
    threshold: float = 0.4
    prefix_padding_ms: int = 500
    silence_duration_ms: int = 700
    server_vad: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.input_rate},
                    "transcription": {
                        "model": self.model,
                        "language": self.language,
                    },
                    "turn_detection": (
                        {
                            "type": "server_vad",
                            "threshold": self.threshold,
                            "prefix_padding_ms": self.prefix_padding_ms,
                            "silence_duration_ms": self.silence_duration_ms,
                        }
                        if self.server_vad
                        else None
                    ),
                }
            },
        }


class RealtimeTranscriptionConnection:
    """State machine for one Realtime transcription WebSocket."""

    def __init__(self, websocket: WebSocket, engine: ASREngine, model: str):
        self.websocket = websocket
        self.engine = engine
        self.session_id = f"sess_{secrets.token_urlsafe(12)}"
        self.config = RealtimeSessionConfig(model=model)
        self._active_item: Optional[str] = None
        self._active_stream: Optional[ASRRealtimeStream] = None
        self._silence_ms = 0.0
        self._audio_ms = 0.0
        self._prefix = bytearray()
        self._relay_tasks: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_text(json.dumps(payload))

    async def error(self, message: str, *, code: str = "invalid_request_error") -> None:
        await self.send({
            "type": "error",
            "error": {"type": code, "message": message},
        })

    async def run(self) -> None:
        await self.websocket.accept()
        await self.send({
            "type": "session.created",
            "session": {"id": self.session_id, **self.config.as_dict()},
        })
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    await self.error("Realtime audio must use JSON input_audio_buffer events")
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
        elif event_type == "input_audio_buffer.append":
            try:
                pcm = base64.b64decode(payload.get("audio", ""), validate=True)
            except (binascii.Error, ValueError):
                await self.error("input_audio_buffer.append contains invalid base64")
                return
            await self.feed_audio(pcm)
        elif event_type == "input_audio_buffer.commit":
            await self.commit()
        elif event_type == "input_audio_buffer.clear":
            self._prefix.clear()
            await self.send({"type": "input_audio_buffer.cleared"})
        else:
            await self.error(f"Unsupported client event '{event_type}'")

    async def _update_session(self, session: Dict[str, Any]) -> None:
        if self._active_stream is not None:
            await self.error("session.update is not allowed during an active utterance")
            return
        if session.get("type", "transcription") != "transcription":
            await self.error("NVIDIA ASR service supports transcription sessions only")
            return
        audio = session.get("audio") or {}
        audio_input = audio.get("input") or {}
        audio_format = audio_input.get("format") or {}
        if audio_format:
            if audio_format.get("type") != "audio/pcm":
                await self.error("Only audio/pcm input is supported")
                return
            rate = int(audio_format.get("rate", INPUT_SAMPLE_RATE))
            if rate not in {16000, 24000}:
                await self.error("PCM input rate must be 16000 or 24000 Hz")
                return
            self.config.input_rate = rate
        transcription = audio_input.get("transcription") or {}
        model = transcription.get("model", self.config.model)
        if model != self.config.model:
            await self.error(f"Unknown model '{model}'")
            return
        self.config.language = transcription.get("language") or "auto"
        turn_detection = audio_input.get("turn_detection", {})
        if turn_detection is None:
            self.config.server_vad = False
        else:
            if turn_detection.get("type", "server_vad") != "server_vad":
                await self.error("Only server_vad turn detection is supported")
                return
            self.config.server_vad = True
            self.config.threshold = max(
                0.0,
                min(float(turn_detection.get("threshold", 0.4)), 1.0),
            )
            self.config.prefix_padding_ms = max(
                0,
                min(int(turn_detection.get("prefix_padding_ms", 500)), 5000),
            )
            self.config.silence_duration_ms = max(
                100,
                min(int(turn_detection.get("silence_duration_ms", 700)), 5000),
            )
        await self.send({
            "type": "session.updated",
            "session": {"id": self.session_id, **self.config.as_dict()},
        })

    async def feed_audio(self, pcm: bytes) -> None:
        pcm16 = _resample_pcm16(
            pcm,
            self.config.input_rate,
            MODEL_SAMPLE_RATE,
        )
        if not pcm16:
            return
        duration_ms = len(pcm16) / (MODEL_SAMPLE_RATE * 2) * 1000.0
        self._audio_ms += duration_ms
        speech = _normalized_rms(pcm16) >= _rms_threshold(self.config.threshold)

        if not self.config.server_vad:
            if self._active_stream is None:
                await self._start_utterance(b"", current_chunk_ms=duration_ms)
            await self._active_stream.send_audio(pcm16)
            return

        if self._active_stream is None:
            if speech:
                prefix = bytes(self._prefix)
                self._prefix.clear()
                await self._start_utterance(prefix, current_chunk_ms=duration_ms)
                await self._active_stream.send_audio(pcm16)
            else:
                self._append_prefix(pcm16)
            return

        await self._active_stream.send_audio(pcm16)
        if speech:
            self._silence_ms = 0.0
            return
        self._silence_ms += duration_ms
        if self._silence_ms >= self.config.silence_duration_ms:
            await self._stop_utterance()

    def _append_prefix(self, pcm16: bytes) -> None:
        self._prefix.extend(pcm16)
        limit = int(MODEL_SAMPLE_RATE * 2 * self.config.prefix_padding_ms / 1000)
        if len(self._prefix) > limit:
            del self._prefix[:-limit]

    async def _start_utterance(
        self,
        prefix: bytes,
        *,
        current_chunk_ms: float,
    ) -> None:
        item_id = f"item_{secrets.token_urlsafe(12)}"
        stream = await self.engine.create_stream(self.config.language)
        self._active_item = item_id
        self._active_stream = stream
        self._silence_ms = 0.0
        if prefix:
            await stream.send_audio(prefix)
        await self.send({
            "type": "input_audio_buffer.speech_started",
            # _audio_ms already includes the chunk that crossed the VAD
            # threshold. Include that chunk and the retained prefix in the
            # reported utterance boundary.
            "audio_start_ms": max(
                0,
                round(
                    self._audio_ms
                    - current_chunk_ms
                    - len(prefix) / (MODEL_SAMPLE_RATE * 2) * 1000.0
                ),
            ),
            "item_id": item_id,
        })
        task = asyncio.create_task(self._relay(item_id, stream))
        self._relay_tasks.add(task)
        task.add_done_callback(self._relay_tasks.discard)

    async def _stop_utterance(self) -> None:
        item_id = self._active_item
        stream = self._active_stream
        if item_id is None or stream is None:
            return
        await self.send({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": round(self._audio_ms - self._silence_ms),
            "item_id": item_id,
        })
        await self.send({
            "type": "input_audio_buffer.committed",
            "item_id": item_id,
        })
        await stream.finish()
        self._active_item = None
        self._active_stream = None
        self._silence_ms = 0.0

    async def _relay(self, item_id: str, stream: ASRRealtimeStream) -> None:
        try:
            async for kind, text in stream.events():
                if kind == "delta" and text:
                    await self.send({
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": text,
                    })
                elif kind == "completed":
                    await self.send({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "content_index": 0,
                        "transcript": text,
                    })
                elif kind == "error":
                    await self.error(text, code="server_error")
        finally:
            await stream.close()

    async def commit(self) -> None:
        if self._active_stream is not None:
            await self._stop_utterance()

    async def close(self) -> None:
        if self._active_stream is not None:
            await self._active_stream.finish()
            self._active_stream = None
            self._active_item = None
        if self._relay_tasks:
            await asyncio.gather(*self._relay_tasks, return_exceptions=True)
