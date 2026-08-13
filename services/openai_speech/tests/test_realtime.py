# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realtime transcription API tests with a lightweight streaming engine."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field

import numpy as np
from fastapi.testclient import TestClient

from openai_speech.app import create_app
from openai_speech.config import ASR_MODEL, Settings


class FakeRealtimeStream:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.sent_partial = False
        self.closed = False

    async def send_audio(self, pcm16: bytes) -> None:
        self.audio.extend(pcm16)
        if pcm16 and not self.sent_partial:
            self.sent_partial = True
            await self.queue.put(("delta", "Hello"))

    async def finish(self) -> None:
        await self.queue.put(("delta", " world."))
        await self.queue.put(("completed", "Hello world."))

    async def events(self):
        while True:
            event = await self.queue.get()
            yield event
            if event[0] in {"completed", "error"}:
                return

    async def close(self) -> None:
        self.closed = True


@dataclass
class FakeRealtimeASR:
    streams: list[FakeRealtimeStream] = field(default_factory=list)

    async def load(self) -> None:
        return None

    async def transcribe(self, audio: bytes, filename: str, language: str) -> str:
        return "REST transcript"

    async def create_stream(self, language: str) -> FakeRealtimeStream:
        assert language == "en-US"
        stream = FakeRealtimeStream()
        self.streams.append(stream)
        return stream


def _audio_event(samples: np.ndarray) -> dict:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(samples.astype("<i2").tobytes()).decode("ascii"),
    }


def _receive_until(websocket, wanted: set[str], limit: int = 20) -> list[dict]:
    events = []
    for _ in range(limit):
        event = websocket.receive_json()
        events.append(event)
        wanted.discard(event["type"])
        if not wanted:
            return events
    raise AssertionError(f"Missing Realtime events: {sorted(wanted)}")


def test_realtime_server_vad_emits_boundaries_partials_and_final() -> None:
    engine = FakeRealtimeASR()
    app = create_app(
        Settings(mode="asr", model_id=ASR_MODEL),
        asr_engine=engine,
    )
    loud = np.full(2400, 12000, dtype=np.int16)  # 100 ms at 24 kHz
    silence = np.zeros(2400, dtype=np.int16)

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/realtime?model={ASR_MODEL}") as websocket:
            created = websocket.receive_json()
            assert created["type"] == "session.created"
            websocket.send_json({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {
                                "model": ASR_MODEL,
                                "language": "en-US",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.4,
                                "prefix_padding_ms": 200,
                                "silence_duration_ms": 100,
                            },
                        }
                    },
                },
            })
            assert websocket.receive_json()["type"] == "session.updated"

            websocket.send_json(_audio_event(loud))
            first = _receive_until(
                websocket,
                {
                    "input_audio_buffer.speech_started",
                    "conversation.item.input_audio_transcription.delta",
                },
            )
            started = next(
                event
                for event in first
                if event["type"] == "input_audio_buffer.speech_started"
            )
            assert started["audio_start_ms"] == 0

            websocket.send_json(_audio_event(silence))
            final = _receive_until(
                websocket,
                {
                    "input_audio_buffer.speech_stopped",
                    "input_audio_buffer.committed",
                    "conversation.item.input_audio_transcription.completed",
                },
            )

    completed = next(
        event
        for event in final
        if event["type"]
        == "conversation.item.input_audio_transcription.completed"
    )
    stopped = next(
        event
        for event in final
        if event["type"] == "input_audio_buffer.speech_stopped"
    )
    assert completed["transcript"] == "Hello world."
    assert completed["content_index"] == 0
    assert completed["item_id"] == started["item_id"]
    assert stopped["audio_end_ms"] == 100
    assert len(engine.streams) == 1
    # Two 100 ms chunks resample from 24 kHz to 16 kHz.
    assert len(engine.streams[0].audio) == 2 * 1600 * 2
    assert engine.streams[0].closed


def test_realtime_manual_commit_without_server_vad() -> None:
    engine = FakeRealtimeASR()
    app = create_app(
        Settings(mode="asr", model_id=ASR_MODEL),
        asr_engine=engine,
    )
    loud = np.full(1600, 12000, dtype=np.int16)  # 100 ms at 16 kHz

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/realtime?model={ASR_MODEL}") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            websocket.send_json({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 16000},
                            "transcription": {
                                "model": ASR_MODEL,
                                "language": "en-US",
                            },
                            "turn_detection": None,
                        }
                    },
                },
            })
            assert websocket.receive_json()["type"] == "session.updated"
            websocket.send_json(_audio_event(loud))
            _receive_until(
                websocket,
                {
                    "input_audio_buffer.speech_started",
                    "conversation.item.input_audio_transcription.delta",
                },
            )
            websocket.send_json({"type": "input_audio_buffer.commit"})
            events = _receive_until(
                websocket,
                {
                    "input_audio_buffer.committed",
                    "conversation.item.input_audio_transcription.completed",
                },
            )

    assert any(
        event.get("transcript") == "Hello world."
        for event in events
    )


def test_realtime_rejects_unknown_model() -> None:
    app = create_app(
        Settings(mode="asr", model_id=ASR_MODEL),
        asr_engine=FakeRealtimeASR(),
    )
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?model=unknown") as websocket:
            message = websocket.receive()
    assert message["type"] == "websocket.close"
    assert message["code"] == 1008
