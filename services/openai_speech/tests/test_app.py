# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API contract tests that use lightweight fake engines."""

import base64
import sys
from dataclasses import dataclass
from types import SimpleNamespace

from fastapi.testclient import TestClient

from openai_speech.app import create_app
from openai_speech.config import ASR_MODEL, TTS_MODEL, Settings
from openai_speech.engines import NemotronASREngine


@dataclass
class FakeASR:
    async def load(self) -> None:
        return None

    async def transcribe(self, audio: bytes, filename: str, language: str) -> str:
        assert audio == b"RIFFfake"
        assert filename == "sample.wav"
        assert language == "ja-JP"
        return "こんにちは。"


@dataclass
class FakeTTS:
    async def load(self) -> None:
        return None

    async def synthesize(self, text: str, voice: str, language: str, response_format: str):
        assert (text, voice, language, response_format) == (
            "Hello",
            "Sofia",
            "en",
            "wav",
        )
        return b"RIFFwave", "audio/wav"


def test_asr_contract() -> None:
    app = create_app(
        Settings(mode="asr", model_id=ASR_MODEL),
        asr_engine=FakeASR(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": ASR_MODEL, "language": "ja-JP"},
            files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
        )
    assert response.status_code == 200
    assert response.json() == {"text": "こんにちは。"}


def test_tts_contract() -> None:
    app = create_app(
        Settings(mode="tts", model_id=TTS_MODEL),
        tts_engine=FakeTTS(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": TTS_MODEL,
                "input": "Hello",
                "voice": "Sofia",
                "language": "en",
                "response_format": "wav",
            },
        )
    assert response.status_code == 200
    assert response.content == b"RIFFwave"
    assert response.headers["content-type"] == "audio/wav"


def test_realtime_tts_contract_streams_exact_text_pcm() -> None:
    class RealtimeFakeTTS:
        async def load(self) -> None:
            return None

        async def synthesize(self, text, voice, language, response_format):
            assert (text, voice, language, response_format) == (
                "Hello from Magpie",
                "Sofia",
                "en",
                "pcm",
            )
            return b"\x01\x00" * 500, "audio/pcm"

    app = create_app(
        Settings(mode="tts", model_id=TTS_MODEL),
        tts_engine=RealtimeFakeTTS(),
    )
    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/realtime?model={TTS_MODEL}") as ws:
            assert ws.receive_json()["type"] == "session.created"
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "model": TTS_MODEL,
                        "audio": {
                            "output": {
                                "format": {"type": "audio/pcm", "rate": 22050},
                                "voice": "Sofia",
                            }
                        },
                    },
                }
            )
            assert ws.receive_json()["type"] == "session.updated"
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "metadata": {"language": "en"},
                        "content": [{"type": "input_text", "text": "Hello from Magpie"}],
                    },
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.created"
            ws.send_json(
                {
                    "type": "response.create",
                    "response": {"output_modalities": ["audio"]},
                }
            )

            events = []
            audio = bytearray()
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "response.output_audio.delta":
                    audio.extend(base64.b64decode(event["delta"]))
                if event["type"] == "response.done":
                    break

    assert events[0]["type"] == "response.created"
    assert bytes(audio) == b"\x01\x00" * 500
    assert events[-1]["response"]["status"] == "completed"
    assert sum(e["type"] == "response.output_audio.done" for e in events) == 1


def test_wrong_mode_returns_404() -> None:
    app = create_app(
        Settings(mode="asr", model_id=ASR_MODEL),
        asr_engine=FakeASR(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"model": TTS_MODEL, "input": "Hello"},
        )
    assert response.status_code == 404


def test_asr_revision_is_forwarded_to_transformers(monkeypatch) -> None:
    observed = {}
    lookaheads = []

    class FakeLoader:
        def __init__(self, kind: str):
            self.kind = kind

        def from_pretrained(self, model_id: str, **kwargs):
            observed[self.kind] = (model_id, kwargs)
            if self.kind == "model":
                return SimpleNamespace(eval=lambda: None)
            return SimpleNamespace(
                set_num_lookahead_tokens=lookaheads.append,
            )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForRNNT=FakeLoader("model"),
            AutoProcessor=FakeLoader("processor"),
        ),
    )
    revision = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
    engine = NemotronASREngine(
        Settings(
            mode="asr",
            model_id=ASR_MODEL,
            model_revision=revision,
        )
    )

    engine._load_sync()

    assert observed["processor"] == (ASR_MODEL, {"revision": revision})
    assert observed["model"] == (ASR_MODEL, {"device_map": "auto", "revision": revision})
    assert lookaheads == [3]
