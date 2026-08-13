# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API contract tests that use lightweight fake engines."""

from dataclasses import dataclass
import sys
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
