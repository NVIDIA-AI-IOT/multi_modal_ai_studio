# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""API contract tests with no model download or CUDA dependency."""

import io
import sys
import threading
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai_server import _audio_chunks, create_app


class FakeRuntime:
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    speaker = "ono_anna"
    backend = "cuda-graph"
    quant = "BF16"
    chunk_size = 2

    def __init__(self) -> None:
        self.languages = []

    def stream(self, _text: str, language: str):
        self.languages.append(language)
        yield np.asarray([0.0, 0.5, -0.5], dtype=np.float32), 24000, {}


def test_pcm_contract_and_japanese_detection() -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": runtime.model_id,
                "input": "こんにちは。",
                "voice": "Ono_Anna",
                "response_format": "pcm",
            },
        )

    assert response.status_code == 200
    assert response.content == b"\x00\x00\x00@\x00\xc0"
    assert response.headers["content-type"].lower().startswith("audio/l16")
    assert runtime.languages == ["Japanese"]


def test_wav_response_has_standard_frame_count() -> None:
    with TestClient(create_app(FakeRuntime())) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "voice": "ono-anna", "response_format": "wav"},
        )

    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    with wave.open(io.BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 3


def test_unsupported_controls_fail_instead_of_silently_falling_back() -> None:
    with TestClient(create_app(FakeRuntime())) as client:
        wrong_voice = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "voice": "alloy", "response_format": "pcm"},
        )
        wrong_speed = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "voice": "Ono_Anna", "speed": 1.25},
        )
        wrong_format = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "voice": "Ono_Anna", "response_format": "mp3"},
        )
        wrong_model = client.post(
            "/v1/audio/speech",
            json={"model": "Qwen3-TTS-INT4", "input": "Hello.", "voice": "Ono_Anna"},
        )

    assert wrong_voice.status_code == 400
    assert wrong_speed.status_code == 400
    assert wrong_format.status_code == 400
    assert wrong_model.status_code == 404


def test_health_and_model_discovery() -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").json()["speaker"] == "ono_anna"
        models = client.get("/v1/models").json()

    assert models["data"][0]["id"] == runtime.model_id


def test_cancelled_stream_releases_producer() -> None:
    stopped = threading.Event()

    class RepeatingRuntime(FakeRuntime):
        def stream(self, _text: str, _language: str):
            try:
                while True:
                    yield np.asarray([0.1], dtype=np.float32), 24000, {}
            finally:
                stopped.set()

    async def cancel_after_first_chunk() -> None:
        chunks = _audio_chunks(RepeatingRuntime(), "hello", "English")
        assert await chunks.__anext__()
        await chunks.aclose()

    import asyncio

    asyncio.run(cancel_after_first_chunk())
    assert stopped.wait(timeout=1.0)
