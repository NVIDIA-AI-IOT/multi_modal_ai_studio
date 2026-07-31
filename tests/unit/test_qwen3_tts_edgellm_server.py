# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Qwen3-TTS TensorRT Edge-LLM adapter."""

import importlib.util
from pathlib import Path
import sys

import pytest


pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


SCRIPT = Path(__file__).parents[2] / "scripts" / "qwen3_tts_edgellm_server.py"
SPEC = importlib.util.spec_from_file_location("qwen3_tts_edgellm_server", SCRIPT)
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class FakeEngine:
    def validate(self):
        return None

    async def synthesize(self, text, voice, response_format):
        assert (text, voice, response_format) == ("こんにちは。", "ono_anna", "pcm")
        return b"\x01\x00", "audio/pcm"


def test_openai_speech_contract():
    settings = SERVER.AdapterSettings(workspace=Path("/unused"))
    app = SERVER.create_app(settings, engine=FakeEngine())
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": SERVER.MODEL_ID,
                "input": "こんにちは。",
                "voice": "Ono_Anna",
                "response_format": "pcm",
            },
        )

    assert response.status_code == 200
    assert response.content == b"\x01\x00"
    assert response.headers["content-type"] == "audio/pcm"


def test_adapter_rejects_unsupported_quantized_alias():
    settings = SERVER.AdapterSettings(workspace=Path("/unused"))
    app = SERVER.create_app(settings, engine=FakeEngine())
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "Qwen3-TTS-0.6B-INT4",
                "input": "Hello",
                "voice": "ono_anna",
            },
        )

    assert response.status_code == 404
