# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Qwen voice controls are explicit, portable, and safe for PCM playback."""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from multi_modal_ai_studio.backends.base import ConfigError, ConnectionError
from multi_modal_ai_studio.backends.tts.openai_rest import (
    OpenAIRestTTSBackend,
    build_speech_request,
)
from multi_modal_ai_studio.config.schema import SessionConfig, TTSConfig
from multi_modal_ai_studio.webui import server
from multi_modal_ai_studio.webui.voice_pipeline import _normalize_frontend_config


def qwen_config(**overrides):
    values = dict(
        scheme="openai-rest", api_base="http://localhost:18083/v1",
        model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", voice="ryan",
        language="en", sample_rate=24000, speed=1.0,
        request_profile="qwen3-customvoice", instructions="Speak warmly and clearly.",
    )
    return TTSConfig(**(values | overrides))


def test_legacy_config_has_no_new_wire_fields():
    config = SessionConfig.from_dict({"tts": {
        "scheme": "openai-rest", "api_base": "http://localhost:18080/v1",
        "model": "kokoro", "voice": "af_heart", "speed": 1.25,
    }}).tts
    assert config.instructions == ""
    assert config.request_profile == "openai"
    assert build_speech_request(config, "Hello.") == {
        "model": "kokoro", "voice": "af_heart", "input": "Hello.",
        "response_format": "pcm", "speed": 1.25,
    }


def test_generic_instructions_are_optional_and_separate_from_spoken_text():
    config = qwen_config(request_profile="openai", instructions="  Be cheerful.  ")
    request = build_speech_request(config, "Hello.")
    assert request["input"] == "Hello."
    assert request["instructions"] == "Be cheerful."
    assert "task_type" not in request and "stream" not in request


@pytest.mark.parametrize("language,expected", [
    ("en-US", "English"), ("english", "English"), ("ja", "Japanese"),
    ("Japanese", "Japanese"), ("zh_CN", "Chinese"), ("ko-KR", "Korean"),
    ("pt-BR", "Portuguese"), ("auto", "Auto"),
])
def test_customvoice_request_uses_raw_audio_stream(language, expected):
    request = build_speech_request(qwen_config(language=language), "Hello.")
    assert request == {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "voice": "ryan",
        "input": "Hello.", "instructions": "Speak warmly and clearly.",
        "language": expected, "task_type": "CustomVoice", "speed": 1.0,
        "response_format": "pcm", "stream": True, "stream_format": "audio",
    }


def test_voicedesign_omits_named_voice():
    request = build_speech_request(qwen_config(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        request_profile="qwen3-voicedesign", voice="stale-ui-voice",
    ), "Hello.")
    assert request["task_type"] == "VoiceDesign"
    assert "voice" not in request
    assert request["instructions"] == "Speak warmly and clearly."


@pytest.mark.parametrize("overrides,message", [
    ({"instructions": None}, "must be text"),
    ({"instructions": "x" * 4001}, "4000"),
    ({"request_profile": "unknown"}, "request profile"),
    ({"model": ""}, "explicit model"),
    ({"sample_rate": 22050}, "24000"),
    ({"speed": 1.25}, "speed 1.0"),
    ({"voice": ""}, "speaker voice"),
    ({"language": "hi"}, "Unsupported Qwen3-TTS language"),
    ({"request_profile": "qwen3-voicedesign", "instructions": "  "}, "requires voice instructions"),
    ({"model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}, "0.6B"),
])
def test_invalid_controls_fail_before_network(overrides, message):
    with pytest.raises(ConfigError, match=message):
        OpenAIRestTTSBackend(qwen_config(**overrides))


def test_06b_customvoice_still_works_without_instructions():
    request = build_speech_request(qwen_config(
        model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", instructions="",
    ), "Hello.")
    assert "instructions" not in request


def test_voice_controls_survive_frontend_session_and_yaml_roundtrips(tmp_path):
    tts = asdict(qwen_config(instructions="Warm robot voice.\n日本語も明瞭に。"))
    normalized = _normalize_frontend_config({"tts": tts | {"backend": "openai-rest"}})
    config = SessionConfig.from_dict(normalized)
    saved = tmp_path / "voice.yaml"
    config.to_yaml(saved)
    for restored in (SessionConfig.from_dict(config.to_dict()), SessionConfig.from_yaml(saved)):
        assert restored.tts.instructions == tts["instructions"]
        assert restored.tts.request_profile == "qwen3-customvoice"


class Response:
    status = 200

    def __init__(self, content_type="application/octet-stream"):
        self.headers = {"Content-Type": content_type}
        self.content = self

    async def iter_chunked(self, _size):
        for chunk in (b"\x01", b"\x00\x02", b"\x00"):
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["qwen3-customvoice", "qwen3-base"])
async def test_all_text_chunks_keep_direction_and_pcm_alignment(profile, monkeypatch):
    from multi_modal_ai_studio.core.voice_library import VoiceLibrary
    monkeypatch.setattr(VoiceLibrary, "speech_fields", lambda *_: {
        "ref_audio": "data:audio/wav;base64,dGVzdA==", "ref_text": "Reference words.",
        "x_vector_only_mode": False,
    })
    backend = OpenAIRestTTSBackend(qwen_config(request_profile=profile,
        instructions="Speak warmly." if profile == "qwen3-customvoice" else "",
        reference_voice_id="a" * 32))
    requests = []

    def post(_url, *, json, headers):
        requests.append(json)
        return Response()

    backend._session = SimpleNamespace(post=post)
    chunks = [chunk async for chunk in backend.synthesize_stream("Hello, friendly robot. " * 60)]
    assert len(requests) > 1
    if profile == "qwen3-customvoice":
        assert all(request["instructions"] == backend.config.instructions for request in requests)
    else:
        assert all(request["ref_text"] == "Reference words." for request in requests)
        assert all("voice" not in request and "instructions" not in request for request in requests)
    assert all(request["stream_format"] == "audio" for request in requests)
    assert all(len(chunk.audio) % 2 == 0 for chunk in chunks)
    assert b"".join(chunk.audio for chunk in chunks) == b"\x01\x00\x02\x00" * len(requests)
    assert sum(chunk.is_final for chunk in chunks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mime", ["text/event-stream", "application/json", "audio/wav", "audio/mpeg"])
async def test_non_pcm_responses_are_not_played_as_noise(mime):
    backend = OpenAIRestTTSBackend(qwen_config())
    backend._session = SimpleNamespace(post=lambda *_a, **_kw: Response(mime))
    chunks = []
    with pytest.raises(ConnectionError, match="non-PCM"):
        async for chunk in backend.synthesize_stream("Hello."):
            chunks.append(chunk)
    assert chunks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["qwen3-customvoice", "qwen3-base"])
async def test_cancel_then_next_turn_keeps_voice_controls(profile, monkeypatch):
    from multi_modal_ai_studio.core.voice_library import VoiceLibrary
    monkeypatch.setattr(VoiceLibrary, "speech_fields", lambda *_: {
        "ref_audio": "data:audio/wav;base64,dGVzdA==", "x_vector_only_mode": True,
    })
    backend = OpenAIRestTTSBackend(qwen_config(request_profile=profile,
        instructions="Speak warmly." if profile == "qwen3-customvoice" else "",
        reference_voice_id="a" * 32, reference_voice_mode="x-vector"))
    started = asyncio.Event()
    released = asyncio.Event()

    class BlockingResponse(Response):
        async def iter_chunked(self, _size):
            started.set()
            try:
                await asyncio.Future()
                yield b""
            finally:
                released.set()

    backend._session = SimpleNamespace(post=lambda *_a, **_kw: BlockingResponse())

    async def consume():
        return [chunk async for chunk in backend.synthesize_stream("Old turn.")]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 1)
    assert backend.cancel_synthesis() == 1
    assert await asyncio.wait_for(task, 1) == []
    assert released.is_set() and not backend._active_tasks
    backend._session = SimpleNamespace(post=lambda *_a, **_kw: Response())
    chunks = [chunk async for chunk in backend.synthesize_stream("New turn.")]
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_preview_uses_same_profile_and_instructions(monkeypatch):
    config = qwen_config(language="Japanese")

    async def synthesize(actual, text):
        assert asdict(actual) == asdict(config)
        assert "日本語" in text
        return b"RIFF-preview"

    async def request_json():
        return {"tts": asdict(config)}

    monkeypatch.setattr(server, "_synthesize_tts_preview", synthesize)
    response = await server.WebUIServer.handle_tts_preview(None, SimpleNamespace(json=request_json))
    assert response.status == 200 and response.body == b"RIFF-preview"


@pytest.mark.asyncio
async def test_missing_design_instructions_returns_actionable_400():
    async def request_json():
        return {"tts": asdict(qwen_config(request_profile="qwen3-voicedesign", instructions=""))}

    response = await server.WebUIServer.handle_tts_preview(None, SimpleNamespace(json=request_json))
    assert response.status == 400
    assert "VoiceDesign requires voice instructions" in json.loads(response.text)["error"]


@pytest.mark.asyncio
async def test_custom_preview_text_returns_exact_transcript_header(monkeypatch):
    text = "Hello!\nこんにちは。"

    async def synthesize(_config, actual):
        assert actual == text
        return b"RIFF-preview"

    async def request_json():
        return {"tts": asdict(qwen_config()), "text": text}

    monkeypatch.setattr(server, "_synthesize_tts_preview", synthesize)
    response = await server.WebUIServer.handle_tts_preview(None, SimpleNamespace(json=request_json))
    assert response.status == 200
    assert unquote(response.headers["X-VAW-Preview-Text"]) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, 123, "x" * 501])
async def test_invalid_preview_text_rejected_before_synthesis(text):
    async def request_json():
        return {"tts": asdict(qwen_config()), "text": text}

    response = await server.WebUIServer.handle_tts_preview(None, SimpleNamespace(json=request_json))
    assert response.status == 400
