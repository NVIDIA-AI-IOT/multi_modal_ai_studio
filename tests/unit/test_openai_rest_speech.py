# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OpenAI-compatible speech adapters."""

import asyncio
import struct
import time
from types import SimpleNamespace

import aiohttp
import pytest

from multi_modal_ai_studio.backends.asr.openai_rest import OpenAIRestASRBackend
from multi_modal_ai_studio.backends.base import split_tts_text
from multi_modal_ai_studio.backends.tts.openai_rest import (
    MAX_REST_TTS_CHARS,
    OpenAIRestTTSBackend,
    _PCM16FrameAligner,
)
from multi_modal_ai_studio.config.schema import ASRConfig, SessionConfig, TTSConfig
from multi_modal_ai_studio.webui.voice_pipeline import (
    TTSChunkBuffer,
    _normalize_frontend_config,
)


def _asr_config() -> ASRConfig:
    return ASRConfig(
        scheme="openai-rest",
        api_base="http://localhost:8081/v1",
        model="nvidia/nemotron-3.5-asr-streaming-0.6b",
    )


def test_openai_rest_vad_and_wav_encoding():
    backend = OpenAIRestASRBackend(_asr_config())
    silence = struct.pack("<160h", *([0] * 160))
    speech = struct.pack("<160h", *([4000] * 160))

    assert not backend._is_speech(silence)
    assert backend._is_speech(speech)
    assert backend._wav_bytes(speech)[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_openai_rest_vad_uses_timeline_clock_across_audio_gaps(monkeypatch):
    timeline = SimpleNamespace(start_time=100.0)
    backend = OpenAIRestASRBackend(_asr_config(), timeline=timeline)
    backend.config.speech_timeout_ms = 200
    backend._session = object()
    backend._results = asyncio.Queue()
    backend._requests = asyncio.Queue()

    now = iter([105.0, 105.1, 110.0, 110.1])
    monkeypatch.setattr(
        "multi_modal_ai_studio.backends.asr.openai_rest.time.time",
        lambda: next(now),
    )
    speech = struct.pack("<1600h", *([4000] * 1600))
    silence = struct.pack("<1600h", *([0] * 1600))

    await backend.send_audio(speech)
    await backend.send_audio(speech)
    await backend.send_audio(silence)
    await backend.send_audio(silence)

    controls = [
        backend._results.get_nowait(),
        backend._results.get_nowait(),
    ]
    assert [item.metadata["control_event"] for item in controls] == [
        "vad_start",
        "vad_end",
    ]
    assert controls[0].metadata["event_timestamp"] == pytest.approx(4.9)
    assert controls[1].metadata["event_timestamp"] == pytest.approx(9.9)

    pcm, start_time, end_time = backend._requests.get_nowait()
    assert pcm
    assert start_time == pytest.approx(4.9)
    assert end_time == pytest.approx(9.9)
    assert end_time > backend._audio_position_ms / 1000.0


@pytest.mark.asyncio
async def test_openai_rest_asr_preserves_utterance_order():
    backend = OpenAIRestASRBackend(_asr_config())
    backend._requests = asyncio.Queue()
    observed = []

    async def fake_transcribe(pcm: bytes) -> None:
        if pcm == b"first":
            await asyncio.sleep(0.01)
        observed.append(pcm)

    backend._transcribe = fake_transcribe
    worker = asyncio.create_task(backend._run_request_worker())
    await backend._requests.put(b"first")
    await backend._requests.put(b"second")
    await backend._requests.put(None)
    await worker

    assert observed == [b"first", b"second"]


@pytest.mark.asyncio
async def test_openai_rest_stop_cancels_stalled_request_worker(monkeypatch):
    backend = OpenAIRestASRBackend(_asr_config())
    backend._results = asyncio.Queue()
    backend._requests = asyncio.Queue()

    class FakeSession:
        async def close(self):
            return None

    backend._session = FakeSession()
    started = asyncio.Event()

    async def stalled_transcribe(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    backend._transcribe = stalled_transcribe
    backend._request_worker = asyncio.create_task(
        backend._run_request_worker()
    )
    await backend._requests.put(b"stalled")
    await started.wait()
    monkeypatch.setattr(
        "multi_modal_ai_studio.backends.asr.openai_rest."
        "_STOP_DRAIN_TIMEOUT_SECONDS",
        0.01,
    )

    await backend.stop_stream()

    assert backend._request_worker is None
    assert backend._session is None


class _FakeASRResponse:
    status = 200

    async def text(self):
        return '{"text":"hello"}'

    async def json(self):
        return {"text": "hello"}


class _FakeASRSession:
    def post(self, *_args, **_kwargs):
        return _FakeRequestContext(_FakeASRResponse())


def test_openai_rest_asr_records_request_timing():
    async def scenario():
        timeline = SimpleNamespace(start_time=time.time() - 1.0)
        backend = OpenAIRestASRBackend(_asr_config(), timeline=timeline)
        backend._session = _FakeASRSession()
        backend._results = asyncio.Queue()

        await backend._transcribe(
            struct.pack("<16000h", *([4000] * 16000)),
            start_time=0.1,
            end_time=1.1,
        )

        result = backend._results.get_nowait()
        metadata = result.metadata
        assert metadata["backend"] == "openai-rest"
        assert metadata["audio_duration_ms"] == pytest.approx(1000.0)
        assert metadata["inference_start_time"] >= 0
        assert (
            metadata["inference_end_time"]
            >= metadata["inference_start_time"]
        )
        assert metadata["inference_duration_ms"] >= 0

    asyncio.run(scenario())


def test_frontend_openai_aliases_normalize_to_rest():
    config = SessionConfig.from_dict(
        {
            "asr": {
                "backend": "openai",
                "openai_url": "http://localhost:8081/v1",
            },
            "tts": {
                "backend": "openai",
                "openai_url": "http://localhost:8082/v1",
                "language": "ja-JP",
            },
        }
    )

    assert config.asr.scheme == "openai-rest"
    assert config.asr.api_base == "http://localhost:8081/v1"
    assert config.tts.scheme == "openai-rest"
    assert config.tts.api_base == "http://localhost:8082/v1"
    assert config.tts.language == "ja-JP"


def test_rest_tts_metadata_keeps_model_and_voice_separate():
    normalized = _normalize_frontend_config(
        {
            "tts_model_name": "Sofia",
            "tts": {
                "backend": "openai-rest",
                "model": "nvidia/magpie_tts_multilingual_357m",
                "voice": "Sofia",
                "riva_model_name": "Sofia",
            },
        }
    )

    assert normalized["tts"]["scheme"] == "openai-rest"
    assert normalized["tts"]["model"] == "nvidia/magpie_tts_multilingual_357m"
    assert normalized["tts"]["voice"] == "Sofia"
    assert "riva_model_name" not in normalized["tts"]
    assert (
        normalized["tts_model_name"]
        == "nvidia/magpie_tts_multilingual_357m"
    )


def test_riva_tts_metadata_keeps_discovered_model_name():
    normalized = _normalize_frontend_config(
        {
            "tts_model_name": "magpie_tts_ensemble-Magpie-Multilingual",
            "tts": {
                "backend": "riva",
                "voice": "Magpie-Multilingual.EN-US.Sofia",
            },
        }
    )

    assert normalized["tts"]["scheme"] == "riva"
    assert (
        normalized["tts"]["riva_model_name"]
        == "magpie_tts_ensemble-Magpie-Multilingual"
    )
    assert (
        normalized["tts_model_name"]
        == "magpie_tts_ensemble-Magpie-Multilingual"
    )


def test_tts_chunk_buffer_counts_cjk_characters():
    buffer = TTSChunkBuffer(first_chunk_words=10)

    assert buffer.add("日本はアジアの") is None
    assert buffer.add("国で、") == "日本はアジアの国で、"


def test_tts_chunk_buffer_keeps_word_count_for_spaced_text():
    buffer = TTSChunkBuffer(first_chunk_words=5)

    assert buffer.add("This is a") is None
    assert buffer.add(" short response.") == "This is a short response."


def test_tts_chunk_buffer_never_splits_an_english_word():
    buffer = TTSChunkBuffer(first_chunk_words=5)

    assert buffer.add("San Francisco is a big") is None
    assert buffer.add(" city") == "San Francisco is a big"
    assert buffer.add(" with a vibrant culture,") == "city with a vibrant culture,"


def test_pcm16_aligner_preserves_samples_across_odd_http_chunks():
    pcm = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    aligner = _PCM16FrameAligner()

    output = [
        aligner.feed(pcm[:3]),
        aligner.feed(pcm[3:6]),
        aligner.feed(pcm[6:]),
    ]

    assert all(len(chunk) % 2 == 0 for chunk in output)
    assert b"".join(output) == pcm
    assert aligner.finish() == b""


def test_multilingual_tts_splitter_prefers_japanese_punctuation():
    text = (
        "ロボットの歴史は古代の自動機械から始まります。"
        "現代では人工知能や高度なセンサーを利用し、"
        "人と安全に協働します。"
    )

    chunks = split_tts_text(text, 24)

    assert all(len(chunk) <= 24 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text
    assert any(chunk.endswith("。") for chunk in chunks)
    assert any(chunk.endswith("、") for chunk in chunks)


def test_multilingual_tts_splitter_supports_arabic_punctuation():
    text = "بدأ تاريخ الروبوتات بآلات قديمة؟ ثم تطورت، وأصبحت أكثر ذكاءً."

    chunks = split_tts_text(text, 24)

    assert all(len(chunk) <= 24 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()
    assert any(chunk.endswith("؟") for chunk in chunks)


def test_multilingual_tts_splitter_keeps_combining_character_cluster():
    text = "か\u3099" * 12

    chunks = split_tts_text(text, 7)

    assert all(len(chunk) <= 7 for chunk in chunks)
    assert all(not chunk.startswith("\u3099") for chunk in chunks)
    assert "".join(chunks) == text


class _FakeResponse:
    status = 200

    def __init__(self, audio=b"\x01\x02\x03\x04"):
        self.content = self
        self.audio = audio

    async def iter_chunked(self, _size):
        yield self.audio


class _FakeRequestContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        return False


class _FakeTTSSession:
    def __init__(self):
        self.inputs = []

    def post(self, _url, *, json, headers):
        self.inputs.append(json["input"])
        return _FakeRequestContext(_FakeResponse())


class _DisconnectOnceTTSSession(_FakeTTSSession):
    def post(self, _url, *, json, headers):
        self.inputs.append(json["input"])
        if len(self.inputs) == 1:
            return _FakeRequestContextThatDisconnects()
        return _FakeRequestContext(_FakeResponse())


class _FakeRequestContextThatDisconnects:
    async def __aenter__(self):
        raise aiohttp.ServerDisconnectedError()

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_openai_rest_tts_preserves_normal_completed_llm_response():
    backend = OpenAIRestTTSBackend(
        TTSConfig(
            scheme="openai-rest",
            api_base="http://localhost:8082/v1",
            model="nvidia/magpie_tts_multilingual_357m",
            voice="Sofia",
            sample_rate=22050,
        )
    )
    fake_session = _FakeTTSSession()
    backend._session = fake_session
    text = (
        "The development of artificial intelligence and machine learning has "
        "significantly advanced robotics, enabling robots to learn from "
        "environments, adapt to new tasks, and perform complex operations "
        "autonomously. This evolution has transformed industries such as "
        "manufacturing, healthcare, logistics, and exploration."
    )

    chunks = [chunk async for chunk in backend.synthesize_stream(text)]

    assert fake_session.inputs == [text]
    assert chunks
    assert not any(chunk.is_final for chunk in chunks[:-1])
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_openai_rest_tts_retries_stale_keepalive_before_audio():
    backend = OpenAIRestTTSBackend(
        TTSConfig(
            scheme="openai-rest",
            api_base="http://localhost:8082/v1",
            model="speaches-ai/Kokoro-82M-v1.0-ONNX-fp16",
            voice="af_heart",
            sample_rate=24000,
        )
    )
    fake_session = _DisconnectOnceTTSSession()
    backend._session = fake_session

    chunks = [chunk async for chunk in backend.synthesize_stream("Hello")]

    assert fake_session.inputs == ["Hello", "Hello"]
    assert chunks
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_openai_rest_tts_only_splits_above_service_input_limit():
    backend = OpenAIRestTTSBackend(
        TTSConfig(
            scheme="openai-rest",
            api_base="http://localhost:8082/v1",
            model="nvidia/magpie_tts_multilingual_357m",
            voice="Sofia",
            sample_rate=22050,
        )
    )
    fake_session = _FakeTTSSession()
    backend._session = fake_session
    text = ("A complete sentence about robotics and artificial intelligence. " * 100)

    chunks = [chunk async for chunk in backend.synthesize_stream(text)]

    assert len(fake_session.inputs) > 1
    assert all(len(item) <= MAX_REST_TTS_CHARS for item in fake_session.inputs)
    assert " ".join(fake_session.inputs).split() == text.split()
    assert chunks[-1].is_final


class _BlockingRequestContext:
    def __init__(self):
        self.started = asyncio.Event()

    async def __aenter__(self):
        self.started.set()
        await asyncio.Future()

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_openai_rest_tts_cancel_synthesis_preserves_caller_task():
    backend = OpenAIRestTTSBackend(
        TTSConfig(
            scheme="openai-rest",
            api_base="http://localhost:8082/v1",
            model="nvidia/magpie_tts_multilingual_357m",
            voice="Sofia",
            sample_rate=22050,
        )
    )
    request = _BlockingRequestContext()
    backend._session = SimpleNamespace(post=lambda *_args, **_kwargs: request)

    async def consume():
        async for _ in backend.synthesize_stream("Long response"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(request.started.wait(), timeout=0.5)
    assert backend.cancel_synthesis() == 1
    await asyncio.wait_for(task, timeout=0.5)
    assert not task.cancelled()
