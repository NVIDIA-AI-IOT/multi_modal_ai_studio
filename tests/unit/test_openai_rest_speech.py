# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OpenAI-compatible speech adapters."""

import asyncio
import struct
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.backends.asr.openai_rest import OpenAIRestASRBackend
from multi_modal_ai_studio.backends.tts.openai_rest import _PCM16FrameAligner
from multi_modal_ai_studio.config.schema import ASRConfig, SessionConfig
from multi_modal_ai_studio.webui.voice_pipeline import TTSChunkBuffer


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
