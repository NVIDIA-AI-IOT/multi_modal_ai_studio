# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import wave

import pytest

from multi_modal_ai_studio.backends.base import TTSChunk
from multi_modal_ai_studio.config.schema import TTSConfig
from multi_modal_ai_studio.webui import server


def test_tts_preview_text_uses_selected_language():
    assert "日本語音声" in server._tts_preview_text("ja-JP")
    assert "中文语音" in server._tts_preview_text("zh-CN")
    assert "Hola" in server._tts_preview_text("es-ES")
    assert "नमस्ते" in server._tts_preview_text("hi-IN")
    assert "Ciao" in server._tts_preview_text("it-IT")
    assert "Olá" in server._tts_preview_text("pt-BR")
    assert "Hello" in server._tts_preview_text("en-US")
    assert "Hello" in server._tts_preview_text("unknown")


def test_pcm16_mono_wav_is_browser_playable():
    result = server._pcm16_mono_wav(b"\x01\x00" * 160, 16000)

    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 160


@pytest.mark.parametrize("scheme", ["riva", "openai-rest", "openai-realtime"])
def test_synthesize_tts_preview_collects_chunks_and_closes(monkeypatch, scheme):
    instances = []

    class FakeTTSBackend:
        def __init__(self, config, **_kwargs):
            self.config = config
            self.closed = False
            instances.append(self)

        async def synthesize_stream(self, text):
            assert text == "Preview sentence."
            yield TTSChunk(audio=b"\x01\x00" * 80, sample_rate=24000)
            yield TTSChunk(audio=b"\x02\x00" * 40, sample_rate=24000, is_final=True)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(server, "RivaTTSBackend", FakeTTSBackend)
    monkeypatch.setattr(server, "OpenAIRestTTSBackend", FakeTTSBackend)
    monkeypatch.setattr(server, "OpenAIRealtimeTTSBackend", FakeTTSBackend)
    config = TTSConfig(
        scheme=scheme,
        server="localhost:50051",
        api_base="http://localhost:8082/v1",
        realtime_url="ws://localhost:8082/v1/realtime",
        sample_rate=24000,
    )

    result = asyncio.run(server._synthesize_tts_preview(config, "Preview sentence."))

    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 120
    assert len(instances) == 1
    assert instances[0].closed is True
