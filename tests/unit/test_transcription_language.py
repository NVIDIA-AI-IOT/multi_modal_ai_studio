# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.backends.asr.language import transcription_language
from multi_modal_ai_studio.backends.asr.openai_realtime import OpenAIRealtimeASRBackend
from multi_modal_ai_studio.backends.asr.openai_rest import OpenAIRestASRBackend
from multi_modal_ai_studio.config.schema import ASRConfig


@pytest.mark.parametrize(
    "locale, expected",
    [
        ("ja-JP", "ja"),
        ("en-US", "en"),
        ("en-GB", "en"),
        ("zh-CN", "zh"),
        ("zh-Hant-TW", "zh"),
        ("pt-BR", "pt"),
        (" EN_us ", "en"),
        ("ja", "ja"),
        ("auto", None),
        (" AUTO ", None),
        ("", None),
        (None, None),
    ],
)
def test_transcription_language_codes(locale, expected):
    assert transcription_language(locale) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "locale, expected",
    [("ja-JP", "ja"), ("en-US", "en"), ("zh-CN", "zh"), ("auto", None), ("", None)],
)
async def test_rest_request_normalizes_language_without_mutating_config(locale, expected):
    fields = {}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def text(self):
            return '{"text": "recognized"}'

        async def json(self):
            return {"text": "recognized"}

    def post(_url, *, data, headers):
        fields.update({meta["name"]: value for meta, _, value in data._fields})
        return Response()

    config = ASRConfig(scheme="openai-rest", api_base="http://localhost/v1", language=locale)
    backend = OpenAIRestASRBackend(config)
    backend._session = SimpleNamespace(post=post)
    backend._results = asyncio.Queue()
    await backend._transcribe(b"\x00\x00" * 160)
    assert fields.get("language") == expected
    assert config.language == locale
    assert backend._results.get_nowait().text == "recognized"


def test_realtime_uses_same_language_normalization():
    config = ASRConfig(
        scheme="openai-realtime",
        realtime_transport="websocket",
        realtime_session_type="transcription",
        realtime_url="ws://localhost/v1/realtime",
        language="ja-JP",
    )
    client = OpenAIRealtimeASRBackend(config)._make_client()
    assert client.input_audio_transcription["language"] == "ja"
    assert config.language == "ja-JP"


def test_riva_language_remains_a_locale():
    assert ASRConfig(scheme="riva", language="ja-JP").language == "ja-JP"
