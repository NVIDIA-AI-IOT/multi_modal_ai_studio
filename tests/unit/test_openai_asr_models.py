# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from multi_modal_ai_studio.webui.server import _parse_openai_asr_models

import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from multi_modal_ai_studio.webui.server import WebUIServer


def test_asr_registry_models_include_local_status_and_all_remote_entries():
    local = {
        "data": [{
            "id": "Systran/faster-whisper-tiny.en",
            "task": "automatic-speech-recognition",
            "language": ["en"],
        }]
    }
    registry = {
        "data": [
            {
                "id": "Systran/faster-whisper-small.en",
                "task": "automatic-speech-recognition",
                "language": ["en"],
            },
            {
                "id": "speaches-ai/Kokoro",
                "task": "text-to-speech",
            },
        ]
    }

    assert _parse_openai_asr_models(local, registry) == [
        {
            "id": "Systran/faster-whisper-tiny.en",
            "downloaded": True,
            "languages": ["en"],
        },
        {
            "id": "Systran/faster-whisper-small.en",
            "downloaded": False,
            "languages": ["en"],
        },
    ]


def test_duplicate_registry_entry_does_not_hide_downloaded_status():
    model = {"id": "model-a", "task": "automatic-speech-recognition"}
    result = _parse_openai_asr_models({"data": [model]}, {"data": [model]})
    assert result == [{"id": "model-a", "downloaded": True, "languages": []}]


def test_realtime_language_metadata_is_optional_and_model_specific():
    result = _parse_openai_asr_models({"data": [
        {"id": "english", "supported_languages": ["en_US", "en_US"]},
        {"id": "bilingual", "languages": ["ja", "en"]},
        {"id": "unknown", "language": "multilingual"},
    ]}, realtime=True)
    assert {m["id"]: m["languages"] for m in result} == {
        "english": ["en-US"], "bilingual": ["en", "ja"], "unknown": [],
    }


@pytest.mark.asyncio
async def test_realtime_discovery_only_reads_models_without_registry_or_inference(tmp_path):
    paths = []

    async def models(request):
        paths.append(request.path_qs)
        return web.json_response({"data": [{"id": "speech", "languages": ["en"]}]})

    app = web.Application()
    app.router.add_get('/v1/models', models)
    async with TestServer(app) as provider:
        async def body():
            return {"api_base": str(provider.make_url('/v1')), "realtime": True}

        response = await WebUIServer(session_dir=tmp_path).handle_openai_asr_models(SimpleNamespace(json=body))
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["models"][0]["languages"] == ["en"]
    assert payload["download_supported"] is False
    assert paths == ['/v1/models']
