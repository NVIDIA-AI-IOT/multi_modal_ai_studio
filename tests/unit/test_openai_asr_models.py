# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from multi_modal_ai_studio.webui.server import _parse_openai_asr_models


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
