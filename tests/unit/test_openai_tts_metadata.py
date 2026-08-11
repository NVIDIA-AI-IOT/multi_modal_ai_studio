# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from multi_modal_ai_studio.webui.server import (
    _is_loopback_api_base,
    _merge_openai_tts_models,
    _parse_openai_tts_metadata,
)


def test_parse_speaches_tts_metadata_uses_selected_model():
    payload = {
        "data": [
            {"id": "whisper", "task": "automatic-speech-recognition"},
            {
                "id": "kokoro-fp16",
                "task": "text-to-speech",
                "sample_rate": 24000,
                "language": ["multilingual"],
                "voices": [
                    {"id": "af_heart", "name": "af_heart", "language": "en-us", "gender": "female"},
                    {"id": "jf_alpha", "name": "jf_alpha", "language": "ja", "gender": "female"},
                ],
            },
        ]
    }

    result = _parse_openai_tts_metadata(payload, "kokoro-fp16")

    assert result == {
        "model": "kokoro-fp16",
        "models": ["kokoro-fp16"],
        "voices": [
            {"id": "af_heart", "name": "af_heart", "language": "en-US", "gender": "female"},
            {"id": "jf_alpha", "name": "jf_alpha", "language": "ja", "gender": "female"},
        ],
        "languages": ["en-US", "ja"],
        "sample_rate": 24000,
    }


def test_parse_generic_openai_models_returns_empty_extensions():
    result = _parse_openai_tts_metadata({"data": [{"id": "tts-1"}]}, "tts-1")

    assert result == {
        "model": None,
        "models": [],
        "voices": [],
        "languages": [],
        "sample_rate": None,
    }


def test_single_tts_model_is_selected_when_requested_name_is_stale():
    result = _parse_openai_tts_metadata({
        "data": [{
            "id": "installed-kokoro",
            "voices": ["voice-a"],
            "language": "en_US",
        }]
    }, "old-name")

    assert result["model"] == "installed-kokoro"
    assert result["voices"] == [{"id": "voice-a", "name": "voice-a"}]
    assert result["languages"] == ["en-US"]


def test_active_language_qualification_is_limited_to_loopback_servers():
    assert _is_loopback_api_base("http://127.0.0.1:18080/v1") is True
    assert _is_loopback_api_base("http://localhost:18080/v1") is True
    assert _is_loopback_api_base("https://api.openai.com/v1") is False


def test_tts_registry_merge_lists_local_first_and_preserves_remote_voice_metadata():
    local = {
        "data": [{
            "id": "kokoro-local",
            "task": "text-to-speech",
            "language": ["multilingual"],
            "voices": [{"id": "af_heart", "language": "en-us"}],
        }]
    }
    registry = {
        "data": [
            local["data"][0],
            {
                "id": "piper-ja",
                "task": "text-to-speech",
                "language": ["ja"],
                "voices": [{"id": "voice-ja", "language": "ja"}],
            },
            {"id": "whisper", "task": "automatic-speech-recognition"},
        ]
    }

    payload, choices = _merge_openai_tts_models(local, registry)

    assert choices == [
        {"id": "kokoro-local", "downloaded": True, "languages": ["multilingual"]},
        {"id": "piper-ja", "downloaded": False, "languages": ["ja"]},
    ]
    assert _parse_openai_tts_metadata(payload, "piper-ja")["voices"] == [
        {"id": "voice-ja", "name": "voice-ja", "language": "ja"},
    ]
