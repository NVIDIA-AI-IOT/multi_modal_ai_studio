# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for lightweight session-list responses."""

import base64

from multi_modal_ai_studio.webui.server import WebUIServer


def test_session_list_summary_omits_heavy_data_and_secrets():
    session = {
        "session_id": "session-1",
        "name": "Test",
        "created_at": "2026-07-30T00:00:00Z",
        "config": {
            "asr_model_name": "Nemotron ASR",
            "asr": {
                "scheme": "openai-rest",
                "model": "nvidia/nemotron-asr",
                "api_key": "secret",
            },
            "llm": {
                "scheme": "openai",
                "model": "Qwen",
                "api_key": "secret",
            },
            "tts": {
                "scheme": "openai-rest",
                "model": "nvidia/magpie",
                "voice": "Sofia",
                "api_key": "secret",
            },
            "devices": {
                "microphone": "browser",
                "speaker": "browser",
                "camera": "none",
            },
        },
        "timeline": [{"event_type": "asr_final"}] * 100,
        "turns": [{"user_text": "large"}] * 10,
        "audio_amplitude_history": [{"amplitude": 1}] * 100,
        "metrics": {"total_turns": 10, "avg_ttl": 1.2},
        "app_version": "0.1.0",
    }

    summary = WebUIServer._session_list_summary(session)

    assert summary["timeline_event_count"] == 100
    assert summary["metrics"]["total_turns"] == 10
    assert summary["config"]["asr"]["model"] == "nvidia/nemotron-asr"
    assert "timeline" not in summary
    assert "turns" not in summary
    assert "audio_amplitude_history" not in summary
    assert "api_key" not in summary["config"]["asr"]
    assert "api_key" not in summary["config"]["llm"]
    assert "api_key" not in summary["config"]["tts"]


def test_session_list_summary_counts_wrapped_timeline_events():
    summary = WebUIServer._session_list_summary(
        {
            "session_id": "wrapped",
            "timeline": {"events": [{}, {}, {}]},
        }
    )

    assert summary["timeline_event_count"] == 3


def test_session_list_exposes_only_thumbnail_presence():
    jpeg = b"\xff\xd8preview\xff\xd9"
    thumbnail = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    summary = WebUIServer._session_list_summary(
        {"session_id": "thumbnail-session", "thumbnail": thumbnail}
    )

    assert summary["has_thumbnail"] is True
    assert "thumbnail" not in summary
