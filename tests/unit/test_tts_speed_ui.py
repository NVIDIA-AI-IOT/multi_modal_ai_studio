# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the OpenAI REST TTS speed control."""

from pathlib import Path


def test_openai_rest_tts_speed_control_is_scoped_and_persisted():
    repository_root = Path(__file__).resolve().parents[2]
    app = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/app.js"
    ).read_text()

    rest_start = app.index("<!-- OpenAI REST TTS Settings -->")
    realtime_start = app.index("<!-- OpenAI Realtime exact-text TTS")
    rest_markup = app[rest_start:realtime_start]

    assert 'id="tts-rest-speed"' in rest_markup
    assert 'min="0.25" max="4.0" step="0.05"' in rest_markup
    assert "updateConfig('tts', 'speed', parseFloat(this.value))" in rest_markup
    assert 'id="tts-rest-speed-value"' in rest_markup
    assert "speed: 1.0" in app

    index = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/index.html"
    ).read_text()
    assert 'app.js?v=20260825-openai-rest-tts-controls-1' in index
