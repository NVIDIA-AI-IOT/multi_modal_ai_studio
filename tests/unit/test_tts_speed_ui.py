# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the OpenAI REST TTS speed control."""

from pathlib import Path
import re


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
    assert re.search(r'app\.js\?v=[^"\s]+', index)


def test_compact_configuration_layout_is_opt_in_and_scoped_to_rest_tts():
    repository_root = Path(__file__).resolve().parents[2]
    app = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/app.js"
    ).read_text()
    index = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/index.html"
    ).read_text()
    styles = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/styles.css"
    ).read_text()

    assert "configurationLayout: 'standard'" in app
    assert "data-configuration-layout" in app
    assert 'data-pane="configuration"' in index
    assert 'name="ui-configuration-layout" value="standard"' in index
    assert 'name="ui-configuration-layout" value="compact"' in index
    assert 'class="backend-content tts-rest-settings"' in app
    assert 'data-lucide="gauge"' in app
    assert 'data-lucide="globe-2"' in app
    assert 'data-lucide="audio-waveform"' in app
    assert 'data-lucide="waveform"' not in app
    assert 'class="config-info-trigger"' in app
    assert 'data-lucide="info"' in app
    assert 'data-lucide="circle-info"' not in app
    assert 'id="tts-rest-metadata-hint" class="config-info-tooltip-content"' in app
    assert 'class="input-hint">Discovering voices and languages' not in app
    assert 'id="tts-rest-sample-rate"' in app
    assert 'body[data-configuration-layout="compact"] .tts-rest-settings' in styles
    assert 'body[data-configuration-layout="compact"] .tts-rest-settings .tts-audio-settings-grid' in styles
    assert 'tts-quality-setting--rest' not in styles
    assert 'styles.css?v=20261107-qwen-voice-panel-1' in index
    assert 'body[data-configuration-layout="compact"] .tts-rest-settings .compact-config-section + .compact-config-section' in styles
    assert '.config-info-trigger:hover + .config-info-tooltip-content' in styles
    assert '.config-info-trigger:focus-visible + .config-info-tooltip-content' in styles


def test_audio_settings_wrap_by_pane_width_and_qwen_group_has_its_own_outline():
    styles = (
        Path(__file__).resolve().parents[2]
        / "src/multi_modal_ai_studio/webui/static/styles.css"
    ).read_text()
    assert 'repeat(auto-fit, minmax(min(100%, 165px), 1fr))' in styles
    assert '.tts-rest-settings .tts-audio-settings-grid > .form-group' in styles
    assert '.tts-rest-settings .tts-speed-control input[type="range"]' in styles
    assert '.tts-rest-settings .qwen-voice-panel {' in styles
    assert 'border: 1px solid rgba(56, 189, 248, 0.65)' in styles
