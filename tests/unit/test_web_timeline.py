# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run dependency-free browser timeline tests from the existing pytest CI job."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_timeline_header_includes_pipeline_and_audio_legend():
    repository_root = Path(__file__).resolve().parents[2]
    html = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/index.html"
    ).read_text()

    for legend in (
        "speech-vad",
        "endpoint-phase",
        "asr-active",
        "llm-prefill",
        "llm-generate",
        "tts",
        "user-audio",
        "ai-audio",
        "barge-in",
    ):
        assert f'data-timeline-legend="{legend}"' in html

    speech_position = html.index('data-timeline-legend="speech-vad"')
    active_position = html.index('data-timeline-legend="asr-active"')
    endpoint_position = html.index('data-timeline-legend="endpoint-phase"')
    assert speech_position < active_position < endpoint_position


def test_barge_in_timeline_records_and_draws_actual_playback_stop():
    repository_root = Path(__file__).resolve().parents[2]
    app = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/app.js"
    ).read_text()

    for behavior in (
        "client_timeline_event",
        "tts_playback_stopped",
        "discarded_audio_end",
        "buildBargeInWindows",
        "timelineBargeInHitRegions",
        "Barge-in playback stopped",
        "livePlaybackHead",
        "syncLiveSessionClock",
        "liveSessionClockSynchronized",
        "buildPeakPreservingPoints",
        "pairTimelineEvents",
        "asr_inference_start",
        "asr_inference_end",
        "getAsrLegendSpec",
        "updateAsrTimelineLegend",
        "dedupeTimelineEventsByTimestamp",
        "speechEndCandidates",
        "localEnergySpeechEnds",
        "physicalSpeechEnd",
        "selectFirstPlaybackTimes",
        "rebuildTtsPlaybackSegments",
        "tts_playback_segments",
    ):
        assert behavior in app


def test_asr_request_timing_is_forwarded_to_the_timeline():
    repository_root = Path(__file__).resolve().parents[2]
    pipeline = (
        repository_root
        / "src/multi_modal_ai_studio/webui/voice_pipeline.py"
    ).read_text()

    for behavior in (
        "inference_start_time",
        "inference_end_time",
        '"asr_inference_start"',
        '"asr_inference_end"',
    ):
        assert behavior in pipeline


def test_web_timeline_regressions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    repository_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            node,
            "--check",
            "src/multi_modal_ai_studio/webui/static/app.js",
        ],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        [
            node,
            "--check",
            "src/multi_modal_ai_studio/webui/static/config_helpers.js",
        ],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        [
            node,
            "--check",
            "src/multi_modal_ai_studio/webui/static/timeline_helpers.js",
        ],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        [
            node,
            "--test",
            "tests/js/config_helpers.test.js",
            "tests/js/timeline_helpers.test.js",
        ],
        cwd=repository_root,
        check=True,
    )


def test_browser_and_server_usb_share_classic_pcm_pipeline():
    repository_root = Path(__file__).resolve().parents[2]
    pipeline = (
        repository_root
        / "src/multi_modal_ai_studio/webui/voice_pipeline.py"
    ).read_text()

    # One call is the Browser WebSocket path and one is the Server USB capture
    # consumer. The common function owns Riva Energy VAD and ASR forwarding.
    assert pipeline.count("await _feed_pcm_to_pipeline(") == 2
    assert "riva_energy_vad.observe(" in pipeline
    assert '"source": "local-energy"' in pipeline


def test_recorded_configuration_note_is_consistent_across_tabs():
    repository_root = Path(__file__).resolve().parents[2]
    app = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/app.js"
    ).read_text()

    assert "Recorded configuration for this session (read-only)" in app
    assert "This is a historical session configuration (read-only)" not in app
    assert "Recorded devices for this session (read-only)" not in app
    assert app.count("renderReadonlySessionConfigNote(") == 7
