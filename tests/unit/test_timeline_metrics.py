# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for streaming voice component latency metrics."""

import pytest

from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core.session import Session, Turn
from multi_modal_ai_studio.core.timeline import Lane, Timeline


def _event(timeline, event_type, timestamp, lane=Lane.SYSTEM):
    timeline.add_event(event_type, lane, timestamp=timestamp)


def test_streaming_metrics_use_first_token_and_tts_start():
    timeline = Timeline()
    timeline.start_time = 1.0
    _event(timeline, "user_speech_end", 4.0)
    _event(timeline, "asr_final", 5.0, Lane.SPEECH)
    _event(timeline, "llm_first_token", 5.2, Lane.LLM)
    _event(timeline, "tts_start", 5.4, Lane.TTS)
    _event(timeline, "tts_first_audio", 6.0, Lane.TTS)
    # Streaming TTS legitimately starts before LLM completion.
    _event(timeline, "llm_complete", 7.0, Lane.LLM)
    _event(timeline, "user_speech_end", 10.0)

    metrics = timeline.calculate_component_latencies(0)

    assert metrics == pytest.approx(
        {
            "asr_latency": 1.0,
            "llm_latency": 0.2,
            "tts_latency": 0.6,
            "ttl": 2.0,
        }
    )


def test_cancelled_turn_does_not_borrow_next_turn_audio():
    timeline = Timeline()
    timeline.start_time = 1.0
    _event(timeline, "user_speech_end", 1.0)
    _event(timeline, "asr_final", 2.0, Lane.SPEECH)
    _event(timeline, "llm_first_token", 2.1, Lane.LLM)
    _event(timeline, "tts_start", 2.2, Lane.TTS)
    _event(timeline, "user_speech_end", 4.0)
    _event(timeline, "asr_final", 5.0, Lane.SPEECH)
    _event(timeline, "llm_first_token", 5.1, Lane.LLM)
    _event(timeline, "tts_start", 5.2, Lane.TTS)
    _event(timeline, "tts_first_audio", 6.0, Lane.TTS)

    first = timeline.calculate_component_latencies(0)
    second = timeline.calculate_component_latencies(1)

    assert "tts_latency" not in first
    assert "ttl" not in first
    assert second["tts_latency"] == pytest.approx(0.8)
    assert second["ttl"] == pytest.approx(2.0)


def test_ttl_bands_reject_negative_band_and_keep_correct_turn():
    session = Session(SessionConfig())
    session.turns = [
        Turn(1, "long answer", "response", start_time=3.9, latencies={}),
        Turn(2, "interrupt", "two", start_time=14.66, latencies={"ttl": -3.42}),
    ]
    _event(session.timeline, "user_speech_end", 3.4)
    _event(session.timeline, "tts_first_audio", 9.9, Lane.TTS)
    _event(session.timeline, "user_speech_end", 14.46)
    _event(session.timeline, "tts_first_audio", 18.06, Lane.TTS)
    session.ttl_bands = [
        {"start": 3.47, "end": 10.39, "ttlMs": 6922},
        {"start": 14.22, "end": 10.80, "ttlMs": -3420},
        {"start": 14.46, "end": 18.51, "ttlMs": 4050},
    ]

    session.apply_ttl_bands()

    assert session.turns[0].latencies["ttl"] == pytest.approx(6.92)
    assert session.turns[1].latencies["ttl"] == pytest.approx(4.05)


def test_cancelled_turn_does_not_claim_following_turn_browser_audio():
    session = Session(SessionConfig())
    session.turns = [
        Turn(1, "long answer", "response", start_time=5.06, latencies={"ttl": 26.99}),
        Turn(2, "interrupt", "two", start_time=29.25, latencies={}),
    ]
    _event(session.timeline, "user_speech_end", 4.13)
    _event(session.timeline, "user_speech_end", 26.79)
    _event(session.timeline, "tts_first_audio", 30.65, Lane.TTS)
    session.ttl_bands = [
        {"start": 4.13, "end": 31.13, "ttlMs": 26997},
    ]

    session.apply_ttl_bands()

    assert "ttl" not in session.turns[0].latencies
    assert session.turns[1].latencies["ttl"] == pytest.approx(3.86)
