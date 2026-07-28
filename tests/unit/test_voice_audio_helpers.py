# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for audio routing helpers shared by browser and server devices."""

import asyncio
import struct

import pytest

from multi_modal_ai_studio.backends.base import ASRResult
from multi_modal_ai_studio.devices.capture import _make_capture_event
from multi_modal_ai_studio.webui.voice_pipeline import (
    BargeInController,
    _asr_result_error_message,
    _capture_event_details,
    _pcm_amplitude_segments,
    _pcm_rms_slices,
    _pcm_rms_to_amplitude,
    _resample_pcm_to_24k,
    _wait_for_task_or_barge_in,
)


def test_asr_backend_error_is_not_treated_as_an_empty_transcript():
    result = ASRResult(
        text="",
        is_final=True,
        metadata={
            "backend": "openai-rest",
            "error": "ASR request failed (404): unknown model",
        },
    )

    assert (
        _asr_result_error_message(result)
        == "ASR request failed (404): unknown model"
    )
    assert _asr_result_error_message(ASRResult(text="", is_final=True)) is None


def test_pcm_helpers_tolerate_odd_length_and_non_bytes_input():
    assert _pcm_rms_to_amplitude(b"\x00") == 0.0
    assert _pcm_rms_slices(b"\x00") == []
    assert _resample_pcm_to_24k(b"\x00", 16000) == b""
    assert _pcm_rms_to_amplitude(None) == 0.0


def test_pcm_helpers_report_signal_and_resample_to_24khz():
    pcm_16k = struct.pack("<160h", *([16384] * 160))

    assert 49.0 <= _pcm_rms_to_amplitude(pcm_16k) <= 51.0
    assert _pcm_rms_slices(pcm_16k, sample_rate=16000, window_s=0.005)
    pcm_24k = _resample_pcm_to_24k(pcm_16k, 16000)
    assert len(pcm_24k) == 240 * 2


def test_pcm_amplitude_segments_are_dense_contiguous_and_resume_at_cursor():
    pcm_16k = struct.pack("<800h", *([16384] * 800))

    first, next_cursor = _pcm_amplitude_segments(
        pcm_16k,
        sample_rate=16000,
        start_time=1.25,
        window_s=0.025,
    )
    second, final_cursor = _pcm_amplitude_segments(
        pcm_16k,
        sample_rate=16000,
        start_time=next_cursor,
        window_s=0.025,
    )

    assert len(first) == 2
    assert first[0]["startTime"] == pytest.approx(1.25)
    assert first[0]["endTime"] == pytest.approx(1.275)
    assert first[1]["startTime"] == pytest.approx(first[0]["endTime"])
    assert first[1]["endTime"] == pytest.approx(1.3)
    assert all(49.0 <= segment["amplitude"] <= 51.0 for segment in first)
    assert second[0]["startTime"] == pytest.approx(first[-1]["endTime"])
    assert final_cursor == pytest.approx(1.35)


def test_pcm_amplitude_segments_leave_cursor_unchanged_for_empty_audio():
    segments, cursor = _pcm_amplitude_segments(
        b"",
        sample_rate=16000,
        start_time=2.5,
    )

    assert segments == []
    assert cursor == 2.5


@pytest.mark.parametrize(
    ("event", "expected_type", "terminal"),
    [
        ("dropped", "capture_dropped", False),
        ("recovered", "capture_recovered", False),
        ("gave_up", "capture_gave_up", True),
    ],
)
def test_capture_event_mapping(event, expected_type, terminal):
    details = _capture_event_details(
        _make_capture_event(event, device="hw:2,0", retry=1)
    )

    assert details is not None
    event_type, data, is_terminal = details
    assert event_type == expected_type
    assert data == {"device": "hw:2,0", "retry": 1}
    assert is_terminal is terminal


def test_final_barge_in_only_interrupts_active_tts():
    controller = BargeInController(enabled=True, trigger="final")
    controller.begin_turn()
    assert not controller.observe_asr(is_final=True, text="before speech")

    controller.start_tts()
    assert not controller.observe_asr(is_final=False, text="partial")
    assert controller.observe_asr(is_final=True, text="interrupt")
    assert controller.requested.is_set()


def test_partial_barge_in_requires_configured_number_of_partials():
    controller = BargeInController(
        enabled=True,
        trigger="partial",
        partial_count=3,
    )
    controller.begin_turn()
    controller.start_tts()

    assert not controller.observe_asr(is_final=False, text="one")
    assert not controller.observe_asr(is_final=False, text="two")
    assert controller.observe_asr(is_final=False, text="three")


@pytest.mark.asyncio
async def test_wait_for_task_returns_true_when_tts_completes():
    task = asyncio.create_task(asyncio.sleep(0))
    requested = asyncio.Event()

    assert await _wait_for_task_or_barge_in(task, requested) is True
    assert task.done()
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_wait_for_task_cancels_tts_on_barge_in():
    started = asyncio.Event()

    async def long_tts():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(long_tts())
    await started.wait()
    requested = asyncio.Event()
    requested.set()

    assert await _wait_for_task_or_barge_in(task, requested) is False
    assert task.cancelled()
