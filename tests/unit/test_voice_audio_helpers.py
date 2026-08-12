# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for audio routing helpers shared by browser and server devices."""

import asyncio
import struct

import pytest

from multi_modal_ai_studio.devices.capture import _make_capture_event
from multi_modal_ai_studio.webui.voice_pipeline import (
    BargeInController,
    _capture_event_details,
    _pcm_rms_slices,
    _pcm_rms_to_amplitude,
    _resample_pcm_to_24k,
    _wait_for_task_or_barge_in,
)


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
