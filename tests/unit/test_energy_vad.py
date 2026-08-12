# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the observation-only PCM energy VAD."""

import struct

import pytest

from multi_modal_ai_studio.backends.asr.energy_vad import EnergyVADObserver
from multi_modal_ai_studio.config.schema import ASRConfig


def _riva_config() -> ASRConfig:
    return ASRConfig(
        scheme="riva",
        server="localhost:50051",
        model="parakeet",
        speech_timeout_ms=200,
    )


def _pcm(amplitude: int, milliseconds: int = 100) -> bytes:
    samples = 16000 * milliseconds // 1000
    return struct.pack(f"<{samples}h", *([amplitude] * samples))


def test_energy_vad_reports_audio_boundary_and_decision_time():
    observer = EnergyVADObserver(_riva_config())

    start = observer.observe(_pcm(4000), chunk_end_time=1.0)
    assert len(start) == 1
    assert start[0].event_type == "vad_start"
    assert start[0].timestamp == pytest.approx(0.9)
    assert start[0].observed_at == pytest.approx(1.0)

    assert observer.observe(_pcm(0), chunk_end_time=1.1) == []
    end = observer.observe(_pcm(0), chunk_end_time=1.2)
    assert len(end) == 1
    assert end[0].event_type == "vad_end"
    assert end[0].timestamp == pytest.approx(1.0)
    assert end[0].observed_at == pytest.approx(1.2)


@pytest.mark.parametrize("capture_source", ["browser", "server-usb"])
def test_energy_vad_is_capture_source_independent(capture_source):
    # Capture source is deliberately not an observer input: both sources have
    # already converged to the same 16 kHz mono PCM contract at this point.
    observer = EnergyVADObserver(_riva_config())
    events = []
    events.extend(observer.observe(_pcm(4000), chunk_end_time=0.1))
    events.extend(observer.observe(_pcm(0), chunk_end_time=0.2))
    events.extend(observer.observe(_pcm(0), chunk_end_time=0.3))

    assert capture_source in {"browser", "server-usb"}
    assert [event.event_type for event in events] == ["vad_start", "vad_end"]


def test_energy_vad_never_gates_or_rewrites_pcm():
    observer = EnergyVADObserver(_riva_config())
    pcm = _pcm(4000)
    original = bytes(pcm)

    observer.observe(pcm, chunk_end_time=0.1)

    assert pcm == original
