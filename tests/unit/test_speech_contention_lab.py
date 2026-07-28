# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the headless LLM/TTS contention benchmark."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "speech_contention_lab.py"
SPEC = importlib.util.spec_from_file_location("speech_contention_lab", SCRIPT)
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def _pcm(seconds, sample_rate=1000):
    return b"\x01\x00" * round(seconds * sample_rate)


def test_simulate_playout_inserts_measured_underrun_silence():
    chunks = [_pcm(0.1), _pcm(0.1), _pcm(0.1)]
    output, metrics = LAB.simulate_playout(
        chunks,
        [0.05, 0.10, 0.40],
        sample_rate=1000,
        prebuffer_ms=100,
    )

    assert metrics["playout_start_s"] == pytest.approx(0.05)
    assert metrics["underrun_count"] == 1
    assert metrics["underrun_s"] == pytest.approx(0.15)
    assert len(output) == len(b"".join(chunks)) + 300


def test_simulate_playout_prebuffer_absorbs_arrival_jitter():
    chunks = [_pcm(0.1), _pcm(0.1), _pcm(0.1)]
    output, metrics = LAB.simulate_playout(
        chunks,
        [0.05, 0.12, 0.25],
        sample_rate=1000,
        prebuffer_ms=200,
    )

    assert metrics["playout_start_s"] == pytest.approx(0.12)
    assert metrics["underrun_count"] == 0
    assert output == b"".join(chunks)


def test_simulate_playout_rejects_mismatched_events():
    with pytest.raises(ValueError):
        LAB.simulate_playout([_pcm(0.1)], [], sample_rate=1000, prebuffer_ms=0)


def test_summarize_trials_compares_isolated_and_overlap_metrics():
    trials = [
        {
            "scenario": "tts-only",
            "error": None,
            "tts": {"ttfa_s": 0.5, "rtf": 0.8},
            "llm": None,
            "playout": {"underrun_count": 0, "underrun_s": 0.0},
        },
        {
            "scenario": "overlap",
            "error": None,
            "tts": {"ttfa_s": 0.9, "rtf": 1.2},
            "llm": {"ttft_s": 0.2, "total_s": 2.0, "tokens_per_s": 30.0},
            "playout": {"underrun_count": 2, "underrun_s": 0.3},
        },
    ]

    summary = LAB.summarize_trials(trials)

    assert summary["tts-only"]["tts_rtf_mean"] == pytest.approx(0.8)
    assert summary["overlap"]["tts_rtf_mean"] == pytest.approx(1.2)
    assert summary["overlap"]["playout_underruns_mean"] == pytest.approx(2)


@pytest.mark.parametrize(
    "mode",
    ["mmas-units", "strict-units", "after-llm", "simultaneous"],
)
def test_parser_accepts_all_overlap_start_modes(mode):
    args = LAB.build_parser().parse_args(["run", "--overlap-start-mode", mode])

    LAB.validate_args(args)

    assert args.overlap_start_mode == mode
