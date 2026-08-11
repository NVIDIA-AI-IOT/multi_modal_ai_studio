# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dependency-light faster-qwen3-tts benchmark helpers."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "faster_qwen3_tts_benchmark.py"
SPEC = importlib.util.spec_from_file_location("faster_qwen3_tts_benchmark", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_transcript_error_normalizes_case_width_and_punctuation():
    metric, rate = BENCH.transcript_error(
        "Today, we test Edge AI.",
        "today we test edge ai",
        "en",
    )

    assert metric == "wer"
    assert rate == 0.0


def test_transcript_error_uses_characters_for_japanese():
    metric, rate = BENCH.transcript_error(
        "実機で、確認します。",
        "実器で確認します",
        "ja",
    )

    assert metric == "cer"
    assert rate == 1 / 8


def test_chunk_sizes_must_be_positive():
    assert BENCH.parse_chunk_sizes("1,2,10") == [1, 2, 10]
