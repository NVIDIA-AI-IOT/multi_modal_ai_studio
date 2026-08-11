# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_parakeet_nemotron.py"
SPEC = importlib.util.spec_from_file_location("benchmark_parakeet_nemotron", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1 B", 1),
        ("1 KiB", 1024),
        ("1.5 MiB", 1572864),
        ("2 GB", 2000000000),
    ],
)
def test_parse_memory_bytes(value, expected):
    assert MODULE.parse_memory_bytes(value) == expected


def test_error_rate_english_wer():
    metric, rate = MODULE.error_rate("A careful robot listens.", "a robot listens", "en")
    assert metric == "wer"
    assert rate == pytest.approx(0.25)


def test_error_rate_japanese_cer():
    metric, rate = MODULE.error_rate("音声認識を確認します。", "音声認識を確認した", "ja-JP")
    assert metric == "cer"
    assert rate == pytest.approx(2 / 10)


def test_normalization_ignores_case_punctuation_and_eou():
    assert MODULE.normalized_text("Hello, WORLD!<EOU>") == "hello world"


def test_parse_models_requires_f16(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="f16 baseline"):
        MODULE.parse_models([f"q8_0={model}"])


def test_parse_manifest(tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"not decoded by the parser")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(f"ja\t{wav}\tja-JP\t音声認識を確認します。\n", encoding="utf-8")
    samples = MODULE.parse_manifest(manifest)
    assert samples[0].sample_id == "ja"
    assert samples[0].reference == "音声認識を確認します。"


def test_markdown_report_marks_orin_nano_as_unqualified():
    report = MODULE.markdown_report(
        {
            "generated_utc": "2026-07-31T00:00:00Z",
            "docker_image": "example",
            "repeats": 1,
            "threads": 1,
            "models": {},
        }
    )
    assert "Orin Nano 8 GB interpretation" in report
    assert "not an Orin Nano qualification" in report
