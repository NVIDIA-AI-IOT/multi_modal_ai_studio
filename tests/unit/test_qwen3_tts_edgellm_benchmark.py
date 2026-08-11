# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3-TTS TensorRT Edge-LLM benchmark helper."""

import importlib.util
import json
from pathlib import Path
import wave

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "qwen3_tts_edgellm_benchmark.py"
SPEC = importlib.util.spec_from_file_location("qwen3_tts_edgellm_benchmark", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_parse_latency_log_reads_each_stream_request():
    output = "\n".join(
        [
            "[stream req 0] chunks=20 TTFC=19.3ms TTFPA=189.9ms",
            "[stream req 1] chunks=18 TTFC=20.1ms TTFPA=211.2ms",
        ]
    )

    assert BENCH.parse_latency_log(output) == [
        {"ttfc_ms": 19.3, "ttfpa_ms": 189.9},
        {"ttfc_ms": 20.1, "ttfpa_ms": 211.2},
    ]


def test_edit_distance_supports_cer_and_wer_units():
    assert BENCH._edit_distance(list("テスト"), list("テキスト")) == 1
    assert BENCH._edit_distance(
        "the response is clear".split(),
        "the response was clear".split(),
    ) == 1


def test_transcript_error_normalizes_punctuation_width_and_case():
    assert BENCH.transcript_error("Ｈｅｌｌｏ, Robot!", "hello robot", "en") == (
        "wer",
        0.0,
    )
    assert BENCH.transcript_error("実機テスト。", "実機 テスト", "ja") == (
        "cer",
        0.0,
    )


def test_inputs_use_fixed_voice_batch_and_common_audio_limit(tmp_path):
    args = type("Args", (), {"output_dir": tmp_path})()

    BENCH.write_inputs(args)

    japanese = json.loads((tmp_path / "ja.json").read_text(encoding="utf-8"))
    english = json.loads((tmp_path / "en.json").read_text(encoding="utf-8"))
    assert japanese["speaker"] == english["speaker"] == "ono_anna"
    assert japanese["batch_size"] == english["batch_size"] == 1
    assert japanese["max_audio_length"] == english["max_audio_length"] == 512
    assert (
        japanese["requests"][0]["messages"][0]["content"]
        == "今日はエッジAIで、低遅延の音声合成を実機検証します。"
    )
    assert (
        english["requests"][0]["messages"][0]["content"]
        == "Today we are validating low-latency text to speech on an edge AI device."
    )


def test_audio_records_reopens_local_wave_instead_of_container_path(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_path = audio_dir / "audio_req0.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\0\0" * 24000)
    output = {
        "responses": [
            {
                "request_idx": 0,
                "audio_file": "/workspace/results/audio/audio_req0.wav",
                "audio_duration_ms": 1000,
                "audio_samples": 24000,
                "audio_sample_rate": 24000,
            }
        ]
    }

    records = BENCH._audio_records(output, audio_dir)

    assert records[0]["wav_duration_sec"] == pytest.approx(1.0)
    assert records[0]["wav_size_bytes"] == wav_path.stat().st_size


def test_manifest_records_recursive_artifact_size(tmp_path):
    root = tmp_path / "engines"
    (root / "talker").mkdir(parents=True)
    (root / "talker" / "llm.engine").write_bytes(b"a" * 10)
    (root / "code2wav.engine").write_bytes(b"b" * 7)
    result = tmp_path / "manifest.json"
    args = type(
        "Args",
        (),
        {"root": root, "kind": "engine", "result": result},
    )()

    BENCH.manifest(args)

    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["total_size_bytes"] == 17
    assert [item["path"] for item in data["files"]] == [
        "code2wav.engine",
        "talker/llm.engine",
    ]
    assert all(len(item["sha256"]) == 64 for item in data["files"])
