# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Keep the documented Jetson speech-service boundary reproducible."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_speaches_launcher_matches_recommended_preset() -> None:
    launcher = _text("scripts/speaches_speech.sh")
    preset = yaml.safe_load(_text("presets/speaches-jetson.yaml"))

    assert "0.9.0-rc.3-cu130-sm87-sm110-auto" in launcher
    assert preset["asr"]["model"] == "Systran/faster-whisper-tiny.en"
    assert preset["tts"]["model"] == (
        "speaches-ai/Kokoro-82M-v1.0-ONNX-fp16"
    )
    assert preset["asr"]["model"] in launcher
    assert preset["tts"]["model"] in launcher
    assert preset["asr"]["api_base"] == "http://localhost:18080/v1"
    assert preset["tts"]["api_base"] == "http://localhost:18080/v1"
    assert "--runtime nvidia" in launcher
    assert 'WHISPER__INFERENCE_DEVICE=cuda' in launcher
    assert '/v1/models/${encoded_id}' in launcher


def test_speaches_quick_start_keeps_llm_independent() -> None:
    preset = yaml.safe_load(_text("presets/speaches-jetson.yaml"))
    realtime_preset = yaml.safe_load(
        _text("presets/speaches-realtime-asr-jetson.yaml")
    )
    readme = _text("README.md")
    launcher = _text("scripts/gemma4_llm.sh")

    assert "Recommended quick start (release candidate)" in readme
    assert preset["llm"]["api_base"] == "http://localhost:8080/v1"
    assert preset["llm"]["model"] == "gemma-4-e2b"
    assert realtime_preset["llm"] == preset["llm"]
    assert "gemma-4-E2B-it-GGUF:Q4_K_S" in launcher
    assert "--no-mmproj" in launcher
    assert "--spec-type draft-mtp" in launcher
    assert "--spec-draft-n-max 3" in launcher
    assert '${GEMMA4_CACHE_DIR}:/root/.cache/huggingface' in launcher
    assert '${GEMMA4_CACHE_DIR}/hub:/data/models/huggingface' in launcher
    assert "latest-jetson-orin" in launcher
    assert "latest-jetson-thor" in launcher
    assert "./scripts/gemma4_llm.sh start" in readme


def test_nvidia_open_models_launcher_owns_only_asr_and_tts() -> None:
    launcher = _text("scripts/nvidia_open_models_speech.sh")
    compose = yaml.safe_load(
        _text("deploy/compose.nvidia-open-models-speech.yaml")
    )

    assert set(compose["services"]) == {"asr", "tts"}
    assert "mmas-nemotron-asr" in launcher
    assert "mmas-magpie-tts" in launcher
    assert "mmas-vllm" not in launcher
    assert "LLM_IMAGE=" not in launcher
    assert "LLM_MODEL=" not in launcher
    assert "verify-llm" not in launcher
