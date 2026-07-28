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
        "asr",
        "vad",
        "llm-prefill",
        "llm-generate",
        "tts",
        "user-audio",
        "ai-audio",
    ):
        assert f'data-timeline-legend="{legend}"' in html


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
