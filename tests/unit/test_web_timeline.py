# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run dependency-free browser timeline tests from the existing pytest CI job."""

import shutil
import subprocess
from pathlib import Path

import pytest


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
            "src/multi_modal_ai_studio/webui/static/timeline_helpers.js",
        ],
        cwd=repository_root,
        check=True,
    )
    subprocess.run(
        [
            node,
            "--test",
            "tests/js/timeline_helpers.test.js",
        ],
        cwd=repository_root,
        check=True,
    )
