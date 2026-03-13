# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for package version and CLI --version."""

import subprocess
import sys
from pathlib import Path

import pytest

from multi_modal_ai_studio import __version__


def test_version_string():
    """__version__ is a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    # Basic semver-like (e.g. 0.1.0 or 0.1.0-alpha.1)
    parts = __version__.split(".")
    assert len(parts) >= 2


def test_version_matches_pyproject():
    """Version in __init__.py should match pyproject.toml (single source of truth)."""
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    for line in content.splitlines():
        if line.strip().startswith("version ="):
            # version = "0.1.0"
            quoted = line.split("=", 1)[1].strip().strip('"').strip("'")
            assert quoted == __version__, (
                f"pyproject.toml version {quoted!r} != __init__.py __version__ {__version__!r}. "
                "Keep both in sync."
            )
            return
    pytest.fail("version = not found in pyproject.toml")


def test_cli_version_exit():
    """CLI --version prints version and exits (smoke test)."""
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    env = {**__import__("os").environ, "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-m", "multi_modal_ai_studio.cli.main", "--version"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout or __version__ in result.stderr
