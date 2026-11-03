# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep index publication opt-in and isolated from code execution."""

from pathlib import Path

import yaml


def test_publishing_requires_manual_dispatch_and_environment():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/build-wheel.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    for destination in ("testpypi", "pypi"):
        job = workflow["jobs"][destination]
        assert "workflow_dispatch" in job["if"]
        assert f"inputs.publish_to == '{destination}'" in job["if"]
        assert job["needs"] == "build"
        assert job["environment"]["name"] == destination
        assert job["permissions"] == {"id-token": "write"}
        assert all("run" not in step for step in job["steps"])
        assert all("checkout" not in step["uses"] for step in job["steps"])


def test_release_build_smokes_distribution_and_preserves_security_gates():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/build-wheel.yml").read_text())
    build = workflow["jobs"]["build"]
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "python scripts/check_distribution.py dist/*.whl" in commands
    assert "python -m build" in commands
    assert "SHA256SUMS" in commands
    assert "uv.lock" in commands and "requirements.txt" in commands
    assert "refs/heads/main" in commands
    assert "gh release view" in commands
    assert "Version(tag[1:]) == Version(version)" in commands
    opt_in = next(
        step for step in build["steps"] if step.get("name") == "Validate manual publishing opt-in"
    )
    assert opt_in["env"]["PYPI_ENABLED"] == "${{ vars.PYPI_PUBLISH_ENABLED }}"
    assert opt_in["env"]["TESTPYPI_ENABLED"] == "${{ vars.TESTPYPI_PUBLISH_ENABLED }}"
