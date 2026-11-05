# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep index publication opt-in and isolated from code execution."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_policy", ROOT / "scripts/release_policy.py")
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def test_publishing_requires_opt_in_and_isolated_environment():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/build-wheel.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    for destination in ("testpypi", "pypi"):
        job = workflow["jobs"][destination]
        assert "workflow_dispatch" in job["if"]
        assert f"inputs.publish_to == '{destination}'" in job["if"]
        assert job["needs"] == (
            "build" if destination == "testpypi" else ["build", "github-release"]
        )
        assert job["environment"]["name"] == destination
        assert job["permissions"] == {"id-token": "write"}
        assert all("run" not in step for step in job["steps"])
        assert all("checkout" not in step["uses"] for step in job["steps"])
    condition = workflow["jobs"]["pypi"]["if"]
    assert "!cancelled()" in condition
    assert "needs.build.result == 'success'" in condition
    assert "needs.build.outputs.publish_pypi == 'true'" in condition
    assert "needs.github-release.result == 'success'" in condition
    assert "needs.github-release.result == 'skipped'" in condition
    assert workflow["jobs"]["build"]["outputs"]["publish_pypi"] == (
        "${{ steps.release-policy.outputs.publish_pypi }}"
    )


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
    assert 'python scripts/release_policy.py >> "$GITHUB_OUTPUT"' in commands
    policy = next(step for step in build["steps"] if step.get("id") == "release-policy")
    assert policy["env"]["PYPI_ENABLED"] == "${{ vars.PYPI_PUBLISH_ENABLED }}"
    assert policy["env"]["RELEASE_PRERELEASE"] == "${{ github.event.release.prerelease }}"
    opt_in = next(
        step for step in build["steps"] if step.get("name") == "Validate manual publishing opt-in"
    )
    assert opt_in["env"]["PYPI_ENABLED"] == "${{ vars.PYPI_PUBLISH_ENABLED }}"
    assert opt_in["env"]["TESTPYPI_ENABLED"] == "${{ vars.TESTPYPI_PUBLISH_ENABLED }}"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"event": "push"}, False),
        ({"event": "pull_request"}, False),
        ({"prerelease": True}, False),
        ({"version": "0.2.0rc2", "tag": "v0.2.0-rc.2"}, False),
        ({"version": "0.2.0.dev1", "tag": "v0.2.0.dev1"}, False),
        ({"event": "workflow_dispatch", "destination": "none"}, False),
        ({"event": "workflow_dispatch", "destination": "testpypi"}, False),
        ({"event": "workflow_dispatch", "destination": "pypi"}, True),
        ({"event": "workflow_dispatch", "destination": "pypi", "enabled": False}, False),
        (
            {
                "event": "workflow_dispatch",
                "destination": "pypi",
                "version": "0.2.0rc2",
                "tag": "v0.2.0-rc.2",
                "prerelease": True,
            },
            True,
        ),
    ],
)
def test_pypi_release_policy(overrides, expected):
    args = {
        "event": "release",
        "version": "0.2.0",
        "tag": "v0.2.0",
        "prerelease": False,
        "destination": "none",
        "enabled": True,
    }
    args.update(overrides)
    assert POLICY.publish_pypi(**args) is expected


@pytest.mark.parametrize("tag", ["main", "0.2.0", "v0.2.1", "v0.2.0-rc.2"])
def test_release_rejects_mismatched_tags_even_when_publishing_disabled(tag):
    with pytest.raises(ValueError):
        POLICY.publish_pypi(
            event="release",
            version="0.2.0",
            tag=tag,
            prerelease=False,
            destination="none",
            enabled=False,
        )
