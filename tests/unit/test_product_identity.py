# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The public rename must not break existing MMAS imports, launchers, or settings."""

import importlib.util
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from multi_modal_ai_studio import __version__
from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.webui.server import get_app_config_dir

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_distribution", ROOT / "scripts/check_distribution.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_project_name_and_both_cli_entrypoints():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "voice-ai-workbench"
    assert project["version"] == __version__
    for command in ("voice-ai-workbench", "multi-modal-ai-studio"):
        assert project["scripts"][command] == "multi_modal_ai_studio.cli.main:main"
    assert project["urls"]["Repository"].endswith("/NVIDIA-AI-IOT/multi_modal_ai_studio")


def test_generated_command_uses_new_cli_without_changing_session_format():
    config = SessionConfig()
    restored = SessionConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert restored.to_cli_args().startswith("voice-ai-workbench ")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux XDG config compatibility")
def test_rename_reuses_existing_config_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    legacy = tmp_path / "multi-modal-ai-studio"
    legacy.mkdir()
    saved = legacy / "existing-settings.json"
    saved.write_text('{"keep": true}')
    assert get_app_config_dir() == legacy
    assert saved.read_text() == '{"keep": true}'
    assert not (tmp_path / "voice-ai-workbench").exists()


@pytest.mark.parametrize("broken", [None, "name", "legacy-cli", "ui-title"])
def test_wheel_identity_guard(tmp_path, broken):
    wheel = tmp_path / "candidate.whl"
    name = "multi-modal-ai-studio" if broken == "name" else "voice-ai-workbench"
    entries = "[console_scripts]\nvoice-ai-workbench = multi_modal_ai_studio.cli.main:main\n"
    if broken != "legacy-cli":
        entries += "multi-modal-ai-studio = multi_modal_ai_studio.cli.main:main\n"
    title = "Multi-modal AI Studio" if broken == "ui-title" else "Voice AI Workbench"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("multi_modal_ai_studio/config/presets/speaches-jetson.yaml", "{}")
        archive.writestr("multi_modal_ai_studio/webui/static/index.html", f"<title>{title}</title>")
        archive.writestr(
            "voice_ai_workbench-0.2.0.dist-info/METADATA", f"Name: {name}\nVersion: 0.2.0\n"
        )
        archive.writestr("voice_ai_workbench-0.2.0.dist-info/entry_points.txt", entries)
    if broken is None:
        assert CHECK.check_wheel_contents(wheel) == "0.2.0"
    else:
        with pytest.raises(AssertionError):
            CHECK.check_wheel_contents(wheel)
