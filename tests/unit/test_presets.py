# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Preset lookup must work outside a source checkout and keep defaults read-only."""

import json
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.config import presets
from multi_modal_ai_studio.webui.server import WebUIServer


@pytest.fixture
def preset_roots(tmp_path, monkeypatch):
    bundled = tmp_path / "installed" / "presets"
    local = tmp_path / "user" / "presets"
    bundled.mkdir(parents=True)
    (bundled / "speaches-jetson.yaml").write_text("name: Bundled\n")
    monkeypatch.setattr(presets, "BUNDLED_PRESETS_DIR", bundled)
    return bundled, local


def test_bundled_lookup_and_local_override(preset_roots):
    bundled, local = preset_roots
    assert presets.find_preset("speaches-jetson", local) == bundled / "speaches-jetson.yaml"
    local.mkdir(parents=True)
    override = local / "speaches-jetson.yaml"
    override.write_text("name: Override\n")
    assert presets.available_presets(local) == {override.name: override}
    assert presets.find_preset("speaches-jetson.yaml", local) == override


@pytest.mark.parametrize("name", ["../secret", "/tmp/secret", "..\\secret", ""])
def test_web_lookup_rejects_paths(preset_roots, name):
    assert presets.find_preset(name, preset_roots[1]) is None


def test_cli_accepts_explicit_file(tmp_path):
    path = tmp_path / "my-preset.yaml"
    path.write_text("name: Explicit\n")
    assert presets.resolve_cli_preset(str(path)) == path


@pytest.mark.asyncio
async def test_web_lists_gets_and_protects_bundled_presets(preset_roots):
    bundled, local = preset_roots
    server = WebUIServer.__new__(WebUIServer)
    server._get_presets_dir = lambda: local
    request = SimpleNamespace(match_info={"name": "speaches-jetson"})
    listed = json.loads((await server.handle_list_presets(request)).text)
    assert [item["filename"] for item in listed["presets"]] == ["speaches-jetson.yaml"]
    assert json.loads((await server.handle_get_preset(request)).text)["name"] == "Bundled"
    assert (await server.handle_delete_preset(request)).status == 403
    assert (bundled / "speaches-jetson.yaml").is_file()

    async def body():
        return {"name": "Speaches Jetson", "config": {"description": "My override"}}

    assert (await server.handle_save_preset(SimpleNamespace(json=body))).status == 200
    assert (
        json.loads((await server.handle_get_preset(request)).text)["description"] == "My override"
    )
    assert (await server.handle_delete_preset(request)).status == 200
    assert json.loads((await server.handle_get_preset(request)).text)["name"] == "Bundled"


def test_installed_writes_stay_outside_package(tmp_path, monkeypatch):
    module_path = (
        tmp_path / "venv" / "site-packages" / "multi_modal_ai_studio" / "config" / "presets.py"
    )
    monkeypatch.setattr(presets, "__file__", str(module_path))
    sessions = tmp_path / "user" / "sessions"
    assert presets.writable_presets_dir(sessions) == tmp_path / "user" / "presets"
