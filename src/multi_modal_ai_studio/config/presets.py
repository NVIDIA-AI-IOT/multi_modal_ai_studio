# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Resolve editable presets before read-only defaults shipped in the wheel."""

from pathlib import Path

BUNDLED_PRESETS_DIR = Path(__file__).with_name("presets")


def writable_presets_dir(session_dir: Path | None = None) -> Path:
    """Preserve source-checkout behavior; never write into site-packages."""
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "src" / "multi_modal_ai_studio").is_dir():
        return repo_root / "presets"
    if session_dir is not None:
        return session_dir.resolve().parent / "presets"
    return Path.cwd() / "presets"


def available_presets(directory: Path | None = None) -> dict[str, Path]:
    """Return a merged filename map, with local files overriding bundled ones."""
    result = {}
    for base in (directory or writable_presets_dir(), BUNDLED_PRESETS_DIR):
        if base.is_dir():
            for path in sorted(base.iterdir()):
                if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
                    result.setdefault(path.name, path)
    return dict(sorted(result.items()))


def find_preset(name: str, directory: Path | None = None) -> Path | None:
    """Find a named preset without allowing traversal outside the preset roots."""
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        return None
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return available_presets(directory).get(filename)


def resolve_cli_preset(value: str) -> Path | None:
    """CLI additionally accepts an explicit file path, as before."""
    path = Path(value)
    return path if path.is_file() else find_preset(value)
