#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Add Apache 2.0 SPDX headers to NVIDIA-authored source files.
Run from repo root: python scripts/add_spdx_headers.py [--dry-run]
"""

from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parent.parent

# SPDX text (no comment syntax)
COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
LICENSE_ID = "SPDX-License-Identifier: Apache-2.0"

# Headers by comment style
HEADERS = {
    "hash": f"# {COPYRIGHT}\n# {LICENSE_ID}\n",
    "slash": f"// {COPYRIGHT}\n// {LICENSE_ID}\n",
    "block_js_css": f"/* {COPYRIGHT}\n * {LICENSE_ID}\n */\n",
    "html": f"<!-- {COPYRIGHT}\n     {LICENSE_ID} -->\n",
}

# Extensions and their comment style; paths relative to repo root
SOURCE_GLOBS = [
    ("src/**/*.py", "hash"),
    ("scripts/**/*.py", "hash"),
    ("scripts/**/*.sh", "hash"),
    ("presets/**/*.yaml", "hash"),
    ("presets/**/*.yml", "hash"),
    ("benchmarks/**/*.py", "hash"),
    ("examples/**/*.py", "hash"),
    ("tests/**/*.py", "hash"),
    ("src/**/*.js", "slash"),
    ("src/**/*.css", "block_js_css"),
    ("src/**/*.html", "html"),
]

# Skip paths (no trailing slash)
SKIP_DIRS = (".venv", "venv", "env", "__pycache__", "node_modules", "docs/cursor")
SKIP_FILES = ("add_spdx_headers.py",)  # this script already has header

SPDX_MARKER = "SPDX-License-Identifier"


def should_skip(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    for d in SKIP_DIRS:
        if d in rel.parts:
            return True
    if path.name in SKIP_FILES:
        return True
    return False


def collect_files() -> list[tuple[Path, str]]:
    out = []
    for glob_pattern, style in SOURCE_GLOBS:
        for p in REPO_ROOT.glob(glob_pattern):
            if not p.is_file() or should_skip(p):
                continue
            out.append((p, style))
    return sorted(set(out))


def has_spdx(content: str) -> bool:
    return SPDX_MARKER in content[:2000]


def add_header(path: Path, style: str, dry_run: bool) -> bool:
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        print(f"  skip (not UTF-8): {path.relative_to(REPO_ROOT)}")
        return False

    if has_spdx(content):
        return False

    header = HEADERS[style]

    # Preserve shebang and put header after it for .py and .sh
    if path.suffix in (".py", ".sh") and content.startswith("#!"):
        first_line, _, rest = content.partition("\n")
        new_content = first_line + "\n" + header + ("\n" if not rest.startswith("\n") else "") + rest
    else:
        new_content = header + content

    if dry_run:
        print(f"  would add: {path.relative_to(REPO_ROOT)}")
        return True

    path.write_bytes(new_content.encode("utf-8"))
    print(f"  added: {path.relative_to(REPO_ROOT)}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Add SPDX headers to source files")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = ap.parse_args()

    files = collect_files()
    print(f"Found {len(files)} source files (excluding already-header and skip list)")
    changed = 0
    for path, style in files:
        if add_header(path, style, args.dry_run):
            changed += 1
    print(f"Done. {'Would update' if args.dry_run else 'Updated'} {changed} files.")


if __name__ == "__main__":
    main()
