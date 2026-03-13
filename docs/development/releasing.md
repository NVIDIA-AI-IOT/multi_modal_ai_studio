# Release Process

This document describes how to create and publish a new release of Multi-modal AI Studio.

## Overview

Releases are driven by **GitHub Releases** and optional automation:

- Create a tagged release with release notes
- Optionally: CI builds the wheel and attaches it to the GitHub Release
- Optionally: Publish to PyPI when ready (see [PyPI (optional)](#pypi-optional) below)

## Version Numbering

Follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **Patch** (`v0.1.1`): Bug fixes, no API changes
- **Minor** (`v0.2.0`): New features, backwards compatible
- **Major** (`v1.0.0`): Breaking changes, incompatible API changes

**Pre-releases:**

- Alpha: `v0.2.0-alpha.1` (early testing, unstable)
- Beta: `v0.2.0-beta.1` (feature complete, testing)
- Release Candidate: `v0.2.0-rc.1` (final testing before release)

## Pre-Release Checklist

Before starting the release process, ensure:

- [ ] All planned features and bug fixes are merged to `main`
- [ ] Tests pass (e.g. `pytest tests/` or `./scripts/pre_commit_check.sh` if configured)
- [ ] Code quality checks pass (Black, Ruff, etc.)
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated with all changes since last release
- [ ] **Version in two places (both must match exactly):**
  - `pyproject.toml` – `version = "X.Y.Z"`
  - `src/multi_modal_ai_studio/__init__.py` – `__version__ = "X.Y.Z"` (this is what `--version` displays)

```bash
# Verify both files have matching versions:
grep '^version =' pyproject.toml
grep '__version__' src/multi_modal_ai_studio/__init__.py

# After installation, verify:
multi-modal-ai-studio --version  # Should show correct version
```

## Release Workflow

### 1. Prepare the Release

On the `main` branch:

```bash
git checkout main
git pull origin main

# 1. Update version in pyproject.toml
#    version = "0.2.0"

# 2. Update version in __init__.py (must match!)
#    __version__ = "0.2.0"

# 3. Verify both match
grep '^version =' pyproject.toml
grep '__version__' src/multi_modal_ai_studio/__init__.py

# 4. Update CHANGELOG.md: move [Unreleased] to [0.2.0] - YYYY-MM-DD

# 5. Commit and push
git add pyproject.toml src/multi_modal_ai_studio/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git push origin main
```

### 2. Create Git Tag (optional)

You can create the tag from the GitHub Releases UI, or locally:

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

Tag format: `vX.Y.Z` (with the `v` prefix).

### 3. Create GitHub Release

1. Go to **Releases** → **Draft a new release**
2. **Tag**: `v0.2.0` (create new or select existing)
3. **Target**: `main`
4. **Title**: `v0.2.0 - Brief description`
5. **Description**: Paste release notes (see [release-checklist.md](release-checklist.md) for templates)
6. Click **Publish release**

If CI is configured to build on release, the workflow will run and can attach the wheel to the release.

### 4. Verify the Release

```bash
# From source (after pull)
pip install -e .
multi-modal-ai-studio --version
multi-modal-ai-studio --help

# From PyPI (only if you publish to PyPI)
pip install --upgrade multi-modal-ai-studio==0.2.0
multi-modal-ai-studio --version
```

### 5. Post-Release Tasks

- Update documentation if needed
- Announce on relevant channels
- Watch for issues; respond to installation/upgrade problems

## Manual Wheel Build

You can build the wheel locally without CI:

```bash
pip install build twine
python -m build
twine check dist/*
# Install and smoke test
pip install dist/*.whl
multi-modal-ai-studio --version
multi-modal-ai-studio --help
```

## PyPI (optional)

When you want `pip install multi-modal-ai-studio` from PyPI:

1. Create the project on [PyPI](https://pypi.org/) (or use [TestPyPI](https://test.pypi.org/) first).
2. Configure [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) so GitHub Actions can publish without API tokens.
3. Add a workflow step that runs only on release (e.g. `if: github.event_name == 'release'`) to run `twine upload dist/*` (or use `pypa/gh-action-pypi-publish`).

See live-vlm-webui’s `build-wheel.yml` and [releasing](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/main/docs/development/releasing.md) doc for a full example.

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github)
