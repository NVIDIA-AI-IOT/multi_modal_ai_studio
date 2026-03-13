# Release Checklist

Quick reference for maintainers creating a release. See [releasing.md](releasing.md) for full details.

## Quick Steps

### 1. Prepare Release

```bash
# Update version (both places must match)
vim pyproject.toml  # version = "X.Y.Z"
vim src/multi_modal_ai_studio/__init__.py  # __version__ = "X.Y.Z"

# Verify both match
grep '^version =' pyproject.toml
grep '__version__' src/multi_modal_ai_studio/__init__.py

# Update changelog
vim CHANGELOG.md    # Move Unreleased → [X.Y.Z] - YYYY-MM-DD

# Commit
git add pyproject.toml src/multi_modal_ai_studio/__init__.py CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
git push origin main
```

### 2. Create GitHub Release

1. Go to: **Releases** → **Draft a new release**
2. **Tag**: `vX.Y.Z` (create tag from main or select existing)
3. **Target**: `main`
4. **Title**: `vX.Y.Z - Brief description`
5. **Description**: Add release notes (see templates below)
6. Click **"Publish release"**

### 3. Verify

```bash
# If published to PyPI (optional): wait 5–10 minutes, then:
pip install --upgrade multi-modal-ai-studio==X.Y.Z
multi-modal-ai-studio --version
multi-modal-ai-studio --help
```

### 4. Monitor

- GitHub Actions (if CI is configured)
- PyPI (only if publishing to PyPI)

## Checklist Template

Copy this checklist when creating a release:

**Pre-Release:**
- [ ] All tests passing on `main`
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `src/multi_modal_ai_studio/__init__.py` (must match)
- [ ] Update `CHANGELOG.md` with release notes (move Unreleased → [X.Y.Z])
- [ ] Review open PRs for any last-minute inclusions
- [ ] Commit and push version bump

**Release:**
- [ ] Create GitHub Release with tag `vX.Y.Z`
- [ ] Monitor GitHub Actions build (if configured)
- [ ] Verify PyPI upload (if publishing to PyPI)

**Post-Release:**
- [ ] Test installation (pip or from source)
- [ ] Verify `multi-modal-ai-studio --version` shows correct version
- [ ] Update documentation if needed
- [ ] Announce release

## Release Notes Template

### For Feature Releases (minor version)

```markdown
## What's New in v0.2.0

### ✨ New Features
- ...

### 🐛 Bug Fixes
- ...

### 📚 Documentation
- ...

## Installation

\```bash
pip install multi-modal-ai-studio==0.2.0
\```

See [CHANGELOG.md](CHANGELOG.md) for complete details.
```

### For Patch Releases (bug fixes)

```markdown
## Bug Fixes in v0.1.1

- ...

## Installation

\```bash
pip install multi-modal-ai-studio==0.1.1
\```
```

## Version Numbering Guide

- **Patch** (`v0.1.1`): Bug fixes only, no new features
- **Minor** (`v0.2.0`): New features, backwards compatible
- **Major** (`v1.0.0`): Breaking changes

**Pre-releases:**
- `v0.2.0-alpha.1` - Early testing
- `v0.2.0-beta.1` - Feature complete, needs testing
- `v0.2.0-rc.1` - Release candidate, final testing

## Done! 🚀

Full documentation: [releasing.md](releasing.md)
