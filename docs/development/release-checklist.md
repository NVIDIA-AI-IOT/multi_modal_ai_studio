# Release checklist

See [releasing.md](releasing.md) for exact setup, approval, build, and publication instructions. Record rc1 evidence in the [qualification record](../releases/v0.2.0-rc.1-qualification.md).

## Before publishing

- [ ] Required changes merged; exact candidate commit recorded.
- [ ] Version fields agree: `pyproject.toml` and `src/multi_modal_ai_studio/__init__.py`.
- [ ] Unit tests, speech-service contracts, and lock/export consistency pass.
- [ ] Wheel built from sdist; clean off-repository installation, CLI preset, preset API, and static Web UI smoke pass.
- [ ] Orin Nano and Thor browser qualification recorded with session JSON filenames and screenshots.
- [ ] Existing USB/Riva coverage mapped to tested commits; omissions marked N/A or explicitly accepted, not silently counted as passing.
- [ ] nSpect inventory contains packages and matches the release dependencies.
- [ ] Every finding is remediated or has the required documented disposition/approval; no unexplained High/Critical findings.
- [ ] Changelog/release date, known limitations, install instructions, and release notes finalized.

## Publishing setup

- [ ] TestPyPI and PyPI project ownership/pending Trusted Publishers configured for `build-wheel.yml`.
- [ ] `testpypi` and `pypi` GitHub environments have required approval and main-branch restrictions.
- [ ] Destination repository opt-in variables enabled only after the above protections exist.
- [ ] Candidate build/TestPyPI installation verified; artifact run URL and hashes recorded.

## Release

- [ ] Tag `v0.2.0-rc.1` points to the exact approved commit.
- [ ] Existing Draft Release reviewed and published as a GitHub Pre-release.
- [ ] Release CI succeeds and wheel, sdist, requirements.txt, uv.lock, and SHA256SUMS are attached.
- [ ] Downloaded artifacts pass checksum and isolated-install checks.
- [ ] Production PyPI workflow explicitly dispatched with the published tag; environment approval granted.
- [ ] `pip install multi-modal-ai-studio==0.2.0rc1` verified from PyPI in a fresh environment.

## After publication

- [ ] JAL tutorial #454 matches the public commands/version and is published.
- [ ] Internal announcement links to the Release, tutorial, and known limitations.
- [ ] Follow-up experimental PRs and playback observability issue #59 stay separate from the RC gate.
