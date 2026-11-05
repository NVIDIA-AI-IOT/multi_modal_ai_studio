# Release checklist

See [releasing.md](releasing.md) for exact setup, approval, build, and publication instructions. The [rc2 qualification record](../releases/v0.2.0-rc.2-qualification.md) documents the tested application candidate; verify the final 0.2.0 artifact separately. Unchecked items below are publication checks, not a claim that earlier candidate tests failed.

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

- [ ] PyPI project ownership/pending Trusted Publisher configured for `build-wheel.yml` and environment `pypi`.
- [ ] The `pypi` environment has the intended approval policy and permits branch `main` plus release tags `v*`.
- [ ] `PYPI_PUBLISH_ENABLED=true` enabled only after the above protections exist.
- [ ] Candidate CI artifact installation verified; run URL and hashes recorded. TestPyPI is not required for this check.
- [ ] Optional TestPyPI rehearsal completed or marked N/A. If used, configure its separate publisher, `testpypi` environment restricted to `main`, and `TESTPYPI_PUBLISH_ENABLED` opt-in before upload.

## Release

- [ ] Tag `v0.2.0` points to the exact approved commit.
- [ ] New 0.2.0 Draft Release reviewed and published as a stable GitHub Release; earlier RC records preserved.
- [ ] Release CI succeeds and wheel, sdist, requirements.txt, uv.lock, and SHA256SUMS are attached.
- [ ] Downloaded artifacts pass checksum and isolated-install checks.
- [ ] Opted-in stable-release PyPI job approved and successful; GitHub/PyPI artifact hashes match. If a manual rebuild was needed instead, its separate provenance/hashes are documented.
- [ ] `pip install voice-ai-workbench==0.2.0` verified from PyPI in a fresh environment.

## After publication

- [ ] JAL tutorial #454 matches the public commands/version and is published.
- [ ] Internal announcement links to the Release, tutorial, and known limitations.
- [ ] Follow-up experimental PRs and playback observability issue #59 stay separate from the release gate.
- [ ] Optional character-persona video demonstrates real model output and measured timing; it does not delay the required release/announcement.
