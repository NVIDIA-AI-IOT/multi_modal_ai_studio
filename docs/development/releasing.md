# Release process

MMAS is distributed as a Python wheel and source distribution (sdist). Speech and LLM services remain independent containers. A new MMAS container is not required for the release candidate.

## Versions and release gate

Use `0.2.0rc1` in `pyproject.toml` and `src/multi_modal_ai_studio/__init__.py`, and `v0.2.0-rc.1` for the GitHub tag and release title. These identify the same Python version.

Before publishing, complete the [release checklist](release-checklist.md) and the [rc1 qualification record](../releases/v0.2.0-rc.1-qualification.md). CI success is not evidence that microphone playback, GPU inference, or barge-in passed on hardware. Record exact commits, container tags/digests, presets, session filenames, screenshots, and any explicitly accepted limitations.

Resolve or disposition current nSpect findings against the exact release dependencies. A successful scan with zero packages is not a clean security result. This workflow does not grant a security exception or release approval.

## What is shipped

| Destination | Contents | Trigger |
|---|---|---|
| GitHub Actions artifacts | Wheel, sdist; separate release-assets bundle with SHA256SUMS, requirements.txt, and uv.lock | Successful CI or manual build |
| GitHub Pre-release | The verified release-assets bundle | Publishing a GitHub Release starts CI; attachment follows successful tests |
| TestPyPI | Wheel and sdist | Explicit manual dispatch, repository opt-in, and testpypi environment approval |
| PyPI | Wheel and sdist | Explicit manual dispatch for an already published release tag, repository opt-in, and pypi environment approval |
| GHCR | Existing independent Speaches / llama.cpp images | Outside the MMAS package-release workflow |

The wheel includes the Web UI and bundled presets. It does not include the repository launch scripts, deployment files, model weights, or generated session/audio data. Use a checkout of the same release tag when building/starting the optional backend services.

Repository `presets/*.yaml` are the canonical preset sources. The setuptools build hook copies them into the wheel, including when building from an sdist. CLI and Web UI load installed defaults without a source checkout. Local presets override installed defaults; saved presets go next to the session directory, never into site-packages. Deleting a local override reveals the bundled default again.

`requirements.txt` is the scanner-compatible export of `uv.lock`, including optional runtime extras. Neither file is automatically enforced by a normal `pip install`; package metadata deliberately retains compatible dependency ranges. To reproduce qualification versions, use the release requirements file as a constraints file (`pip install -c requirements.txt ...`), or use the tagged source checkout with `uv sync --locked`. Do not install every optional device/video dependency on a speech-only host solely to satisfy a scanner.

## One-time maintainer setup

1. Create GitHub environments named **testpypi** and **pypi**. Configure required reviewers, prevent self-review where available, and restrict deployment to the `main` branch (the branch containing the manually dispatched workflow). Check your GitHub plan supports the intended protections before enabling publishing.
2. Configure a pending Trusted Publisher separately on [TestPyPI](https://test.pypi.org/manage/account/publishing/) and [PyPI](https://pypi.org/manage/account/publishing/). Use project **multi-modal-ai-studio**, owner **NVIDIA-AI-IOT**, repository **multi_modal_ai_studio**, workflow filename **build-wheel.yml**, and the matching environment **testpypi** or **pypi**. Ensure the appropriate team maintainers have project access.
3. Only after those protections and publishers are configured, set repository Actions variables `TESTPYPI_PUBLISH_ENABLED=true` and `PYPI_PUBLISH_ENABLED=true` for the destinations you intend to enable. Leave them unset otherwise.

No PyPI API token is required. Only the isolated publishing jobs receive `id-token: write`; they download tested artifacts and do not check out or execute project code. Builds and pull requests have read-only repository permission. The GitHub asset-attachment job alone receives `contents: write`.

Publishing is **not** triggered by merging a PR, pushing to main, or publishing a GitHub Release. An explicit Actions dispatch is still required for either package index. A disabled destination fails with an explanatory message before building.

## Candidate build and TestPyPI

1. Merge the release preparation PR.
2. Open **Actions → Build and Test → Run workflow**. Select workflow branch **main**, set **ref** to the candidate's full commit SHA, and **publish_to** to **testpypi**. For a build without any index upload, select **none**.
3. CI runs the unit and speech-service contract tests, builds an sdist and a wheel from it, checks package metadata, and installs the wheel in a fresh virtual environment outside the checkout. It exercises CLI version/help, the Speaches CLI preset, preset HTTP endpoints, and UI static files without contacting GPU backends.
4. Approve the **testpypi** environment only after reviewing the build. Validate the downloaded TestPyPI wheel on a clean environment and qualify the final commit on Orin Nano and Thor.
5. Preserve the run URL, hashes, and device evidence in the qualification record.

TestPyPI and PyPI are independent indexes. Reusing an uploaded filename/version is not supported; do not silently skip existing files. If another candidate must be uploaded after a package has been published, increment the RC version and update both version fields. Local wheel artifacts can be rebuilt without consuming an index version.

For a TestPyPI installation, avoid mixing TestPyPI and PyPI as competing dependency indexes. Download only MMAS from TestPyPI and install that wheel with dependencies resolved from PyPI:

```bash
python -m pip download --no-deps --index-url https://test.pypi.org/simple/ multi-modal-ai-studio==0.2.0rc1
python -m pip install ./multi_modal_ai_studio-0.2.0rc1-py3-none-any.whl
```

For local validation:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist/*.whl
```

Use a clean `dist/` directory containing only the intended version.

## GitHub Pre-release

After the security and hardware gates are satisfied:

1. Finalize the changelog date, release notes, and qualification evidence. Any code/dependency change after qualification needs proportionate revalidation.
2. Create tag **v0.2.0-rc.1** at the exact approved commit, not at a moving main. Update the existing Draft Release to that tag and mark it as a **Pre-release**.
3. Publish the Draft Release. The release-triggered build verifies tag/package version agreement, reruns tests, and attaches the wheel, sdist, lock/constraints files, and SHA256SUMS.
4. Confirm the release workflow completed and attachments exist. GitHub publication precedes artifact attachment; a failed build does not automatically unpublish the release.
5. Download the release artifacts and run `sha256sum -c SHA256SUMS` with all listed files in the same directory. Install the downloaded wheel outside the repository and confirm version, presets, and Web UI.

Avoid publishing an unqualified release solely to trigger a build; use the manual **none** build first.

## Production PyPI

1. Open **Actions → Build and Test → Run workflow** on workflow branch **main**.
2. Set **ref** to the published **v0.2.0-rc.1** tag and **publish_to** to **pypi**.
3. CI requires a published, non-draft GitHub Release for that tag, matching Python version, and checkout of the actual tag commit.
4. Review the build and approve the **pypi** environment. The publishing job uploads its tested wheel/sdist via Trusted Publishing.
5. Confirm the version on PyPI, then install in a fresh environment:

```bash
python -m pip install multi-modal-ai-studio==0.2.0rc1
multi-modal-ai-studio --version
multi-modal-ai-studio --preset speaches-jetson --port 8092
```

The last command expects separately running speech/LLM endpoints for conversation. Opening the UI and selecting presets does not prove that inference services are installed. Wheel/sdist builds in different runs are not assumed byte-for-byte identical; record each run's checksums and compare provenance/version.

## Documentation and announcement

Update the [Jetson AI Lab tutorial PR #454](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/pull/454) with the actual published version and validated setup commands. Publish the tutorial and send the internal announcement after the download/install path is verified. Clearly distinguish the recommended Speaches + Gemma browser path, NVIDIA open-model reference services, experimental USB/headless coverage, and optional Qwen3-TTS integrations.

## References

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PyPA GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
