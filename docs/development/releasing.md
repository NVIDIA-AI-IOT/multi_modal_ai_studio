# Release process

Voice AI Workbench (formerly Multi-modal AI Studio / MMAS) is distributed as the `voice-ai-workbench` Python wheel and source distribution (sdist). Speech and LLM services remain independent containers. A new application container is not required for the 0.2.0 release. The repository and Python import remain `multi_modal_ai_studio`; the new `voice-ai-workbench` CLI and legacy `multi-modal-ai-studio` alias invoke the same application.

## Versions and release gate

Use `0.2.0` in `pyproject.toml` and `src/multi_modal_ai_studio/__init__.py`, and `v0.2.0` for the stable GitHub tag. For future release candidates, Python version `0.2.1rc1` corresponds to GitHub tag `v0.2.1-rc.1`. Keep rc1 and rc2 qualification records as historical evidence rather than relabeling their artifacts as 0.2.0.

Before publishing, complete the [release checklist](release-checklist.md) and review the [0.2.0 qualification record](../releases/v0.2.0-qualification.md), earlier [rc2 qualification record](../releases/v0.2.0-rc.2-qualification.md), and [0.2.0 release notes](../releases/v0.2.0.md). CI success is not evidence that microphone playback, GPU inference, or barge-in passed on hardware. Record exact commits, container tags/digests, presets, session filenames, screenshots, and any explicitly accepted limitations. Revalidate changes between the tested candidate and the final artifact in proportion to their scope.

Resolve or disposition current nSpect findings against the exact release dependencies. A successful scan with zero packages is not a clean security result. This workflow does not grant a security exception or release approval.

## What is shipped

| Destination | Contents | Trigger |
|---|---|---|
| GitHub Actions artifacts | Wheel, sdist; separate release-assets bundle with SHA256SUMS, requirements.txt, and uv.lock | Successful CI or manual build |
| GitHub Release or Pre-release | The verified release-assets bundle | Publishing a GitHub Release starts CI; attachment follows successful tests |
| TestPyPI | Wheel and sdist | Explicit manual dispatch, repository opt-in, and the configured testpypi environment protections |
| PyPI | The same wheel and sdist as the GitHub assets | Stable release publication, repository opt-in, successful asset attachment, and the configured pypi environment protections; explicit manual promotion is also available |
| GHCR | Existing independent Speaches / llama.cpp images | Outside the MMAS package-release workflow |

The wheel includes the Web UI and bundled presets. It does not include the repository launch scripts, deployment files, model weights, or generated session/audio data. Use a checkout of the same release tag when building/starting the optional backend services.

Repository `presets/*.yaml` are the canonical preset sources. The setuptools build hook copies them into the wheel, including when building from an sdist. CLI and Web UI load installed defaults without a source checkout. Local presets override installed defaults; saved presets go next to the session directory, never into site-packages. Deleting a local override reveals the bundled default again.

`requirements.txt` is the scanner-compatible export of `uv.lock`, including optional runtime extras. Neither file is automatically enforced by a normal `pip install`; package metadata deliberately retains compatible dependency ranges. To reproduce qualification versions, use the release requirements file as a constraints file (`pip install -c requirements.txt ...`), or use the tagged source checkout with `uv sync --locked`. Do not install every optional device/video dependency on a speech-only host solely to satisfy a scanner.

## One-time maintainer setup

1. Create the GitHub environment **pypi**. If your maintainer team requires a separate approval before uploading, configure required reviewers and the appropriate self-review policy. Allow branch `main` for manual promotion and release tags `v*` for release-triggered publishing. In **Deployment branches and tags**, add `main` with type **Branch** and `v*` with type **Tag**; a `v*` branch rule does not authorize release tags. Check your GitHub plan supports the intended protections before enabling publishing. Merely naming the environment `pypi` does not create an approval step: without required reviewers, an allowed publishing job can proceed automatically.
2. Configure a pending Trusted Publisher on [PyPI](https://pypi.org/manage/account/publishing/). Use project **voice-ai-workbench**, owner **NVIDIA-AI-IOT**, repository **multi_modal_ai_studio**, workflow filename **build-wheel.yml**, and environment **pypi**. Ensure the appropriate team maintainers have project access. PyPI permits an unspecified environment, displayed as `(Any)`, but explicitly naming `pypi` restricts trust to that environment; see [PyPI's environment guidance](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
3. Only after those protections and the publisher are configured, set the repository Actions variable `PYPI_PUBLISH_ENABLED=true`. Leave it unset to build GitHub assets without uploading to PyPI.

**TestPyPI is optional.** To rehearse index publishing there, separately create a **testpypi** GitHub environment restricted to branch `main`, configure a matching pending publisher on [TestPyPI](https://test.pypi.org/manage/account/publishing/), and then set `TESTPYPI_PUBLISH_ENABLED=true`. This is not required for a candidate build or production PyPI publication.

No PyPI API token is required. Only the isolated publishing jobs receive `id-token: write`; they download tested artifacts and do not check out or execute project code. Builds and pull requests have read-only repository permission. The GitHub asset-attachment job alone receives `contents: write`.

Merging a PR or pushing to main never publishes to a package index. A stable GitHub Release automatically starts the **pypi** publishing job after the build and asset attachment succeed when `PYPI_PUBLISH_ENABLED=true`; it promotes the exact artifacts attached by that same workflow run. That job waits for human approval only if the environment has required reviewers. Without that protection, publishing the stable GitHub Release is the action that authorizes the automated PyPI upload; do not rely on a later confirmation prompt. Both the GitHub prerelease flag and the Python package version are checked, so an RC/dev version does not auto-publish even if the release is accidentally marked stable. With PyPI disabled, the GitHub release artifacts can still be built and attached. An explicitly dispatched but disabled index destination fails before building.

This follows [Live VLM WebUI's release-triggered Trusted Publishing pattern](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/main/.github/workflows/build-wheel.yml), with additional artifact verification, isolated publishing jobs, and opt-in/environment gates. Its PyPI publisher registration does not grant access to the MMAS project. A missing public PyPI project does not prove that no account-private pending publisher exists; check the intended maintainer's PyPI account before registering another one.

## Candidate build

1. Merge the release preparation PR.
2. Open **Actions → Build and Test → Run workflow**. Select workflow branch **main**, set **ref** to the candidate's full commit SHA, and **publish_to** to **none**. No PyPI or TestPyPI account setup is needed for this build.
3. CI runs the unit and speech-service contract tests, builds an sdist and a wheel from it, checks package metadata, and installs the wheel in a fresh virtual environment outside the checkout. It exercises CLI version/help, the Speaches CLI preset, preset HTTP endpoints, and UI static files without contacting GPU backends.
4. Download the **release-assets** artifact and verify `SHA256SUMS`. Validate its wheel in a clean environment and qualify the final commit on Orin Nano and Thor.
5. Preserve the run URL, hashes, and device evidence in the qualification record.

For an optional TestPyPI rehearsal, repeat the manual dispatch with **publish_to=testpypi** after configuring that destination. Review the build, complete any configured **testpypi** environment approval, and verify installation from that index. Without required reviewers, the upload proceeds automatically after the build succeeds.

TestPyPI and PyPI are independent indexes. Reusing an uploaded filename/version on the same index is not supported; do not silently skip existing files. If another candidate must be uploaded to that index, increment the RC version and update both version fields. Local wheel artifacts can be rebuilt without consuming an index version.

For a TestPyPI installation, avoid mixing TestPyPI and PyPI as competing dependency indexes. After uploading the 0.2.0 candidate there, download only Voice AI Workbench from TestPyPI and install that wheel with dependencies resolved from PyPI:

```bash
python -m pip download --no-deps --index-url https://test.pypi.org/simple/ voice-ai-workbench==0.2.0
python -m pip install ./voice_ai_workbench-0.2.0-py3-none-any.whl
```

For local validation:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist/*.whl
```

Use a clean `dist/` directory containing only the intended version.

## GitHub Release

After the security and hardware gates are satisfied:

1. Finalize the changelog date, release notes, and qualification evidence. Any code/dependency change after qualification needs proportionate revalidation.
2. Create tag **v0.2.0** at the exact approved commit, not at a moving main. Prepare a new Draft Release with the finalized 0.2.0 notes; preserve earlier RC records. For this stable version, leave **Pre-release** unchecked. Future RC tags must be marked as pre-releases.
3. Publish the Draft Release. The release-triggered build verifies tag/package version agreement, reruns tests, and attaches the wheel, sdist, lock/constraints files, and SHA256SUMS.
4. Confirm the build and asset-attachment jobs succeeded. If PyPI publishing is enabled, complete any configured environment approval and verify the **pypi** job in the same run. Without required reviewers, it uploads automatically after successful asset attachment. GitHub publication precedes artifact attachment and PyPI upload; a failed job does not automatically unpublish the GitHub release.
5. Download the release artifacts and run `sha256sum -c SHA256SUMS` with all listed files in the same directory. Install the downloaded wheel outside the repository and confirm version, presets, and Web UI.

Avoid publishing an unqualified release solely to trigger a build; use the manual **none** build first.

## Production PyPI verification and manual fallback

The preferred stable path is the **pypi** job in the release-triggered run. It waits for successful GitHub asset attachment and uploads that run's tested wheel/sdist through Trusted Publishing. Compare the SHA256 hashes on PyPI with the GitHub `SHA256SUMS` before announcing availability.

If PyPI was not enabled at GitHub publication, or an RC is intentionally being published to PyPI, an explicit fallback is available:

1. Open **Actions → Build and Test → Run workflow** on workflow branch **main**.
2. Set **ref** to an already published release tag, such as **v0.2.0**, and **publish_to** to **pypi**.
3. CI requires a published, non-draft GitHub Release for that tag, matching Python version, and checkout of the actual tag commit. The tag must contain the release-policy script and workflow-compatible build files.
4. Review the new build and complete any configured **pypi** environment approval. Without required reviewers, dispatching this workflow authorizes an upload after the build succeeds. This fallback rebuilds from the tag; it does not reuse a previous run's files. Do not assume byte-for-byte identity with existing GitHub assets. Record and verify both sets of hashes and provenance, and clearly identify any difference.

After either path succeeds, confirm the version on PyPI, then install in a fresh environment:

```bash
python -m pip install voice-ai-workbench==0.2.0
voice-ai-workbench --version
voice-ai-workbench --preset speaches-jetson --port 8092
```

The last command expects separately running speech/LLM endpoints for conversation. Opening the UI and selecting presets does not prove that inference services are installed. Do not rerun a successful upload or enable silent skipping of existing versions; an index does not permit replacing a published artifact.

## Upgrading an earlier MMAS installation

Prefer a fresh virtual environment when testing the renamed distribution. If reusing an environment with an earlier `multi-modal-ai-studio` wheel or editable install, uninstall that distribution **before** installing `voice-ai-workbench`: both distributions provide the same Python package, so leaving both installed can cause a later uninstall to remove shared files.

After publication, in the activated environment:

```bash
python -m pip uninstall multi-modal-ai-studio
python -m pip install voice-ai-workbench==0.2.0
voice-ai-workbench --version
multi-modal-ai-studio --version  # retained compatibility alias
```

For pre-publication validation, replace the install argument with the verified local `voice_ai_workbench-0.2.0-py3-none-any.whl`. Uninstalling the Python distribution does not delete externally stored sessions, user presets, or model caches. Keep the same session-directory argument/working directory when switching launchers. The OS-specific `multi-modal-ai-studio` configuration directory is intentionally unchanged, as are `multi_modal_ai_studio` imports and the GitHub repository name. Earlier rc1/rc2 artifacts and qualification records retain their original names; they are not retroactively renamed or republished.

If a pending Trusted Publisher was registered under `multi-modal-ai-studio`, configure one for **voice-ai-workbench** before publication; the old project name does not authorize upload to the new project. The owner, repository, workflow filename, and environment remain as listed above.

## Documentation and announcement

Update the [Jetson AI Lab tutorial PR #454](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/pull/454) with the actual published version and validated setup commands. Publish the tutorial and send the internal announcement after the download/install path is verified. Clearly distinguish the recommended Speaches + Gemma browser path, NVIDIA open-model reference services, experimental USB/headless coverage, and optional Qwen3-TTS integrations.

## References

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PyPA GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
