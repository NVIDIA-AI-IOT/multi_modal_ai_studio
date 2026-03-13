# Testing Guide

## Overview

Multi-modal AI Studio uses a tiered testing strategy:

- **Tier 1 (CI / quick):** Unit tests in `tests/unit/` – no live Riva or LLM. Fast, suitable for every commit and CI.
- **Tier 2 (pre-commit):** Format (Black), lint (Ruff), and unit tests – run locally before committing.
- **Tier 3 (pre-release):** Optional integration tests with live backends (e.g. `scripts/test_integration.py`, `scripts/test_backends.py`) – run manually before releases.
- **Tier 4 (performance):** Optional latency/throughput tests if added later.

## Test Layout

| Directory | Purpose | Requires |
|-----------|---------|----------|
| `tests/unit/` | Version, config schema, pure logic | Nothing |
| `tests/integration/` | Component interactions, optional mocks | Optional live services |
| `scripts/test_*.py` | Manual / integration scripts | Riva, Ollama, etc. |

## Markers

- **`slow`** – Excluded by quick runs and CI. Use for tests that are slow or depend on external services.
- **`integration`** – Marks integration tests; can be excluded with `-m "not integration"`.

Example:

```bash
# Only unit, no slow
pytest tests/unit -m "not slow"

# All except integration
pytest -m "not integration"
```

## Running Tests

```bash
# Quick (unit only)
./scripts/test_quick.sh

# All discovered tests
pytest

# With coverage
pytest --cov=multi_modal_ai_studio --cov-report=term-missing

# Pre-commit (format + lint + unit tests)
./scripts/pre_commit_check.sh
```

## Version and Config Tests

Unit tests include:

- **Version:** `__version__` is set, matches `pyproject.toml`, and CLI `--version` prints it.
- **Config:** `ASRConfig`, `LLMConfig`, `TTSConfig`, `AppConfig` can be constructed with expected defaults.

These require no external services and run in CI.

## Integration and Manual Scripts

Existing scripts under `scripts/` (e.g. `test_backends.py`, `test_integration.py`) remain for manual runs and pre-release checks. They are not part of the default `pytest` run. You can later move or duplicate some of them into `tests/integration/` and mark them with `@pytest.mark.integration` if you want them in the test suite with an option to exclude.

## CI

When CI is configured (e.g. GitHub Actions), run:

1. Unit tests: `pytest tests/unit -v -m "not slow"`
2. Optionally: build wheel, `twine check`, install wheel and run `multi-modal-ai-studio --version` / `--help`.

See the repository’s CI workflow for the exact commands.
