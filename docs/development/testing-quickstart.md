# Testing Quick Start

## One-Line Commands

```bash
# Quick tests (unit only, fast)
./scripts/test_quick.sh

# All tests in tests/
pytest

# With coverage
pytest --cov=multi_modal_ai_studio --cov-report=term-missing

# Pre-commit checks (format + lint + tests)
./scripts/pre_commit_check.sh
```

## Test Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_quick.sh` | Fast unit tests only | During development, before each commit |
| `pre_commit_check.sh` | Format + lint + unit tests | Before committing code |

## Test Layout

- **`tests/unit/`** – Unit tests (no live Riva/Ollama). Safe for CI and quick runs.
- **`tests/integration/`** – Optional; tests that need mocks or live services (mark with `@pytest.mark.integration` or `@pytest.mark.slow`).

Run only fast tests:

```bash
pytest tests/unit -v -m "not slow"
```

## Common Workflows

**During development:**

```bash
./scripts/test_quick.sh
```

**Before committing:**

```bash
./scripts/pre_commit_check.sh
```

**Single test or file:**

```bash
pytest tests/unit/test_version.py -v
pytest tests/unit/test_version.py::test_version_string -v
```

## Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Verify
python -c "import multi_modal_ai_studio; print(multi_modal_ai_studio.__version__)"
pytest --version
```

See [testing.md](testing.md) for the full testing guide and tier strategy.
