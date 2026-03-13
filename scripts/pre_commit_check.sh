#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Pre-commit checks: formatting, linting, and tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Pre-Commit Checks${NC}"
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""

CHECKS_FAILED=0

# 1. Black formatting check
echo -e "${YELLOW}[1/5] Checking code formatting (black)...${NC}"
if black --check src/ tests/ 2>/dev/null; then
    echo -e "${GREEN}✓ Code formatting check passed${NC}"
else
    echo -e "${RED}✗ Code formatting check failed${NC}"
    echo "  Run: black src/ tests/"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

# 2. Ruff linting
echo -e "${YELLOW}[2/5] Running linter (ruff)...${NC}"
if ruff check src/ tests/ 2>/dev/null; then
    echo -e "${GREEN}✓ Linter check passed${NC}"
else
    echo -e "${RED}✗ Linter check failed${NC}"
    echo "  Run: ruff check --fix src/ tests/"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

# 3. Type checking (optional)
if command -v mypy &> /dev/null; then
    echo -e "${YELLOW}[3/5] Running type checker (mypy)...${NC}"
    if mypy src/ --ignore-missing-imports 2>/dev/null; then
        echo -e "${GREEN}✓ Type check passed${NC}"
    else
        echo -e "${YELLOW}⚠ Type check has warnings (not blocking)${NC}"
    fi
else
    echo -e "${YELLOW}[3/5] Skipping type check (mypy not installed)${NC}"
fi
echo ""

# 4. Unit tests (fast tests only)
echo -e "${YELLOW}[4/5] Running unit tests...${NC}"
if pytest tests/unit -q --tb=line -m "not slow" 2>/dev/null; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

# 5. Import check
echo -e "${YELLOW}[5/5] Checking package imports...${NC}"
if python -c "import multi_modal_ai_studio" 2>/dev/null; then
    echo -e "${GREEN}✓ Package imports successfully${NC}"
else
    echo -e "${RED}✗ Package import failed${NC}"
    echo "  Run from repo root: pip install -e ."
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

echo -e "${BLUE}════════════════════════════════════════════${NC}"
if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All pre-commit checks passed!${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}✗ $CHECKS_FAILED check(s) failed${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    echo ""
    echo "Please fix the issues above before committing."
    exit 1
fi
