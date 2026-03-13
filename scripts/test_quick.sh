#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Quick test run (unit tests only, exclude slow/integration)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# So tests can import multi_modal_ai_studio without pip install -e .
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

echo "Running quick tests (unit, excluding slow)..."
echo ""

pytest tests/unit \
    -v \
    -m "not slow" \
    --tb=short \
    --maxfail=3 \
    -x

echo ""
echo "Quick tests completed!"
