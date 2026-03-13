#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Quick test run (unit tests only, exclude slow/integration)

set -e

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
