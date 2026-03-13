#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate a list of 3rd party Python dependencies with license information
# for NVIDIA OSRB / security review (e.g. nSpect, dependency list submission).
#
# Usage (from repo root):
#   ./scripts/generate_third_party_licenses.sh
#
# Uses a temporary venv so it works even when system Python is externally
# managed (PEP 668). No active venv required.
#
# Output: third_party_licenses.csv, third_party_licenses.md, and THIRD-PARTY-NOTICES.md in repo root.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_CSV="$REPO_ROOT/third_party_licenses.csv"
OUT_MD="$REPO_ROOT/third_party_licenses.md"
OUT_NOTICES="$REPO_ROOT/THIRD-PARTY-NOTICES.md"

TMPVENV=$(mktemp -d)
trap "rm -rf '$TMPVENV'" EXIT

echo "Using temporary venv..."
python3 -m venv "$TMPVENV"
"$TMPVENV/bin/pip" install -q --upgrade pip
"$TMPVENV/bin/pip" install -q pip-licenses
"$TMPVENV/bin/pip" install -q -e .

echo "Writing $OUT_CSV"
"$TMPVENV/bin/pip-licenses" --format=csv --output-file="$OUT_CSV" 2>/dev/null || "$TMPVENV/bin/pip-licenses" --format=csv > "$OUT_CSV"
echo "Writing $OUT_MD"
"$TMPVENV/bin/pip-licenses" --format=markdown --output-file="$OUT_MD" 2>/dev/null || "$TMPVENV/bin/pip-licenses" --format=markdown > "$OUT_MD"
cp "$OUT_MD" "$OUT_NOTICES"
echo "Writing $OUT_NOTICES (copy for OSRB link)"

echo "Done. Use third_party_licenses.csv for nSpect; link to THIRD-PARTY-NOTICES.md for OSRB."
