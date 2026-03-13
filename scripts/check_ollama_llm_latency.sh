#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Quick LLM latency check: same request as voice pipeline (Ollama OpenAI-compatible API).
# Usage: ./scripts/check_ollama_llm_latency.sh [OLLAMA_BASE]
# Default OLLAMA_BASE=http://localhost:11434

set -e
BASE="${1:-http://localhost:11434}"
URL="${BASE}/v1/chat/completions"

echo "=== Ollama LLM latency check (same as voice pipeline) ==="
echo "URL: $URL"
echo ""

# Non-streaming: total time for "Hello" -> short reply (like Turn 1)
echo "1. Non-streaming (total time for one short reply):"
echo "   curl -s -o /dev/null -w 'Total time: %{time_total}s\n' ..."
time curl -s -o /tmp/ollama_check.json -w "Total time: %{time_total}s\n" \
  -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": false
  }'
echo "Response (first 500 chars):"
head -c 500 /tmp/ollama_check.json
echo ""
echo ""

echo "=== Manual curl (copy-paste) ==="
echo "  time curl -s -X POST '$URL' -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"llama3.2:3b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"temperature\": 0.7, \"max_tokens\": 512, \"stream\": false}'"
