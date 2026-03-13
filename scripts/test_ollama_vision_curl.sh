#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Vision probe for Ollama (same style as test_cosmos_vision_curl.sh for vLLM).
# Prints: curl equivalent, request JSON (formatted), response JSON (formatted).
# Usage: ./scripts/test_ollama_vision_curl.sh [MODEL] [IMAGE] [OLLAMA_URL]
#   MODEL      default: gemma3:4b
#   IMAGE      default: cat.jpg (or any small JPEG/PNG in cwd)
#   OLLAMA_URL default: http://localhost:11434 (e.g. http://other-host:11434)
# Env: OLLAMA_URL overrides default if 3rd arg not provided.
# Requires: Python 3

set -e
MODEL="${1:-gemma3:4b}"
IMAGE="${2:-cat.jpg}"
OLLAMA_URL="${3:-${OLLAMA_URL:-http://localhost:11434}}"

if [[ ! -f "$IMAGE" ]]; then
  echo "Image not found: $IMAGE" >&2
  exit 1
fi

echo "Testing model: $MODEL at $OLLAMA_URL/api/chat (vision probe)" >&2

# Do request and parse in one Python process; print curl, request payload, response payload.
MODEL="$MODEL" IMAGE="$IMAGE" OLLAMA_URL="$OLLAMA_URL" python3 - << 'PY'
import base64
import json
import os
import sys
import urllib.request

model = os.environ["MODEL"]
image_path = os.environ["IMAGE"]
url = os.environ["OLLAMA_URL"].rstrip("/") + "/api/chat"

with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Describe this image in one word.", "images": [b64]}],
    "stream": False,
}
# For display: same structure but images truncated so output is readable
payload_display = {
    "model": payload["model"],
    "messages": [{
        "role": payload["messages"][0]["role"],
        "content": payload["messages"][0]["content"],
        "images": ["<base64 %d chars>" % len(b64)],
    }],
    "stream": payload["stream"],
}

print("--- Curl equivalent ---")
print("curl -X POST '%s' \\" % url)
print("  -H 'Content-Type: application/json' \\")
print("  -d @payload.json")
print("")
print("--- Request JSON (payload.json, image as base64 in messages[].images[]) ---")
print(json.dumps(payload_display, indent=2))
print("")
print("--- Response JSON ---")

body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    print(json.dumps(data, indent=2))
    content = (data.get("message") or {}).get("content", "")
    if content:
        print("", file=sys.stderr)
        print("Vision probe OK (model supports images): %s" % content.strip(), file=sys.stderr)
        sys.exit(0)
    else:
        print("Vision probe failed (no message content)", file=sys.stderr)
        sys.exit(1)
except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8")
    try:
        err_json = json.loads(err_body)
        print(json.dumps(err_json, indent=2))
    except Exception:
        print(err_body)
    print("Vision probe failed (HTTP %s)" % e.code, file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print("Vision probe error: %s" % e, file=sys.stderr)
    sys.exit(1)
PY
