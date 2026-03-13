#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Vision probe for vLLM OpenAI-compatible /v1/chat/completions (e.g. Cosmos, Nemotron).
# Prints: curl equivalent, request JSON (formatted), response JSON (formatted).
# Usage: ./scripts/test_cosmos_vision_curl.sh [IMAGE] [BASE_URL] [MODEL]
#   IMAGE     default: cat.jpg
#   BASE_URL  default: http://10.110.50.167:8000  (e.g. http://localhost:9000 for Nemotron)
#   MODEL     default: nvidia/cosmos-reason2-8b-fp8  (e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
# Env: VLLM_URL, MODEL override defaults if args not provided.
# Requires: Python 3

set -e
IMAGE="${1:-cat.jpg}"
VLLM_URL="${2:-${VLLM_URL:-http://10.110.50.167:8000}}"
MODEL="${3:-${MODEL:-nvidia/cosmos-reason2-8b-fp8}}"

if [[ ! -f "$IMAGE" ]]; then
  echo "Image not found: $IMAGE" >&2
  exit 1
fi

echo "Testing model: $MODEL at $VLLM_URL/v1/chat/completions (vision probe)" >&2

IMAGE="$IMAGE" VLLM_URL="$VLLM_URL" MODEL="$MODEL" python3 - << 'PY'
import base64
import json
import os
import sys
import urllib.request

model = os.environ["MODEL"]
image_path = os.environ["IMAGE"]
base_url = os.environ["VLLM_URL"].rstrip("/")
url = base_url + "/v1/chat/completions"

with open(image_path, "rb") as f:
    raw = f.read()
b64 = base64.b64encode(raw).decode("utf-8")
# Infer data URL media type from extension
ext = (os.path.splitext(image_path)[1] or "").lower()
mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"
data_url = "data:%s;base64,%s" % (mime, b64)

# OpenAI-compatible vision message content (array of parts)
content_parts = [
    {"type": "text", "text": "Describe this image in one word."},
    {"type": "image_url", "image_url": {"url": data_url}},
]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": content_parts}],
    "max_tokens": 64,
    "stream": False,
}
# For display: same structure but image URL truncated
payload_display = {
    "model": payload["model"],
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": content_parts[0]["text"]},
            {"type": "image_url", "image_url": {"url": "<data:%s;base64,... %d chars>" % (mime, len(data_url))}},
        ],
    }],
    "max_tokens": payload["max_tokens"],
    "stream": payload["stream"],
}

print("--- Curl equivalent ---")
print("curl -X POST '%s' \\" % url)
print("  -H 'Content-Type: application/json' \\")
print("  -d @payload.json")
print("")
print("--- Request JSON (payload.json, image as data URL in messages[].content[].image_url) ---")
print(json.dumps(payload_display, indent=2))
print("")
print("--- Response JSON ---")

body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    print(json.dumps(data, indent=2))
    content = ""
    if "choices" in data and len(data["choices"]) > 0:
        msg = data["choices"][0].get("message") or {}
        content = (msg.get("content") or "").strip()
    if content:
        print("", file=sys.stderr)
        print("Vision probe OK (model supports images): %s" % content, file=sys.stderr)
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
