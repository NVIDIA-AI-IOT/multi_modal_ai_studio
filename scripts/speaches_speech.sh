#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Run the recommended Speaches release-candidate speech service on Jetson.

set -euo pipefail

SPEACHES_IMAGE="${SPEACHES_IMAGE:-ghcr.io/nvidia-ai-iot/speaches:0.9.0-rc.3-cu130-sm87-sm110-auto}"
SPEACHES_CONTAINER_NAME="${SPEACHES_CONTAINER_NAME:-mmas-speaches}"
SPEACHES_MODEL_VOLUME="${SPEACHES_MODEL_VOLUME:-mmas-speaches-models}"
SPEACHES_PORT="${SPEACHES_PORT:-18080}"
SPEACHES_PULL="${SPEACHES_PULL:-always}"
ASR_MODEL="${ASR_MODEL:-Systran/faster-whisper-tiny.en}"
TTS_MODEL="${TTS_MODEL:-speaches-ai/Kokoro-82M-v1.0-ONNX-fp16}"
TTS_VOICE="${TTS_VOICE:-af_heart}"
BASE_URL="${SPEACHES_BASE_URL:-http://127.0.0.1:${SPEACHES_PORT}}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        return 1
    fi
}

doctor() {
    require_command docker
    require_command curl
    require_command python3

    echo "Architecture: $(uname -m)"
    if [[ -r /etc/nv_tegra_release ]]; then
        echo -n "L4T: "
        head -n 1 /etc/nv_tegra_release
    else
        echo "L4T: not detected"
    fi
    echo "Docker: $(docker version --format '{{.Server.Version}}')"
    echo -n "NVIDIA runtime: "
    if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
        echo "available"
    else
        echo "missing"
        return 1
    fi
    echo "Image: ${SPEACHES_IMAGE}"
    echo "ASR model: ${ASR_MODEL}"
    echo "TTS model: ${TTS_MODEL}"
    echo "API: ${BASE_URL}/v1"
}

wait_for_health() {
    local attempts="${SPEACHES_HEALTH_ATTEMPTS:-120}"
    local delay="${SPEACHES_HEALTH_DELAY_SECONDS:-2}"
    local attempt

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if curl --fail --silent "${BASE_URL}/health" >/dev/null; then
            echo "Speaches is healthy at ${BASE_URL}."
            return 0
        fi
        sleep "${delay}"
    done

    echo "Speaches did not become healthy after $((attempts * delay)) seconds." >&2
    docker logs --tail 100 "${SPEACHES_CONTAINER_NAME}" >&2 || true
    return 1
}

encoded_model_id() {
    python3 - "$1" <<'PY'
import sys
from urllib.parse import quote

print(quote(sys.argv[1], safe=""))
PY
}

download_model() {
    local model_id="$1"
    local encoded_id
    encoded_id="$(encoded_model_id "${model_id}")"

    if curl --fail --silent "${BASE_URL}/v1/models/${encoded_id}" >/dev/null 2>&1; then
        echo "Model already available: ${model_id}"
        return 0
    fi

    echo "Downloading model: ${model_id}"
    curl --fail --silent --show-error \
        -X POST "${BASE_URL}/v1/models/${encoded_id}" >/dev/null
}

start() {
    doctor

    if [[ "${SPEACHES_PULL}" != "never" ]]; then
        docker pull "${SPEACHES_IMAGE}"
    fi
    docker volume inspect "${SPEACHES_MODEL_VOLUME}" >/dev/null 2>&1 ||
        docker volume create "${SPEACHES_MODEL_VOLUME}" >/dev/null
    docker rm -f "${SPEACHES_CONTAINER_NAME}" >/dev/null 2>&1 || true

    docker run -d \
        --name "${SPEACHES_CONTAINER_NAME}" \
        --restart unless-stopped \
        --runtime nvidia \
        --network host \
        --ipc host \
        -e "PORT=${SPEACHES_PORT}" \
        -e ENABLE_UI=false \
        -e "LOOPBACK_HOST_URL=http://127.0.0.1:${SPEACHES_PORT}" \
        -e WHISPER__INFERENCE_DEVICE=cuda \
        -e WHISPER__COMPUTE_TYPE=float16 \
        -v "${SPEACHES_MODEL_VOLUME}:/data/models/huggingface" \
        "${SPEACHES_IMAGE}" >/dev/null

    wait_for_health
    download_model "${ASR_MODEL}"
    download_model "${TTS_MODEL}"
    echo "Speaches ASR and TTS are ready. Connect a separate OpenAI-compatible LLM."
    models
}

stop() {
    docker rm -f "${SPEACHES_CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Removed ${SPEACHES_CONTAINER_NAME}; model volume ${SPEACHES_MODEL_VOLUME} was preserved."
}

status() {
    docker ps -a \
        --filter "name=^/${SPEACHES_CONTAINER_NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

models() {
    curl --fail --silent --show-error "${BASE_URL}/v1/models"
    echo
}

logs() {
    docker logs --follow --tail 100 "${SPEACHES_CONTAINER_NAME}"
}

verify() {
    require_command curl
    require_command python3
    wait_for_health

    local work_dir
    work_dir="$(mktemp -d)"
    trap 'rm -rf -- "${work_dir}"' RETURN

    curl --fail --silent --show-error \
        "${BASE_URL}/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d "$(printf \
            '{"model":"%s","input":"Hello from the Jetson speech quick start.","voice":"%s","response_format":"mp3"}' \
            "${TTS_MODEL}" "${TTS_VOICE}")" \
        --output "${work_dir}/speech.mp3"

    curl --fail --silent --show-error \
        "${BASE_URL}/v1/audio/transcriptions" \
        -F "file=@${work_dir}/speech.mp3" \
        -F "model=${ASR_MODEL}" \
        -F language=en \
        -F response_format=json \
        --output "${work_dir}/transcription.json"

    python3 - "${work_dir}/transcription.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
text = payload.get("text", "").strip()
if not text:
    raise SystemExit("ASR returned an empty transcript")
print(f"Speech API smoke test passed. Transcript: {text}")
PY
}

case "${1:-}" in
    doctor)
        doctor
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    status)
        status
        ;;
    models)
        models
        ;;
    verify)
        verify
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {doctor|start|stop|restart|status|models|verify|logs}" >&2
        exit 2
        ;;
esac
