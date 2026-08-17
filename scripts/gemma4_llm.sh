#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Run the recommended text-only Gemma 4 E2B llama.cpp service on Jetson.

set -euo pipefail

GEMMA4_CONTAINER_NAME="${GEMMA4_CONTAINER_NAME:-mmas-gemma4-llm}"
GEMMA4_CACHE_DIR="${GEMMA4_CACHE_DIR:-${HOME%/}/.cache/huggingface}"
GEMMA4_MODEL="${GEMMA4_MODEL:-unsloth/gemma-4-E2B-it-GGUF:Q4_K_S}"
GEMMA4_MODEL_ALIAS="${GEMMA4_MODEL_ALIAS:-gemma-4-e2b}"
GEMMA4_PORT="${GEMMA4_PORT:-8080}"
GEMMA4_CONTEXT_SIZE="${GEMMA4_CONTEXT_SIZE:-2048}"
GEMMA4_ENABLE_MTP="${GEMMA4_ENABLE_MTP:-true}"
GEMMA4_REASONING="${GEMMA4_REASONING:-off}"
GEMMA4_PULL="${GEMMA4_PULL:-always}"
BASE_URL="${GEMMA4_BASE_URL:-http://127.0.0.1:${GEMMA4_PORT}}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        return 1
    fi
}

device_model() {
    if [[ -r /proc/device-tree/model ]]; then
        tr -d '\0' </proc/device-tree/model
    else
        echo "unknown"
    fi
}

default_image() {
    local model
    model="$(device_model)"
    case "${model,,}" in
        *thor*)
            echo "ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor"
            ;;
        *orin*)
            echo "ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin"
            ;;
        *)
            echo "Cannot select a Jetson llama.cpp image for: ${model}" >&2
            echo "Set LLAMA_CPP_IMAGE explicitly." >&2
            return 1
            ;;
    esac
}

llama_cpp_image() {
    if [[ -n "${LLAMA_CPP_IMAGE:-}" ]]; then
        echo "${LLAMA_CPP_IMAGE}"
    else
        default_image
    fi
}

doctor() {
    require_command docker
    require_command curl
    require_command python3

    echo "Device: $(device_model)"
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
    echo "Image: $(llama_cpp_image)"
    echo "Model: ${GEMMA4_MODEL}"
    echo "Model cache: ${GEMMA4_CACHE_DIR}"
    echo "Text-only: yes (--no-mmproj)"
    echo "MTP: ${GEMMA4_ENABLE_MTP}"
    echo "Reasoning: ${GEMMA4_REASONING}"
    echo "API: ${BASE_URL}/v1"
}

wait_for_health() {
    local attempts="${GEMMA4_HEALTH_ATTEMPTS:-300}"
    local delay="${GEMMA4_HEALTH_DELAY_SECONDS:-2}"
    local attempt

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if curl --fail --silent "${BASE_URL}/health" >/dev/null; then
            echo "Gemma 4 is healthy at ${BASE_URL}."
            return 0
        fi
        sleep "${delay}"
    done

    echo "Gemma 4 did not become healthy after $((attempts * delay)) seconds." >&2
    docker logs --tail 100 "${GEMMA4_CONTAINER_NAME}" >&2 || true
    return 1
}

start() {
    local image
    local -a mtp_args=()

    doctor
    image="$(llama_cpp_image)"
    if [[ "${GEMMA4_PULL}" != "never" ]]; then
        docker pull "${image}"
    fi
    mkdir -p "${GEMMA4_CACHE_DIR}/hub"
    docker rm -f "${GEMMA4_CONTAINER_NAME}" >/dev/null 2>&1 || true

    if [[ "${GEMMA4_ENABLE_MTP}" == "true" ]]; then
        mtp_args=(--spec-type draft-mtp --spec-draft-n-max 3)
    fi

    docker run -d \
        --name "${GEMMA4_CONTAINER_NAME}" \
        --restart unless-stopped \
        --runtime nvidia \
        --network host \
        -v "${GEMMA4_CACHE_DIR}:/root/.cache/huggingface" \
        -v "${GEMMA4_CACHE_DIR}/hub:/data/models/huggingface" \
        "${image}" \
        llama-server \
        -hf "${GEMMA4_MODEL}" \
        --no-mmproj \
        --alias "${GEMMA4_MODEL_ALIAS}" \
        --ctx-size "${GEMMA4_CONTEXT_SIZE}" \
        --parallel 1 \
        --gpu-layers all \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --reasoning "${GEMMA4_REASONING}" \
        --no-webui \
        --host 0.0.0.0 \
        --port "${GEMMA4_PORT}" \
        "${mtp_args[@]}" >/dev/null

    wait_for_health
    echo "Gemma 4 E2B is ready at ${BASE_URL}/v1."
    models
}

stop() {
    docker rm -f "${GEMMA4_CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Removed ${GEMMA4_CONTAINER_NAME}; model cache ${GEMMA4_CACHE_DIR} was preserved."
}

status() {
    docker ps -a \
        --filter "name=^/${GEMMA4_CONTAINER_NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

models() {
    curl --fail --silent --show-error "${BASE_URL}/v1/models"
    echo
}

logs() {
    docker logs --follow --tail 100 "${GEMMA4_CONTAINER_NAME}"
}

verify() {
    local response

    require_command curl
    require_command python3
    wait_for_health

    response="$(curl --fail --silent --show-error \
        "${BASE_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${GEMMA4_MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: Jetson voice ready\"}],\"temperature\":0,\"max_tokens\":20}")"

    python3 - "${response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
text = payload["choices"][0]["message"]["content"].strip()
if not text:
    raise SystemExit("LLM returned an empty response")
print(f"LLM API smoke test passed. Response: {text}")
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
