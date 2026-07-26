#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Build and run the public-model speech stack without Riva or NGC credentials.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPEECH_DIR="${REPO_ROOT}/services/openai_speech"
CACHE_VOLUME="${CACHE_VOLUME:-mmas-hf-cache}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-4B}"
LLM_SERVED_NAME="${LLM_SERVED_NAME:-${LLM_MODEL}}"
LLM_IMAGE="${LLM_IMAGE:-vllm/vllm-openai:v0.25.0}"
ASR_REVISION="${ASR_REVISION:-f3d333391852ba876df169dcc9ba902d25b6ab0b}"
LLM_REVISION="${LLM_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"
MAGPIE_REVISION="${MAGPIE_REVISION:-v2607}"

DEFAULT_SPEECH_BASE_IMAGE="nvcr.io/nvidia/pytorch:26.06-py3"
if [[ -r /etc/nv_tegra_release ]] &&
    grep -q '^# R39 ' /etc/nv_tegra_release; then
    DEFAULT_SPEECH_BASE_IMAGE="nvcr.io/nvidia/pytorch:26.04-py3"
fi
SPEECH_BASE_IMAGE="${SPEECH_BASE_IMAGE:-${DEFAULT_SPEECH_BASE_IMAGE}}"
TTS_NEMO_REF="${TTS_NEMO_REF:-2639d4bef8d1450782263a8f616242acfb6fecb9}"

docker_common=(
    --runtime nvidia
    --network host
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
)

build() {
    docker build \
        -f "${SPEECH_DIR}/Dockerfile.asr" \
        --build-arg "BASE_IMAGE=${SPEECH_BASE_IMAGE}" \
        -t mmas/nemotron-asr-openai:0.1.0 \
        "${SPEECH_DIR}"
    docker build \
        -f "${SPEECH_DIR}/Dockerfile.tts" \
        --build-arg "BASE_IMAGE=${SPEECH_BASE_IMAGE}" \
        --build-arg "NEMO_REF=${TTS_NEMO_REF}" \
        -t mmas/magpie-tts-openai:0.1.0 \
        "${SPEECH_DIR}"
}

doctor() {
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
    echo "LLM image: ${LLM_IMAGE}"
    echo "LLM model: ${LLM_MODEL}"
    echo "Speech base image: ${SPEECH_BASE_IMAGE}"
    echo "NeMo Speech revision: ${TTS_NEMO_REF}"
}

replace_container() {
    local name="$1"
    shift
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run -d --name "${name}" --restart unless-stopped "$@"
}

start() {
    doctor

    docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1 ||
        docker volume create "${CACHE_VOLUME}" >/dev/null

    local hf_token_args=()
    local llm_cache_args=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf_token_args=(-e "HF_TOKEN=${HF_TOKEN}")
    fi
    if [[ "${LLM_KV_CACHE_MEMORY_BYTES:-2G}" != "auto" ]]; then
        llm_cache_args=(
            --kv-cache-memory-bytes
            "${LLM_KV_CACHE_MEMORY_BYTES:-2G}"
        )
    fi

    replace_container mmas-nemotron-asr \
        "${docker_common[@]}" \
        "${hf_token_args[@]}" \
        -e SPEECH_EAGER_LOAD="${SPEECH_EAGER_LOAD:-1}" \
        -e SPEECH_MODEL_REVISION="${ASR_REVISION}" \
        -e HF_HOME=/models/huggingface \
        -v "${CACHE_VOLUME}:/models/huggingface" \
        mmas/nemotron-asr-openai:0.1.0

    replace_container mmas-vllm \
        "${docker_common[@]}" \
        "${hf_token_args[@]}" \
        -e HF_HOME=/models/huggingface \
        -e HF_HUB_CACHE=/models/huggingface/hub \
        -v "${CACHE_VOLUME}:/models/huggingface" \
        --entrypoint vllm \
        "${LLM_IMAGE}" \
        serve "${LLM_MODEL}" \
        --revision "${LLM_REVISION}" \
        --served-model-name "${LLM_SERVED_NAME}" \
        --port 8000 \
        --dtype "${LLM_DTYPE:-bfloat16}" \
        --max-model-len "${LLM_MAX_MODEL_LEN:-8192}" \
        --max-num-seqs "${LLM_MAX_NUM_SEQS:-2}" \
        --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION:-0.30}" \
        "${llm_cache_args[@]}"

    replace_container mmas-magpie-tts \
        "${docker_common[@]}" \
        "${hf_token_args[@]}" \
        -e SPEECH_EAGER_LOAD="${SPEECH_EAGER_LOAD:-1}" \
        -e SPEECH_MODEL_REVISION="${MAGPIE_REVISION}" \
        -e HF_HOME=/models/huggingface \
        -v "${CACHE_VOLUME}:/models/huggingface" \
        mmas/magpie-tts-openai:0.1.0

    echo "Services started. First model load can take several minutes."
    status
}

stop() {
    docker rm -f mmas-nemotron-asr mmas-vllm mmas-magpie-tts \
        >/dev/null 2>&1 || true
    echo "Removed MMAS open-model service containers; model cache was preserved."
}

status() {
    docker ps -a \
        --filter name=mmas-nemotron-asr \
        --filter name=mmas-vllm \
        --filter name=mmas-magpie-tts \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

verify_llm() {
    curl --fail --silent --show-error \
        http://localhost:8000/v1/models
    echo
    curl --fail --silent --show-error \
        http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "$(printf \
            '{"model":"%s","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":64}' \
            "${LLM_SERVED_NAME}")"
    echo
}

case "${1:-}" in
    build)
        build
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
    doctor)
        doctor
        ;;
    verify-llm)
        verify_llm
        ;;
    *)
        echo "Usage: $0 {build|start|stop|restart|status|doctor|verify-llm}" >&2
        exit 2
        ;;
esac
