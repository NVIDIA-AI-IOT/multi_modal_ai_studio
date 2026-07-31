#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Isolated build, benchmark, and service lifecycle for faster-qwen3-tts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_DIR="${REPO_ROOT}/services/faster_qwen3_tts"
IMAGE="${FQWEN_IMAGE:-mmas/faster-qwen3-tts:a70afc0}"
BASE_IMAGE="${FQWEN_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1}"
CONTAINER="${FQWEN_CONTAINER:-mmas-faster-qwen3-tts}"
PORT="${FQWEN_PORT:-18082}"
MODEL="${FQWEN_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
MODEL_REVISION="${FQWEN_MODEL_REVISION:-85e237c12c027371202489a0ec509ded67b5e4b5}"
SPEAKER="${FQWEN_SPEAKER:-ono_anna}"
BACKEND="${FQWEN_BACKEND:-cuda-graph}"
QUANT="${FQWEN_QUANT:-BF16}"
CHUNK_SIZE="${FQWEN_CHUNK_SIZE:-2}"
CACHE_ROOT="${FQWEN_CACHE_ROOT:-${HOME%/}/.cache/mmas-faster-qwen3-tts}"
RESULTS_DIR="${FQWEN_RESULTS_DIR:-${REPO_ROOT}/benchmark-results/faster-qwen3-tts}"
UPSTREAM_COMMIT="a70afc0f81f7f5f8801c3227968f1102f43f211c"
QWENTTS_PYTHON_COMMIT="b0b2da11293fb5a3f84fafc0a4c64524d7635b88"
QWENTTS_CPP_COMMIT="7df559a8ca25f66fee02970514ebe5f01dee9055"
QWENTTS_SM110_LIB="${CACHE_ROOT}/qwentts-sm110/qwentts-cpp-python/src/qwentts_cpp/lib/libqwen.so"

mkdir -p \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/qwentts" \
    "${CACHE_ROOT}/qwentts-sm110" \
    "${RESULTS_DIR}"

docker_common=(
    --runtime nvidia
    --network host
    --ipc host
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp/mmas-faster-qwen3-tts
    -e USER=jetson
    -e LOGNAME=jetson
    -e HF_HOME=/models/huggingface
    -e HF_HUB_CACHE=/models/huggingface/hub
    -v "${CACHE_ROOT}/huggingface:/models/huggingface"
    -v "${CACHE_ROOT}/qwentts:/models/qwentts"
    -v "${CACHE_ROOT}/qwentts-sm110:${CACHE_ROOT}/qwentts-sm110"
)

build() {
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "FASTER_QWEN3_TTS_COMMIT=${UPSTREAM_COMMIT}" \
        -t "${IMAGE}" \
        "${SERVICE_DIR}"
}

build_ggml_sm110() {
    local build_root="${CACHE_ROOT}/qwentts-sm110"
    mkdir -p "${build_root}"
    docker run --rm \
        "${docker_common[@]}" \
        -v "${build_root}:/work" \
        --entrypoint bash \
        "${IMAGE}" \
        -lc "
            set -euo pipefail
            if [[ ! -d /work/qwentts-cpp-python/.git ]]; then
                git clone https://github.com/andimarafioti/qwentts-cpp-python.git /work/qwentts-cpp-python
            fi
            git -C /work/qwentts-cpp-python fetch origin ${QWENTTS_PYTHON_COMMIT}
            git -C /work/qwentts-cpp-python checkout --detach ${QWENTTS_PYTHON_COMMIT}
            if [[ ! -d /work/qwentts.cpp/.git ]]; then
                git clone https://github.com/ServeurpersoCom/qwentts.cpp.git /work/qwentts.cpp
            fi
            git -C /work/qwentts.cpp fetch origin ${QWENTTS_CPP_COMMIT}
            git -C /work/qwentts.cpp checkout --detach ${QWENTTS_CPP_COMMIT}
            git -C /work/qwentts.cpp submodule update --init --recursive
            cd /work/qwentts-cpp-python
            QWENTTS_CPP_NO_STRIP=1 python3 scripts/build_native.py \
                --source /work/qwentts.cpp \
                --build-dir /work/build-sm110 \
                --backend cuda \
                --clean \
                --jobs '${FQWEN_BUILD_JOBS:-2}' \
                --cmake-arg=-G \
                --cmake-arg=Ninja \
                --cmake-arg=-DCMAKE_CUDA_ARCHITECTURES=110
            test -f src/qwentts_cpp/lib/libqwen.so
        "
    echo "Built pinned SM 11.0 qwentts.cpp libraries: ${QWENTTS_SM110_LIB}"
}

doctor() {
    echo "Host: $(hostname)"
    echo "Architecture: $(uname -m)"
    if [[ -r /etc/nv_tegra_release ]]; then
        head -n 1 /etc/nv_tegra_release
    fi
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "Image: ${IMAGE}"
    echo "Model: ${MODEL}@${MODEL_REVISION}"
    echo "Cache: ${CACHE_ROOT}"
    echo "Results: ${RESULTS_DIR}"
    echo "Service: http://127.0.0.1:${PORT}/v1"
}

start() {
    docker image inspect "${IMAGE}" >/dev/null
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${CONTAINER}" \
        --restart unless-stopped \
        "${docker_common[@]}" \
        -e "QWEN_TTS_MODEL=${MODEL}" \
        -e "QWEN_TTS_MODEL_REVISION=${MODEL_REVISION}" \
        -e "QWEN_TTS_SPEAKER=${SPEAKER}" \
        -e "QWEN_TTS_BACKEND=${BACKEND}" \
        -e "QWEN_TTS_QUANT=${QUANT}" \
        -e "QWEN_TTS_CHUNK_SIZE=${CHUNK_SIZE}" \
        -e "QWEN_TTS_GGML_CACHE_DIR=/models/qwentts" \
        -e "QWEN_TTS_QWENTTS_LIBRARY_PATH=${FQWEN_QWENTTS_LIB:-}" \
        -e "QWEN_TTS_PORT=${PORT}" \
        "${IMAGE}" >/dev/null
    echo "Started ${CONTAINER}; model initialization continues in the background."
}

stop() {
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    echo "Removed ${CONTAINER}; caches and benchmark results were preserved."
}

status() {
    docker ps -a \
        --filter "name=^/${CONTAINER}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

logs() {
    docker logs --tail "${FQWEN_LOG_LINES:-120}" "${CONTAINER}"
}

benchmark_one() {
    local backend="$1"
    local quant="${2:-BF16}"
    local extra=()
    if [[ -n "${FQWEN_ASR_URL:-}" ]]; then
        extra+=(--asr-url "${FQWEN_ASR_URL}")
    fi
    if [[ -n "${FQWEN_QWENTTS_LIB:-}" ]]; then
        extra+=(--qwentts-library-path "${FQWEN_QWENTTS_LIB}")
    fi
    docker run --rm \
        "${docker_common[@]}" \
        -v "${REPO_ROOT}:/workspace" \
        --entrypoint python3 \
        "${IMAGE}" \
        /workspace/benchmarks/faster_qwen3_tts_benchmark.py \
        --backend "${backend}" \
        --quant "${quant}" \
        --model "${MODEL}" \
        --model-revision "${MODEL_REVISION}" \
        --speaker "${SPEAKER}" \
        --chunk-sizes "${FQWEN_CHUNK_SIZES:-1,2,10}" \
        --max-seq-len "${FQWEN_MAX_SEQ_LEN:-2048}" \
        --max-new-tokens "${FQWEN_MAX_NEW_TOKENS:-192}" \
        --ggml-cache-dir /models/qwentts \
        --output-dir /workspace/benchmark-results/faster-qwen3-tts \
        "${extra[@]}"
}

benchmark_all() {
    benchmark_one baseline BF16
    benchmark_one cuda-graph BF16
    benchmark_one ggml Q8_0
    benchmark_one ggml Q4_K_M
}

verify_api() {
    curl --fail --silent --show-error \
        "http://127.0.0.1:${PORT}/health"
    echo
    local output="${RESULTS_DIR}/api-smoke.wav"
    curl --fail --silent --show-error \
        "http://127.0.0.1:${PORT}/v1/audio/speech" \
        -H 'Content-Type: application/json' \
        -d "$(printf \
            '{"model":"%s","input":"今日は実機で音声合成を検証します。","voice":"Ono_Anna","response_format":"wav"}' \
            "${MODEL}")" \
        --output "${output}"
    file "${output}"
}

case "${1:-}" in
    build)
        build
        ;;
    build-ggml-sm110)
        build_ggml_sm110
        ;;
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
    logs)
        logs
        ;;
    benchmark)
        benchmark_one "${2:-cuda-graph}" "${3:-BF16}"
        ;;
    benchmark-all)
        benchmark_all
        ;;
    verify-api)
        verify_api
        ;;
    *)
        echo "Usage: $0 {build|build-ggml-sm110|doctor|start|stop|restart|status|logs|benchmark [baseline|cuda-graph|ggml] [BF16|Q8_0|Q4_K_M]|benchmark-all|verify-api}" >&2
        exit 2
        ;;
esac
