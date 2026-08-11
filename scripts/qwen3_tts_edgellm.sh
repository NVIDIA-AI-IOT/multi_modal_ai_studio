#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible Qwen3-TTS 0.6B workflow for TensorRT Edge-LLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EDGE_LLM_REF="${EDGE_LLM_REF:-7f061f21f0a581ba234a1e233c9315b89d8e47d6}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
MODEL_REVISION="${MODEL_REVISION:-85e237c12c027371202489a0ec509ded67b5e4b5}"
WORKSPACE_DIR="${EDGE_LLM_WORKSPACE:-${HOME%/}/tensorrt-edgellm-workspace}"
SOURCE_DIR="${EDGE_LLM_SOURCE_DIR:-${WORKSPACE_DIR}/TensorRT-Edge-LLM}"
MODEL_DIR="${WORKSPACE_DIR}/model"
ONNX_DIR="${WORKSPACE_DIR}/onnx-fp16"
BUILD_DIR="${WORKSPACE_DIR}/build-v0.9.1"
ENGINE_ROOT="${WORKSPACE_DIR}/engines-fp16"
RESULTS_DIR="${EDGE_LLM_RESULTS_DIR:-${REPO_ROOT}/benchmark-results/qwen3-tts-edgellm}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
BENCH_HELPER="${SCRIPT_DIR}/qwen3_tts_edgellm_benchmark.py"

# This SBSA base supplies CUDA 13.0 and TensorRT matching JetPack 7.0/7.1.
# The small local dev image adds the standard C/C++ headers needed by nvcc.
RUNTIME_BASE_IMAGE="${EDGE_LLM_RUNTIME_BASE_IMAGE:-ghcr.io/nvidia-ai-iot/vllm@sha256:08ec0a116a09e50b74e26eca91f52bfe2c95977f0eb4cb3e079cfb4b3c3932c6}"
RUNTIME_IMAGE="${EDGE_LLM_RUNTIME_IMAGE:-mmas/tensorrt-edgellm-dev:v0.9.1-cu130}"
EXPORT_IMAGE="${EDGE_LLM_EXPORT_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
BUILD_JOBS="${BUILD_JOBS:-4}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-1024 2048 4096}"
CHUNK_FRAMES="${CHUNK_FRAMES:-1 2 10}"

runtime_docker=(
    docker run --rm
    --runtime nvidia
    --network host
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${SOURCE_DIR}:/src"
    -v "${WORKSPACE_DIR}:/workspace"
    -v "${BUILD_DIR}:/build"
    -v "${RESULTS_DIR}:/results"
    -w /workspace
    "${RUNTIME_IMAGE}"
)

usage() {
    cat <<EOF
Usage: $0 COMMAND

Commands:
  doctor          Print platform, dependency, revision, and artifact checks
  prepare-source  Clone TensorRT Edge-LLM at the pinned v0.9.1 commit
  export-onnx     Download the pinned model and export FP16 ONNX on x86_64
  build-runtime   Build the three required C++ tools on this Thor
  build-engines   Build Talker, CodePredictor, and Code2Wav for 1024/2048/4096
  benchmark       Run Japanese/English, non-streaming and chunkFrames=1/2/10
  score-quality   Add CER/WER using an already-running OpenAI-compatible ASR
  summarize       Write CSV and Markdown summaries from recorded JSON

Large model, ONNX, engine, WAV, and benchmark artifacts stay under:
  ${WORKSPACE_DIR}
  ${RESULTS_DIR}
EOF
}

skip() {
    local reason="$1"
    mkdir -p "${RESULTS_DIR}"
    python3 "${BENCH_HELPER}" skip \
        --result "${RESULTS_DIR}/skip-$(date -u +%Y%m%dT%H%M%SZ).json" \
        --reason "${reason}"
    echo "SKIP: ${reason}" >&2
}

require_file() {
    local path="$1"
    local reason="$2"
    if [[ ! -f "${path}" ]]; then
        skip "${reason}: ${path}"
        return 1
    fi
}

prepare_source() {
    mkdir -p "${WORKSPACE_DIR}"
    if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
        git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git "${SOURCE_DIR}"
    fi
    git -C "${SOURCE_DIR}" fetch origin "${EDGE_LLM_REF}"
    git -C "${SOURCE_DIR}" checkout --detach "${EDGE_LLM_REF}"
    git -C "${SOURCE_DIR}" submodule update --init --recursive
    local actual_ref
    actual_ref="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
    if [[ "${actual_ref}" != "${EDGE_LLM_REF}" ]]; then
        echo "TensorRT Edge-LLM revision mismatch: ${actual_ref}" >&2
        return 1
    fi
}

doctor() {
    echo "Host architecture: $(uname -m)"
    if [[ -r /etc/nv_tegra_release ]]; then
        echo -n "L4T: "
        head -n 1 /etc/nv_tegra_release
    else
        echo "L4T: not detected"
    fi
    echo "TensorRT Edge-LLM ref: ${EDGE_LLM_REF}"
    echo "Model: ${MODEL_ID}@${MODEL_REVISION}"
    echo "Precision: FP16 (the only upstream-supported Qwen3-TTS precision)"
    echo "Batch size: 1"
    echo "Runtime image: ${RUNTIME_IMAGE}"
    echo "Runtime base image: ${RUNTIME_BASE_IMAGE}"
    echo "Workspace: ${WORKSPACE_DIR}"
    echo "Results: ${RESULTS_DIR}"
    command -v docker >/dev/null || {
        echo "docker is missing" >&2
        return 1
    }
    docker info >/dev/null
    if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
        echo "NVIDIA container runtime: available"
    else
        echo "NVIDIA container runtime: missing" >&2
        return 1
    fi
    if [[ -d "${SOURCE_DIR}/.git" ]]; then
        echo "Source ref: $(git -C "${SOURCE_DIR}" rev-parse HEAD)"
    else
        echo "Source: not prepared"
    fi
    for artifact in \
        "${ONNX_DIR}/llm/model.onnx" \
        "${ONNX_DIR}/code_predictor/model.onnx" \
        "${ONNX_DIR}/code2wav/model.onnx" \
        "${BUILD_DIR}/examples/llm/llm_build" \
        "${BUILD_DIR}/examples/multimodal/audio_build" \
        "${BUILD_DIR}/examples/omni/qwen3_tts_inference"; do
        if [[ -f "${artifact}" ]]; then
            echo "Found: ${artifact}"
        else
            echo "Missing: ${artifact}"
        fi
    done
}

prepare_runtime_image() {
    docker build \
        --build-arg "BASE_IMAGE=${RUNTIME_BASE_IMAGE}" \
        -f "${REPO_ROOT}/services/qwen3_tts_edgellm/Dockerfile" \
        -t "${RUNTIME_IMAGE}" \
        "${REPO_ROOT}/services/qwen3_tts_edgellm"
}

export_onnx() {
    if [[ "$(uname -m)" != "x86_64" ]]; then
        skip "upstream FP16 ONNX export requires an x86_64 Linux host with an NVIDIA GPU"
        return 3
    fi
    prepare_source
    mkdir -p "${MODEL_DIR}" "${ONNX_DIR}" "${HF_CACHE_DIR}" "${RESULTS_DIR}"

    local token_args=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        token_args=(--env HF_TOKEN)
    fi
    local force_export="${FORCE:-0}"
    docker run --rm --gpus all --ipc host \
        "${token_args[@]}" \
        -e "MODEL_ID=${MODEL_ID}" \
        -e "MODEL_REVISION=${MODEL_REVISION}" \
        -e "FORCE_EXPORT=${force_export}" \
        -v "${SOURCE_DIR}:/src:ro" \
        -v "${WORKSPACE_DIR}:/workspace" \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        -w /workspace \
        "${EXPORT_IMAGE}" \
        bash -lc '
            set -euo pipefail
            if [[ -f /workspace/onnx-fp16/code2wav/model.onnx &&
                  "${FORCE_EXPORT}" != "1" ]]; then
                echo "ONNX export already complete; set FORCE=1 to rebuild"
                exit 0
            fi
            # setuptools writes egg-info beside the source during a direct
            # install. Keep the pinned host checkout read-only and install
            # from a disposable container-local copy instead.
            cp -a /src /tmp/TensorRT-Edge-LLM
            pip install --disable-pip-version-check /tmp/TensorRT-Edge-LLM
            # The export dependencies resolve Torch 2.12.0. Replace the
            # export image development torchvision build with the
            # matching release before importing transformers image modules.
            pip install --disable-pip-version-check torchvision==0.27.0
            python3 -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=\"/workspace/model\")" \
                "${MODEL_ID}" "${MODEL_REVISION}"
            tensorrt-edgellm-export \
                /workspace/model \
                /workspace/onnx-fp16 \
                --dtype float16
        '

    python3 "${BENCH_HELPER}" manifest \
        --root "${ONNX_DIR}" \
        --kind onnx \
        --result "${RESULTS_DIR}/onnx-manifest.json"
    (
        cd "${WORKSPACE_DIR}"
        find onnx-fp16 -type f -print0 \
            | sort -z \
            | xargs -0 sha256sum > onnx-fp16.SHA256SUMS.tmp
        mv onnx-fp16.SHA256SUMS.tmp onnx-fp16.SHA256SUMS
    )
}

build_runtime() {
    prepare_source
    prepare_runtime_image
    mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"
    "${runtime_docker[@]}" cmake -U 'TensorRT_*' -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/opt/tensorrt \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/aarch64_linux_toolchain.cmake \
        -DEMBEDDED_TARGET=jetson-thor \
        -DCUDA_CTK_VERSION=13.0 \
        -DENABLE_CUTE_DSL=ALL
    "${runtime_docker[@]}" cmake --build /build \
        --parallel "${BUILD_JOBS}" \
        --target llm_build audio_build qwen3_tts_inference
}

measure_build() {
    local sequence_length="$1"
    local component="$2"
    shift 2
    local result_dir="${RESULTS_DIR}/build/${sequence_length}"
    mkdir -p "${result_dir}"
    python3 "${BENCH_HELPER}" measure \
        --result "${result_dir}/${component}.json" \
        --label "build-${component}-mxsl${sequence_length}" \
        --metadata "component=${component}" \
        --metadata "sequence_length=${sequence_length}" \
        -- "$@"
}

build_engines() {
    require_file "${ONNX_DIR}/llm/model.onnx" \
        "Talker ONNX is missing; run export-onnx on x86_64 and transfer the workspace" || return 3
    require_file "${ONNX_DIR}/code_predictor/model.onnx" \
        "CodePredictor ONNX is missing" || return 3
    require_file "${ONNX_DIR}/code2wav/model.onnx" \
        "Code2Wav ONNX is missing" || return 3
    require_file "${BUILD_DIR}/examples/llm/llm_build" \
        "runtime is missing; run build-runtime on the Thor" || return 3
    if [[ -f "${WORKSPACE_DIR}/onnx-fp16.SHA256SUMS" ]]; then
        (
            cd "${WORKSPACE_DIR}"
            sha256sum --check --strict onnx-fp16.SHA256SUMS
        )
    else
        echo "WARNING: onnx-fp16.SHA256SUMS is absent; transfer it with ONNX artifacts" >&2
    fi

    local sequence_length engine_dir
    for sequence_length in ${SEQUENCE_LENGTHS}; do
        engine_dir="${ENGINE_ROOT}/mxsl${sequence_length}"
        mkdir -p "${engine_dir}"

        if [[ "${FORCE:-0}" == "1" ||
              ! -f "${engine_dir}/talker/llm.engine" ]]; then
            measure_build "${sequence_length}" talker \
                "${runtime_docker[@]}" \
                /workspace/build-v0.9.1/examples/llm/llm_build \
                --onnxDir=/workspace/onnx-fp16/llm \
                --engineDir="/workspace/engines-fp16/mxsl${sequence_length}/talker" \
                --maxInputLen="${sequence_length}" \
                --maxKVCacheCapacity="${sequence_length}" \
                --maxBatchSize=1
        fi

        if [[ "${FORCE:-0}" == "1" ||
              ! -f "${engine_dir}/code_predictor/llm.engine" ]]; then
            measure_build "${sequence_length}" code_predictor \
                "${runtime_docker[@]}" \
                /workspace/build-v0.9.1/examples/llm/llm_build \
                --onnxDir=/workspace/onnx-fp16/code_predictor \
                --engineDir="/workspace/engines-fp16/mxsl${sequence_length}/code_predictor" \
                --maxInputLen="${sequence_length}" \
                --maxKVCacheCapacity="${sequence_length}" \
                --maxBatchSize=1
        fi

        if [[ "${FORCE:-0}" == "1" ||
              ! -f "${engine_dir}/code2wav/code2wav.engine" ]]; then
            measure_build "${sequence_length}" code2wav \
                "${runtime_docker[@]}" \
                /workspace/build-v0.9.1/examples/multimodal/audio_build \
                --onnxDir=/workspace/onnx-fp16/code2wav \
                --engineDir="/workspace/engines-fp16/mxsl${sequence_length}"
        fi

        python3 "${BENCH_HELPER}" manifest \
            --root "${engine_dir}" \
            --kind engine \
            --result "${RESULTS_DIR}/build/${sequence_length}/engine-manifest.json"
    done
}

write_inputs() {
    local input_dir="${WORKSPACE_DIR}/inputs"
    mkdir -p "${input_dir}"
    python3 "${BENCH_HELPER}" inputs --output-dir "${input_dir}"
}

run_trial() {
    local sequence_length="$1"
    local language="$2"
    local chunk="$3"
    local mode="nonstream"
    local stream_args=()
    if [[ "${chunk}" != "0" ]]; then
        mode="chunk${chunk}"
        stream_args=(--streaming "--chunkFrames=${chunk}")
    fi

    local trial_dir="${RESULTS_DIR}/inference/mxsl${sequence_length}/${language}/${mode}"
    local workspace_trial="/results/inference/mxsl${sequence_length}/${language}/${mode}"
    mkdir -p "${trial_dir}/audio"

    python3 "${BENCH_HELPER}" measure \
        --result "${trial_dir}/trial.json" \
        --label "infer-mxsl${sequence_length}-${language}-${mode}" \
        --metadata "sequence_length=${sequence_length}" \
        --metadata "language=${language}" \
        --metadata "chunk_frames=${chunk}" \
        --inference-output "${trial_dir}/output.json" \
        --profile-output "${trial_dir}/profile.json" \
        --audio-dir "${trial_dir}/audio" \
        -- \
        "${runtime_docker[@]}" \
        /workspace/build-v0.9.1/examples/omni/qwen3_tts_inference \
        "--talkerEngineDir=/workspace/engines-fp16/mxsl${sequence_length}/talker" \
        "--code2wavEngineDir=/workspace/engines-fp16/mxsl${sequence_length}/code2wav" \
        "--tokenizerDir=/workspace/engines-fp16/mxsl${sequence_length}/talker" \
        "--inputFile=/workspace/inputs/${language}.json" \
        "--outputFile=${workspace_trial}/output.json" \
        "--outputAudioDir=${workspace_trial}/audio" \
        "--profileOutputFile=${workspace_trial}/profile.json" \
        --dumpProfile \
        --batchSize=1 \
        "${stream_args[@]}"
}

benchmark() {
    require_file "${BUILD_DIR}/examples/omni/qwen3_tts_inference" \
        "runtime is missing; run build-runtime" || return 3
    write_inputs
    mkdir -p "${RESULTS_DIR}"
    # The runtime has no seed option. Run each language independently and retain
    # the WAV so stochastic quality differences remain auditable.
    local sequence_length language chunk engine_dir
    for sequence_length in ${SEQUENCE_LENGTHS}; do
        engine_dir="${ENGINE_ROOT}/mxsl${sequence_length}"
        if [[ ! -f "${engine_dir}/talker/llm.engine" ||
              ! -f "${engine_dir}/code_predictor/llm.engine" ||
              ! -f "${engine_dir}/code2wav/code2wav.engine" ]]; then
            skip "mxsl${sequence_length} engine set is incomplete"
            continue
        fi
        for language in ja en; do
            run_trial "${sequence_length}" "${language}" 0
            for chunk in ${CHUNK_FRAMES}; do
                run_trial "${sequence_length}" "${language}" "${chunk}"
            done
        done
    done
    summarize
}

score_quality() {
    local asr_url="${EDGE_LLM_ASR_URL:-http://127.0.0.1:8081/v1/audio/transcriptions}"
    local asr_model="${EDGE_LLM_ASR_MODEL:-nvidia/nemotron-3.5-asr-streaming-0.6b}"
    python3 "${BENCH_HELPER}" score-quality \
        --results-root "${RESULTS_DIR}" \
        --asr-url "${asr_url}" \
        --asr-model "${asr_model}"
    summarize
}

summarize() {
    mkdir -p "${RESULTS_DIR}"
    python3 "${BENCH_HELPER}" summarize \
        --results-root "${RESULTS_DIR}" \
        --csv "${RESULTS_DIR}/summary.csv" \
        --markdown "${RESULTS_DIR}/summary.md"
    echo "Summary: ${RESULTS_DIR}/summary.md"
}

case "${1:-}" in
    doctor)
        doctor
        ;;
    prepare-source)
        prepare_source
        ;;
    export-onnx)
        export_onnx
        ;;
    build-runtime)
        build_runtime
        ;;
    build-engines)
        build_engines
        ;;
    benchmark)
        benchmark
        ;;
    score-quality)
        score_quality
        ;;
    summarize)
        summarize
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
