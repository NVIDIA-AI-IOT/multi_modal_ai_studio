#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible Nemotron 3.5 ASR -> parakeet.cpp GGUF workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARAKEET_REPO="${PARAKEET_REPO:-https://github.com/mudler/parakeet.cpp.git}"
PARAKEET_REVISION="${PARAKEET_REVISION:-1bfbebfaaf493866f49597cd3b7901959d395c60}"
GGUF_REPO_REVISION="${GGUF_REPO_REVISION:-bf0af9f425fa01809cadec671b3cb672709d13e9}"
NEMOTRON_REVISION="${NEMOTRON_REVISION:-f3d333391852ba876df169dcc9ba902d25b6ab0b}"
NEMO_TOOLKIT_REVISION="${NEMO_TOOLKIT_REVISION:-1c82990befeb0f44640d460b2dde75fd47fa9b2f}"
MODEL_ID="${MODEL_ID:-nvidia/nemotron-3.5-asr-streaming-0.6b}"

WORK_ROOT="${PARAKEET_WORK_ROOT:-${REPO_ROOT}/benchmark-results/parakeet-nemotron}"
SOURCE_DIR="${WORK_ROOT}/parakeet.cpp"
MODEL_DIR="${WORK_ROOT}/models"
RESULT_DIR="${WORK_ROOT}/results"
CONVERTER_VENV="${WORK_ROOT}/converter-venv"
CONVERTER_SOURCE_ROOT="${WORK_ROOT}/sources"
NEMO_TOOLKIT_ARCHIVE="${CONVERTER_SOURCE_ROOT}/nemo-${NEMO_TOOLKIT_REVISION}.tar.gz"
NEMO_TOOLKIT_SOURCE="${CONVERTER_SOURCE_ROOT}/nemo-${NEMO_TOOLKIT_REVISION}"
NEMO_TOOLKIT_MARKER="${NEMO_TOOLKIT_SOURCE}/.mmas-archive-sha256"

CLI_IMAGE="${PARAKEET_CLI_IMAGE:-mmas/parakeet-nemotron-cli:${PARAKEET_REVISION}-cuda}"
SERVER_IMAGE="${PARAKEET_SERVER_IMAGE:-mmas/parakeet-nemotron-server:${PARAKEET_REVISION}-cuda}"
CUDA_ARCHS="${PARAKEET_CUDA_ARCHS:-87;110}"

HF_GGUF_BASE="https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/${GGUF_REPO_REVISION}"
HF_SOURCE_BASE="https://huggingface.co/${MODEL_ID}/resolve/${NEMOTRON_REVISION}"

F16_NAME="nemotron-3.5-asr-streaming-0.6b-f16.gguf"
Q8_NAME="nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
Q4_NAME="nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"
NEMO_NAME="nemotron-3.5-asr-streaming-0.6b.nemo"
F32_NAME="nemotron-3.5-asr-streaming-0.6b-f32.gguf"
LOCAL_Q8_NAME="nemotron-3.5-asr-streaming-0.6b-from-f32-q8_0.gguf"
LOCAL_Q4_NAME="nemotron-3.5-asr-streaming-0.6b-from-f32-q4_k.gguf"

declare -Ar PREBUILT_SHA256=(
    ["${F16_NAME}"]="b64413c3886edf2b45eb3e757f911f1bc8020b7cf157622cd0bd0452c6d84aac"
    ["${Q8_NAME}"]="ba2f13eccd4a5245be728f77e6149bd6a4fdcdd133ff2e08ac6005bcef7a99f1"
    ["${Q4_NAME}"]="5ad85eb3f3014c1a300d67b7ccbd23c38c4c952405cbe33a861e19fb2775e84b"
)
NEMO_SHA256="210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74"
NEMO_TOOLKIT_SHA256="79b2355001fed99621b04ede58f45376d8ae67cb01c3a9783f4a522ac8284b49"
SMOKE_API_CONTAINER=""

mkdir -p "${WORK_ROOT}" "${MODEL_DIR}" "${RESULT_DIR}" "${CONVERTER_SOURCE_ROOT}"

usage() {
    cat <<EOF
Usage: $0 COMMAND [ARGS]

Commands:
  doctor                       Print pinned revisions and prerequisites
  checkout                     Checkout pinned parakeet.cpp + ggml submodule
  build                        Build pinned CUDA 13 CLI and server images
  fetch-prebuilt [all|f16|q8_0|q4_k]
                               Download checksum-verified published GGUFs
  fetch-source                 Download the pinned original NeMo checkpoint
  setup-converter              Create the pinned Python conversion environment
  convert-f32                  Convert the pinned .nemo checkpoint to F32 GGUF
  quantize-local [all|q8_0|q4_k]
                               Quantize local F32 with the upstream CLI
  verify-models                Verify published checksums and print local hashes
  benchmark MANIFEST           Compare published f16/q8_0/q4_k on common WAVs
  benchmark-local MANIFEST     Compare f16 with locally generated q8_0/q4_k
  smoke-api WAV [MODEL]        Test POST /v1/audio/transcriptions on port 18083

The benchmark manifest is UTF-8 TSV:
  id<TAB>wav_path<TAB>language<TAB>reference transcript

All checkouts, models, and results remain under:
  ${WORK_ROOT}
EOF
}

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

cleanup_smoke_api() {
    if [[ -n "${SMOKE_API_CONTAINER}" ]]; then
        docker rm -f "${SMOKE_API_CONTAINER}" >/dev/null 2>&1 || true
    fi
}

sha_matches() {
    local path="$1"
    local expected="$2"
    [[ -f "${path}" ]] &&
        printf '%s  %s\n' "${expected}" "${path}" | sha256sum --check --status
}

download_checked() {
    local url="$1"
    local destination="$2"
    local expected="$3"
    local partial="${destination}.part"

    if sha_matches "${destination}" "${expected}"; then
        echo "Verified existing $(basename "${destination}")"
        return
    fi
    rm -f "${partial}"
    echo "Downloading $(basename "${destination}")"
    curl --fail --location --retry 5 --retry-all-errors \
        --output "${partial}" "${url}"
    printf '%s  %s\n' "${expected}" "${partial}" | sha256sum --check
    mv "${partial}" "${destination}"
}

checkout() {
    need git
    if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
        git clone --filter=blob:none "${PARAKEET_REPO}" "${SOURCE_DIR}"
    fi
    if [[ "$(git -C "${SOURCE_DIR}" remote get-url origin)" != "${PARAKEET_REPO}" ]]; then
        echo "Unexpected origin in ${SOURCE_DIR}; refusing to reuse it" >&2
        exit 1
    fi
    git -C "${SOURCE_DIR}" fetch --depth 1 origin "${PARAKEET_REVISION}"
    git -C "${SOURCE_DIR}" checkout --detach "${PARAKEET_REVISION}"
    git -C "${SOURCE_DIR}" submodule sync --recursive
    git -C "${SOURCE_DIR}" submodule update --init --recursive --depth 1
    [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${PARAKEET_REVISION}" ]]
    echo "Checked out parakeet.cpp ${PARAKEET_REVISION}"
    git -C "${SOURCE_DIR}" submodule status
}

build() {
    need docker
    checkout
    local common_args=(
        --build-arg BUILD_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04
        --build-arg RUNTIME_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04
        --build-arg "CMAKE_EXTRA_ARGS=-DPARAKEET_GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON"
        --build-arg "CUDA_ARCHS=${CUDA_ARCHS}"
    )
    docker build "${common_args[@]}" --target runtime -t "${CLI_IMAGE}" "${SOURCE_DIR}"
    docker build "${common_args[@]}" --target runtime-server -t "${SERVER_IMAGE}" "${SOURCE_DIR}"
    echo "Built ${CLI_IMAGE}"
    echo "Built ${SERVER_IMAGE}"
}

fetch_prebuilt() {
    need curl
    need sha256sum
    local selection="${1:-all}"
    local names=()
    case "${selection}" in
        all) names=("${F16_NAME}" "${Q8_NAME}" "${Q4_NAME}") ;;
        f16) names=("${F16_NAME}") ;;
        q8_0) names=("${Q8_NAME}") ;;
        q4_k) names=("${Q4_NAME}") ;;
        *) echo "Unknown prebuilt variant: ${selection}" >&2; exit 2 ;;
    esac
    local name
    for name in "${names[@]}"; do
        download_checked \
            "${HF_GGUF_BASE}/${name}" \
            "${MODEL_DIR}/${name}" \
            "${PREBUILT_SHA256[${name}]}"
    done
}

fetch_source() {
    need curl
    need sha256sum
    download_checked \
        "${HF_SOURCE_BASE}/${NEMO_NAME}" \
        "${MODEL_DIR}/${NEMO_NAME}" \
        "${NEMO_SHA256}"
}

setup_converter() {
    need python3
    need curl
    need sha256sum
    need tar
    checkout
    download_checked \
        "https://github.com/NVIDIA-NeMo/NeMo/archive/${NEMO_TOOLKIT_REVISION}.tar.gz" \
        "${NEMO_TOOLKIT_ARCHIVE}" \
        "${NEMO_TOOLKIT_SHA256}"
    if [[ ! -f "${NEMO_TOOLKIT_MARKER}" ]] ||
        [[ "$(<"${NEMO_TOOLKIT_MARKER}")" != "${NEMO_TOOLKIT_SHA256}" ]]; then
        mkdir -p "${NEMO_TOOLKIT_SOURCE}"
        tar -xzf "${NEMO_TOOLKIT_ARCHIVE}" \
            --strip-components=1 \
            -C "${NEMO_TOOLKIT_SOURCE}"
        printf '%s\n' "${NEMO_TOOLKIT_SHA256}" >"${NEMO_TOOLKIT_MARKER}"
    fi
    [[ -f "${NEMO_TOOLKIT_SOURCE}/pyproject.toml" ]]
    if [[ ! -x "${CONVERTER_VENV}/bin/pip" ]]; then
        if ! python3 -m venv "${CONVERTER_VENV}"; then
            local bootstrap="${WORK_ROOT}/virtualenv-bootstrap"
            echo "stdlib venv unavailable; bootstrapping virtualenv under ${bootstrap}"
            python3 -m pip install --target "${bootstrap}" "virtualenv==20.35.4"
            PYTHONPATH="${bootstrap}" python3 -m virtualenv "${CONVERTER_VENV}"
        fi
    fi
    "${CONVERTER_VENV}/bin/pip" install --upgrade "pip==26.1.1"
    "${CONVERTER_VENV}/bin/pip" install \
        "torch==2.8.0" \
        "gguf==0.19.0" \
        "${NEMO_TOOLKIT_SOURCE}[asr]"
    "${CONVERTER_VENV}/bin/python" -c \
        "from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt"
    echo "Converter ready with NeMo Toolkit ${NEMO_TOOLKIT_REVISION}"
}

convert_f32() {
    fetch_source
    checkout
    local converter_python="${PARAKEET_CONVERTER_PYTHON:-${CONVERTER_VENV}/bin/python}"
    if [[ ! -x "${converter_python}" ]]; then
        echo "Converter Python not found: ${converter_python}" >&2
        echo "Run setup-converter or set PARAKEET_CONVERTER_PYTHON." >&2
        exit 1
    fi
    if [[ -s "${MODEL_DIR}/${F32_NAME}" ]]; then
        echo "Keeping existing ${MODEL_DIR}/${F32_NAME}"
        return
    fi
    "${converter_python}" "${SOURCE_DIR}/scripts/convert_parakeet_to_gguf.py" \
        --model "${MODEL_DIR}/${NEMO_NAME}" \
        --dtype f32 \
        --output "${MODEL_DIR}/${F32_NAME}.part"
    mv "${MODEL_DIR}/${F32_NAME}.part" "${MODEL_DIR}/${F32_NAME}"
}

run_cli() {
    need docker
    docker run --rm --runtime nvidia --network none --ipc host \
        --user "$(id -u):$(id -g)" \
        -v "${WORK_ROOT}:/work" \
        "${CLI_IMAGE}" "$@"
}

quantize_local() {
    local selection="${1:-all}"
    [[ -s "${MODEL_DIR}/${F32_NAME}" ]] || {
        echo "Missing F32 model; run convert-f32 first" >&2
        exit 1
    }
    local variants=()
    case "${selection}" in
        all) variants=("q8_0:${LOCAL_Q8_NAME}" "q4_k:${LOCAL_Q4_NAME}") ;;
        q8_0) variants=("q8_0:${LOCAL_Q8_NAME}") ;;
        q4_k) variants=("q4_k:${LOCAL_Q4_NAME}") ;;
        *) echo "Unknown local quantization: ${selection}" >&2; exit 2 ;;
    esac
    local item quant name
    for item in "${variants[@]}"; do
        quant="${item%%:*}"
        name="${item#*:}"
        if [[ -s "${MODEL_DIR}/${name}" ]]; then
            echo "Keeping existing ${MODEL_DIR}/${name}"
            continue
        fi
        run_cli quantize \
            "/work/models/${F32_NAME}" \
            "/work/models/${name}.part" \
            "${quant}"
        mv "${MODEL_DIR}/${name}.part" "${MODEL_DIR}/${name}"
    done
}

verify_models() {
    need sha256sum
    local name path
    for name in "${F16_NAME}" "${Q8_NAME}" "${Q4_NAME}"; do
        path="${MODEL_DIR}/${name}"
        [[ -f "${path}" ]] || continue
        printf '%s  %s\n' "${PREBUILT_SHA256[${name}]}" "${path}" |
            sha256sum --check
    done
    for name in "${NEMO_NAME}" "${F32_NAME}" "${LOCAL_Q8_NAME}" "${LOCAL_Q4_NAME}"; do
        path="${MODEL_DIR}/${name}"
        [[ -f "${path}" ]] && sha256sum "${path}"
    done
    return 0
}

benchmark() {
    local manifest="${1:?benchmark requires a TSV manifest}"
    fetch_prebuilt all
    run_benchmark "${manifest}" "published" \
        "f16=${MODEL_DIR}/${F16_NAME}" \
        "q8_0=${MODEL_DIR}/${Q8_NAME}" \
        "q4_k=${MODEL_DIR}/${Q4_NAME}"
}

benchmark_local() {
    local manifest="${1:?benchmark-local requires a TSV manifest}"
    local name
    for name in "${F16_NAME}" "${LOCAL_Q8_NAME}" "${LOCAL_Q4_NAME}"; do
        [[ -s "${MODEL_DIR}/${name}" ]] || {
            echo "Missing ${MODEL_DIR}/${name}" >&2
            echo "Run fetch-prebuilt f16, convert-f32, and quantize-local all." >&2
            exit 1
        }
    done
    run_benchmark "${manifest}" "local" \
        "f16=${MODEL_DIR}/${F16_NAME}" \
        "local_q8_0=${MODEL_DIR}/${LOCAL_Q8_NAME}" \
        "local_q4_k=${MODEL_DIR}/${LOCAL_Q4_NAME}"
}

run_benchmark() {
    local manifest="$1"
    local result_label="$2"
    shift 2
    need python3
    python3 "${SCRIPT_DIR}/benchmark_parakeet_nemotron.py" \
        --docker-image "${CLI_IMAGE}" \
        --models "$@" \
        --manifest "${manifest}" \
        --output-dir "${RESULT_DIR}/$(date -u +%Y%m%dT%H%M%SZ)-${result_label}"
}

smoke_api() {
    local wav="${1:?smoke-api requires a WAV path}"
    local model="${2:-${MODEL_DIR}/${Q8_NAME}}"
    local name="mmas-parakeet-smoke-$$"
    local port="${PARAKEET_SERVER_PORT:-18083}"
    [[ -f "${wav}" ]] || { echo "WAV not found: ${wav}" >&2; exit 1; }
    [[ -f "${model}" ]] || { echo "Model not found: ${model}" >&2; exit 1; }
    need realpath
    wav="$(realpath "${wav}")"
    model="$(realpath "${model}")"
    need docker
    need curl
    docker run -d --name "${name}" --runtime nvidia --network host --ipc host \
        -v "${model}:/models/model.gguf:ro" \
        "${SERVER_IMAGE}" --model /models/model.gguf --port "${port}" >/dev/null
    SMOKE_API_CONTAINER="${name}"
    trap cleanup_smoke_api EXIT
    local ready=0
    local _attempt
    for _attempt in {1..120}; do
        if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
            ready=1
            break
        fi
        sleep 0.25
    done
    [[ "${ready}" == 1 ]] || {
        docker logs "${name}" >&2
        echo "Server did not become ready" >&2
        exit 1
    }
    curl --fail --silent --show-error \
        -F "file=@${wav}" \
        -F "model=parakeet" \
        -F "response_format=verbose_json" \
        -F "timestamp_granularities[]=word" \
        "http://127.0.0.1:${port}/v1/audio/transcriptions"
    echo
    docker rm -f "${name}" >/dev/null 2>&1 || true
    SMOKE_API_CONTAINER=""
    trap - EXIT
}

doctor() {
    echo "parakeet.cpp: ${PARAKEET_REVISION}"
    echo "ggml: $(git -C "${SOURCE_DIR}" rev-parse HEAD:third_party/ggml 2>/dev/null || echo e705c5fed490514458bdd2eaddc43bd098fcce9b)"
    echo "GGUF collection: ${GGUF_REPO_REVISION}"
    echo "Nemotron checkpoint: ${NEMOTRON_REVISION}"
    echo "NeMo Toolkit converter: ${NEMO_TOOLKIT_REVISION}"
    echo "CUDA architectures: ${CUDA_ARCHS}"
    echo "Work root: ${WORK_ROOT}"
    echo "Architecture: $(uname -m)"
    if [[ -r /etc/nv_tegra_release ]]; then
        head -n 1 /etc/nv_tegra_release
    fi
    command -v docker >/dev/null && docker version --format 'Docker: {{.Server.Version}}'
    command -v nvidia-smi >/dev/null &&
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
}

case "${1:-}" in
    doctor) doctor ;;
    checkout) checkout ;;
    build) build ;;
    fetch-prebuilt) fetch_prebuilt "${2:-all}" ;;
    fetch-source) fetch_source ;;
    setup-converter) setup_converter ;;
    convert-f32) convert_f32 ;;
    quantize-local) quantize_local "${2:-all}" ;;
    verify-models) verify_models ;;
    benchmark) benchmark "${2:-}" ;;
    benchmark-local) benchmark_local "${2:-}" ;;
    smoke-api) smoke_api "${2:-}" "${3:-}" ;;
    -h|--help|help|"") usage ;;
    *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
