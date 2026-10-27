#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${repo_dir}/requirements.txt"
uv_bin="${UV_BIN:-uv}"

usage() {
    echo "Usage: $0 [--write|--check]" >&2
}

mode="${1:---write}"
if [[ "${mode}" != "--write" && "${mode}" != "--check" ]]; then
    usage
    exit 2
fi

if ! command -v "${uv_bin}" >/dev/null 2>&1; then
    echo "error: uv was not found; install it or set UV_BIN" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
body="${tmp_dir}/requirements.body"
candidate="${tmp_dir}/requirements.txt"

"${uv_bin}" export \
    --directory "${repo_dir}" \
    --format requirements.txt \
    --frozen \
    --no-dev \
    --extra audio \
    --extra camera \
    --extra video-input \
    --extra webrtc-camera \
    --no-emit-project \
    --no-hashes \
    --no-annotate \
    --no-header \
    --output-file "${body}" >/dev/null

{
    echo "# Generated from uv.lock for nSpect and other requirements.txt-based scanners."
    echo "# Includes core and shipped runtime extras; excludes the dev extra."
    echo "# Do not edit directly; run scripts/generate_nspect_requirements.sh."
    echo
    cat "${body}"
} >"${candidate}"

if [[ "${mode}" == "--check" ]]; then
    if ! diff -u "${target}" "${candidate}"; then
        echo "error: requirements.txt is out of date" >&2
        echo "run scripts/generate_nspect_requirements.sh" >&2
        exit 1
    fi
    echo "requirements.txt matches uv.lock"
else
    cp "${candidate}" "${target}"
    echo "wrote ${target}"
fi
