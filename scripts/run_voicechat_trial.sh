#!/usr/bin/env bash
set -euo pipefail
VAW_TRIAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VAW_TRIAL_ROOT"
mkdir -p .runtime/sessions
export XDG_CONFIG_HOME="$VAW_TRIAL_ROOT/.runtime/config"
export PYTHONPATH="$VAW_TRIAL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${VAW_PYTHON:-python3}" -m multi_modal_ai_studio.cli.main \
    --host "${VAW_TRIAL_HOST:-127.0.0.1}" --port "${VAW_TRIAL_PORT:-8110}" \
    --preset "$VAW_TRIAL_ROOT/presets/nemotron-voicechat-thor.yaml" \
    --session-dir "$VAW_TRIAL_ROOT/.runtime/sessions"
