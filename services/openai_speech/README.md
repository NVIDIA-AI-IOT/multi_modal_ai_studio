<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenAI-compatible NVIDIA speech services

This directory exposes NVIDIA's public speech checkpoints without NVIDIA Riva:

- ASR: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- TTS: `nvidia/magpie_tts_multilingual_357m`, pinned to revision `v2607`

The Jetson launcher pins the exact ASR, Qwen, and NeMo Speech revisions used
for the Thor/Orin qualification. See
[`docs/setup_open_models_jetson.md`](../../docs/setup_open_models_jetson.md)
for the revision table, performance results, and the Magpie real-time-factor
limitation.

The same application image runs in either `asr` or `tts` mode.  Run two
containers to keep model memory, health checks, and benchmarking independent.

Implemented endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions` in ASR mode
- `POST /v1/audio/speech` in TTS mode

The first milestone intentionally implements the standard REST endpoints.
Realtime microphone transcription and barge-in are layered on top in MMAS and
will use the same engines.

## Local development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Heavy model dependencies are optional so API tests do not download checkpoints:

```bash
pip install -e ".[asr]"  # ASR service
pip install -e ".[tts]"  # TTS service
```

## Run

```bash
# ASR
SPEECH_SERVICE_MODE=asr \
python -m openai_speech

# TTS
SPEECH_SERVICE_MODE=tts \
python -m openai_speech
```

Models are loaded lazily on the first inference request. Set
`SPEECH_EAGER_LOAD=1` to load during startup and make readiness reflect model
availability.

## API examples

```bash
curl http://localhost:8081/v1/audio/transcriptions \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F language=en-US \
  -F file=@sample.wav

curl http://localhost:8082/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/magpie_tts_multilingual_357m",
    "input": "Hello. This voice pipeline is running on Jetson Thor.",
    "voice": "Sofia",
    "response_format": "wav",
    "language": "en"
  }' \
  --output speech.wav
```

These examples match the English-first public Jetson smoke test. For P2
multilingual validation, change both ASR and TTS language fields together.
