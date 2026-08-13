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
- `WS /v1/realtime` in ASR mode (OpenAI Realtime GA transcription sessions)
- `POST /v1/audio/speech` and `WS /v1/realtime` in TTS mode

The Realtime ASR endpoint uses Nemotron's native cache-aware streaming path.
It emits VAD speech boundaries, incremental transcript deltas, and a final
transcript without repeatedly transcribing a growing audio file. The default
three-token right context corresponds to a 320 ms model chunk. Set
`SPEECH_REALTIME_LOOKAHEAD_TOKENS` to `0`, `1`, `3`, `6`, or `13` to choose the
model's 80, 160, 320, 560, or 1120 ms latency/accuracy point.

Magpie supports exact-text REST and Realtime speech service boundaries. Realtime
sessions accept a user `input_text` item followed by `response.create`, emit
`response.output_audio.delta` PCM frames, and accept `response.cancel`. The
current public Magpie checkpoint still produces a complete waveform internally;
therefore the first delta follows model completion rather than model-native
incremental generation. The Realtime boundary nevertheless provides standard
event framing, exact response correlation, and prompt cancellation of audio
delivery. Cancellation does not yet interrupt an already-running Magpie model
call, so the GPU may remain occupied until that internal call returns. The ASR
session continues to feed MMAS's independent LLM and TTS stages.

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

# Realtime GA transcription through the MMAS provider-contract client
python ../../scripts/test_realtime_transcription.py sample.wav \
  --url 'ws://127.0.0.1:8081/v1/realtime' \
  --api-style openai-ga \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --language en-US
```

These examples match the English-first public Jetson smoke test. For P2
multilingual validation, change both ASR and TTS language fields together.
