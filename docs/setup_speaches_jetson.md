<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speaches on Jetson: recommended quick start (release candidate)

This is the recommended path for bringing up a local MMAS voice assistant on
Jetson. One Speaches container provides OpenAI-compatible Faster-Whisper ASR
and Kokoro TTS. The LLM remains an independent OpenAI-compatible Chat
Completions service, so it can be sized for the device and changed without
restarting speech.

```text
browser microphone
  -> Speaches Faster-Whisper (:18080)
  -> text-only Gemma 4 E2B llama.cpp (:8080)
  -> Speaches Kokoro (:18080)
  -> browser speaker
```

No Riva installation, NGC entitlement, or hosted API key is required. The
Jetson Speaches image is currently a release candidate, not a production or GA
claim.

## Supported images

The launcher pins this ARM64 CUDA 13 image:

```text
ghcr.io/nvidia-ai-iot/speaches:0.9.0-rc.3-cu130-sm87-sm110-auto
```

It contains CUDA code for SM 8.7 (Jetson Orin) and SM 11.0 (Jetson Thor). The
published digest used for qualification is:

```text
sha256:abb7d669e73a32f8055500100cc05aa1b5e2ef1b7280d71a5391d458681e4d69
```

The Gemma launcher follows the device-specific llama.cpp images used by the
[Jetson AI Lab Gemma 4 tutorial](https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/):

| Device | llama.cpp image |
| --- | --- |
| Jetson Orin | `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` |
| Jetson Thor | `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor` |

Set `LLAMA_CPP_IMAGE` to pin or replace that image without editing the script.

Use a Jetson Linux release whose GPU driver can run CUDA 13 containers. Keep
enough free storage for the container and downloaded model cache. The starter
models are intentionally small:

| Stage | Default model | Purpose |
| --- | --- | --- |
| ASR | `Systran/faster-whisper-tiny.en` | Small English smoke-test model |
| TTS | `speaches-ai/Kokoro-82M-v1.0-ONNX-fp16` | Multilingual Kokoro ONNX model |
| Voice | `af_heart` | English Kokoro voice |
| LLM | `unsloth/gemma-4-E2B-it-GGUF:Q4_K_S` | Text-only Gemma 4 E2B with optional MTP |

## 1. Install MMAS

From the repository root:

```bash
./scripts/setup_dev.sh
source .venv/bin/activate
```

Docker and the NVIDIA container runtime must already be available. Check the
host before downloading anything:

```bash
./scripts/speaches_speech.sh doctor
```

## 2. Start and verify speech

```bash
./scripts/speaches_speech.sh start
./scripts/speaches_speech.sh verify
```

`start` pulls the pinned image, starts one GPU container on port `18080`, and
downloads the two default models into the persistent Docker volume
`mmas-speaches-models`. Re-running it is safe. `verify` synthesizes a short MP3
and transcribes it through the two public APIs.

Useful lifecycle commands are:

```bash
./scripts/speaches_speech.sh status
./scripts/speaches_speech.sh models
./scripts/speaches_speech.sh logs
./scripts/speaches_speech.sh stop
```

`stop` removes the container but preserves downloaded models.

## 3. Start the independent LLM

The recommended LLM is Gemma 4 E2B Q4_K_S served by the Jetson llama.cpp
container:

```bash
./scripts/gemma4_llm.sh start
./scripts/gemma4_llm.sh verify
```

The launcher detects Jetson Orin or Thor and pulls the corresponding Jetson AI
Lab llama.cpp image. It limits context to 4096 tokens, uses one server slot,
enables Gemma's small MTP draft, and disables hidden reasoning by default.
Reasoning is disabled because a short voice response should not wait several
seconds for an internal thinking trace before its first audible token.
`--no-mmproj` prevents the combined Vision/Audio projector (about 986 MB) from
being downloaded or loaded; MMAS vision is disabled in this preset.

The model, quantization, and MTP arguments match the command planned for the
Jetson AI Lab model page:

```bash
llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --reasoning off
```

MMAS adds `--no-mmproj` because this voice pipeline uses Gemma only as a text
LLM. The launcher also mounts `${HOME}/.cache/huggingface` at the documented
container path and maps its `hub` directory to the cache path currently set by
the Jetson llama.cpp image. This keeps downloads reusable across both layouts.
Useful lifecycle commands are:

```bash
./scripts/gemma4_llm.sh status
./scripts/gemma4_llm.sh models
./scripts/gemma4_llm.sh logs
./scripts/gemma4_llm.sh stop
```

Set `GEMMA4_ENABLE_MTP=false` before `start` to disable MTP. You can instead
set `GEMMA4_REASONING=auto` (or `on`) before `start` when a use case benefits
from a reasoning trace and can accept the extra time to first speech. You can
instead use vLLM, Ollama, a hosted service, or another compatible server;
change `llm.api_base` and `llm.model` in the UI or in a copied preset.

For a smaller-memory fallback, serve Qwen 2.5 0.5B with any compatible server
and point the preset at it. Gemma 4 E2B is the default because it matches the
Jetson AI Lab path and the pipeline used for this project's Orin Nano tests.

## 4. Start MMAS

```bash
multi-modal-ai-studio --preset speaches-jetson --port 8092
```

Open `https://localhost:8092`, or `https://JETSON_IP:8092` from another machine,
accept the local development certificate, and allow browser microphone access.
The recommended preset uses the portable REST ASR and REST TTS path.

To exercise Realtime transcription instead, use:

```bash
multi-modal-ai-studio --preset speaches-realtime-asr-jetson --port 8092
```

Speaches currently performs VAD and sends completed speech chunks to
Faster-Whisper. This provides a Realtime WebSocket session and speech-boundary
events, but it is not model-native token-by-token Faster-Whisper decoding and
may not provide partial transcript deltas. The REST preset is therefore the
clearest baseline; use the Realtime preset to evaluate endpointing and
barge-in.

## API endpoints

The same service supports both preset choices:

| API | Endpoint |
| --- | --- |
| Health | `GET http://localhost:18080/health` |
| Models | `GET http://localhost:18080/v1/models` |
| REST ASR | `POST http://localhost:18080/v1/audio/transcriptions` |
| REST TTS | `POST http://localhost:18080/v1/audio/speech` |
| Realtime ASR | `ws://localhost:18080/v1/realtime?intent=transcription` |

The launcher exposes the service on the host network without authentication.
Use it only on a trusted development network or add an authenticated reverse
proxy before broader exposure.

## Model and port overrides

Override launcher defaults with environment variables. Keep the preset model
IDs and API port in sync:

```bash
ASR_MODEL=Systran/faster-whisper-small \
TTS_MODEL=speaches-ai/Kokoro-82M-v1.0-ONNX-fp16 \
SPEACHES_PORT=18080 \
./scripts/speaches_speech.sh start
```

To pin the qualified digest rather than the readable tag:

```bash
SPEACHES_IMAGE=ghcr.io/nvidia-ai-iot/speaches@sha256:abb7d669e73a32f8055500100cc05aa1b5e2ef1b7280d71a5391d458681e4d69 \
./scripts/speaches_speech.sh start
```

English is the P1 smoke-test language. The image also exposes Kokoro voices
for Japanese, Chinese, and other languages; treat those as separate P2 quality
tests and change the ASR model, language, and TTS voice together.

## Troubleshooting

- `No models found`: run `./scripts/speaches_speech.sh models`; if the defaults
  are absent, run `start` again and inspect `logs`.
- CUDA or GPU startup errors: run `doctor`, confirm the NVIDIA Docker runtime,
  and verify that the host driver supports CUDA 13 containers.
- Out of memory on smaller Orin devices: stop other GPU services, retain the
  tiny ASR model, use the 4096-token Gemma default, and start each service
  sequentially. If needed, choose a smaller OpenAI-compatible LLM.
- No LLM models in MMAS: Speaches serves only ASR and TTS. Start the independent
  Gemma service and reload `http://localhost:8080/v1` in the LLM tab.
- Remote browser microphone denied: access MMAS over HTTPS and grant microphone
  permission for the Jetson URL.
