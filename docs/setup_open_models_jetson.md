<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA open speech models on Jetson Thor and Orin

This reference setup runs NVIDIA public ASR and TTS checkpoints without Riva,
an NGC entitlement, or a hosted speech API:

```text
microphone
  -> Nemotron 3.5 ASR (OpenAI REST or Realtime, :8081)
  -> separately managed OpenAI-compatible LLM
  -> Magpie multilingual v2607 (OpenAI REST or Realtime, :8082)
  -> speaker
```

`scripts/nvidia_open_models_speech.sh` deliberately owns only the two speech
services. LLM selection, memory policy, and lifecycle remain independent. This
keeps the same speech pair usable with Ollama, llama.cpp, vLLM, a hosted API,
or any other OpenAI-compatible Chat Completions service.

For the shortest first-time setup, use the
[Speaches recommended quick start](setup_speaches_jetson.md). Use this page to
evaluate NVIDIA Open Speech Models directly or compare them with Riva.

“Open” here means publicly downloadable weights and source-based local
inference, not that every component has an OSI license. Nemotron 3.5 ASR uses
Open Model Development Watch License 1.1, Magpie uses the NVIDIA Open Model
License, and the API wrapper in this repository is Apache-2.0. Review all model
and container terms for the intended product.

## Qualified speech models and platforms

| Stage | Model | Default port |
| --- | --- | ---: |
| ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b` | 8081 |
| TTS | `nvidia/magpie_tts_multilingual_357m`, revision `v2607` | 8082 |

Initial functional qualification was performed on a Jetson AGX Thor Developer
Kit running L4T R38.4 and a Jetson AGX Orin 64GB running L4T R39.2. Orin on
JetPack 6 / L4T R36 remains unverified for these CUDA 13 containers.

## 1. Build the speech images

```bash
./scripts/nvidia_open_models_speech.sh doctor
./scripts/nvidia_open_models_speech.sh build
```

The script selects the qualified `nvcr.io/nvidia/pytorch:26.06-py3` base on
L4T R38 and `26.04-py3` on L4T R39. Override `SPEECH_BASE_IMAGE` only for
deliberate upgrade testing. The first build is large because it includes CUDA,
PyTorch, and NeMo Speech.

## 2. Start and verify ASR/TTS

```bash
./scripts/nvidia_open_models_speech.sh start
./scripts/nvidia_open_models_speech.sh status
./scripts/nvidia_open_models_speech.sh verify
```

Public checkpoints are downloaded from Hugging Face into the persistent Docker
volume `mmas-hf-cache`. `HF_TOKEN` is optional for higher download limits; the
listed checkpoints do not require an access grant.

Docker Compose provides the same two-service boundary:

```bash
docker compose -f deploy/compose.nvidia-open-models-speech.yaml up -d --build
```

Compose cannot inspect L4T while expanding YAML. On L4T R39, select the
qualified base explicitly:

```bash
SPEECH_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3 \
docker compose -f deploy/compose.nvidia-open-models-speech.yaml up -d --build
```

## 3. Check the speech APIs

REST ASR:

```bash
curl http://localhost:8081/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F language=en-US \
  -F response_format=json
```

Native cache-aware Realtime ASR:

```bash
python scripts/test_realtime_transcription.py sample.wav \
  --url 'ws://127.0.0.1:8081/v1/realtime' \
  --api-style openai-ga \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --language en-US
```

The default `SPEECH_REALTIME_LOOKAHEAD_TOKENS=3` selects the model's 320 ms
streaming point. Supported values `0`, `1`, `3`, `6`, and `13` map to 80, 160,
320, 560, and 1120 ms. This only affects native Realtime ASR; the REST file
endpoint remains available for comparison.

REST TTS:

```bash
curl http://localhost:8082/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/magpie_tts_multilingual_357m",
    "input": "Hello. This speech service is running on Jetson.",
    "voice": "Sofia",
    "language": "en-US",
    "response_format": "wav"
  }' \
  --output magpie-en.wav
```

Health and model discovery are available at `/health` and `/v1/models` on both
services. The Realtime preset also uses the Magpie WebSocket adapter on port
8082. Its completed text chunks are returned as PCM delta events; the current
`do_tts()` model path is still phrase-level generation, not model-native
incremental waveform synthesis.

## 4. Connect an LLM

Start an independent server that implements `POST /v1/chat/completions`. The
bundled NVIDIA presets retain the qualified Qwen3-4B/vLLM example at
`http://localhost:8000/v1`, but the speech launcher does not start it. You may
edit the preset or use the MMAS LLM tab to select a different server and model.

See the LLM/VLM section in [INSTALL.md](../INSTALL.md) for Ollama, llama.cpp,
vLLM, and other choices. Verify the selected service independently before
starting a voice session:

```bash
curl http://localhost:8000/v1/models
```

## 5. Run MMAS

For native Realtime Nemotron ASR and the Magpie Realtime adapter:

```bash
source .venv/bin/activate
multi-modal-ai-studio \
  --preset nvidia-open-models-realtime-speech-jetson \
  --port 8092
```

For the portable REST speech baseline:

```bash
multi-modal-ai-studio \
  --preset nvidia-open-models-speech-jetson \
  --port 8092
```

Open `https://localhost:8092`, accept the local certificate, and select the
browser microphone and speaker. English is the P1 public smoke test. Treat
Japanese and other languages as P2 validation and change ASR and TTS language
settings together.

The presets send `chat_template_kwargs.enable_thinking=false` to the qualified
Qwen3 example so reasoning tokens are not spoken. Remove or replace that extra
request body when using an LLM server that does not accept it.

## Qualification measurements

These are warm functional measurements, not a sustained-load benchmark.
Qwen3-4B/vLLM was used as the independently served LLM during the historical
full-cascade qualification.

### Thor

| Stage | Test | Result |
| --- | --- | --- |
| Nemotron ASR | 14.848 s English WAV | 0.24 s response; correct command transcript |
| Qwen3-4B reference LLM | short Japanese response | 0.69 s |
| Magpie v2607 | 4.737 s Japanese WAV | 3.04 s response; 22.05 kHz mono |
| Resident memory | ASR + reference LLM + TTS | about 8.8 GiB total |

### AGX Orin 64GB

| Stage | Test | Mean / p50 / p95 |
| --- | --- | --- |
| Nemotron 3.5 ASR | real WAV, GPU transcription | 0.407 / 0.399 / 0.433 s |
| Qwen3-4B reference LLM | Japanese Chat Completions | 0.946 / 0.945 / 0.948 s |
| Magpie v2607 | Japanese WAV generation | 5.387 / 5.249 / 5.911 s |
| Historical full cascade | ASR -> LLM -> TTS, offline | 6.643 s total |

The generated TTS audio averaged 3.864 seconds. Magpie's mean real-time factor
was 1.394 (p95 1.411), using `generation time / audio duration`; a value above
1 means synthesis was slower than audio real time. The ordinary NeMo/PyTorch
path is functionally complete but is not yet a sustained real-time TTS result.

| Resident container memory | Result |
| --- | ---: |
| ASR | 4.05 GiB |
| Independently managed vLLM reference | 16.33 GiB |
| TTS | 5.93 GiB |

No CUDA, PTX, kernel-image, invalid-device, or OOM errors occurred in the Orin
qualification. PyTorch emitted an SM 8.7 architecture warning, but actual ASR,
TTS, vLLM native extensions, CUDA graphs, and repeated inference completed.

## Evaluation checklist

- ASR WER/CER, language detection, first-word retention, endpoint errors, and
  speech-final latency.
- TTS time to first audio, real-time factor, pronunciation, voice consistency,
  and listening preference.
- Full-turn latency from end of user speech to first assistant audio.
- Peak and resident memory, power mode, clocks, temperature, and throttling.
- Barge-in behavior and echo leakage for browser and physical USB audio.
- REST versus Realtime endpointing and transcript behavior with identical WAVs.

## Stop

```bash
./scripts/nvidia_open_models_speech.sh stop
```

This removes only `mmas-nemotron-asr` and `mmas-magpie-tts`. It never stops an
LLM and preserves the shared model cache.
