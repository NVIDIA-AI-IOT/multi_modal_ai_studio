<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Open speech models on Jetson Thor and Orin

This setup runs the complete voice cascade without Riva SDK, an NGC
entitlement, or a hosted API key:

```text
microphone
  -> Nemotron 3.5 ASR (OpenAI Transcriptions, :8081)
  -> Qwen3 4B on vLLM (OpenAI Chat Completions, :8000)
  -> Magpie multilingual v2607 (OpenAI Speech, :8082)
  -> speaker
```

The checkpoints are public. Each stage is an independent OpenAI-compatible
service, so it can be started, measured, replaced, and debugged separately.

“Open” here means publicly downloadable model weights and source-based local
inference, with no Riva/EA application or service credential. Checkpoint terms
still apply: Nemotron 3.5 ASR uses Open Model Development Watch License 1.1,
and Magpie uses the NVIDIA Open Model License. The API wrapper in this
repository is Apache-2.0. Review those model licenses for the intended product;
this stack should not be described as uniformly OSI-licensed software. The
container base also includes NVIDIA CUDA/PyTorch components under their
respective license terms.

## Qualified platforms

The initial qualification was run on 2026-07-25:

| Device | L4T | Driver/CUDA capability | Verified scope |
|---|---|---|---|
| NVIDIA Jetson AGX Thor Developer Kit | R38.4.0 | CUDA 13 compatible | Complete ASR -> LLM -> TTS stack |
| NVIDIA Jetson AGX Orin 64GB | R39.2.0 | Driver 595.78 / CUDA 13.2 | Complete ASR -> LLM -> TTS stack |

Both systems use aarch64. Orin running JetPack 6 / L4T R36 remains unverified.

The qualified model versions are:

| Stage | Model |
|---|---|
| ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| LLM | `Qwen/Qwen3-4B`, vLLM 0.25.0 |
| TTS | `nvidia/magpie_tts_multilingual_357m`, revision `v2607` |

## 1. Build

From the repository root:

```bash
./scripts/nvidia_open_models_speech.sh doctor
./scripts/nvidia_open_models_speech.sh build
```

The speech images use the public multi-architecture
NVIDIA PyTorch base. The script selects the qualified `26.06-py3` base on
L4T R38 and `26.04-py3` on L4T R39. Override `SPEECH_BASE_IMAGE` only for
deliberate upgrade testing. The first build is large because it includes CUDA
and PyTorch.

## 2. Start the three services

```bash
./scripts/nvidia_open_models_speech.sh start
./scripts/nvidia_open_models_speech.sh status
```

Models are downloaded from Hugging Face into the persistent Docker volume
`mmas-hf-cache`. Set `HF_TOKEN` only if higher Hugging Face download limits are
needed; the listed models do not require an access grant.

The default `vllm/vllm-openai:v0.25.0` image is the upstream multi-architecture
OpenAI server. The image contains a CUDA 13.0 user-space runtime; the host does
not need a separately installed CUDA 13 toolkit, but its JetPack/L4T GPU driver
must be able to run CUDA 13 applications.

On Orin R39.2, PyTorch warns that SM 8.7 is not explicitly listed in its build
architecture list. Despite that warning, CUDA tensor operations, the vLLM
native extension, FlashAttention 2, FlashInfer, CUDA graphs, Qwen3-4B loading,
and repeated OpenAI Chat Completions requests all completed successfully. No
PTX, `no kernel image`, `invalid device function`, or custom-kernel execution
errors were observed.

The launch profile uses a fixed 2 GiB KV cache in addition to a conservative
GPU-memory utilization limit. Jetson uses unified memory, and an absolute KV
cache avoids startup profiling races when the ASR and TTS containers initialize
or release memory at the same time. Override `LLM_KV_CACHE_MEMORY_BYTES` and
`LLM_GPU_MEMORY_UTILIZATION` for a larger model or higher concurrency.

The repository also contains
`deploy/compose.nvidia-open-models-speech.yaml` for systems with the Docker Compose
plugin:

```bash
docker compose -f deploy/compose.nvidia-open-models-speech.yaml up -d --build
```

Compose cannot inspect L4T while expanding YAML. On an R39 host, select the
qualified speech base explicitly:

```bash
SPEECH_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3 \
docker compose -f deploy/compose.nvidia-open-models-speech.yaml up -d --build
```

## 3. Verify each standard endpoint

ASR:

```bash
curl http://localhost:8081/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F language=en-US \
  -F response_format=json
```

LLM:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 64
  }'
```

Or run the same LLM check with:

```bash
./scripts/nvidia_open_models_speech.sh verify-llm
```

TTS:

```bash
curl http://localhost:8082/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/magpie_tts_multilingual_357m",
    "input": "Hello. This voice pipeline is running on Jetson Thor.",
    "voice": "Sofia",
    "language": "en-US",
    "response_format": "wav"
  }' \
  --output magpie-en.wav
```

Health and model discovery are available at `/health` and `/v1/models` on all
speech services.

## 4. Run Multi-modal AI Studio

Install MMAS, then load the bundled preset:

```bash
./scripts/setup_dev.sh
source .venv/bin/activate
multi-modal-ai-studio \
  --preset nvidia-open-models-speech-jetson \
  --port 8092
```

Open `https://localhost:8092`, accept the local certificate, and select the
browser microphone and speaker. In the configuration UI, each ASR and TTS
backend can also be changed to **OpenAI REST API** manually.

The bundled preset intentionally uses English for the public P1 smoke test.
Treat Japanese and other languages as P2 multilingual validation, changing
both ASR and TTS language settings together.

The preset sends `chat_template_kwargs.enable_thinking=false` to Qwen3. This
keeps chain-of-thought out of the spoken response and avoids spending the
voice-demo token budget before a short answer is produced.

The preset starts TTS from short text chunks before the LLM finishes. The
public Magpie v2607 `do_tts()` path still returns each chunk only after that
chunk has been fully synthesized; it is not model-native incremental audio
streaming.

## Initial Thor measurements

These are smoke-test measurements, not a complete benchmark. Timings are warm
single requests and should be repeated under the intended power mode.

| Stage | Test | Result |
|---|---|---|
| Nemotron ASR | 14.848 s English WAV | 0.24 s response; correct command transcript |
| Qwen3 4B | short Japanese Chat Completions response (P2 multilingual check) | 0.69 s on vLLM 0.25.0 |
| Magpie v2607 | 4.737 s Japanese WAV (P2 multilingual check) | 3.04 s response; 22.05 kHz mono |
| Resident memory | all three containers | about 8.8 GiB total |

The first request is slower because it downloads and initializes the model.
Observed first-request times were about 25 seconds for ASR and 50–75 seconds
for TTS.

## Initial Orin measurements

These measurements used an AGX Orin 64GB with L4T R39.2.0. All three services
ran concurrently as independent localhost OpenAI-compatible APIs. The
measurements are five warm sequential requests per service, not a concurrency
or sustained-load benchmark.

| Stage | Test | Mean / p50 / p95 |
|---|---|---|
| Nemotron 3.5 ASR | real WAV, GPU transcription | 0.407 / 0.399 / 0.433 s |
| Qwen3-4B | Japanese Chat Completions (P2 multilingual check) | 0.946 / 0.945 / 0.948 s |
| Magpie v2607 | Japanese WAV generation (P2 multilingual check) | 5.387 / 5.249 / 5.911 s |
| Full cascade | ASR -> LLM -> TTS, offline | 6.643 s total |

The generated TTS audio averaged 3.864 seconds. Magpie's mean real-time factor
was 1.394 and p95 was 1.411, where an RTF above 1 means synthesis is slower
than audio real time. The ordinary NeMo/PyTorch path is therefore functionally
complete but is not yet a streaming real-time voice-agent configuration.
The first TTS request took 166.27 seconds because it included initial
OpenJTalk dictionary download and setup; the table reports warm requests.

The cascade used a 3.529-second input WAV. Its ASR output confused the Japanese
word `実機` with `実器`, so this is a functional and latency qualification, not
a WER/CER quality claim.

| Resident container memory | Result |
|---|---:|
| ASR | 4.05 GiB |
| vLLM | 16.33 GiB |
| TTS | 5.93 GiB |

During the run, GPU temperature averaged 42.1 C and peaked at 44.7 C.
`VDD_GPU_SOC` averaged 12.88 W and peaked at 24.77 W in the existing MAXN
power mode. No CUDA, PTX, kernel-image, invalid-device, or OOM errors occurred.
The known SM 8.7 warning was present, but vLLM completed FlashAttention 2,
`torch.compile`, CUDA graph capture, and real inference.

The exact qualified revisions were:

| Component | Revision |
|---|---|
| Nemotron 3.5 ASR | `f3d333391852ba876df169dcc9ba902d25b6ab0b` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Magpie v2607 checkpoint | `5023df68bd3f5b5ce6d666a50979bc501af145cc` |
| NeMo Speech | `2639d4bef8d1450782263a8f616242acfb6fecb9` |

The common stack defaults to a smaller fixed 2 GiB KV cache so ASR, LLM, and
TTS can start together conservatively. This is the profile used by the
three-service Orin qualification. The script accepts
`LLM_KV_CACHE_MEMORY_BYTES=auto` to omit the fixed-cache option:

```bash
LLM_KV_CACHE_MEMORY_BYTES=auto \
LLM_GPU_MEMORY_UTILIZATION=0.45 \
./scripts/nvidia_open_models_speech.sh start
```

Use that larger profile only when enough unified memory remains for ASR, TTS,
MMAS, and other processes.

## What to evaluate next

For every candidate ASR/TTS pair, save the MMAS session and compare:

- ASR WER/CER, language detection, endpointing errors, and speech-final latency.
- TTS time to first audio, real-time factor, pronunciation, voice consistency,
  and MOS or preference scores.
- Full-turn latency from end-of-user-speech to first assistant audio.
- Peak and resident memory, power mode, clocks, temperature, and throttling.
- Barge-in behavior and echo leakage with the target microphone/speaker.

The next engineering step for Nemotron 3.5 ASR is a native streaming adapter
using its cache-aware chunks. The current REST baseline deliberately uses the
portable file-transcription contract and local VAD.

## Stop

```bash
./scripts/nvidia_open_models_speech.sh stop
```

This removes only the three named service containers and preserves the model
cache volume.
