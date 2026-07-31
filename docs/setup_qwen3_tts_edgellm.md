<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3-TTS 0.6B with TensorRT Edge-LLM

This is the reproducible FP16 path for
[`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
on Jetson AGX Thor. It is intentionally separate from
`nvidia_open_models_speech.sh`: ONNX export must run on x86, while TensorRT
engine build and inference must run on the target Jetson.

The implementation follows
[`NVIDIA/TensorRT-Edge-LLM`](https://github.com/NVIDIA/TensorRT-Edge-LLM)
v0.9.1 at commit
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`. The model snapshot is pinned to
`85e237c12c027371202489a0ec509ded67b5e4b5`.

## Supported boundary

The upstream Qwen3-TTS support matrix lists:

- `Qwen3-TTS-12Hz-0.6B-CustomVoice`
- three components: Talker, CodePredictor, and Code2Wav
- FP16 only
- runtime batch size 1

INT8, INT4, FP8, and other quantized configurations are not claimed here.
The scripts do not silently substitute one of those precisions.

The upstream v0.9.1 standalone TTS CLI supports streaming with
`--streaming --chunkFrames=N`. Smaller chunks lower time-to-first-playable
audio but call Code2Wav more often. Each chunk is vocoded without left-context
overlap, so very small chunks can introduce boundary artifacts. The benchmark
keeps the WAV from `chunkFrames=1`, `2`, and `10` for listening or ASR-based
quality review; `10` is the first normal-quality candidate.

## Artifact layout

Large files are outside git under
`${EDGE_LLM_WORKSPACE:-$HOME/tensorrt-edgellm-workspace}`:

```text
tensorrt-edgellm-workspace/
├── TensorRT-Edge-LLM/       # pinned source checkout
├── model/                   # pinned Hugging Face snapshot
├── onnx-fp16/
│   ├── llm/                 # Talker
│   ├── code_predictor/
│   └── code2wav/
├── build-v0.9.1/            # target C++ tools
├── engines-fp16/
│   ├── mxsl1024/
│   ├── mxsl2048/
│   └── mxsl4096/
└── inputs/                  # fixed Japanese and English JSON
```

Measurements and generated WAV files go to the already ignored
`benchmark-results/qwen3-tts-edgellm/` directory.

## 1. Check both hosts

On the Thor:

```bash
./scripts/qwen3_tts_edgellm.sh doctor
```

The qualified target is `aarch64`, Jetson AGX Thor, L4T R38.4, with a CUDA
13.0-compatible driver. Existing ASR, LLM, TTS, and Riva services do not need
to be stopped. They do make the whole-system memory delta conservative.

The ONNX host must be x86-64 Linux with an NVIDIA GPU. The script rejects an
ARM export attempt with a recorded `SKIP` result.

## 2. Export FP16 ONNX on x86

Copy this repository checkout to the x86 host, then run:

```bash
./scripts/qwen3_tts_edgellm.sh export-onnx
```

The command:

1. checks out the pinned TensorRT Edge-LLM commit;
2. starts NVIDIA's `nvcr.io/nvidia/pytorch:25.12-py3` export image;
3. downloads the exact model revision with `snapshot_download`;
4. runs the current official command:

   ```bash
   tensorrt-edgellm-export \
     /workspace/model \
     /workspace/onnx-fp16 \
     --dtype float16
   ```

It also writes `onnx-manifest.json` with per-file SHA-256 values and
per-file/total sizes. A Hugging Face token is optional for this public
checkpoint. If `HF_TOKEN` is present, Docker receives it by environment name;
it is not written into a command, result, or archive.

The pinned package install resolves Torch 2.12.0. The `25.12-py3` export
image contains an older development torchvision build that fails while
registering `torchvision::nms` beside that Torch version, so the reproducible
container command installs the matching `torchvision==0.27.0` release before
running the exporter. The host Python environment is not modified.

Transfer the ONNX directory without flattening it. External ONNX data paths
are relative:

```bash
rsync -a --info=progress2 \
  "$HOME/tensorrt-edgellm-workspace/onnx-fp16/" \
  thor:"$HOME/tensorrt-edgellm-workspace/onnx-fp16/"
rsync -a \
  "$HOME/tensorrt-edgellm-workspace/onnx-fp16.SHA256SUMS" \
  thor:"$HOME/tensorrt-edgellm-workspace/"
```

`export-onnx` creates `onnx-fp16.SHA256SUMS`, and `build-engines` verifies it
when present. For a manually prepared offline transfer, the equivalent is:

```bash
cd "$HOME/tensorrt-edgellm-workspace"
find onnx-fp16 -type f -print0 |
  sort -z |
  xargs -0 sha256sum > onnx-fp16.SHA256SUMS
tar --zstd -cf qwen3-tts-0.6b-edgellm-v091-fp16-onnx.tar.zst \
  onnx-fp16 onnx-fp16.SHA256SUMS
sha256sum qwen3-tts-0.6b-edgellm-v091-fp16-onnx.tar.zst
```

Do not include the Hugging Face cache, token, or full source checkpoint in
the transfer bundle.

## 3. Build the runtime on Thor

```bash
BUILD_JOBS=4 ./scripts/qwen3_tts_edgellm.sh build-runtime
```

The derived development container is pinned to the CUDA 13.0 SBSA base digest
and adds only the missing standard C development headers. CMake uses the
official Thor settings:

```text
EMBEDDED_TARGET=jetson-thor
CUDA_CTK_VERSION=13.0
TRT_PACKAGE_DIR=/opt/tensorrt
ENABLE_CUTE_DSL=ALL
```

The derived container mirrors the TensorRT headers under `/opt/tensorrt`.
This avoids adding `/usr/include` as an explicit CUDA system include ahead of
GCC's standard-library include order; the underlying TensorRT libraries remain
the same JetPack-compatible files from the pinned base.

Only `llm_build`, `audio_build`, and `qwen3_tts_inference` are requested.

## 4. Build all engine profiles

After the transferred ONNX checksums pass:

```bash
./scripts/qwen3_tts_edgellm.sh build-engines
```

For each maximum sequence length (`1024`, `2048`, and the official `4096`),
the script builds:

```text
Talker:        maxInputLen=N, maxKVCacheCapacity=N, maxBatchSize=1
CodePredictor: maxInputLen=N, maxKVCacheCapacity=N, maxBatchSize=1
Code2Wav:      upstream audio profile defaults
```

Each component has a build result containing wall time and whole-system peak
unified-memory delta. Each profile also gets an engine size manifest. Existing
complete engines are reused; set `FORCE=1` for a deliberate rebuild.

The fixed prompts use `max_audio_length=512` for every engine profile. At 12
codec frames per second this is much longer than either prompt, while fitting
all three cache capacities. A successful short-prompt run does not prove that
a 1024 engine can accept a 4096-frame request.

## 5. Benchmark the same Japanese and English prompts

```bash
./scripts/qwen3_tts_edgellm.sh benchmark
```

Both prompts use `ono_anna`, batch 1, and the same sampling parameters. For
each sequence length and language, the harness runs:

- non-streaming;
- streaming `chunkFrames=1`;
- streaming `chunkFrames=2`;
- streaming `chunkFrames=10`.

The result records:

- wall-clock total generation time;
- upstream TTFC and TTFPA log values for streaming;
- output duration and total-time RTF;
- upstream peak unified memory and a conservative `/proc/meminfo` delta;
- ONNX, engine, and WAV sizes;
- full command, upstream profile JSON, and retained log.

TTFC means first codec token. TTFPA means completion of the first vocoded
chunk. The current standalone non-streaming CLI does not publish either value,
so the summary shows `—` instead of inventing a measurement. RTF here is
wall-clock request time divided by output audio duration; values below 1 are
faster than real time.

The runtime exposes no seed option. Every generated WAV is therefore retained
for transcript comparison and listening. Latency comparisons should use
several warm repetitions when making a product decision.

After the GPU benchmark is finished, an already-running OpenAI-compatible ASR
can score intelligibility without overlapping TTS timing:

```bash
EDGE_LLM_ASR_URL=http://127.0.0.1:8081/v1/audio/transcriptions \
  ./scripts/qwen3_tts_edgellm.sh score-quality
```

This stores the returned transcript and Japanese CER or English WER in each
trial JSON and adds the error rate to the CSV/Markdown summary. CER/WER can
detect severe chunk-boundary intelligibility loss; it does not measure speaker
similarity or naturalness, so retain and listen to the WAVs when comparing
`chunkFrames=1`, `2`, and `10`.

## 6. Optional OpenAI Speech adapter for MMAS

TensorRT Edge-LLM v0.9.1 does not ship an OpenAI-compatible Qwen3-TTS server.
Start the thin local adapter:

```bash
pip install -e '.[qwen-edge-tts]'
python3 scripts/qwen3_tts_edgellm_server.py --host 127.0.0.1 --port 8083
```

Then run MMAS with:

```bash
multi-modal-ai-studio --preset qwen3-tts-edgellm --port 8092
```

The adapter validates the FP16 engine layout, serializes requests because the
upstream runtime is batch 1, and accepts `pcm` or `wav`. It passes arguments
as an argv list rather than a shell command and mounts the workspace
read-write only for per-request output.

This is a compatibility adapter, not a production serving implementation. It
launches the upstream CLI for each request, and the HTTP response is returned
after the CLI has written the final WAV. Setting `EDGE_LLM_CHUNK_FRAMES=10`
exercises upstream chunked vocoding but does not turn this HTTP adapter into
true incremental audio delivery.

## Current qualification status

On 2026-07-31, the following was verified on `jat-c93c-r384` without stopping
the existing MMAS/Riva containers:

- the host is Jetson AGX Thor / L4T R38.4;
- TensorRT Edge-LLM v0.9.1 configures against CUDA 13.0 and TensorRT;
- `llm_build`, `audio_build`, and `qwen3_tts_inference` compile on the target;
- all three executables resolve their shared-library dependencies, and their
  actual `--help` output exposes the build, batch-1, tokenizer, streaming, and
  `chunkFrames` options used by the scripts;
- the initial general-purpose SBSA image lacked `libc6-dev`, which stopped the
  final CUDA compilation at `math.h`; the derived runtime Dockerfile records
  and fixes that prerequisite.

Full engine and inference qualification remains gated on the x86-generated
official ONNX bundle. Until those results exist, do not quote a Thor TTFPA,
RTF, or memory number from this document.

Jetson Orin Nano 8GB was not used for these measurements. The upstream release
supports FP16 on Orin, but neither engine build nor runtime memory for this
three-engine Qwen3-TTS path should be inferred from Thor unified-memory data.
