<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# faster-qwen3-tts on Jetson

This path compares the official PyTorch Qwen3-TTS implementation, the
`faster-qwen3-tts` Torch/CUDA-graph path, and its experimental qwentts.cpp
GGML adapter on the same 0.6B CustomVoice workload. It also exposes the
CUDA-graph path to Multi-modal AI Studio through a strict
`POST /v1/audio/speech` adapter.

## Pinned inputs and support boundary

| Input | Pinned revision |
|---|---|
| `andimarafioti/faster-qwen3-tts` | `a70afc0f81f7f5f8801c3227968f1102f43f211c` (v0.3.2 merge, 2026-07-17) |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `85e237c12c027371202489a0ec509ded67b5e4b5` |
| `Serveurperso/Qwen3-TTS-GGUF` | `e0f336a048a3de02b29b8ad92969217d9ecffe3e` (audit reference) |

The upstream README reports the following Jetson AGX Orin 64GB results for the
0.6B Base model, using `chunk_size=8`: baseline `RTF=0.179`,
baseline `TTFA=3641 ms`, CUDA graphs `RTF=1.307`, and CUDA graphs
`TTFA=597 ms`. In that table, RTF means **audio seconds / compute seconds**,
so values above one are faster than real time. These are upstream results, not
measurements from this repository.

Current upstream behavior matters:

- the fast Torch path is BF16 and uses `torch.cuda.CUDAGraph`;
- the official Qwen baseline does not provide CustomVoice streaming, so the
  harness records its time-to-full-playable-audio (`TTFPA`) and leaves `TTFA`
  unsupported instead of relabeling TTFPA;
- the GGML adapter is explicitly experimental and supports `BF16`, `Q8_0`, and
  `Q4_K_M` when matching qwentts.cpp model files and native wheels exist;
- the upstream `examples/openai_server.py` implements `/v1/audio/speech`, but
  it loads the Base voice-clone model and requires reference audio. It does not
  route the CustomVoice speaker `Ono_Anna`.

The local adapter fills only that last integration gap. It does not modify the
upstream repository or claim unsupported quantization for the Torch path.

Primary sources:

- [faster-qwen3-tts repository and Results](https://github.com/andimarafioti/faster-qwen3-tts/tree/a70afc0f81f7f5f8801c3227968f1102f43f211c)
- [upstream OpenAI server](https://github.com/andimarafioti/faster-qwen3-tts/blob/a70afc0f81f7f5f8801c3227968f1102f43f211c/examples/openai_server.py)
- [upstream GGML backend notes](https://github.com/andimarafioti/faster-qwen3-tts/blob/a70afc0f81f7f5f8801c3227968f1102f43f211c/docs/ggml-backend.md)

## Build and inspect

On L4T R38 / CUDA 13:

```bash
./scripts/faster_qwen3_tts.sh doctor
./scripts/faster_qwen3_tts.sh build
```

The image inherits NVIDIA PyTorch `26.06-py3` pinned to digest
`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`,
checks out the exact upstream commit, and installs the CUDA 13.0 aarch64
qwentts.cpp wrapper. Model and GGUF files are kept under
`~/.cache/mmas-faster-qwen3-tts`, outside Git.

Qwen's package eagerly imports its 25 Hz tokenizer, which imports
`torchaudio.compliance.kaldi`, even though this qualification uses only the
12 Hz model. The NVIDIA PyTorch 26.06 image ships Torch built for CUDA 13.3,
while the public torchaudio wheel is built for CUDA 13.0 and refuses to import
beside it. The image therefore supplies a 12 Hz-only import shim for that
unused Kaldi module. It raises immediately if a 25 Hz path calls `fbank`; no
Qwen model or inference source is patched.

The service and benchmark use a dedicated cache and port. They do not stop
Riva, the existing MMAS speech services, or vLLM.

## Reproduce the comparison

Run individual paths:

```bash
./scripts/faster_qwen3_tts.sh benchmark baseline BF16
./scripts/faster_qwen3_tts.sh benchmark cuda-graph BF16
./scripts/faster_qwen3_tts.sh benchmark ggml Q8_0
./scripts/faster_qwen3_tts.sh benchmark ggml Q4_K_M
```

Or run all four:

```bash
./scripts/faster_qwen3_tts.sh benchmark-all
```

The fixed workload is batch 1, `Ono_Anna` / `ono_anna`, one Japanese sentence,
one English sentence, and streaming chunks 1, 2, and 10. JSON and WAV results
are written under the Git-ignored
`benchmark-results/faster-qwen3-tts/` directory.

Each result records:

- load time, TTFA where actual streaming exists, TTFPA, total generation time;
- audio duration and both conventional `compute/audio` RTF and upstream-style
  `audio/compute`;
- process RSS, system unified-memory peak/delta, and Torch allocator peaks;
- WAV/model sizes, checksum, RMS, clipping, silence, and non-finite checks.

For transcript-based quality, point the harness at an already-running
OpenAI-compatible ASR endpoint:

```bash
FQWEN_ASR_URL=http://127.0.0.1:8081/v1/audio/transcriptions \
  ./scripts/faster_qwen3_tts.sh benchmark cuda-graph BF16
```

This adds Japanese CER and English WER. Without that option, the result says
`not_run`; waveform sanity metrics still run, but they are not a substitute
for intelligibility or speaker-similarity listening tests. CER/WER uses NFKC,
lowercase text, and ignores punctuation and whitespace differences before
computing character units for Japanese or word units for English.

If an exact GGUF combination or native library is unavailable, the GGML run
writes a JSON result with `status: skipped` and the actual error. It is never
reported as a successful quantized run.

On the Thor SM 11.0 qualification host, the published
`qwentts-cpp-python 0.3.1+cu130` wheel initializes CUDA and loads both Q8_0
GGUFs, but then reports that it was compiled only for SM
75/80/86/90/121 and fails at the first flash-attention kernel. The harness
therefore records a published-wheel GGML run as unsupported on SM 11.0 rather
than repeatedly launching a process that the native library aborts.

The upstream wrapper officially exposes arbitrary CMake arguments in
`scripts/build_native.py`, and its wheel workflow fixes qwentts.cpp to
`7df559a8ca25f66fee02970514ebe5f01dee9055`. A reproducible, unpatched SM 11.0
source build is therefore also provided:

```bash
./scripts/faster_qwen3_tts.sh build-ggml-sm110

FQWEN_QWENTTS_LIB="$HOME/.cache/mmas-faster-qwen3-tts/qwentts-sm110/qwentts-cpp-python/src/qwentts_cpp/lib/libqwen.so" \
  ./scripts/faster_qwen3_tts.sh benchmark ggml Q8_0

FQWEN_QWENTTS_LIB="$HOME/.cache/mmas-faster-qwen3-tts/qwentts-sm110/qwentts-cpp-python/src/qwentts_cpp/lib/libqwen.so" \
  ./scripts/faster_qwen3_tts.sh benchmark ggml Q4_K_M
```

That build fixes `qwentts-cpp-python` to
`b0b2da11293fb5a3f84fafc0a4c64524d7635b88`, qwentts.cpp to the commit above,
and its ggml submodule transitively to
`d0aac01c793cf6005c7fd17cdd65fef0062f6041`. It changes no source and passes
only upstream's supported `-DCMAKE_CUDA_ARCHITECTURES=110` build option.

The source-built SM 11.0 library completes Q8_0 inference. Q4_K_M currently
loads the model but aborts on the first synthesis call because the pinned
qwentts.cpp CUDA `get_rows` implementation does not support the model's
`q6_K` source tensor. After observing that native failure, the harness
preflights this exact combination and writes a `status: skipped` JSON result
with the reason. Q4_K_M is not presented as working, and no local qwentts.cpp
source patch is carried.

## Measured Thor results

The following is the isolated batch-1 run on `jat-4cbb4701c93c`, NVIDIA Thor
SM 11.0, L4T R38.4, and Torch 2.13 / CUDA 13.3. The table uses upstream-style
RTF (`audio seconds / compute seconds`); higher is faster. TTFA is measured at
the Python streaming boundary. Baseline has no streaming TTFA, so only TTFPA
is reported.

| Backend | Language | Chunk | TTFA ms | Total s | Audio s | RTF | ASR quality |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch BF16 baseline | ja | buffered | — | 13.252 | 6.56 | 0.495 | CER 0.240 |
| PyTorch BF16 baseline | en | buffered | — | 17.484 | 8.80 | 0.503 | WER 0.000 |
| CUDA Graph BF16 | ja | 1 | 340 | 7.242 | 5.60 | 0.773 | CER 0.360 |
| CUDA Graph BF16 | ja | 2 | 231 | 4.831 | 5.60 | 1.159 | CER 0.360 |
| CUDA Graph BF16 | ja | 10 | 531 | 3.356 | 5.60 | 1.669 | CER 0.360 |
| CUDA Graph BF16 | en | 1 | 197 | 19.543 | 15.20 | 0.778 | WER 1.000 |
| CUDA Graph BF16 | en | 2 | 257 | 12.976 | 15.20 | 1.171 | WER 1.000 |
| CUDA Graph BF16 | en | 10 | 556 | 8.494 | 15.20 | 1.790 | WER 1.000 |
| GGML Q8_0, source SM110 | ja | 1 | 37 | 1.593 | 5.68 | 3.565 | CER 0.360 |
| GGML Q8_0, source SM110 | ja | 2 | 40 | 1.934 | 7.04 | 3.639 | CER 0.320 |
| GGML Q8_0, source SM110 | ja | 10 | 30 | 1.649 | 6.00 | 3.639 | CER 0.280 |
| GGML Q8_0, source SM110 | en | 1 | 30 | 2.039 | 7.52 | 3.687 | WER 0.071 |
| GGML Q8_0, source SM110 | en | 2 | 34 | 1.744 | 6.40 | 3.670 | WER 0.071 |
| GGML Q8_0, source SM110 | en | 10 | 29 | 1.631 | 6.00 | 3.680 | WER 0.000 |

The Nemotron ASR transcript was empty for all three CUDA Graph English WAVs.
Their waveform sanity checks passed, so this is an intelligibility failure,
not a missing output file. CUDA Graph BF16 is therefore not the English P1
recommendation at this pinned revision. Q8_0 gives the best observed latency
and speed; `chunk_size=10` also reproduced the complete normalized English
reference. The BF16 baseline had the best Japanese score but was slower than
real time.

The BF16 model snapshot is 2,498,388,392 bytes. The Q8_0 talker plus tokenizer
GGUFs total 1,259,739,168 bytes; the currently unusable Q4_K_M pair totals
859,852,832 bytes. Peak process RSS was 2.50 GiB for baseline, 2.63 GiB for
CUDA Graph, and 2.10 GiB for Q8_0. Q8_0 loaded in 1.50 seconds, compared with
7.66 seconds for baseline and 6.83 seconds for the CUDA Graph model before
its warmup/capture.

qwentts.cpp sampling is stochastic and its pinned Python adapter does not
expose a native seed. Chunk comparisons can therefore include output-length
and transcript variation as well as chunking overhead; repeat trials before
using small differences for capacity planning.

## Start the MMAS endpoint

For the measured English-first P1 path, start the source-built Q8_0 backend:

```bash
FQWEN_BACKEND=ggml \
FQWEN_QUANT=Q8_0 \
FQWEN_QWENTTS_LIB="$HOME/.cache/mmas-faster-qwen3-tts/qwentts-sm110/qwentts-cpp-python/src/qwentts_cpp/lib/libqwen.so" \
  ./scripts/faster_qwen3_tts.sh start
./scripts/faster_qwen3_tts.sh logs
./scripts/faster_qwen3_tts.sh verify-api
```

The default endpoint is `http://127.0.0.1:18082/v1/audio/speech`. It streams
24 kHz mono PCM and accepts only the configured `Ono_Anna` voice, `speed=1.0`,
and `pcm` or `wav`. A WAV request is buffered until the final frame so the
response has standard RIFF chunk sizes; use `pcm` for incremental delivery.
Unsupported voice, speed, language, and format values return HTTP 400 rather
than silently changing behavior. Japanese/English is inferred from text unless
the optional `language` extension is present.

If a PCM client disconnects, the adapter signals the producer and releases its
bounded queue at the next upstream yield. The pinned native generator does not
expose a hard cancellation primitive for an already-running kernel, so this is
cooperative request cancellation rather than immediate GPU preemption.

Launch MMAS with the corresponding preset:

```bash
multi-modal-ai-studio \
  --preset faster-qwen3-tts-jetson \
  --port 8096
```

The preset is English-first for the public P1 path. Japanese remains a P2
multilingual check in the fixed benchmark and can be exercised directly:

```bash
curl http://127.0.0.1:18082/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"今日は実機で音声合成を検証します。","voice":"Ono_Anna","response_format":"wav"}' \
  --output qwen3-tts-ja.wav
```

Stop only this service when finished:

```bash
./scripts/faster_qwen3_tts.sh stop
```

## Interpreting Jetson memory

Jetson uses unified memory. `nvidia-smi` process GPU memory and the Torch
allocator do not represent the full resident footprint, so the harness records
system used-memory peaks alongside process RSS. An Orin Nano 8GB fit claim
requires an actual 8GB run. A Thor result can show a lower-bound footprint, but
must not be presented as an Orin Nano qualification.
