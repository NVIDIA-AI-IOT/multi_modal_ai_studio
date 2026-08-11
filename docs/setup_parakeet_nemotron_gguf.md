<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 ASR with parakeet.cpp GGUF

This is the reproducible, dependency-light ASR comparison path for
`nvidia/nemotron-3.5-asr-streaming-0.6b`. It is separate from
`scripts/nvidia_open_models_speech.sh`: the existing NeMo/PyTorch service and
the parakeet.cpp experiment can coexist, and no existing container is stopped
or replaced.

All large checkouts, checkpoints, GGUFs, and generated reports go under the
git-ignored `benchmark-results/parakeet-nemotron/` directory.

## Pinned upstream inputs

| Input | Revision |
|---|---|
| `mudler/parakeet.cpp` | `1bfbebfaaf493866f49597cd3b7901959d395c60` |
| vendored `ggml` submodule | `e705c5fed490514458bdd2eaddc43bd098fcce9b` |
| `mudler/parakeet-cpp-gguf` | `bf0af9f425fa01809cadec671b3cb672709d13e9` |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | `f3d333391852ba876df169dcc9ba902d25b6ab0b` |
| `NVIDIA-NeMo/NeMo` converter | `1c82990befeb0f44640d460b2dde75fd47fa9b2f` |

The selected parakeet.cpp revision is the upstream HEAD inspected on
2026-07-31. Its real interfaces are:

```text
parakeet-cli transcribe --model MODEL.gguf --input AUDIO.wav --lang ja-JP
parakeet-cli bench --model MODEL.gguf --manifest FILE --decoder tdt --lang en
parakeet-cli quantize F32.gguf OUTPUT.gguf q8_0
parakeet-cli quantize F32.gguf OUTPUT.gguf q4_k
parakeet-server --model MODEL.gguf --port 8080
POST /v1/audio/transcriptions
```

`q4_k` is the formal upstream four-bit K-quant name. Do not substitute
`Q4_K_M`, `q4k-q8`, or another patched layout.

The quantizer only changes eligible F32 linear weights consumed directly by
`ggml_mul_mat`. Convolution, LSTM, featurizer, normalization, bias, and other
hand-read tensors remain F32.

## Checksum-verified published models

The fastest qualification path uses the upstream-published models first:

| Variant | Bytes | SHA-256 |
|---|---:|---|
| f16 | 1,484,324,992 | `b64413c3886edf2b45eb3e757f911f1bc8020b7cf157622cd0bd0452c6d84aac` |
| q8_0 | 983,696,512 | `ba2f13eccd4a5245be728f77e6149bd6a4fdcdd133ff2e08ac6005bcef7a99f1` |
| q4_k | 718,102,624 | `5ad85eb3f3014c1a300d67b7ccbd23c38c4c952405cbe33a861e19fb2775e84b` |

```bash
./scripts/parakeet_nemotron_gguf.sh doctor
./scripts/parakeet_nemotron_gguf.sh checkout
./scripts/parakeet_nemotron_gguf.sh build
./scripts/parakeet_nemotron_gguf.sh fetch-prebuilt all
./scripts/parakeet_nemotron_gguf.sh verify-models
```

The CUDA build passes `-DPARAKEET_GGML_CUDA=ON` and defaults to CUDA 13 code
for both `sm_87` and `sm_110`. Override `PARAKEET_CUDA_ARCHS` only for a
deliberate platform-specific build.

The local CUDA 13 image was built on Jetson AGX Thor R38.4. Inspection of
`/usr/local/lib/libggml-cuda.so` with `cuobjdump --list-elf` found cubins for
both `sm_87` and `sm_110`; execution on Thor reported compute capability 11.0.
This establishes the requested dual-architecture artifact contents, while the
separate runtime measurements below remain platform-specific.

## Reproduce the conversion and quantization

The original `.nemo` download is also revision- and checksum-pinned:

```bash
./scripts/parakeet_nemotron_gguf.sh fetch-source
./scripts/parakeet_nemotron_gguf.sh setup-converter
./scripts/parakeet_nemotron_gguf.sh convert-f32
./scripts/parakeet_nemotron_gguf.sh quantize-local all
```

This produces a lossless F32 intermediate followed by upstream CLI-generated
`q8_0` and `q4_k`. The F32 file is intentionally retained for auditability.
The original NVIDIA checkpoint is BF16/FP16 source data; the published f16
GGUF is the primary full-precision-size baseline for runtime comparison.

On the Thor reproduction, conversion emitted 657 F32 tensors. The upstream
CLI quantized 219 tensors for `q8_0` and copied 438 unchanged. For `q4_k`, it
quantized 218 and copied 439; `joint.pred.weight` remained F32 because its
leading dimension 640 is not divisible by the K-quant superblock size 256.
The locally produced files were:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,552,332,448 | `7358e5d0872f2eadd8e212d4d56029b62edcaf6c964d78890236693b8d3d5080` |
| local q8_0 | 983,696,544 | `234639c499b532a9dfb6c771285a8403060daa8eb452c184d56750b5eca26059` |
| local q4_k | 718,102,688 | `f792d7be4cc3762419baa2a9fa25417f3988dbc894a2f729acf4f1afc7b5cba0` |

The local and published hashes are not expected to match. Their
`general.name` metadata records different source paths, accounting for the
small file-size difference. All 657 local q8_0 tensor payloads matched the
published q8_0 payloads byte-for-byte. The ARM64-generated q4_k had the same
tensor names, shapes, and types, but 1,064,326 of 717,891,776 tensor payload
bytes differed from the published artifact; use the transcript comparison
below as the functional parity gate rather than claiming byte identity.

Conversion is memory- and disk-intensive. A separate x86 conversion host may
be used by setting `PARAKEET_CONVERTER_PYTHON` to a Python environment with
PyTorch 2.8.0, gguf 0.19.0, and NeMo Toolkit commit
`1c82990befeb0f44640d460b2dde75fd47fa9b2f`. Copy only the resulting F32 GGUF
back, then run `quantize-local` on Jetson. `setup-converter` downloads the
NeMo source archive with SHA-256
`79b2355001fed99621b04ede58f45376d8ae67cb01c3a9783f4a522ac8284b49`
and installs that exact source.

Do not replace this pin with PyPI `nemo_toolkit[asr]==2.7.3`: that release
does not include `nemo.collections.asr.models.rnnt_bpe_models_prompt`, which
the inspected parakeet.cpp Nemotron 3.5 converter imports.

## Common English/Japanese benchmark

Create a UTF-8 TSV. Every variant receives exactly the same WAV paths:

```text
en<TAB>/absolute/path/fixed-en.wav<TAB>en<TAB>A helpful robot listens carefully.
ja<TAB>/absolute/path/fixed-ja.wav<TAB>ja-JP<TAB>音声認識の動作を実機で確認します。
```

Then run:

```bash
./scripts/parakeet_nemotron_gguf.sh benchmark /absolute/path/manifest.tsv
```

After reproducing F32 and both quants, run the same harness against the local
outputs:

```bash
./scripts/parakeet_nemotron_gguf.sh benchmark-local /absolute/path/manifest.tsv
```

The harness records, for every model and language:

- exact model bytes and SHA-256;
- model load/startup time from upstream `parakeet-cli bench`;
- peak container RAM sampled during the process;
- mean/p50 processing time and RTF after the upstream warmup;
- transcript plus English WER or Japanese CER against the supplied reference;
- normalized transcript edit distance against the f16 output.

JSON and Markdown reports are written below
`benchmark-results/parakeet-nemotron/results/`. Jetson uses unified memory, so
container RAM includes allocations charged to its cgroup; record the active
power mode and concurrent workloads alongside any published number.

### Thor R38.4 final solo measurements

The final runs used the fixed local image
`mmas/parakeet-nemotron-cli:1bfbebfaaf493866f49597cd3b7901959d395c60-cuda`
(image ID `sha256:416b9db9179102c10fbf4093a25c9a6ab5c6b3abb2f60065f67c2d2404ed5fb0`)
on NVIDIA Thor, compute capability 11.0, kernel `6.8.12-tegra`, and the 120 W
power mode. Existing MMAS service containers remained running, but no other
GPU build or benchmark issued work during these measurements. Each processing
number is the mean of three post-warmup runs with eight CPU threads.

Checksum-verified published artifacts:

| Variant | Size MiB | Language | Load ms | Peak cgroup RAM MiB | Processing ms | RTF | WER/CER | Normalized diff vs f16 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| f16 | 1415.6 | en | 465.9 | 385.2 | 314.6 | 0.0188 | 0.0000 WER | 0.0000 |
| f16 | 1415.6 | ja-JP | 404.0 | 448.7 | 178.1 | 0.0193 | 0.0213 CER | 0.0000 |
| q8_0 | 938.1 | en | 327.2 | 386.2 | 332.4 | 0.0199 | 0.0000 WER | 0.0000 |
| q8_0 | 938.1 | ja-JP | 358.1 | 414.7 | 185.5 | 0.0201 | 0.0213 CER | 0.0000 |
| q4_k | 684.8 | en | 289.6 | 376.5 | 318.8 | 0.0191 | 0.0000 WER | 0.0000 |
| q4_k | 684.8 | ja-JP | 306.3 | 400.8 | 161.1 | 0.0174 | 0.0213 CER | 0.0000 |

Locally reproduced quants, using the same f16 baseline:

| Variant | Size MiB | Language | Load ms | Peak cgroup RAM MiB | Processing ms | RTF | WER/CER | Normalized diff vs f16 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| local q8_0 | 938.1 | en | 341.1 | 387.7 | 313.5 | 0.0188 | 0.0000 WER | 0.0000 |
| local q8_0 | 938.1 | ja-JP | 369.7 | 415.6 | 186.9 | 0.0202 | 0.0213 CER | 0.0000 |
| local q4_k | 684.8 | en | 281.5 | 372.0 | 295.2 | 0.0177 | 0.0000 WER | 0.0000 |
| local q4_k | 684.8 | ja-JP | 287.5 | 399.2 | 184.1 | 0.0199 | 0.0213 CER | 0.0000 |

Every variant made the same Japanese substitution, `実機` → `実器`. The
English q4_k outputs changed punctuation boundaries, but normalization removes
punctuation and the word sequence remained equal to f16. The locally generated
q4_k's platform-dependent byte differences therefore passed the functional
transcript parity gate on both fixed WAVs.

For this workload, q4_k is the recommended deployment default: it is 51.6%
smaller than f16, loaded 24–38% faster in the published-artifact run, and did
not change normalized transcript accuracy. Choose q8_0 when a more conservative
quantization is worth an additional 253.3 MiB; its locally generated tensor
payloads also reproduce the published q8_0 byte-for-byte. Keep f16 as the
comparison baseline, not the memory-efficient serving default.

The peak column is the process cgroup sample, not total unified-memory
residency. In particular, memory-mapped model pages and GPU allocations make
it invalid to infer an Orin Nano fit by adding or comparing that column alone.

### Orin Nano 8 GB interpretation

The dual-architecture binary contains both `sm_87` and `sm_110` cubins, so it
is an execution candidate for AGX Orin/Orin Nano and Thor. This is not an Orin
Nano qualification. A Thor cgroup peak plus the model artifact size can
establish only a conservative model-only fit floor; memory-mapped model pages,
CUDA allocations, the OS, MMAS, audio buffers, and other services still compete
for the same 8 GB unified-memory pool. Keep these categories separate:

- **Measured:** model size, process/cgroup peak, load time, and RTF on the named
  Thor run.
- **Estimated:** whether the observed model-only footprint leaves plausible
  room on an otherwise idle 8 GB Orin Nano.
- **Unverified:** Orin Nano kernel execution, end-to-end MMAS coexistence,
  sustained thermals, and OOM behavior.

Do not publish an Orin Nano support claim until the same harness and API smoke
test pass on that device with the intended concurrent services.

## OpenAI-compatible API smoke test

The upstream example server accepts WAV uploads and serializes inference. It
supports `json`, `text`, and `verbose_json`, plus word timestamps. It does not
provide production authentication, metrics, or multi-model scheduling.

```bash
./scripts/parakeet_nemotron_gguf.sh \
  smoke-api /absolute/path/fixed-en.wav
```

The smoke server uses host port `18083` by default and is removed afterward.
Set `PARAKEET_SERVER_PORT` if that port is occupied. The endpoint is:

```text
POST http://127.0.0.1:18083/v1/audio/transcriptions
```

The model's default language prompt is `auto`. The inspected upstream server
does not currently map an OpenAI multipart `language` field to parakeet.cpp's
`--lang`; use the CLI benchmark for explicit `en` and `ja-JP` qualification.
The example server's `verbose_json.language` value is currently a placeholder,
not language-detection evidence.

The fixed local server image returned HTTP 200 for the 9.242-second Japanese
qualification WAV using the published q8_0 model. It transcribed both
sentences and made one reference substitution, `実機` → `実器`; the saved
`verbose_json` also reported the placeholder language `"en"`. This is a
successful API/inference smoke test, not a zero-error accuracy result.
