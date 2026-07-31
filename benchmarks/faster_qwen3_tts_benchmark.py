#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Repeatable Qwen3-TTS 0.6B baseline/CUDA-graph/GGML benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any, Self

import numpy as np

UPSTREAM_COMMIT = "a70afc0f81f7f5f8801c3227968f1102f43f211c"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
GGUF_REVISION = "e0f336a048a3de02b29b8ad92969217d9ecffe3e"
SPEAKER = "ono_anna"
FIXED_TEXTS = {
    "ja": "今日はエッジAIで、低遅延の音声合成を実機検証します。",
    "en": "Today we are validating low-latency text to speech on an edge AI device.",
}
LANGUAGES = {"ja": "Japanese", "en": "English"}


def _proc_rss_bytes() -> int:
    status = Path("/proc/self/status").read_text()
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else 0


def _system_used_bytes() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values["MemTotal"] - values["MemAvailable"]


class PeakMemory:
    """Sample process RSS and Jetson unified system memory during one scope."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.start_rss = 0
        self.start_system_used = 0
        self.peak_rss = 0
        self.peak_system_used = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self.start_rss = self.peak_rss = _proc_rss_bytes()
        self.start_system_used = self.peak_system_used = _system_used_bytes()

        def sample() -> None:
            while not self._stop.wait(self.interval_s):
                self.peak_rss = max(self.peak_rss, _proc_rss_bytes())
                self.peak_system_used = max(self.peak_system_used, _system_used_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.peak_rss = max(self.peak_rss, _proc_rss_bytes())
        self.peak_system_used = max(self.peak_system_used, _system_used_bytes())

    def result(self) -> dict[str, float]:
        gib = 1024**3
        return {
            "process_rss_start_gib": self.start_rss / gib,
            "process_rss_peak_gib": self.peak_rss / gib,
            "system_used_start_gib": self.start_system_used / gib,
            "system_used_peak_gib": self.peak_system_used / gib,
            "system_used_peak_delta_gib": max(
                0, self.peak_system_used - self.start_system_used
            )
            / gib,
        }


def sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio) * 32768.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def audio_quality(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite = np.isfinite(samples)
    clean = samples[finite]
    if clean.size == 0:
        return {"valid": False, "reason": "no finite samples"}
    peak = float(np.max(np.abs(clean)))
    rms = float(np.sqrt(np.mean(np.square(clean))))
    return {
        "valid": bool(len(clean) > sample_rate // 10 and peak > 1e-4),
        "duration_s": len(samples) / sample_rate,
        "sample_rate_hz": sample_rate,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": float(np.mean(np.abs(clean) >= 0.999)),
        "near_silence_ratio": float(np.mean(np.abs(clean) < 1e-4)),
        "nonfinite_samples": int(len(samples) - len(clean)),
    }


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, observed in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def transcript_error(
    reference: str,
    hypothesis: str,
    language_code: str,
) -> tuple[str, float]:
    """Return punctuation-insensitive CER or WER for a generated utterance."""

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    normalized_reference = normalize(reference)
    normalized_hypothesis = normalize(hypothesis)
    if language_code == "ja":
        ref_units = list(normalized_reference.replace(" ", ""))
        hyp_units = list(normalized_hypothesis.replace(" ", ""))
        metric = "cer"
    else:
        ref_units = normalized_reference.split()
        hyp_units = normalized_hypothesis.split()
        metric = "wer"
    return metric, _distance(ref_units, hyp_units) / max(1, len(ref_units))


def optional_asr(
    wav_path: Path,
    expected: str,
    language_code: str,
    asr_url: str | None,
    asr_model: str,
) -> dict[str, Any]:
    if not asr_url:
        return {
            "status": "not_run",
            "reason": "pass --asr-url for transcript CER/WER; waveform checks still ran",
        }
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        asr_url,
        "-F",
        f"file=@{wav_path}",
        "-F",
        f"model={asr_model}",
        "-F",
        f"language={'ja-JP' if language_code == 'ja' else 'en-US'}",
        "-F",
        "response_format=json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    transcript = str(payload.get("text", "")).strip()
    metric, rate = transcript_error(expected, transcript, language_code)
    return {
        "status": "ok",
        "transcript": transcript,
        metric: rate,
    }


def model_size_bytes(model_path: str) -> int | None:
    path = Path(model_path)
    if not path.is_dir():
        return None
    total = 0
    seen: set[tuple[int, int]] = set()
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        stat = item.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def load_model(args: argparse.Namespace) -> tuple[object, str]:
    import torch
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    if args.backend == "baseline":
        from qwen_tts import Qwen3TTSModel

        return (
            Qwen3TTSModel.from_pretrained(
                snapshot,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="eager",
            ),
            snapshot,
        )

    from faster_qwen3_tts import FasterQwen3TTS

    if args.backend == "cuda-graph":
        return (
            FasterQwen3TTS.from_pretrained(
                snapshot,
                device="cuda",
                dtype=torch.bfloat16,
                attn_implementation="eager",
                max_seq_len=args.max_seq_len,
            ),
            snapshot,
        )
    return (
        FasterQwen3TTS.from_pretrained(
            args.model,
            backend="ggml",
            quant=args.quant,
            cache_dir=args.ggml_cache_dir,
            local_files_only=args.local_files_only,
            qwentts_library_path=args.qwentts_library_path,
        ),
        snapshot,
    )


def generate_baseline(
    model: object,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    sync_cuda()
    started = time.perf_counter()
    audio_list, sample_rate = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        max_new_tokens=max_new_tokens,
    )
    sync_cuda()
    elapsed = time.perf_counter() - started
    audio = np.asarray(audio_list[0]).reshape(-1)
    return audio, sample_rate, {
        "ttfa_supported": False,
        "ttfa_ms": None,
        "ttfpa_ms": elapsed * 1000,
        "generation_s": elapsed,
        "chunks": 1,
    }


def generate_streaming(
    model: object,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
    chunk_size: int,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    sync_cuda()
    started = time.perf_counter()
    chunks: list[np.ndarray] = []
    sample_rate = 24000
    ttfa_ms = None
    for chunk, sample_rate, _timing in model.generate_custom_voice_streaming(
        text=text,
        speaker=speaker,
        language=language,
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
    ):
        if ttfa_ms is None:
            sync_cuda()
            ttfa_ms = (time.perf_counter() - started) * 1000
        chunks.append(np.asarray(chunk).reshape(-1))
    sync_cuda()
    elapsed = time.perf_counter() - started
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, sample_rate, {
        "ttfa_supported": True,
        "ttfa_ms": ttfa_ms,
        "ttfpa_ms": elapsed * 1000,
        "generation_s": elapsed,
        "chunks": len(chunks),
    }


def gpu_metadata() -> dict[str, Any]:
    import torch

    l4t = Path("/etc/nv_tegra_release")
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "l4t": l4t.read_text().splitlines()[0] if l4t.exists() else None,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if (
        args.backend == "ggml"
        and torch.cuda.get_device_capability(0) == (11, 0)
        and not args.qwentts_library_path
    ):
        raise RuntimeError(
            "qwentts-cpp-python 0.3.1+cu130 aarch64 reports CUDA kernels for "
            "SM 75/80/86/90/121 but not Thor SM 110; Q8_0/Q4_K_M are unsupported "
            "by this pinned upstream wheel"
        )
    if (
        args.backend == "ggml"
        and args.quant == "Q4_K_M"
        and torch.cuda.get_device_capability(0) == (11, 0)
        and args.qwentts_library_path
    ):
        raise RuntimeError(
            "the pinned source-built qwentts.cpp SM 110 library aborts Q4_K_M "
            "in ggml_cuda_get_rows_switch_src0_type because src0 type q6_K is "
            "unsupported; Q8_0 is the supported GGML path on this device"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    with PeakMemory() as load_memory:
        load_started = time.perf_counter()
        model, model_path = load_model(args)
        sync_cuda()
        load_s = time.perf_counter() - load_started

    supported = (
        model.get_supported_speakers()
        if args.backend == "ggml"
        else model.get_supported_speakers()
        if args.backend == "baseline"
        else model.model.get_supported_speakers()
    )
    speaker_lookup = {value.lower(): value for value in supported or []}
    if args.speaker.lower() not in speaker_lookup:
        raise RuntimeError(f"speaker {args.speaker!r} unavailable; supported={supported}")
    speaker = speaker_lookup[args.speaker.lower()]

    warmup_text = FIXED_TEXTS["en"][:32]
    with PeakMemory() as warmup_memory:
        if args.backend == "baseline":
            generate_baseline(
                model, warmup_text, "English", speaker, min(32, args.max_new_tokens)
            )
        else:
            generate_streaming(
                model,
                warmup_text,
                "English",
                speaker,
                min(32, args.max_new_tokens),
                args.chunk_sizes[0],
            )

    results: list[dict[str, Any]] = []
    for language_code, text in FIXED_TEXTS.items():
        chunk_sizes: list[int | None] = (
            [None] if args.backend == "baseline" else args.chunk_sizes
        )
        for chunk_size in chunk_sizes:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
            with PeakMemory() as request_memory:
                if args.backend == "baseline":
                    audio, sample_rate, timings = generate_baseline(
                        model,
                        text,
                        LANGUAGES[language_code],
                        speaker,
                        args.max_new_tokens,
                    )
                else:
                    audio, sample_rate, timings = generate_streaming(
                        model,
                        text,
                        LANGUAGES[language_code],
                        speaker,
                        args.max_new_tokens,
                        int(chunk_size),
                    )
            quality = audio_quality(audio, sample_rate)
            audio_s = quality.get("duration_s", 0.0)
            timings["rtf_compute_per_audio"] = (
                timings["generation_s"] / audio_s if audio_s else None
            )
            timings["audio_per_compute"] = (
                audio_s / timings["generation_s"] if timings["generation_s"] else None
            )

            suffix = "buffered" if chunk_size is None else f"chunk{chunk_size}"
            wav_path = (
                args.output_dir
                / f"{args.backend}-{args.quant.lower()}-{language_code}-{suffix}.wav"
            )
            write_wav(wav_path, audio, sample_rate)
            results.append(
                {
                    "language": language_code,
                    "text": text,
                    "speaker": speaker,
                    "chunk_size": chunk_size,
                    **timings,
                    "memory": {
                        **request_memory.result(),
                        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated()
                        / 1024**3,
                        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved()
                        / 1024**3,
                    },
                    "quality": quality,
                    "asr": optional_asr(
                        wav_path,
                        text,
                        language_code,
                        args.asr_url,
                        args.asr_model,
                    ),
                    "wav": str(wav_path),
                    "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                    "wav_bytes": wav_path.stat().st_size,
                }
            )

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream": {
            "repository": "https://github.com/andimarafioti/faster-qwen3-tts",
            "commit": UPSTREAM_COMMIT,
        },
        "model": {
            "id": args.model,
            "revision": args.model_revision,
            "local_path": model_path,
            "files_bytes": model_size_bytes(model_path),
        },
        "gguf_repository_revision": GGUF_REVISION,
        "backend": args.backend,
        "quant": args.quant,
        "qwentts_library_path": args.qwentts_library_path,
        "precision_note": (
            "BF16 PyTorch"
            if args.backend in {"baseline", "cuda-graph"}
            else f"experimental qwentts.cpp GGML {args.quant}"
        ),
        "device": gpu_metadata(),
        "batch_size": 1,
        "seed": args.seed,
        "seed_scope": (
            "PyTorch CPU and CUDA RNG"
            if args.backend != "ggml"
            else "PyTorch RNG only; the pinned qwentts.cpp adapter does not expose "
            "a native sampling seed"
        ),
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "load_s": load_s,
        "load_memory": load_memory.result(),
        "warmup_memory": warmup_memory.result(),
        "results": results,
    }


def parse_chunk_sizes(value: str) -> list[int]:
    result = [int(part) for part in value.split(",") if part.strip()]
    if not result or any(size <= 0 for size in result):
        raise argparse.ArgumentTypeError("chunk sizes must be positive")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("baseline", "cuda-graph", "ggml"),
        required=True,
    )
    parser.add_argument("--quant", choices=("BF16", "Q8_0", "Q4_K_M"), default="BF16")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--speaker", default=SPEAKER)
    parser.add_argument(
        "--chunk-sizes",
        type=parse_chunk_sizes,
        default=parse_chunk_sizes("1,2,10"),
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ggml-cache-dir", default="/models/qwentts")
    parser.add_argument("--qwentts-library-path")
    parser.add_argument("--asr-url")
    parser.add_argument("--asr-model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/faster-qwen3-tts"),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.backend != "ggml" and args.quant != "BF16":
        parser.error("baseline and cuda-graph only support BF16")
    return args


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.json_output or args.output_dir / f"{args.backend}-{args.quant.lower()}.json"
    try:
        payload = run(args)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        if args.backend != "ggml":
            raise
        payload = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": args.backend,
            "quant": args.quant,
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
            "upstream_commit": UPSTREAM_COMMIT,
            "gguf_repository_revision": GGUF_REVISION,
            "qwentts_library_path": args.qwentts_library_path,
        }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
