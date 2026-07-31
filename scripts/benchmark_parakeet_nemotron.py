#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark common English/Japanese WAVs across parakeet.cpp GGUF variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Sample:
    sample_id: str
    wav: Path
    language: str
    reference: str


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"<(?:eou|eob)>", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for col, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col] + 1,
                    previous[col - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, language: str) -> Tuple[str, float]:
    ref = normalized_text(reference)
    hyp = normalized_text(hypothesis)
    if language.lower().startswith("ja"):
        metric = "cer"
        ref_tokens = list(ref.replace(" ", ""))
        hyp_tokens = list(hyp.replace(" ", ""))
    else:
        metric = "wer"
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
    denominator = max(1, len(ref_tokens))
    return metric, edit_distance(ref_tokens, hyp_tokens) / denominator


def transcript_difference(baseline: str, candidate: str, language: str) -> float:
    _metric, rate = error_rate(baseline, candidate, language)
    return rate


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or (row[0].lstrip().startswith("#")):
                continue
            if len(row) != 4:
                raise ValueError(f"{path}:{line_number}: expected exactly 4 TSV fields")
            sample_id, wav_text, language, reference = row
            wav = Path(wav_text).expanduser().resolve()
            if not wav.is_file():
                raise ValueError(f"{path}:{line_number}: WAV not found: {wav}")
            if wav.suffix.lower() != ".wav":
                raise ValueError(f"{path}:{line_number}: input must be WAV: {wav}")
            samples.append(Sample(sample_id, wav, language, reference))
    if not samples:
        raise ValueError(f"{path}: no samples")
    return samples


def parse_models(items: Iterable[str]) -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"model must be LABEL=PATH: {item}")
        label, path_text = item.split("=", 1)
        path = Path(path_text).expanduser().resolve()
        if not label or not path.is_file():
            raise ValueError(f"invalid model: {item}")
        models[label] = path
    if "f16" not in models:
        raise ValueError("models must include the f16 baseline")
    return models


def parse_memory_bytes(value: str) -> int:
    number_text, unit = re.match(r"^\s*([\d.]+)\s*([kmgt]?i?b)\s*$", value, re.I).groups()
    multiplier = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }[unit.lower()]
    return int(float(number_text) * multiplier)


def command_output(command: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = " | ".join(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    return value if result.returncode == 0 and value else None


def host_metadata(image: str) -> dict:
    tegra_release = Path("/etc/nv_tegra_release")
    return {
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "tegra_release": (
            tegra_release.read_text(encoding="utf-8").splitlines()[0]
            if tegra_release.is_file()
            else None
        ),
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]
        ),
        "power_mode": command_output(["nvpmodel", "-q"]),
        "docker_image_id": command_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image]
        ),
    }


def monitor_docker_memory(container_id: str, stop: threading.Event, peak: List[int]) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            used = result.stdout.strip().split("/", 1)[0]
            try:
                peak[0] = max(peak[0], parse_memory_bytes(used))
            except (AttributeError, KeyError, ValueError):
                pass
        stop.wait(0.02)


def docker_bench(
    image: str,
    model: Path,
    samples: Sequence[Sample],
    language: str,
    repeats: int,
    threads: int,
    output_dir: Path,
) -> Tuple[dict, int, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="parakeet-bench-", dir=output_dir) as temp_text:
        temp = Path(temp_text)
        manifest = temp / "manifest.txt"
        bench_json = temp / "bench.json"
        manifest.write_text(
            "".join(f"{sample.wav}\n" for sample in samples for _ in range(repeats)),
            encoding="utf-8",
        )

        mounts = {model.parent, manifest.parent}
        mounts.update(sample.wav.parent for sample in samples)
        command = [
            "docker",
            "create",
            "--runtime",
            "nvidia",
            "--network",
            "none",
            "--ipc",
            "host",
        ]
        for directory in sorted(mounts):
            command.extend(["-v", f"{directory}:{directory}"])
        command.extend(
            [
                image,
                "bench",
                "--model",
                str(model),
                "--manifest",
                str(manifest),
                "--decoder",
                "tdt",
                "--lang",
                language,
                "--threads",
                str(threads),
                "--json",
                str(bench_json),
            ]
        )
        container_id = subprocess.check_output(command, text=True).strip()
        stop = threading.Event()
        peak = [0]
        monitor = threading.Thread(
            target=monitor_docker_memory,
            args=(container_id, stop, peak),
            daemon=True,
        )
        started = time.monotonic()
        monitor.start()
        try:
            completed = subprocess.run(
                ["docker", "start", "-a", container_id],
                capture_output=True,
                text=True,
                check=False,
            )
            wall_seconds = time.monotonic() - started
        finally:
            stop.set()
            monitor.join(timeout=3)
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"parakeet.cpp bench failed ({completed.returncode}): {completed.stderr}"
            )
        return json.loads(bench_json.read_text(encoding="utf-8")), peak[0], wall_seconds


def summarize(
    models: Dict[str, Path],
    samples: Sequence[Sample],
    image: str,
    repeats: int,
    threads: int,
    output_dir: Path,
) -> dict:
    result: dict = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "docker_image": image,
        "repeats": repeats,
        "threads": threads,
        "host": host_metadata(image),
        "models": {},
        "samples": [
            {
                "id": sample.sample_id,
                "wav": str(sample.wav),
                "language": sample.language,
                "reference": sample.reference,
                "audio_seconds": wav_duration(sample.wav),
            }
            for sample in samples
        ],
    }
    languages = sorted({sample.language for sample in samples})
    for label, model in models.items():
        model_result = {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": sha256_file(model),
            "languages": {},
        }
        for language in languages:
            language_samples = [sample for sample in samples if sample.language == language]
            raw, peak_bytes, wall_seconds = docker_bench(
                image, model, language_samples, language, repeats, threads, output_dir
            )
            rows = []
            for sample_index, sample in enumerate(language_samples):
                file_rows = raw["files"][
                    sample_index * repeats : (sample_index + 1) * repeats
                ]
                proc_values = [row["proc_ms"] for row in file_rows]
                transcript = file_rows[-1]["text"]
                metric, rate = error_rate(sample.reference, transcript, language)
                rows.append(
                    {
                        "id": sample.sample_id,
                        "audio_seconds": file_rows[-1]["audio_sec"],
                        "proc_ms_mean": statistics.mean(proc_values),
                        "proc_ms_p50": statistics.median(proc_values),
                        "rtf_mean": statistics.mean(proc_values)
                        / 1000.0
                        / file_rows[-1]["audio_sec"],
                        "transcript": transcript,
                        metric: rate,
                    }
                )
            model_result["languages"][language] = {
                "load_ms": raw["load_ms"],
                "peak_ram_bytes": peak_bytes,
                "process_wall_seconds": wall_seconds,
                "samples": rows,
            }
        result["models"][label] = model_result

    baseline = result["models"]["f16"]
    for label, model_result in result["models"].items():
        for language in languages:
            baseline_rows = {
                row["id"]: row for row in baseline["languages"][language]["samples"]
            }
            for row in model_result["languages"][language]["samples"]:
                row["transcript_diff_vs_f16"] = transcript_difference(
                    baseline_rows[row["id"]]["transcript"], row["transcript"], language
                )
    return result


def markdown_report(result: dict) -> str:
    lines = [
        "# Nemotron 3.5 ASR / parakeet.cpp benchmark",
        "",
        f"- Generated: `{result['generated_utc']}`",
        f"- Image: `{result['docker_image']}`",
        f"- Repeats: {result['repeats']}; threads: {result['threads']}",
    ]
    host = result.get("host", {})
    if host:
        lines.extend(
            [
                f"- Host: `{host.get('gpu') or 'unknown GPU'}`; "
                f"`{host.get('architecture') or 'unknown architecture'}`",
                f"- Kernel: `{host.get('kernel') or 'unknown'}`",
                f"- Power mode: `{host.get('power_mode') or 'unavailable'}`",
            ]
        )
    lines.extend(
        [
            "",
        "| Variant | Size MiB | Language | Load ms | Peak RAM MiB | Sample | Proc ms | RTF | WER/CER | Diff vs f16 |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for label, model in result["models"].items():
        for language, language_result in model["languages"].items():
            for row in language_result["samples"]:
                rate = row.get("wer", row.get("cer", 0.0))
                lines.append(
                    "| {label} | {size:.1f} | {language} | {load:.1f} | {ram:.1f} | "
                    "{sample} | {proc:.1f} | {rtf:.4f} | {rate:.4f} | {diff:.4f} |".format(
                        label=label,
                        size=model["size_bytes"] / 1024**2,
                        language=language,
                        load=language_result["load_ms"],
                        ram=language_result["peak_ram_bytes"] / 1024**2,
                        sample=row["id"],
                        proc=row["proc_ms_mean"],
                        rtf=row["rtf_mean"],
                        rate=rate,
                        diff=row["transcript_diff_vs_f16"],
                    )
                )
    lines.extend(
        [
            "",
            "Transcripts and exact SHA-256 values are in `results.json`.",
            "",
            "## Orin Nano 8 GB interpretation",
            "",
            "These measurements apply only to the named benchmark host. A dual-sm_87/sm_110 "
            "binary is an Orin Nano execution candidate, not an Orin Nano qualification. "
            "Treat model size and process/cgroup peak as measured inputs; any 8 GB fit "
            "assessment is an estimate until this harness and the API smoke test pass on "
            "the target Orin Nano with its intended concurrent services.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--models", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if args.repeats < 1 or args.threads < 1:
        parser.error("--repeats and --threads must be positive")
    try:
        models = parse_models(args.models)
        samples = parse_manifest(args.manifest)
    except ValueError as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = summarize(
        models,
        samples,
        args.docker_image,
        args.repeats,
        args.threads,
        args.output_dir,
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = markdown_report(result)
    (args.output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
