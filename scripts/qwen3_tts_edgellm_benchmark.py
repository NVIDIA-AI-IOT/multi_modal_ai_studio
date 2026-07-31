#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measurement and report helpers for the TensorRT Edge-LLM Qwen3-TTS path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional
import unicodedata
import wave


TTFC_RE = re.compile(r"TTFC=([0-9.]+)ms\s+TTFPA=([0-9.]+)ms")

FIXED_INPUTS = {
    "ja": "今日はエッジAIで、低遅延の音声合成を実機検証します。",
    "en": "Today we are validating low-latency text to speech on an edge AI device.",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def parse_latency_log(output: str) -> List[Dict[str, float]]:
    """Extract upstream standalone streaming TTFC/TTFPA log records."""
    return [
        {"ttfc_ms": float(match.group(1)), "ttfpa_ms": float(match.group(2))}
        for match in TTFC_RE.finditer(output)
    ]


def _load_json_if_present(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _audio_records(
    output: Optional[Dict[str, Any]],
    audio_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    if not output:
        return []
    records = []
    for response in output.get("responses", []):
        record = {
            "request_idx": response.get("request_idx"),
            "audio_duration_ms": response.get("audio_duration_ms"),
            "audio_samples": response.get("audio_samples"),
            "audio_sample_rate": response.get("audio_sample_rate"),
        }
        source_name = response.get("audio_file")
        if source_name and audio_dir:
            wav_path = audio_dir / Path(source_name).name
            if wav_path.is_file():
                record["wav_path"] = str(wav_path)
                record["wav_size_bytes"] = wav_path.stat().st_size
                with wave.open(str(wav_path), "rb") as wav_file:
                    record["wav_duration_sec"] = (
                        wav_file.getnframes() / float(wav_file.getframerate())
                    )
        records.append(record)
    return records


def _metadata(values: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must use key=value: {value}")
        key, raw = value.split("=", 1)
        if raw.isdigit():
            result[key] = int(raw)
        else:
            result[key] = raw
    return result


def _edit_distance(reference: List[str], hypothesis: List[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(hypothesis, start=1):
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
    """Use the same punctuation-insensitive CER/WER definition as the GGML path."""

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    normalized_reference = normalize(reference)
    normalized_hypothesis = normalize(hypothesis)
    if language_code == "ja":
        reference_units = list(normalized_reference.replace(" ", ""))
        observed_units = list(normalized_hypothesis.replace(" ", ""))
        metric = "cer"
    else:
        reference_units = normalized_reference.split()
        observed_units = normalized_hypothesis.split()
        metric = "wer"
    return metric, _edit_distance(reference_units, observed_units) / max(
        1,
        len(reference_units),
    )


def score_quality(args: argparse.Namespace) -> int:
    """Add transcript CER/WER to completed trials using an existing ASR API."""
    failures = 0
    for path in sorted(args.results_root.glob("inference/**/trial.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata") or {}
        language = str(metadata.get("language") or "")
        audios = record.get("audio") or []
        if record.get("status") != "success" or not audios:
            continue
        wav_path = Path(str(audios[0].get("wav_path") or ""))
        if not wav_path.is_file():
            record["quality"] = {
                "status": "failed",
                "reason": f"WAV not found: {wav_path}",
            }
            _write_json(path, record)
            failures += 1
            continue
        command = [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            args.asr_url,
            "-F",
            f"file=@{wav_path}",
            "-F",
            f"model={args.asr_model}",
            "-F",
            f"language={'ja-JP' if language == 'ja' else 'en-US'}",
            "-F",
            "response_format=json",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            payload = json.loads(completed.stdout) if completed.stdout else {}
        except json.JSONDecodeError:
            payload = {}
        if completed.returncode or not isinstance(payload.get("text"), str):
            record["quality"] = {
                "status": "failed",
                "reason": (completed.stderr or completed.stdout or "ASR returned no text")[
                    -2000:
                ],
                "endpoint": args.asr_url,
            }
            failures += 1
        else:
            transcript = payload["text"].strip()
            expected = FIXED_INPUTS[language]
            metric, error_rate = transcript_error(expected, transcript, language)
            record["quality"] = {
                "status": "success",
                "endpoint": args.asr_url,
                "model": args.asr_model,
                "reference": expected,
                "transcript": transcript,
                "metric": metric,
                "error_rate": error_rate,
            }
        _write_json(path, record)
        print(f"{path}: {record['quality']}")
    return 1 if failures else 0


def measure(args: argparse.Namespace) -> int:
    start_available = _read_mem_available_bytes()
    minimum_available = start_available
    stop_event = threading.Event()

    def sample_memory() -> None:
        nonlocal minimum_available
        while not stop_event.wait(0.05):
            try:
                minimum_available = min(minimum_available, _read_mem_available_bytes())
            except (OSError, RuntimeError):
                pass

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started_at = time.monotonic()
    lines: List[str] = []
    try:
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    except OSError as exc:
        lines.append(f"{type(exc).__name__}: {exc}\n")
        return_code = 127
    finally:
        elapsed = time.monotonic() - started_at
        stop_event.set()
        sampler.join(timeout=1)

    end_available = _read_mem_available_bytes()
    log_text = "".join(lines)
    inference_output = _load_json_if_present(args.inference_output)
    profile = _load_json_if_present(args.profile_output)
    audios = _audio_records(inference_output, args.audio_dir)
    audio_seconds = sum(float(item.get("wav_duration_sec") or 0) for item in audios)
    record: Dict[str, Any] = {
        "schema_version": 1,
        "status": "success" if return_code == 0 else "failed",
        "label": args.label,
        "return_code": return_code,
        "elapsed_sec": elapsed,
        "rtf": (elapsed / audio_seconds) if audio_seconds > 0 else None,
        "metadata": _metadata(args.metadata),
        "command": args.command,
        "system_memory": {
            "available_start_bytes": start_available,
            "available_end_bytes": end_available,
            "available_min_bytes": minimum_available,
            # Whole-system unified-memory delta. Other running services make
            # this conservative; the upstream profile is retained separately.
            "peak_delta_bytes": max(0, start_available - minimum_available),
        },
        "stream_latency": parse_latency_log(log_text),
        "audio": audios,
        "upstream_profile": profile,
        "log": log_text,
    }
    _write_json(args.result, record)
    return return_code


def write_inputs(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for language, text in FIXED_INPUTS.items():
        value = {
            "batch_size": 1,
            "talker_temperature": 0.9,
            "talker_top_k": 50,
            "talker_top_p": 1.0,
            "repetition_penalty": 1.05,
            "speaker": "ono_anna",
            # 512 codec frames are about 42 seconds at 12 Hz and fit all
            # engine profiles while remaining far above these fixed prompts.
            "max_audio_length": 512,
            "requests": [
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": text,
                        }
                    ]
                }
            ],
        }
        _write_json(args.output_dir / f"{language}.json", value)
    return 0


def manifest(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    files = []
    total = 0
    if root.is_dir():
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            size = path.stat().st_size
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
            total += size
    _write_json(
        args.result,
        {
            "schema_version": 1,
            "status": "success" if root.is_dir() else "skipped",
            "kind": args.kind,
            "root": str(root),
            "total_size_bytes": total,
            "files": files,
        },
    )
    return 0


def write_skip(args: argparse.Namespace) -> int:
    _write_json(
        args.result,
        {
            "schema_version": 1,
            "status": "skipped",
            "reason": args.reason,
        },
    )
    return 0


def _first_audio(record: Dict[str, Any]) -> Dict[str, Any]:
    audios = record.get("audio") or []
    return audios[0] if audios else {}


def _first_latency(record: Dict[str, Any]) -> Dict[str, Any]:
    values = record.get("stream_latency") or []
    return values[0] if values else {}


def _peak_profile_memory(record: Dict[str, Any]) -> Optional[float]:
    profile = record.get("upstream_profile") or {}
    value = profile.get("peak_unified_memory_bytes")
    if value is None:
        return None
    return float(value) / (1024**3)


def _trial_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(results_root.glob("inference/**/trial.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata") or {}
        audio = _first_audio(record)
        latency = _first_latency(record)
        quality = record.get("quality") or {}
        rows.append(
            {
                "status": record.get("status"),
                "sequence_length": metadata.get("sequence_length"),
                "language": metadata.get("language"),
                "chunk_frames": metadata.get("chunk_frames"),
                "elapsed_sec": record.get("elapsed_sec"),
                "ttfc_ms": latency.get("ttfc_ms"),
                "ttfpa_ms": latency.get("ttfpa_ms"),
                "audio_sec": audio.get("wav_duration_sec"),
                "rtf": record.get("rtf"),
                "peak_unified_gib": _peak_profile_memory(record),
                "wav_size_bytes": audio.get("wav_size_bytes"),
                "quality_metric": quality.get("metric"),
                "quality_error_rate": quality.get("error_rate"),
                "transcript": quality.get("transcript"),
                "result": str(path.relative_to(results_root)),
            }
        )
    return rows


def _build_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(results_root.glob("build/*/*.json")):
        if path.name == "engine-manifest.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata") or {}
        memory = record.get("system_memory") or {}
        rows.append(
            {
                "status": record.get("status"),
                "sequence_length": metadata.get("sequence_length"),
                "component": metadata.get("component"),
                "elapsed_sec": record.get("elapsed_sec"),
                "peak_delta_gib": (
                    float(memory["peak_delta_bytes"]) / (1024**3)
                    if memory.get("peak_delta_bytes") is not None
                    else None
                ),
            }
        )
    return rows


def _artifact_rows(results_root: Path) -> List[Dict[str, Any]]:
    candidates = [results_root / "onnx-manifest.json"]
    candidates.extend(sorted(results_root.glob("build/*/engine-manifest.json")))
    rows = []
    for path in candidates:
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "kind": record.get("kind"),
                "profile": path.parent.name if record.get("kind") == "engine" else "fp16",
                "size_gib": float(record.get("total_size_bytes") or 0) / (1024**3),
                "manifest": str(path.relative_to(results_root)),
            }
        )
    return rows


SUMMARY_FIELDS = [
    "status",
    "sequence_length",
    "language",
    "chunk_frames",
    "elapsed_sec",
    "ttfc_ms",
    "ttfpa_ms",
    "audio_sec",
    "rtf",
    "peak_unified_gib",
    "wav_size_bytes",
    "quality_metric",
    "quality_error_rate",
    "transcript",
    "result",
]


def summarize(args: argparse.Namespace) -> int:
    rows = _trial_rows(args.results_root)
    build_rows = _build_rows(args.results_root)
    artifact_rows = _artifact_rows(args.results_root)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Qwen3-TTS TensorRT Edge-LLM benchmark",
        "",
        "| status | max seq | lang | chunkFrames | total s | TTFC ms | "
        "TTFPA ms | audio s | RTF | peak unified GiB | CER/WER |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        display = []
        for key in (
            "status",
            "sequence_length",
            "language",
            "chunk_frames",
            "elapsed_sec",
            "ttfc_ms",
            "ttfpa_ms",
            "audio_sec",
            "rtf",
            "peak_unified_gib",
            "quality_error_rate",
        ):
            value = row.get(key)
            if value is None:
                display.append("—")
            elif isinstance(value, float):
                display.append(f"{value:.3f}")
            else:
                display.append(str(value))
        lines.append("| " + " | ".join(display) + " |")
    if not rows:
        lines.extend(
            [
                "",
                "No completed inference trials were found. Inspect `skip-*.json` "
                "for a recorded prerequisite blocker.",
            ]
        )
    lines.extend(
        [
            "",
            "## Engine build",
            "",
            "| status | max seq | component | build s | whole-system peak delta GiB |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in build_rows:
        elapsed = row.get("elapsed_sec")
        peak = row.get("peak_delta_gib")
        lines.append(
            "| {status} | {seq} | {component} | {elapsed} | {peak} |".format(
                status=row.get("status") or "—",
                seq=row.get("sequence_length") or "—",
                component=row.get("component") or "—",
                elapsed="—" if elapsed is None else f"{elapsed:.3f}",
                peak="—" if peak is None else f"{peak:.3f}",
            )
        )
    if not build_rows:
        lines.append("| — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Artifact size",
            "",
            "| kind | profile | size GiB | manifest |",
            "|---|---|---:|---|",
        ]
    )
    for row in artifact_rows:
        manifest_path = row["manifest"]
        lines.append(
            f"| {row['kind']} | {row['profile']} | "
            f"{row['size_gib']:.3f} | `{manifest_path}` |"
        )
    if not artifact_rows:
        lines.append("| — | — | — | — |")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--result", type=Path, required=True)
    measure_parser.add_argument("--label", required=True)
    measure_parser.add_argument("--metadata", action="append", default=[])
    measure_parser.add_argument("--inference-output", type=Path)
    measure_parser.add_argument("--profile-output", type=Path)
    measure_parser.add_argument("--audio-dir", type=Path)
    measure_parser.add_argument("command", nargs=argparse.REMAINDER)
    measure_parser.set_defaults(func=measure)

    inputs_parser = subparsers.add_parser("inputs")
    inputs_parser.add_argument("--output-dir", type=Path, required=True)
    inputs_parser.set_defaults(func=write_inputs)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--root", type=Path, required=True)
    manifest_parser.add_argument("--kind", choices=("onnx", "engine"), required=True)
    manifest_parser.add_argument("--result", type=Path, required=True)
    manifest_parser.set_defaults(func=manifest)

    skip_parser = subparsers.add_parser("skip")
    skip_parser.add_argument("--result", type=Path, required=True)
    skip_parser.add_argument("--reason", required=True)
    skip_parser.set_defaults(func=write_skip)

    quality_parser = subparsers.add_parser("score-quality")
    quality_parser.add_argument("--results-root", type=Path, required=True)
    quality_parser.add_argument("--asr-url", required=True)
    quality_parser.add_argument("--asr-model", required=True)
    quality_parser.set_defaults(func=score_quality)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--results-root", type=Path, required=True)
    summary_parser.add_argument("--csv", type=Path, required=True)
    summary_parser.add_argument("--markdown", type=Path, required=True)
    summary_parser.set_defaults(func=summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command_name == "measure":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            raise SystemExit("measure requires a command after --")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
