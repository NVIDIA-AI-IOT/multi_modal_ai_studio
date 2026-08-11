#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless LLM/TTS contention benchmark with a browser report.

The benchmark deliberately uses a fixed LLM prompt and fixed TTS text.  This
separates GPU/resource contention from conversational turn detection and makes
MIG/no-MIG runs directly comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin
import wave

import aiohttp
from aiohttp import web


DEFAULT_LLM_PROMPT = (
    "Explain in about 120 words how a conversational robot can safely help a "
    "person in a home."
)
DEFAULT_TTS_TEXT = (
    "A helpful robot should listen carefully, explain what it is doing, and ask "
    "before taking important actions. It should move slowly near people, protect "
    "private information, and stop immediately when asked. If it is uncertain, "
    "it should say so and request help instead of guessing."
)
SCENARIOS = ("tts-only", "llm-only", "overlap")
PCM_WIDTH_BYTES = 2
UNDERRUN_THRESHOLD_SECONDS = 0.005


def _endpoint(api_base: str, relative_path: str) -> str:
    """Build an endpoint from either a /v1 base URL or server root."""
    return urljoin(api_base.rstrip("/") + "/", relative_path.lstrip("/"))


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write mono PCM16 audio as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def simulate_playout(
    audio_chunks: Sequence[bytes],
    arrival_seconds: Sequence[float],
    sample_rate: int,
    prebuffer_ms: float,
) -> Tuple[bytes, Dict[str, Any]]:
    """Reconstruct live playout, inserting silence when the buffer underruns.

    Arrival timestamps are relative to the start of the TTS request.  Playback
    starts once the requested amount of audio has arrived, or when the final
    chunk arrives if the response is shorter than the prebuffer.
    """
    if len(audio_chunks) != len(arrival_seconds):
        raise ValueError("audio_chunks and arrival_seconds must have equal length")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not audio_chunks:
        return b"", {
            "playout_start_s": None,
            "underrun_count": 0,
            "underrun_s": 0.0,
            "inserted_silence_samples": 0,
        }

    bytes_per_second = sample_rate * PCM_WIDTH_BYTES
    durations = [len(chunk) / bytes_per_second for chunk in audio_chunks]
    target_seconds = max(0.0, prebuffer_ms / 1000.0)
    buffered_seconds = 0.0
    start_index = len(audio_chunks) - 1
    for index, duration in enumerate(durations):
        buffered_seconds += duration
        if buffered_seconds >= target_seconds:
            start_index = index
            break

    playout_start = max(0.0, float(arrival_seconds[start_index]))
    output = bytearray()
    playhead_wall_time = playout_start

    # Everything received before playback starts is already queued.
    for index in range(start_index + 1):
        output.extend(audio_chunks[index])
        playhead_wall_time += durations[index]

    underrun_count = 0
    underrun_seconds = 0.0
    silence_samples = 0
    underruns = []
    for index in range(start_index + 1, len(audio_chunks)):
        arrival = max(playout_start, float(arrival_seconds[index]))
        gap = max(0.0, arrival - playhead_wall_time)
        if gap >= UNDERRUN_THRESHOLD_SECONDS:
            gap_samples = round(gap * sample_rate)
            output.extend(b"\x00\x00" * gap_samples)
            actual_gap = gap_samples / sample_rate
            silence_samples += gap_samples
            underrun_count += 1
            underrun_seconds += actual_gap
            underruns.append(
                {
                    "before_chunk": index,
                    "start_s": playhead_wall_time,
                    "duration_s": actual_gap,
                }
            )
            playhead_wall_time += actual_gap
        output.extend(audio_chunks[index])
        playhead_wall_time += durations[index]

    return bytes(output), {
        "playout_start_s": playout_start,
        "underrun_count": underrun_count,
        "underrun_s": underrun_seconds,
        "inserted_silence_samples": silence_samples,
        "underruns": underruns,
    }


async def run_llm(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    max_tokens: int,
    on_text_delta: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Run one streaming OpenAI-compatible Chat Completions request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token = None
    text_parts: List[str] = []
    token_events = []
    completion_tokens = None
    async with session.post(
        _endpoint(api_base, "chat/completions"),
        json=request,
        headers=headers,
    ) as response:
        body_on_error = None
        if response.status >= 400:
            body_on_error = await response.text()
        if body_on_error is not None:
            raise RuntimeError(
                "LLM request failed ({}): {}".format(response.status, body_on_error[:500])
            )
        async for raw_line in response.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not isinstance(content, str) or not content:
                continue
            now = time.perf_counter()
            if first_token is None:
                first_token = now
            text_parts.append(content)
            arrival_seconds = now - started
            token_events.append({"arrival_s": arrival_seconds, "text": content})
            if on_text_delta is not None:
                on_text_delta(content, arrival_seconds)

    completed = time.perf_counter()
    generated_text = "".join(text_parts)
    generation_seconds = (
        completed - first_token if first_token is not None else None
    )
    tokens_per_second = None
    if completion_tokens and generation_seconds and generation_seconds > 0:
        tokens_per_second = completion_tokens / generation_seconds
    return {
        "ttft_s": first_token - started if first_token is not None else None,
        "total_s": completed - started,
        "generation_s": generation_seconds,
        "completion_tokens": completion_tokens,
        "tokens_per_s": tokens_per_second,
        "stream_events": token_events,
        "text": generated_text,
    }


async def run_openai_tts(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    api_key: Optional[str],
    model: str,
    voice: str,
    language: str,
    text: str,
    sample_rate: int,
) -> Tuple[Dict[str, Any], List[bytes], List[float]]:
    """Run one OpenAI-compatible PCM speech request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = {
        "model": model,
        "input": text,
        "voice": voice,
        "language": language.split("-")[0].lower(),
        "response_format": "pcm",
        "speed": 1.0,
    }
    started = time.perf_counter()
    chunks: List[bytes] = []
    arrivals: List[float] = []
    carry = b""
    async with session.post(
        _endpoint(api_base, "audio/speech"),
        json=request,
        headers=headers,
    ) as response:
        body_on_error = None
        if response.status >= 400:
            body_on_error = await response.text()
        if body_on_error is not None:
            raise RuntimeError(
                "TTS request failed ({}): {}".format(response.status, body_on_error[:500])
            )
        async for chunk in response.content.iter_chunked(16384):
            if not chunk:
                continue
            aligned = carry + chunk
            aligned_length = len(aligned) - (len(aligned) % PCM_WIDTH_BYTES)
            carry = aligned[aligned_length:]
            aligned = aligned[:aligned_length]
            if not aligned:
                continue
            chunks.append(bytes(aligned))
            arrivals.append(time.perf_counter() - started)
    if carry:
        raise RuntimeError("TTS response ended with an incomplete PCM16 sample")
    completed = time.perf_counter()
    return _tts_metrics(chunks, arrivals, sample_rate, completed - started), chunks, arrivals


async def run_riva_tts(
    *,
    server: str,
    voice: str,
    language: str,
    text: str,
    sample_rate: int,
) -> Tuple[Dict[str, Any], List[bytes], List[float]]:
    """Run one Riva streaming synthesis request through the MMAS adapter."""
    # Keep Riva optional for OpenAI-only benchmark environments.
    from multi_modal_ai_studio.backends.tts.riva import RivaTTSBackend
    from multi_modal_ai_studio.config.schema import TTSConfig

    backend = RivaTTSBackend(
        TTSConfig(
            scheme="riva",
            server=server,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
        )
    )
    started = time.perf_counter()
    chunks: List[bytes] = []
    arrivals: List[float] = []
    async for chunk in backend.synthesize_stream(text):
        if chunk.audio:
            chunks.append(chunk.audio)
            arrivals.append(time.perf_counter() - started)
    completed = time.perf_counter()
    return _tts_metrics(chunks, arrivals, sample_rate, completed - started), chunks, arrivals


def _tts_metrics(
    chunks: Sequence[bytes],
    arrivals: Sequence[float],
    sample_rate: int,
    total_seconds: float,
) -> Dict[str, Any]:
    audio_bytes = sum(len(chunk) for chunk in chunks)
    audio_seconds = audio_bytes / (sample_rate * PCM_WIDTH_BYTES)
    arrival_gaps = [
        arrivals[index] - arrivals[index - 1]
        for index in range(1, len(arrivals))
    ]
    events = [
        {
            "arrival_s": float(arrival),
            "bytes": len(chunk),
            "audio_s": len(chunk) / (sample_rate * PCM_WIDTH_BYTES),
        }
        for chunk, arrival in zip(chunks, arrivals)
    ]
    return {
        "ttfa_s": arrivals[0] if arrivals else None,
        "total_s": total_seconds,
        "audio_s": audio_seconds,
        "rtf": total_seconds / audio_seconds if audio_seconds > 0 else None,
        "chunk_count": len(chunks),
        "max_arrival_gap_s": max(arrival_gaps) if arrival_gaps else 0.0,
        "chunk_events": events,
    }


def _query_gpu_once() -> Optional[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    rows = []
    for line in result.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) < 6:
            continue
        row = {
            "index": values[0],
            "uuid": values[1],
            "gpu_util_percent": _optional_float(values[2]),
            "memory_util_percent": _optional_float(values[3]),
            "memory_used_mib": _optional_float(values[4]),
            "power_w": _optional_float(values[5]),
        }
        rows.append(row)
    return {"devices": rows} if rows else None


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def sample_gpu(stop: asyncio.Event, interval: float) -> List[Dict[str, Any]]:
    """Sample nvidia-smi without blocking the inference event loop."""
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    samples = []
    while not stop.is_set():
        sample = await loop.run_in_executor(None, _query_gpu_once)
        if sample:
            sample["time_s"] = time.perf_counter() - started
            samples.append(sample)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    return samples


async def run_trial(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    scenario: str,
    repeat_index: int,
    run_dir: Path,
) -> Dict[str, Any]:
    """Run one controlled scenario and persist its audio artifacts."""
    stop_gpu = asyncio.Event()
    gpu_task = asyncio.create_task(sample_gpu(stop_gpu, args.gpu_sample_interval))
    trial_started = time.perf_counter()

    async def tts_call():
        start_offset = time.perf_counter() - trial_started
        if args.tts_backend == "riva":
            result = await run_riva_tts(
                server=args.tts_server,
                voice=args.tts_voice,
                language=args.tts_language,
                text=args.tts_text,
                sample_rate=args.tts_sample_rate,
            )
        else:
            result = await run_openai_tts(
                session,
                api_base=args.tts_api_base,
                api_key=args.tts_api_key,
                model=args.tts_model,
                voice=args.tts_voice,
                language=args.tts_language,
                text=args.tts_text,
                sample_rate=args.tts_sample_rate,
            )
        result[0]["start_offset_s"] = start_offset
        return result

    async def llm_call(on_text_delta=None):
        start_offset = time.perf_counter() - trial_started
        result = await run_llm(
            session,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            model=args.llm_model,
            prompt=args.llm_prompt,
            max_tokens=args.llm_max_tokens,
            on_text_delta=on_text_delta,
        )
        result["start_offset_s"] = start_offset
        return result

    llm_result = None
    tts_result = None
    tts_chunks: List[bytes] = []
    tts_arrivals: List[float] = []
    overlap_trigger = None
    error = None
    try:
        if scenario == "tts-only":
            tts_result, tts_chunks, tts_arrivals = await tts_call()
        elif scenario == "llm-only":
            llm_result = await llm_call()
        elif scenario == "overlap":
            if args.overlap_start_mode == "simultaneous":
                llm_output, tts_output = await asyncio.gather(llm_call(), tts_call())
                overlap_trigger = {
                    "mode": "simultaneous",
                    "requested_units": 0,
                    "actual_units": 0,
                    "llm_elapsed_s": 0.0,
                }
            elif args.overlap_start_mode == "after-llm":
                llm_output = await llm_call()
                overlap_trigger = {
                    "mode": "after-llm",
                    "requested_units": None,
                    "actual_units": None,
                    "llm_elapsed_s": llm_output.get("total_s", 0.0),
                    "triggered_at_completion": True,
                }
                tts_output = await tts_call()
            else:
                # Keep the TTS text fixed so that trigger and MIG comparisons
                # retain an identical synthesis load.
                from multi_modal_ai_studio.webui.voice_pipeline import TTSChunkBuffer

                trigger = asyncio.Event()
                trigger_buffer = (
                    TTSChunkBuffer(first_chunk_words=args.overlap_trigger_units)
                    if args.overlap_start_mode == "mmas-units"
                    else None
                )
                strict_parts: List[str] = []
                trigger_state: Dict[str, Any] = {}

                def on_llm_delta(delta: str, llm_elapsed_s: float) -> None:
                    if trigger_buffer is not None:
                        ready = trigger_buffer.add(delta)
                    else:
                        strict_parts.append(delta)
                        candidate = "".join(strict_parts)
                        ready = (
                            candidate
                            if TTSChunkBuffer._speech_units(candidate)
                            >= args.overlap_trigger_units
                            else None
                        )
                    if ready and not trigger.is_set():
                        trigger_state.update(
                            {
                                "mode": args.overlap_start_mode,
                                "requested_units": args.overlap_trigger_units,
                                "actual_units": TTSChunkBuffer._speech_units(ready),
                                "llm_elapsed_s": llm_elapsed_s,
                                "trigger_text": ready,
                            }
                        )
                        trigger.set()

                async def llm_with_trigger():
                    result = await llm_call(on_llm_delta)
                    if not trigger.is_set():
                        remainder = (
                            trigger_buffer.flush()
                            if trigger_buffer is not None
                            else "".join(strict_parts)
                        ) or ""
                        trigger_state.update(
                            {
                                "mode": args.overlap_start_mode,
                                "requested_units": args.overlap_trigger_units,
                                "actual_units": TTSChunkBuffer._speech_units(remainder),
                                "llm_elapsed_s": result.get("total_s", 0.0),
                                "trigger_text": remainder,
                                "triggered_at_completion": True,
                            }
                        )
                        trigger.set()
                    return result

                async def tts_after_trigger():
                    await trigger.wait()
                    return await tts_call()

                llm_output, tts_output = await asyncio.gather(
                    llm_with_trigger(),
                    tts_after_trigger(),
                )
                overlap_trigger = trigger_state
            llm_result = llm_output
            tts_result, tts_chunks, tts_arrivals = tts_output
        else:
            raise ValueError("Unknown scenario: " + scenario)
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
    finally:
        stop_gpu.set()
        gpu_samples = await gpu_task

    artifact_prefix = "{}-{:02d}".format(scenario, repeat_index + 1)
    artifacts = {}
    playout = None
    if tts_result is not None and tts_chunks:
        clean_pcm = b"".join(tts_chunks)
        clean_name = artifact_prefix + "-clean.wav"
        write_pcm_wav(run_dir / clean_name, clean_pcm, args.tts_sample_rate)
        artifacts["clean_audio"] = clean_name

        playout_pcm, playout = simulate_playout(
            tts_chunks,
            tts_arrivals,
            args.tts_sample_rate,
            args.playout_prebuffer_ms,
        )
        playout_name = artifact_prefix + "-simulated-playout.wav"
        write_pcm_wav(run_dir / playout_name, playout_pcm, args.tts_sample_rate)
        artifacts["simulated_playout_audio"] = playout_name

    return {
        "scenario": scenario,
        "repeat": repeat_index + 1,
        "wall_s": time.perf_counter() - trial_started,
        "llm": llm_result,
        "tts": tts_result,
        "playout": playout,
        "overlap_trigger": overlap_trigger,
        "gpu_samples": gpu_samples,
        "artifacts": artifacts,
        "error": error,
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def summarize_trials(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the most useful metrics by scenario."""
    summary = {}
    for scenario in SCENARIOS:
        selected = [trial for trial in trials if trial["scenario"] == scenario]
        if not selected:
            continue
        summary[scenario] = {
            "trials": len(selected),
            "errors": sum(1 for trial in selected if trial.get("error")),
            "llm_ttft_mean_s": _mean(
                (trial.get("llm") or {}).get("ttft_s") for trial in selected
            ),
            "llm_total_mean_s": _mean(
                (trial.get("llm") or {}).get("total_s") for trial in selected
            ),
            "llm_tokens_per_s_mean": _mean(
                (trial.get("llm") or {}).get("tokens_per_s") for trial in selected
            ),
            "tts_ttfa_mean_s": _mean(
                (trial.get("tts") or {}).get("ttfa_s") for trial in selected
            ),
            "tts_rtf_mean": _mean(
                (trial.get("tts") or {}).get("rtf") for trial in selected
            ),
            "playout_underruns_mean": _mean(
                _optional_float((trial.get("playout") or {}).get("underrun_count"))
                for trial in selected
            ),
            "playout_underrun_mean_s": _mean(
                (trial.get("playout") or {}).get("underrun_s") for trial in selected
            ),
        }
    return summary


def _command_output(command: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    value = (result.stdout or result.stderr).strip()
    return value or None


def host_metadata() -> Dict[str, Any]:
    release_path = Path("/etc/nv_tegra_release")
    return {
        "hostname": _command_output(["hostname"]),
        "jetson_release": (
            release_path.read_text(encoding="utf-8", errors="replace").strip()
            if release_path.exists()
            else None
        ),
        "nvidia_smi_list": _command_output(["nvidia-smi", "-L"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,mig.mode.current,mig.mode.pending",
                "--format=csv,noheader",
            ]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


async def execute_benchmark(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    label = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in args.label
    ).strip("-")
    run_name = timestamp + ("-" + label if label else "")
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max(4, args.repeats * 2))
    trials = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.warmup:
            print("Warming up LLM and TTS...")
            warmup_args = argparse.Namespace(**vars(args))
            warmup_args.tts_text = "The speech service is warming up."
            warmup_args.llm_prompt = "Reply with one short sentence."
            warmup_dir = run_dir / ".warmup"
            warmup_dir.mkdir()
            await run_trial(warmup_args, session, "overlap", 0, warmup_dir)

        for repeat_index in range(args.repeats):
            for scenario in args.scenarios:
                print(
                    "[{}/{}] {}".format(
                        repeat_index + 1,
                        args.repeats,
                        scenario,
                    ),
                    flush=True,
                )
                trial = await run_trial(
                    args,
                    session,
                    scenario,
                    repeat_index,
                    run_dir,
                )
                trials.append(trial)
                if trial["error"]:
                    print("  ERROR: " + trial["error"], file=sys.stderr)
                else:
                    tts = trial.get("tts") or {}
                    llm = trial.get("llm") or {}
                    playout = trial.get("playout") or {}
                    print(
                        "  TTFT={} TTFA={} RTF={} underruns={}".format(
                            _format_metric(llm.get("ttft_s")),
                            _format_metric(tts.get("ttfa_s")),
                            _format_metric(tts.get("rtf")),
                            playout.get("underrun_count", "-"),
                        )
                    )

    manifest = {
        "schema_version": 1,
        "run_id": run_name,
        "label": args.label,
        "display_order": args.display_order,
        "comparison_role": args.comparison_role,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host_metadata(),
        "configuration": {
            "scenarios": args.scenarios,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "llm_api_base": args.llm_api_base,
            "llm_model": args.llm_model,
            "llm_prompt": args.llm_prompt,
            "llm_max_tokens": args.llm_max_tokens,
            "tts_backend": args.tts_backend,
            "tts_api_base": (
                args.tts_api_base if args.tts_backend == "openai-rest" else None
            ),
            "tts_server": args.tts_server if args.tts_backend == "riva" else None,
            "tts_model": args.tts_model,
            "tts_voice": args.tts_voice,
            "tts_language": args.tts_language,
            "tts_text": args.tts_text,
            "tts_sample_rate": args.tts_sample_rate,
            "playout_prebuffer_ms": args.playout_prebuffer_ms,
            "overlap_start_mode": args.overlap_start_mode,
            "overlap_trigger_units": args.overlap_trigger_units,
        },
        "summary": summarize_trials(trials),
        "trials": trials,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Results: " + str(manifest_path))
    return run_dir


def _format_metric(value: Optional[float]) -> str:
    return "-" if value is None else "{:.3f}".format(value)


def discover_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Load valid manifests below a result directory."""
    results_dir = results_dir.resolve()
    candidates = []
    direct = results_dir / "manifest.json"
    if direct.exists():
        candidates.append(direct)
    candidates.extend(sorted(results_dir.glob("*/manifest.json"), reverse=True))
    runs = []
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        relative_parent = manifest_path.parent.relative_to(results_dir)
        manifest["_artifact_base"] = str(relative_parent) or "."
        runs.append(manifest)
    explicit = sorted(
        (run for run in runs if run.get("display_order") is not None),
        key=lambda run: run["display_order"],
    )
    implicit = [run for run in runs if run.get("display_order") is None]
    return explicit + implicit


async def serve_results(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir).expanduser().resolve()
    dashboard_path = Path(__file__).with_name("speech_contention_dashboard.html")

    async def index(_: web.Request) -> web.FileResponse:
        response = web.FileResponse(dashboard_path)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response

    async def runs(_: web.Request) -> web.Response:
        response = web.json_response(discover_runs(results_dir))
        response.headers["Cache-Control"] = "no-store"
        return response

    async def artifact(request: web.Request) -> web.FileResponse:
        run_name = request.match_info["run"]
        relative = request.match_info["path"]
        candidate = (results_dir / run_name / relative).resolve()
        if results_dir not in candidate.parents or not candidate.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(candidate)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/runs", runs)
    app.router.add_get("/artifacts/{run}/{path:.*}", artifact)
    print("Speech Contention Lab: http://{}:{}".format(args.host, args.port))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a headless contention benchmark")
    run.add_argument("--label", default="no-mig", help="Run label shown in comparisons")
    run.add_argument(
        "--display-order",
        type=int,
        default=None,
        help="Optional stable order in the browser experiment matrix",
    )
    run.add_argument(
        "--comparison-role",
        default="",
        help="Short description such as baseline, backend variant, or load variant",
    )
    run.add_argument("--output-dir", default="./benchmark-results/speech-contention")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    run.add_argument("--warmup", action="store_true")
    run.add_argument("--request-timeout", type=float, default=240.0)
    run.add_argument("--gpu-sample-interval", type=float, default=0.25)
    run.add_argument("--playout-prebuffer-ms", type=float, default=100.0)
    run.add_argument(
        "--overlap-start-mode",
        choices=("mmas-units", "strict-units", "after-llm", "simultaneous"),
        default="mmas-units",
        help=(
            "Start overlap TTS from the MMAS phrase-aware trigger, an exact "
            "unit threshold, after LLM completion, or at LLM request start"
        ),
    )
    run.add_argument(
        "--overlap-trigger-units",
        type=int,
        default=10,
        help="Speech units before TTS for mmas-units or strict-units modes",
    )

    run.add_argument("--llm-api-base", default="http://localhost:8000/v1")
    run.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY"))
    run.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    run.add_argument("--llm-prompt", default=DEFAULT_LLM_PROMPT)
    run.add_argument("--llm-max-tokens", type=int, default=192)

    run.add_argument(
        "--tts-backend",
        choices=("openai-rest", "riva"),
        default="openai-rest",
    )
    run.add_argument("--tts-api-base", default="http://localhost:8082/v1")
    run.add_argument("--tts-api-key", default=os.environ.get("TTS_API_KEY"))
    run.add_argument("--tts-server", default="localhost:50051")
    run.add_argument("--tts-model", default="nvidia/magpie_tts_multilingual_357m")
    run.add_argument("--tts-voice", default="Sofia")
    run.add_argument("--tts-language", default="en-US")
    run.add_argument("--tts-text", default=DEFAULT_TTS_TEXT)
    run.add_argument("--tts-sample-rate", type=int, default=22050)

    serve = subparsers.add_parser("serve", help="Serve waveform/audio reports")
    serve.add_argument(
        "--results-dir",
        default="./benchmark-results/speech-contention",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8097)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.tts_sample_rate <= 0:
        raise SystemExit("--tts-sample-rate must be positive")
    if args.gpu_sample_interval <= 0:
        raise SystemExit("--gpu-sample-interval must be positive")
    if args.playout_prebuffer_ms < 0:
        raise SystemExit("--playout-prebuffer-ms cannot be negative")
    if args.overlap_trigger_units <= 0:
        raise SystemExit("--overlap-trigger-units must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.command == "run":
        asyncio.run(execute_benchmark(args))
    else:
        asyncio.run(serve_results(args))


if __name__ == "__main__":
    main()
