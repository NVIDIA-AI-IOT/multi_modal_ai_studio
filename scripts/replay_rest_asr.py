#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Replay identical WAV bytes alone, during TTS, and just after TTS.

Uses existing REST services; does not download models or change servers.
No audio is played. Results contain transcripts: keep them private.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
import wave
from pathlib import Path

import aiohttp


async def transcribe(client, args, wav, clock):
    form = aiohttp.FormData()
    form.add_field("file", wav, filename="replay.wav", content_type="audio/wav")
    form.add_field("model", args.model)
    if args.language != "auto":
        form.add_field("language", args.language)
    form.add_field("response_format", "json")
    start = time.perf_counter()
    async with client.post(args.api_base.rstrip("/") + "/audio/transcriptions", data=form) as res:
        body = await res.text()
        result = {
            "http_status": res.status,
            "start_s": start - clock,
            "end_s": time.perf_counter() - clock,
            "seconds": time.perf_counter() - start,
        }
        if res.status == 200:
            result["transcript"] = json.loads(body).get("text", "")
        else:
            result["error"] = body[:500]
    return result


async def synthesize(client, args, clock, issued):
    start = time.perf_counter()
    issued.set()
    payload = {
        "model": args.tts_model,
        "voice": args.voice,
        "input": args.tts_text,
        "response_format": "pcm",
        "speed": 1.0,
    }
    async with client.post(args.api_base.rstrip("/") + "/audio/speech", json=payload) as res:
        result = {"http_status": res.status, "start_s": start - clock, "first_audio_s": None}
        size = 0
        async for chunk in res.content.iter_chunked(4096):
            if chunk and result["first_audio_s"] is None and res.status == 200:
                result["first_audio_s"] = time.perf_counter() - clock
            size += len(chunk)
        result.update(end_s=time.perf_counter() - clock, bytes=size)
    return result


async def run(args):
    wav = args.wav.read_bytes()
    with wave.open(str(args.wav), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Use the captured mono PCM16 WAV, without re-encoding")
        duration = audio.getnframes() / audio.getframerate()
    headers = {}
    if key := os.environ.get("SPEECH_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    report = {
        "input_sha256": hashlib.sha256(wav).hexdigest(),
        "audio_seconds": duration,
        "model": args.model,
        "language": args.language,
        "tts_model": args.tts_model,
        "voice": args.voice,
        "note": "Request latency includes server queueing. Overlap is measured, not assumed.",
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting evidence, and checkpoint each run for interrupted tests.
    with args.output.open("x") as output:
        os.chmod(args.output, 0o600)
        async with aiohttp.ClientSession(
            headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as client:
            for repeat in range(args.repeats):
                # Rotate order to avoid confounding a mode with warm-up/drift.
                modes = list(args.modes)
                modes = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
                for mode in modes:
                    clock = time.perf_counter()
                    row = {"mode": mode, "repeat": repeat + 1}
                    try:
                        if mode == "asr-only":
                            row["asr"] = await transcribe(client, args, wav, clock)
                        elif mode == "after-tts":
                            row["tts"] = await synthesize(client, args, clock, asyncio.Event())
                            row["asr"] = await transcribe(client, args, wav, clock)
                        else:
                            issued = asyncio.Event()
                            task = asyncio.create_task(synthesize(client, args, clock, issued))
                            try:
                                await issued.wait()
                                await asyncio.sleep(args.overlap_delay)
                                row["asr"] = await transcribe(client, args, wav, clock)
                                row["tts"] = await task
                            finally:
                                if not task.done():
                                    task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                            row["measured_overlap_s"] = max(
                                0,
                                min(row["asr"]["end_s"], row["tts"]["end_s"])
                                - max(row["asr"]["start_s"], row["tts"]["start_s"]),
                            )
                    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                        row["error_type"] = type(exc).__name__
                    report["runs"].append(row)
                    output.seek(0)
                    json.dump(report, output, ensure_ascii=False, indent=2)
                    output.truncate()
                    output.flush()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    await asyncio.sleep(args.pause)
        for mode in args.modes:
            values = [
                r["asr"]["seconds"]
                for r in report["runs"]
                if r["mode"] == mode and r.get("asr", {}).get("http_status") == 200
            ]
            if values:
                print(
                    f"{mode}: n={len(values)} median={statistics.median(values):.3f}s "
                    f"max={max(values):.3f}s",
                    flush=True,
                )
    return (
        1
        if any(
            r.get("error_type")
            or r.get("asr", {}).get("http_status") != 200
            or ("tts" in r and r["tts"]["http_status"] != 200)
            for r in report["runs"]
        )
        else 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="Systran/faster-whisper-small")
    parser.add_argument("--language", default="ja", help="ISO language, or auto")
    parser.add_argument("--tts-model", default="speaches-ai/Kokoro-82M-v1.0-ONNX-fp16")
    parser.add_argument("--voice", default="jm_kumo")
    parser.add_argument(
        "--tts-text",
        default=(
            "ロボティクスの歴史は古代の機械から始まりました。現代では産業、医療、探査など、"
            "さまざまな分野でロボットが活躍しています。安全性と人間との協調が重要です。"
        )
        * 3,
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["asr-only", "tts-overlap", "after-tts"],
        default=["asr-only", "tts-overlap", "after-tts"],
    )
    parser.add_argument("--overlap-delay", type=float, default=0.05)
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or min(args.overlap_delay, args.pause) < 0:
        parser.error("repeats must be 1..100; delays must be non-negative")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
