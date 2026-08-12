#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream a PCM16 mono WAV through MMAS's Realtime ASR adapter."""

import argparse
import asyncio
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_modal_ai_studio.backends.asr.openai_realtime import (  # noqa: E402
    OpenAIRealtimeASRBackend,
    _resample_pcm16,
)
from multi_modal_ai_studio.config.schema import ASRConfig  # noqa: E402


async def run(args: argparse.Namespace) -> str:
    with wave.open(str(args.wav), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Input must be PCM16 mono WAV")
        source_rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    pcm_16k = _resample_pcm16(pcm, source_rate, 16000)

    backend = OpenAIRealtimeASRBackend(ASRConfig(
        scheme="openai-realtime",
        realtime_url=args.url,
        api_key=os.environ.get(args.api_key_env, ""),
        model=args.model,
        language=args.language,
        realtime_transport="websocket",
        realtime_session_type="transcription",
        realtime_api_style=args.api_style,
        speech_timeout_ms=args.silence_duration_ms,
    ))
    await backend.start_stream()
    final_text = ""
    try:
        async def consume() -> None:
            nonlocal final_text
            async for result in backend.receive_results():
                control = result.metadata.get("control_event")
                if control:
                    print(f"[{control}]")
                elif result.metadata.get("error"):
                    raise RuntimeError(result.metadata["error"])
                elif result.is_final:
                    final_text = result.text
                    print(f"[final] {result.text}")
                    return
                elif result.text:
                    print(f"[partial] {result.text}")

        consumer = asyncio.create_task(consume())
        frames_per_chunk = int(16000 * args.chunk_ms / 1000)
        chunk_bytes = frames_per_chunk * 2
        for offset in range(0, len(pcm_16k), chunk_bytes):
            await backend.send_audio(pcm_16k[offset : offset + chunk_bytes])
            await asyncio.sleep(args.chunk_ms / 1000)
        silence_chunks = max(
            1,
            (args.silence_duration_ms + 500 + args.chunk_ms - 1) // args.chunk_ms,
        )
        for _ in range(silence_chunks):
            await backend.send_audio(b"\x00\x00" * frames_per_chunk)
            await asyncio.sleep(args.chunk_ms / 1000)
        await asyncio.wait_for(consumer, timeout=args.timeout)
    finally:
        await backend.stop_stream()
    if not final_text:
        raise RuntimeError("Provider returned no final transcript")
    return final_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--api-style",
        choices=("openai-ga", "openai-beta"),
        default="openai-ga",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--silence-duration-ms", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
