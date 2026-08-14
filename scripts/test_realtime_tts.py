#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesize exact text through MMAS's OpenAI Realtime TTS adapter."""

import argparse
import asyncio
import os
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_modal_ai_studio.backends.tts.openai_realtime import (  # noqa: E402
    OpenAIRealtimeTTSBackend,
)
from multi_modal_ai_studio.config.schema import TTSConfig  # noqa: E402


async def run(args: argparse.Namespace) -> None:
    backend = OpenAIRealtimeTTSBackend(
        TTSConfig(
            scheme="openai-realtime",
            realtime_url=args.url,
            realtime_transport="websocket",
            realtime_api_style=args.api_style,
            api_key=os.environ.get(args.api_key_env, ""),
            model=args.model,
            voice=args.voice,
            language=args.language,
            sample_rate=args.sample_rate,
        )
    )
    started = time.perf_counter()
    first_audio = None
    pcm = bytearray()
    sample_rate = args.sample_rate
    try:
        async for chunk in backend.synthesize_stream(args.text):
            if chunk.audio and first_audio is None:
                first_audio = time.perf_counter()
            pcm.extend(chunk.audio)
            sample_rate = chunk.sample_rate
    finally:
        await backend.close()
    completed = time.perf_counter()
    if not pcm or first_audio is None:
        raise RuntimeError("Provider returned no audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    duration = len(pcm) / (sample_rate * 2)
    print(f"TTFA: {first_audio - started:.3f}s")
    print(f"Total: {completed - started:.3f}s")
    print(f"Audio: {duration:.3f}s at {sample_rate} Hz")
    print(f"Output: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--url", default="ws://localhost:8082/v1/realtime")
    parser.add_argument("--model", default="nvidia/magpie_tts_multilingual_357m")
    parser.add_argument("--voice", default="Sofia")
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--api-style",
        choices=("openai-ga", "openai-beta"),
        default="openai-ga",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, default=Path("realtime-tts.wav"))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
