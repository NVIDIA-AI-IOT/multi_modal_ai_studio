#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the workbench's Qwen3-TTS adapter with a saved preset."""

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_modal_ai_studio.backends.tts.openai_rest import OpenAIRestTTSBackend  # noqa: E402
from multi_modal_ai_studio.config.schema import SessionConfig  # noqa: E402


async def run(args: argparse.Namespace) -> None:
    config = SessionConfig.from_yaml(args.preset).tts
    if config.request_profile not in ("qwen3-customvoice", "qwen3-voicedesign", "qwen3-base"):
        raise ValueError("Choose a preset with an explicit Qwen3-TTS request profile")
    if args.api_base:
        config.api_base = args.api_base
    if args.reference_voice_id:
        config.reference_voice_id = args.reference_voice_id
    if args.reference_voice_mode:
        config.reference_voice_mode = args.reference_voice_mode
    # Refuse accidental replacement of prior listening/benchmark evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = OpenAIRestTTSBackend(config)
    started = time.perf_counter()
    first_audio = None
    pcm = bytearray()
    chunks = 0
    try:
        async with asyncio.timeout(args.timeout):
            async for chunk in backend.synthesize_stream(args.text):
                if chunk.audio:
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    if chunk.sample_rate != 24000 or len(chunk.audio) % 2:
                        raise RuntimeError("Expected aligned 24 kHz PCM16")
                    pcm.extend(chunk.audio)
                    chunks += 1
        completed = time.perf_counter()
    finally:
        await backend.close()
    if not pcm or first_audio is None:
        raise RuntimeError("Provider returned no audio")
    output = args.output_dir / "speech.wav"
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    duration = len(pcm) / 48000
    metrics = {
        "profile": config.request_profile,
        "model": config.model,
        "language": config.language,
        "voice": config.voice if config.request_profile == "qwen3-customvoice" else None,
        "instructions": config.instructions,
        "reference_voice_id": config.reference_voice_id,
        "reference_voice_mode": config.reference_voice_mode if config.request_profile == "qwen3-base" else None,
        "text": args.text,
        "sample_rate": 24000,
        "client_first_pcm_seconds": first_audio - started,
        "total_seconds": completed - started,
        "audio_seconds": duration,
        "rtf": (completed - started) / duration,
        "pcm_chunks": chunks,
        "wav_path": str(output.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--text", default="Hello! I am ready for a little adventure.")
    parser.add_argument("--api-base", help="Override only the TTS endpoint")
    parser.add_argument("--reference-voice-id", help="Server-local reference ID for Qwen3-TTS Base")
    parser.add_argument("--reference-voice-mode", choices=("icl", "x-vector"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new output directory")
    parser.add_argument("--timeout", type=float, default=120)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
