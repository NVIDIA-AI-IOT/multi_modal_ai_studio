#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Send a WAV file (16 kHz or 44.1/48 kHz) to the OpenAI Realtime API.
Resamples to 24 kHz if needed, then streams PCM to the client and prints
transcript + saves response audio to a file.

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/test_realtime_with_wav.py what-is-1-plus-1.wav
  python scripts/test_realtime_with_wav.py what-is-1-plus-1.wav --output response.wav
"""

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multi_modal_ai_studio.backends.realtime import (
    REALTIME_SAMPLE_RATE,
    OpenAIRealtimeClient,
    RealtimeEvent,
)


def load_wav(path: Path) -> Tuple[bytes, int]:
    """Load WAV as 16-bit mono PCM. Returns (pcm_bytes, sample_rate)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        if wav.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit")
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    return frames, rate


def resample_pcm(pcm_bytes: bytes, from_rate: int, to_rate: int = REALTIME_SAMPLE_RATE) -> bytes:
    """Resample 16-bit mono PCM from from_rate to to_rate (default 24 kHz)."""
    if from_rate == to_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    n_old = len(samples)
    n_new = int(round(n_old * to_rate / from_rate))
    x_old = np.arange(n_old)
    x_new = np.linspace(0, n_old - 1, n_new)
    resampled = np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.int16)
    return resampled.tobytes()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Send WAV to Realtime API and get response")
    parser.add_argument("wav", type=Path, help="Input WAV (16 kHz or other; will resample to 24 kHz)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Save response audio to this WAV")
    parser.add_argument("--timeout", type=float, default=30.0, help="Max seconds to wait for response")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every event type received")
    parser.add_argument("--debug", action="store_true", help="Log every raw server event type (implies -v)")
    parser.add_argument("--model", default="gpt-realtime", help="Realtime model (default: gpt-realtime; try gpt-realtime-2025-08-28)")
    args = parser.parse_args()

    if args.verbose or args.debug:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("multi_modal_ai_studio.backends.realtime.client").setLevel(logging.INFO)

    api_key = __import__("os").environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    if not args.wav.exists():
        print(f"File not found: {args.wav}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.wav}...")
    pcm_bytes, rate = load_wav(args.wav)
    n_frames = len(pcm_bytes) // 2
    duration = n_frames / rate
    print(f"  {rate} Hz, {duration:.2f} s, {n_frames} samples")

    if rate != REALTIME_SAMPLE_RATE:
        print(f"Resampling {rate} Hz -> {REALTIME_SAMPLE_RATE} Hz...")
        pcm_bytes = resample_pcm(pcm_bytes, rate, REALTIME_SAMPLE_RATE)
        n_frames = len(pcm_bytes) // 2
        print(f"  -> {n_frames} samples at {REALTIME_SAMPLE_RATE} Hz")

    url = "wss://api.openai.com/v1/realtime"
    client = OpenAIRealtimeClient(
        url=url,
        api_key=api_key,
        model=args.model,
        instructions="You are a helpful assistant. Answer concisely.",
        log_all_events=args.debug,
    )

    print("Connecting...")
    await client.connect()

    # Collect response audio and transcript
    response_audio: list = []
    transcript_parts: list = []
    verbose = args.verbose or args.debug
    chunk_frames = 480  # 20 ms at 24 kHz

    async def run_and_consume() -> None:
        event_iter = client.events()
        # Wait for session_ready before sending any audio (server ignores appends until then)
        async for ev in event_iter:
            if ev is None:
                return
            if verbose and ev.raw:
                print(f"  [ev] {ev.kind} (type={ev.raw.get('type', '')})")
            if ev.kind == "error":
                print("Error:", ev.message)
                return
            if ev.kind == "session_ready":
                print("  Session ready")
                break
        else:
            print("Error: session ready never received")
            return

        # Send all audio. With server VAD (default), the server auto-commits when it
        # detects end of speech. Append a short silence so VAD reliably sees "speech stopped".
        n_sent = 0
        for i in range(0, len(pcm_bytes), chunk_frames * 2):
            chunk = pcm_bytes[i : i + chunk_frames * 2]
            if chunk:
                await client.send_audio(chunk)
                n_sent += len(chunk) // 2
        # ~200 ms silence so server VAD commits (do not call commit_audio() or we get "buffer too small")
        silence_ms = 200
        silence_frames = int(REALTIME_SAMPLE_RATE * silence_ms / 1000)
        await client.send_audio(b"\x00\x00" * silence_frames)
        n_sent += silence_frames
        print(f"Sent {n_sent} samples ({n_sent / REALTIME_SAMPLE_RATE:.2f} s), waiting for server VAD to commit...")

        # Drain remaining events (transcript, audio, response_done)
        async for ev in event_iter:
            if ev is None:
                return
            if verbose and ev.raw:
                print(f"  [ev] {ev.kind} (type={ev.raw.get('type', '')})")
            if ev.kind == "error":
                print("Error:", ev.message)
                return
            if ev.kind == "transcript_delta":
                if ev.text:
                    print(f"  [partial] {ev.text}", end="", flush=True)
            elif ev.kind == "output_transcript_delta":
                if ev.text:
                    print(ev.text, end="", flush=True)
            elif ev.kind == "transcript_completed":
                if ev.text:
                    transcript_parts.append(ev.text)
                    print(f"\n  [transcript] {ev.text}")
            elif ev.kind == "audio":
                if ev.audio:
                    response_audio.append(ev.audio)
            elif ev.kind == "response_done":
                print("  [response.done]")
                break  # End of turn; exit so we don't wait forever for more events

    try:
        await asyncio.wait_for(run_and_consume(), timeout=args.timeout)
    except asyncio.TimeoutError:
        print(f"  (timeout after {args.timeout}s)")
    finally:
        await client.disconnect()

    print(f"  Received {len(response_audio)} audio chunks, {len(transcript_parts)} transcript(s)")
    if not response_audio and not transcript_parts:
        print("  (Empty response: server sent no response.output_audio.delta or transcript.)")
        print("  Try: --model gpt-realtime-2025-08-28 | different WAV | paid tier (Realtime not on free tier) | --debug")

    if response_audio and args.output:
        out_path = args.output
        combined = b"".join(response_audio)
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(REALTIME_SAMPLE_RATE)
            wav.writeframes(combined)
        print(f"Saved response audio to {out_path} ({len(combined)//2} samples)")

    if transcript_parts:
        print("Final transcript:", " ".join(transcript_parts))


if __name__ == "__main__":
    asyncio.run(main())
