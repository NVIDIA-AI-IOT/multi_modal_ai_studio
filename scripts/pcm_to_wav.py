#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert 24 kHz 16-bit mono PCM (s16le) to WAV. Same format as Realtime debug PCM files."""

import argparse
import sys
from pathlib import Path

try:
    import wave
except ImportError:
    sys.exit("Python wave module not available.")

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def pcm_to_wav(pcm_path: Path, wav_path: Path | None = None) -> Path:
    if wav_path is None:
        wav_path = pcm_path.with_suffix(".wav")
    with open(pcm_path, "rb") as f:
        data = f.read()
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)
    return wav_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert 24 kHz 16-bit mono PCM to WAV")
    ap.add_argument("pcm", type=Path, help="Input .pcm file")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output .wav file (default: same name as input with .wav)")
    args = ap.parse_args()
    if not args.pcm.exists():
        print(f"Error: not found: {args.pcm}", file=sys.stderr)
        sys.exit(1)
    out = pcm_to_wav(args.pcm, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
