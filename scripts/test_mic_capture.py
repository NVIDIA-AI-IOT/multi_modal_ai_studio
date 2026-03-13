#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test server mic capture on Jetson: list devices and record a short 16kHz 16-bit mono sample.
Run from repo root: python scripts/test_mic_capture.py [--device DEV] [--seconds N] [--out FILE]
Uses same format as the app (SAMPLE_RATE=16000, CHANNELS=1, S16_LE).
"""
import argparse
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

# Match app constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_BYTES = 2048 * 2  # 2048 samples * 2 bytes


def list_alsa():
    """List ALSA capture devices (arecord -l)."""
    print("=== ALSA capture devices (arecord -l) ===\n")
    try:
        subprocess.run(["arecord", "-l"], check=True)
    except FileNotFoundError:
        print("arecord not found. Install alsa-utils.")
        return
    except subprocess.CalledProcessError as e:
        print("arecord -l failed:", e)
        return
    print("\nUse -D plughw:CARD,DEV (e.g. plughw:2,0) or -D hw:CARD,DEV")
    print("plughw does sample-rate conversion (use for USB mics that only do 48kHz).\n")


def list_pyaudio():
    """List PyAudio input devices if available."""
    try:
        import pyaudio
    except ImportError:
        print("PyAudio not installed; skipping PyAudio device list.")
        return
    pa = pyaudio.PyAudio()
    print("=== PyAudio input devices ===\n")
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        if dev.get("maxInputChannels", 0) > 0:
            print(f"  Index {i}: {dev.get('name', '?')} (inputs={dev.get('maxInputChannels')}, defaultSampleRate={dev.get('defaultSampleRate')})")
    pa.terminate()
    print()


def capture_alsa(device: str, seconds: float, out_path: Optional[str]) -> int:
    """Capture via arecord; return bytes captured."""
    dev = device.strip()
    if dev.startswith("hw:") and not dev.startswith("plughw:"):
        dev = "plug" + dev
        print(f"Using {dev} for rate conversion (same as app).")
    cmd = [
        "arecord",
        "-D", dev,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-t", "raw",
        "-d", str(int(seconds)),
    ]
    if out_path:
        # Always capture raw to stdout; then write .wav with header or raw as requested
        cmd.append("-")
        print(f"Capturing {seconds}s to {out_path} ...")
        result = subprocess.run(cmd, capture_output=True, timeout=int(seconds) + 5)
        if result.returncode != 0:
            print("arecord stderr:", result.stderr.decode("utf-8", errors="replace"))
            return 0
        raw_data = result.stdout
        size = len(raw_data)
        if out_path.lower().endswith(".wav"):
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(raw_data)
            print(f"Wrote WAV {out_path} ({size / (SAMPLE_RATE * 2):.2f}s at 16kHz 16-bit mono).")
        else:
            Path(out_path).write_bytes(raw_data)
            print(f"Wrote raw PCM {out_path} ({size} bytes, {size / (SAMPLE_RATE * 2):.2f}s).")
        return size
    else:
        cmd.append("-")
        print(f"Capturing {seconds}s to stdout (counting bytes) ...")
        result = subprocess.run(cmd, capture_output=True, timeout=int(seconds) + 5)
        if result.returncode != 0:
            print("arecord stderr:", result.stderr.decode("utf-8", errors="replace"))
            return 0
        size = len(result.stdout)
        print(f"Captured {size} bytes ({size / (SAMPLE_RATE * 2):.2f}s).")
        return size


def capture_pyaudio(device_index: int, seconds: float, out_path: Optional[str]) -> int:
    """Capture via PyAudio; return bytes captured."""
    try:
        import pyaudio
        import wave
    except ImportError as e:
        print("PyAudio (or wave) not installed:", e)
        return 0
    CHUNK_SAMPLES = 2048
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SAMPLES,
        )
    except Exception as e:
        print("Failed to open PyAudio device:", e)
        pa.terminate()
        return 0
    total = 0
    frames = []
    n_chunks = int(seconds * SAMPLE_RATE / CHUNK_SAMPLES) + 1
    print(f"Capturing {seconds}s from PyAudio device {device_index} ({n_chunks} chunks) ...")
    for _ in range(n_chunks):
        try:
            data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
        except Exception as e:
            print("Read error:", e)
            break
        total += len(data)
        if out_path:
            frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    if out_path and frames:
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        print(f"Wrote {total} bytes to {out_path} ({total / (SAMPLE_RATE * 2):.2f}s).")
    else:
        print(f"Captured {total} bytes ({total / (SAMPLE_RATE * 2):.2f}s).")
    return total


def main():
    ap = argparse.ArgumentParser(description="Test mic capture (16kHz 16-bit mono, same as app)")
    ap.add_argument("--list", action="store_true", help="List ALSA and PyAudio devices and exit")
    ap.add_argument("--device", "-D", default="plughw:2,0", help="ALSA device (e.g. plughw:2,0 or hw:2,0)")
    ap.add_argument("--pyaudio-index", type=int, default=None, help="Use PyAudio device index instead of ALSA")
    ap.add_argument("--seconds", "-d", type=float, default=3.0, help="Seconds to record")
    ap.add_argument("--out", "-o", default=None, help="Output file (.raw or .wav); if omitted, only count bytes")
    args = ap.parse_args()

    if args.list:
        list_alsa()
        list_pyaudio()
        return 0

    if args.pyaudio_index is not None:
        total = capture_pyaudio(args.pyaudio_index, args.seconds, args.out)
    else:
        total = capture_alsa(args.device, args.seconds, args.out)

    if total == 0:
        print("No data captured; check device ID (reconnect can change hw:X,Y). Run with --list to see devices.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
