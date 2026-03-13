#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Capture 6 frames from a USB webcam and write a 3-second MP4 at 2 fps.
Suitable for NVIDIA Cosmos-reason2 (MP4 input). Uses H.264 when available.
Usage:
  python scripts/capture_short_webcam_video.py [--device 0] [--out output.mp4]
"""
from __future__ import annotations

import argparse
import sys
import time

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture 6 frames from webcam → 3s MP4 @ 2 fps")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--out", "-o", default="webcam_6frames.mp4", help="Output MP4 path")
    parser.add_argument("--spaced", action="store_true", help="Space captures by 0.5s for true 2 fps timing")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("Install opencv: pip install opencv-python-headless", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Cannot open camera device {args.device}", file=sys.stderr)
        return 1

    num_frames = 6
    fps = 2.0
    frames = []

    # Warm up: discard a few frames so we get fresh ones
    for _ in range(3):
        cap.read()

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {i+1}", file=sys.stderr)
            cap.release()
            return 1
        frames.append(frame)
        if args.spaced and i < num_frames - 1:
            time.sleep(0.5)

    cap.release()

    if not frames:
        return 1

    h, w = frames[0].shape[:2]
    # Prefer H.264 for Cosmos/wide compatibility; fallback to mp4v
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        if out.isOpened():
            break
        out.release()
    else:
        print("No suitable codec (tried avc1, H264, mp4v)", file=sys.stderr)
        return 1

    for f in frames:
        out.write(f)
    out.release()

    print(f"Wrote {args.out} ({num_frames} frames, {fps} fps, {w}x{h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
