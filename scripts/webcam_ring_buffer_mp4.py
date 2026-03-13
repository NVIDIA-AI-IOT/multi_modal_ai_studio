#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ring buffer of (timestamp, frame) from a webcam; sample frames at fixed time
offsets (0s, 0.5s, 1s, 1.5s, 2s, 2.5s) and encode to in-memory MP4 → base64.
No file written; encode time is typically 20–80 ms for 6 frames (resolution-dependent).
"""
from __future__ import annotations

import base64
import io
import sys
import time
from collections import deque
from typing import Deque

# Default: 6 frames at 0, 0.5, 1, 1.5, 2, 2.5 sec → 3 s video @ 2 fps
DEFAULT_OFFSETS_SEC = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
FPS = 2.0


class FrameRingBuffer:
    """Ring buffer of (timestamp, frame). Drops oldest when full."""

    def __init__(self, max_duration_sec: float = 3.5, max_frames: int = 120):
        self.max_duration_sec = max_duration_sec
        self.max_frames = max_frames
        self._buf: Deque[tuple[float, object]] = deque(maxlen=max_frames)

    def push(self, timestamp: float, frame: object) -> None:
        self._buf.append((timestamp, frame))
        # Optional: trim by time so we don't keep frames older than max_duration_sec
        if self._buf:
            t0 = self._buf[0][0]
            while len(self._buf) > 1 and (self._buf[-1][0] - self._buf[0][0]) > self.max_duration_sec:
                self._buf.popleft()

    def sample_at_offsets(self, offsets_sec: tuple[float, ...]) -> list:
        """Return frames at 'offsets_sec' seconds before the latest timestamp (oldest to newest)."""
        if not self._buf:
            return []
        latest_ts = self._buf[-1][0]
        frames_ordered: list[tuple[float, object]] = []  # (offset, frame)

        for offset in sorted(offsets_sec):
            target_ts = latest_ts - offset
            best = min(
                self._buf,
                key=lambda x: abs(x[0] - target_ts),
            )
            frames_ordered.append((offset, best[1]))
        # Video chronological order: oldest (largest offset) first
        return [f for _, f in reversed(frames_ordered)]

    def latest_ts(self) -> float | None:
        return self._buf[-1][0] if self._buf else None

    def __len__(self) -> int:
        return len(self._buf)


def _encode_mp4_pyav(frames: list, fps: float, width: int, height: int) -> bytes:
    """Encode BGR frames to MP4 bytes using PyAV (no file)."""
    import av

    out = io.BytesIO()
    container = av.open(out, "w", format="mp4")
    stream = container.add_stream("libx264", rate=int(fps) if fps == int(fps) else 2)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    for i, bgr in enumerate(frames):
        # PyAV from_ndarray: OpenCV frames are BGR
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)
    container.close()
    return out.getvalue()


def _encode_mp4_tempfile(frames: list, fps: float, width: int, height: int) -> bytes:
    """Fallback: encode with OpenCV to temp file, read back (no PyAV)."""
    import os
    import tempfile

    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError("VideoWriter could not open temp file")
        for f in frames:
            out.write(f)
        out.release()
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def frames_to_mp4_base64(
    frames: list,
    fps: float = FPS,
    use_pyav: bool | None = None,
) -> tuple[str, float]:
    """
    Encode a list of BGR frames (e.g. from OpenCV) to MP4 in memory, return (base64_str, encode_time_sec).
    Prefer PyAV for true in-memory; fallback to temp file + OpenCV.
    """
    if not frames:
        return "", 0.0
    h, w = frames[0].shape[:2]
    if use_pyav is None:
        try:
            import av  # noqa: F401
            use_pyav = True
        except ImportError:
            use_pyav = False

    t0 = time.perf_counter()
    if use_pyav:
        raw = _encode_mp4_pyav(frames, fps, w, h)
    else:
        raw = _encode_mp4_tempfile(frames, fps, w, h)
    elapsed = time.perf_counter() - t0
    return base64.b64encode(raw).decode("ascii"), elapsed


def main() -> int:
    """Run a short test: fill buffer from webcam, sample, encode to base64, print timing."""
    import argparse

    parser = argparse.ArgumentParser(description="Ring-buffer webcam → sample 6 frames → MP4 base64")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=float, default=3.5, help="Seconds to fill buffer before sampling")
    parser.add_argument("--pyav", action="store_true", help="Force PyAV (fail if not installed)")
    parser.add_argument("--no-pyav", action="store_true", help="Use temp-file fallback")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv required: pip install opencv-python-headless", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Cannot open camera {args.device}", file=sys.stderr)
        return 1

    buf = FrameRingBuffer(max_duration_sec=4.0, max_frames=120)
    print("Filling ring buffer for {:.1f}s...".format(args.duration))
    start = time.perf_counter()
    while (time.perf_counter() - start) < args.duration:
        ret, frame = cap.read()
        if not ret:
            break
        buf.push(time.perf_counter(), frame)
    cap.release()

    frames = buf.sample_at_offsets(DEFAULT_OFFSETS_SEC)
    if len(frames) < len(DEFAULT_OFFSETS_SEC):
        print("Not enough frames in buffer", file=sys.stderr)
        return 1

    use_pyav = None if not args.pyav and not args.no_pyav else args.pyav
    b64, encode_sec = frames_to_mp4_base64(frames, use_pyav=use_pyav)
    print(f"Encoded {len(frames)} frames → base64 in {encode_sec*1000:.1f} ms")
    print(f"Base64 length: {len(b64)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
