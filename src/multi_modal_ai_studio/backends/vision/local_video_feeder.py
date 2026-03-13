# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Local Video Feeder — reads an MP4 file and feeds FrameBroker in a loop.

Works exactly like the USB camera path in server.py (handle_camera_stream):
  cap.read() → frame_broker.store_frame()
except the source is a local video file instead of /dev/video0.
"""

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalVideoFeeder:
    """Background thread that loops an MP4 file and stores frames in FrameBroker."""

    def __init__(self, video_path: str, frame_broker, fps: float = 5.0, jpeg_quality: int = 70):
        self._video_path = str(video_path)
        self._frame_broker = frame_broker
        self._fps = max(0.5, fps)
        self._jpeg_quality = jpeg_quality
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[LocalVideoFeeder] Already running")
            return True
        path = Path(self._video_path)
        if not path.is_file():
            logger.error("[LocalVideoFeeder] Video file not found: %s", self._video_path)
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="local-video-feeder")
        self._thread.start()
        logger.info("[LocalVideoFeeder] Started: %s @ %.1f fps", path.name, self._fps)
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("[LocalVideoFeeder] Stopped")

    def _loop(self) -> None:
        try:
            import cv2
        except ImportError:
            logger.error("[LocalVideoFeeder] opencv-python-headless not installed")
            return

        interval = 1.0 / self._fps
        frames_fed = 0

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                logger.error("[LocalVideoFeeder] Cannot open: %s", self._video_path)
                break

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            logger.info(
                "[LocalVideoFeeder] Opened %s (%.1f fps, %d frames, feeding at %.1f fps)",
                Path(self._video_path).name, video_fps, total_frames, self._fps,
            )

            # Calculate frame step: skip frames to match target FPS
            # e.g. 30fps video at 5fps target → read every 6th frame
            frame_step = max(1, round(video_fps / self._fps))

            frame_idx = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break  # end of video → will re-open for loop

                frame_idx += 1
                if frame_idx % frame_step != 0:
                    continue

                try:
                    self._frame_broker.store_frame(frame, jpeg_quality=self._jpeg_quality)
                    frames_fed += 1
                    if frames_fed <= 3 or frames_fed % 50 == 0:
                        logger.debug("[LocalVideoFeeder] Fed frame #%d", frames_fed)
                except Exception as e:
                    logger.warning("[LocalVideoFeeder] store_frame error: %s", e)

                self._stop_event.wait(interval)

            cap.release()
            if not self._stop_event.is_set():
                logger.debug("[LocalVideoFeeder] Looping video")

        logger.info("[LocalVideoFeeder] Thread exiting (fed %d frames total)", frames_fed)
