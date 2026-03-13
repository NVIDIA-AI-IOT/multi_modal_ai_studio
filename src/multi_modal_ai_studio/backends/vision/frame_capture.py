# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Frame Capture for VLM (Vision Language Models).

Production-ready, low-latency frame capture for Jetson cameras.
Maintains a background thread that continuously captures frames,
providing instant access to the latest frame when needed.

Design principles:
- Zero-latency frame access (frame already captured when ASR final arrives)
- Minimal resource usage (configurable capture rate)
- Generic interface (works with any V4L2 camera)
- Thread-safe (safe to call get_frame from any async context)
"""

import base64
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CapturedFrame:
    """A captured video frame with metadata."""
    
    # Base64-encoded JPEG image data
    image_base64: str
    # MIME type (always image/jpeg for now)
    media_type: str = "image/jpeg"
    # Capture timestamp (time.time())
    timestamp: float = 0.0
    # Frame dimensions
    width: int = 0
    height: int = 0
    
    def to_data_url(self) -> str:
        """Convert to data URL for OpenAI-compatible VLM API."""
        return f"data:{self.media_type};base64,{self.image_base64}"


class FrameCapture:
    """
    Low-latency frame capture for Jetson cameras.
    
    Usage:
        capture = FrameCapture(device="/dev/video0")
        capture.start()
        
        # Later, when ASR final arrives:
        frame = capture.get_latest_frame()
        if frame:
            # Send frame.to_data_url() to VLM
            ...
        
        capture.stop()
    
    For best latency:
    - Start capture when session begins
    - Call get_latest_frame() when you need a frame (no wait)
    - Stop capture when session ends
    """
    
    def __init__(
        self,
        device: str = "/dev/video0",
        target_fps: float = 5.0,
        resolution: Optional[Tuple[int, int]] = None,
        jpeg_quality: int = 85,
    ):
        """
        Initialize frame capture.
        
        Args:
            device: V4L2 device path (e.g., /dev/video0)
            target_fps: Target capture rate (lower = less CPU usage)
            resolution: Optional (width, height) to resize frames
            jpeg_quality: JPEG compression quality (1-100, higher = larger)
        """
        self.device = device
        self.target_fps = max(1.0, min(30.0, target_fps))
        self.resolution = resolution
        self.jpeg_quality = jpeg_quality
        
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[CapturedFrame] = None
        self._frame_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
    
    def start(self) -> bool:
        """
        Start background frame capture.
        
        Returns:
            True if started successfully, False if camera unavailable.
        """
        if self._running:
            return True
        
        try:
            import cv2
        except ImportError:
            logger.error("FrameCapture requires opencv-python: pip install opencv-python-headless")
            self._last_error = "OpenCV not installed"
            return False
        
        # Check device exists
        if not Path(self.device).exists():
            logger.warning(f"Camera device not found: {self.device}")
            self._last_error = f"Device not found: {self.device}"
            return False
        
        # Open camera
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            logger.warning(f"Failed to open camera: {self.device}")
            self._last_error = f"Failed to open: {self.device}"
            return False
        
        # Configure camera (optional)
        if self.resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"FrameCapture started: {self.device} @ {self.target_fps} fps")
        return True
    
    def stop(self) -> None:
        """Stop background frame capture and release camera."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        logger.info(f"FrameCapture stopped: {self._frame_count} frames captured, {self._error_count} errors")
    
    def get_latest_frame(self) -> Optional[CapturedFrame]:
        """
        Get the most recently captured frame.
        
        This is a non-blocking call - returns immediately with the
        latest frame (or None if no frame captured yet).
        
        Returns:
            CapturedFrame or None if no frame available.
        """
        with self._lock:
            return self._latest_frame
    
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running and self._cap is not None
    
    def get_status(self) -> dict:
        """Get capture status for debugging."""
        return {
            "device": self.device,
            "running": self._running,
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "has_frame": self._latest_frame is not None,
        }
    
    def _capture_loop(self) -> None:
        """Background thread: continuously capture frames."""
        import cv2
        
        frame_interval = 1.0 / self.target_fps
        last_capture = 0.0
        
        while self._running:
            now = time.time()
            
            # Rate limiting
            elapsed = now - last_capture
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue
            
            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self._error_count += 1
                    if self._error_count % 100 == 1:
                        logger.warning(f"Frame capture failed: {self.device} (error #{self._error_count})")
                    time.sleep(0.1)
                    continue
                
                # Resize if needed
                if self.resolution:
                    frame = cv2.resize(frame, self.resolution)
                
                # Encode to JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                _, jpeg_data = cv2.imencode(".jpg", frame, encode_params)
                
                # Create frame object
                height, width = frame.shape[:2]
                captured = CapturedFrame(
                    image_base64=base64.b64encode(jpeg_data.tobytes()).decode("ascii"),
                    media_type="image/jpeg",
                    timestamp=now,
                    width=width,
                    height=height,
                )
                
                # Update latest frame (thread-safe)
                with self._lock:
                    self._latest_frame = captured
                
                self._frame_count += 1
                last_capture = now
                
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                if self._error_count % 100 == 1:
                    logger.exception(f"Frame capture error: {e}")
                time.sleep(0.1)
        
        logger.debug("Capture loop ended")


class FrameCaptureManager:
    """
    Manages frame capture for multiple devices.
    
    Use this when you need to support both browser and Jetson cameras,
    or multiple Jetson cameras.
    """
    
    def __init__(self):
        self._captures: dict[str, FrameCapture] = {}
        self._lock = threading.Lock()
    
    def get_or_create(
        self,
        device: str,
        target_fps: float = 5.0,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Optional[FrameCapture]:
        """
        Get existing capture or create new one for device.
        
        Returns None if device is invalid or not a Jetson camera path.
        """
        # Only handle Jetson camera paths
        if not device or not device.startswith("/dev/video"):
            return None
        
        with self._lock:
            if device not in self._captures:
                capture = FrameCapture(
                    device=device,
                    target_fps=target_fps,
                    resolution=resolution,
                )
                if capture.start():
                    self._captures[device] = capture
                else:
                    return None
            
            return self._captures.get(device)
    
    def stop_all(self) -> None:
        """Stop all active captures."""
        with self._lock:
            for capture in self._captures.values():
                capture.stop()
            self._captures.clear()
    
    def stop(self, device: str) -> None:
        """Stop capture for a specific device."""
        with self._lock:
            if device in self._captures:
                self._captures[device].stop()
                del self._captures[device]


# Global manager instance (optional, for convenience)
_global_manager: Optional[FrameCaptureManager] = None


def get_frame_capture_manager() -> FrameCaptureManager:
    """Get the global FrameCaptureManager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = FrameCaptureManager()
    return _global_manager


