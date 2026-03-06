"""
Frame Broker - Shared frame buffer for server-side camera.

This singleton stores frames captured by WebRTC in a ring buffer,
allowing VLM to read frames without opening the camera again.

Inspired by live-vlm-webui's video_processor pattern.
See: https://github.com/NVIDIA-AI-IOT/live-vlm-webui
"""

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TimestampedFrame:
    """A frame with its capture timestamp."""
    timestamp: float  # time.time() when captured
    jpeg_bytes: bytes  # JPEG-encoded frame


class FrameBroker:
    """
    Singleton ring buffer for server-side camera frames.
    
    WebRTC camera track stores frames here; VLM reads from here.
    This avoids the camera lock issue where two processes try to
    access /dev/video0 simultaneously.
    """
    
    _instance: Optional["FrameBroker"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Ring buffer - stores last N seconds of frames
        self._buffer: deque[TimestampedFrame] = deque(maxlen=100)  # ~10s at 10fps
        self._buffer_lock = threading.Lock()
        
        # Configuration
        self._max_age_seconds = 10.0  # Discard frames older than this
        self._jpeg_quality = 70  # Default JPEG quality
        
        # Stats
        self._frames_stored = 0
        self._frames_retrieved = 0
        self._last_store_log = 0  # Throttle logging
        
        logger.info("[FrameBroker] Initialized (max_frames=100, max_age=%.1fs)", self._max_age_seconds)
    
    def store_frame(self, frame_bgr, jpeg_quality: int = 70) -> None:
        """
        Store a BGR frame (from OpenCV) in the ring buffer.
        
        Args:
            frame_bgr: OpenCV BGR frame (numpy array)
            jpeg_quality: JPEG compression quality (0-100)
        """
        try:
            import cv2
            
            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            success, jpeg_bytes = cv2.imencode('.jpg', frame_bgr, encode_params)
            
            if not success:
                logger.warning("[FrameBroker] JPEG encode failed")
                return
            
            # Store with timestamp
            frame = TimestampedFrame(
                timestamp=time.time(),
                jpeg_bytes=jpeg_bytes.tobytes()
            )
            
            with self._buffer_lock:
                self._buffer.append(frame)
                self._frames_stored += 1
                
                # Prune old frames
                cutoff = time.time() - self._max_age_seconds
                while self._buffer and self._buffer[0].timestamp < cutoff:
                    self._buffer.popleft()
                
                # Log every 30 frames (~1s at 30fps)
                if self._frames_stored % 30 == 0:
                    logger.debug("[FrameBroker] Stored frame #%d, buffer_size=%d", 
                                self._frames_stored, len(self._buffer))
                    
        except Exception as e:
            logger.warning("[FrameBroker] store_frame error: %s", e)
    
    def get_frames(
        self,
        t_start: float,
        t_end: float,
        n_frames: int,
        max_width: int = 640
    ) -> List[str]:
        """
        Get n_frames evenly spaced between t_start and t_end.
        
        Args:
            t_start: Start timestamp (time.time())
            t_end: End timestamp (time.time())
            n_frames: Number of frames to return
            max_width: Maximum frame width (for resizing)
            
        Returns:
            List of base64 data URLs (data:image/jpeg;base64,...)
        """
        if n_frames <= 0:
            return []
            
        with self._buffer_lock:
            if not self._buffer:
                logger.warning("[FrameBroker] Buffer empty, no frames available")
                return []
            
            # Find frames in time range
            frames_in_range = [
                f for f in self._buffer
                if t_start <= f.timestamp <= t_end
            ]
            
            if not frames_in_range:
                # Fall back to most recent frames
                logger.info("[FrameBroker] No frames in range [%.2f, %.2f], using recent frames",
                           t_start, t_end)
                frames_in_range = list(self._buffer)[-n_frames:]
            
            # Evenly sample n_frames
            if len(frames_in_range) <= n_frames:
                selected = frames_in_range
            else:
                step = len(frames_in_range) / n_frames
                indices = [int(i * step) for i in range(n_frames)]
                selected = [frames_in_range[i] for i in indices]
            
            self._frames_retrieved += len(selected)
        
        # Convert to base64 data URLs
        result = []
        for frame in selected:
            try:
                b64 = base64.b64encode(frame.jpeg_bytes).decode('ascii')
                result.append(f"data:image/jpeg;base64,{b64}")
            except Exception as e:
                logger.warning("[FrameBroker] Base64 encode error: %s", e)
        
        logger.info("[FrameBroker] Retrieved %d frames (requested %d, range %.2f-%.2f)",
                   len(result), n_frames, t_start, t_end)
        return result
    
    def get_latest_frame(self) -> Optional[str]:
        """Get the most recent frame as a data URL."""
        with self._buffer_lock:
            if not self._buffer:
                return None
            frame = self._buffer[-1]
        
        try:
            b64 = base64.b64encode(frame.jpeg_bytes).decode('ascii')
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.warning("[FrameBroker] Base64 encode error: %s", e)
            return None
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._buffer_lock:
            return {
                "buffer_size": len(self._buffer),
                "frames_stored": self._frames_stored,
                "frames_retrieved": self._frames_retrieved,
                "oldest_frame_age": time.time() - self._buffer[0].timestamp if self._buffer else 0,
            }
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._buffer_lock:
            self._buffer.clear()
        logger.info("[FrameBroker] Buffer cleared")


# Global instance accessor
def get_frame_broker() -> FrameBroker:
    """Get the global FrameBroker instance."""
    return FrameBroker()

