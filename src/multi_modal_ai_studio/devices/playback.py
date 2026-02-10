"""
Server-side audio playback for voice pipeline.

When the user selects a Server USB speaker (ALSA), the server plays TTS audio
to that device instead of streaming to the browser.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def start_server_speaker_playback(
    device: str,
    sample_rate: int = 24000,
) -> Optional[subprocess.Popen]:
    """
    Start an aplay subprocess for streaming TTS audio to an ALSA device.

    Args:
        device: ALSA device (e.g. "hw:3,0" or "plughw:3,0")
        sample_rate: Audio sample rate (default 24000 for TTS)

    Returns:
        The subprocess.Popen object (with stdin pipe), or None if failed.
        Write 16-bit signed PCM data to proc.stdin.
        Call stop_server_speaker_playback when done.
    """
    if not device:
        return None
    
    dev = (device or "default").strip()
    # Use plughw for rate conversion if needed
    if dev.startswith("hw:") and not dev.startswith("plughw:"):
        dev = "plug" + dev
        logger.debug("ALSA using %s for rate conversion (sample_rate=%d)", dev, sample_rate)
    
    cmd = ["aplay", "-D", dev, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw"]
    logger.info("ALSA playback starting: %s (device=%s)", " ".join(cmd), device)
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return proc
    except FileNotFoundError:
        logger.warning("aplay not found; cannot play to ALSA device %s", device)
        return None
    except Exception as e:
        logger.warning("Failed to start aplay for %s: %s", device, e)
        return None


def stop_server_speaker_playback(proc: Optional[subprocess.Popen]) -> None:
    """
    Stop the aplay subprocess and release the device.

    Args:
        proc: The subprocess returned by start_server_speaker_playback.
    """
    if proc is None:
        return
    
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    
    logger.debug("ALSA playback stopped")

