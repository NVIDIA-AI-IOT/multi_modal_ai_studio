# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Server-side audio playback for TTS (voice pipeline).

When the user selects a Server USB speaker (ALSA), the server plays TTS PCM
to that device via aplay in addition to sending audio to the browser.
"""

import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

CHANNELS = 1
PLAYBACK_RETRIES = 3
PLAYBACK_RETRY_DELAY = 0.3  # seconds between retries

_RED = "\033[91m"
_RESET = "\033[0m"


def start_server_speaker_playback(
    device: str,
    sample_rate: int,
    proc_holder: Optional[list] = None,
) -> Optional[subprocess.Popen]:
    """Start aplay for TTS playback to an ALSA device.

    Caller must write 16-bit LE mono PCM to the returned process's stdin,
    then close stdin when done so aplay exits. Use plughw when device is
    hw:X,Y so ALSA can do sample-rate conversion if needed.

    Retries up to PLAYBACK_RETRIES times for transient device errors
    (e.g. USB audio device momentarily unavailable).

    Args:
        device: ALSA device (e.g. hw:2,0).
        sample_rate: PCM sample rate in Hz (e.g. 24000 from TTS).
        proc_holder: If provided, the Popen is appended so caller can terminate
            it to stop playback (e.g. on disconnect).

    Returns:
        Popen with stdin=PIPE, or None if aplay could not be started.
    """
    if not device:
        return None

    dev = (device or "default").strip()
    if dev.startswith("hw:") and not dev.startswith("plughw:"):
        dev = "plug" + dev
        logger.debug("ALSA playback using %s for rate conversion", dev)
    cmd = [
        "aplay",
        "-D", dev,
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", str(CHANNELS),
        "-t", "raw",
    ]

    last_err = ""
    for attempt in range(1, PLAYBACK_RETRIES + 1):
        logger.info("ALSA playback starting: %s (device=%s, rate=%s)", " ".join(cmd), device, sample_rate)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.15)
            if proc.poll() is not None:
                last_err = (proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else "") or "(no stderr)"
                if attempt < PLAYBACK_RETRIES:
                    logger.error(
                        "%saplay exited immediately for %s (attempt %d/%d): %s — retrying in %.1fs%s",
                        _RED, device, attempt, PLAYBACK_RETRIES, last_err, PLAYBACK_RETRY_DELAY, _RESET,
                    )
                    time.sleep(PLAYBACK_RETRY_DELAY)
                    continue
                logger.error("%saplay exited immediately for %s after %d attempts: %s%s", _RED, device, attempt, last_err, _RESET)
                return None
            if proc_holder is not None:
                proc_holder.append(proc)
            return proc
        except FileNotFoundError:
            logger.error("%saplay not found; cannot play to ALSA device %s%s", _RED, device, _RESET)
            return None
        except Exception as e:
            last_err = str(e)
            if attempt < PLAYBACK_RETRIES:
                logger.error(
                    "%sFailed to start aplay for %s (attempt %d/%d): %s — retrying in %.1fs%s",
                    _RED, device, attempt, PLAYBACK_RETRIES, e, PLAYBACK_RETRY_DELAY, _RESET,
                )
                time.sleep(PLAYBACK_RETRY_DELAY)
                continue
            logger.error("%sFailed to start aplay for %s after %d attempts: %s%s", _RED, device, attempt, e, _RESET)
            return None
    return None


def stop_server_speaker_playback(proc: Optional[subprocess.Popen]) -> None:
    """Close stdin and wait for aplay to finish, or terminate if it doesn't exit."""
    if proc is None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        logger.debug("Stop server speaker playback: %s", e)
        try:
            proc.terminate()
        except Exception:
            pass
