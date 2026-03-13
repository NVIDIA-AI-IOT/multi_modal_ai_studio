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


def start_server_speaker_playback(
    device: str,
    sample_rate: int,
    proc_holder: Optional[list] = None,
) -> Optional[subprocess.Popen]:
    """Start aplay for TTS playback to an ALSA device.

    Caller must write 16-bit LE mono PCM to the returned process's stdin,
    then close stdin when done so aplay exits. Use plughw when device is
    hw:X,Y so ALSA can do sample-rate conversion if needed.

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
    logger.info("ALSA playback starting: %s (device=%s, rate=%s)", " ".join(cmd), device, sample_rate)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # If same device is used for mic (arecord) and speaker, aplay may exit with "Device or resource busy"
        time.sleep(0.15)
        if proc.poll() is not None:
            err = (proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else "") or "(no stderr)"
            logger.warning("aplay exited immediately for %s: %s", device, err)
            return None
        if proc_holder is not None:
            proc_holder.append(proc)
        return proc
    except FileNotFoundError:
        logger.warning("aplay not found; cannot play to ALSA device %s", device)
        return None
    except Exception as e:
        logger.warning("Failed to start aplay for %s: %s", device, e)
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
