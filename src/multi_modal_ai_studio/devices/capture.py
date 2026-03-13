# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Server-side audio capture for voice pipeline.

When the user selects a Server USB microphone (ALSA or PyAudio), the server
captures from that device and feeds PCM to ASR instead of using browser WebSocket audio.
"""

import logging
import queue
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
# Chunk size matching browser: 2048 samples = 4096 bytes @ 16-bit
CHUNK_SAMPLES = 2048
CHUNK_BYTES = CHUNK_SAMPLES * 2


def _capture_alsa(
    device: str,
    out_queue: "queue.Queue[Optional[bytes]]",
    stop_event: threading.Event,
    proc_holder: Optional[list] = None,
) -> None:
    """Capture from ALSA device via arecord; put PCM chunks in out_queue. Runs in thread.
    Uses plughw when device is hw:X,Y so ALSA can do sample-rate conversion (many USB mics only support 48kHz).
    If proc_holder is a list, the subprocess is stored as proc_holder[0] so the caller can terminate it to release the device quickly.
    """
    dev = (device or "default").strip()
    if dev.startswith("hw:") and not dev.startswith("plughw:"):
        dev = "plug" + dev
        logger.debug("ALSA using %s for rate conversion (requested 16kHz)", dev)
    cmd = ["arecord", "-D", dev, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(CHANNELS), "-t", "raw"]
    logger.info("ALSA capture starting: %s (device=%s)", " ".join(cmd), device)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=CHUNK_BYTES,
        )
        if proc_holder is not None:
            proc_holder.append(proc)
    except FileNotFoundError:
        logger.warning("arecord not found; cannot capture from ALSA device %s", device)
        out_queue.put(None)
        return
    except Exception as e:
        logger.warning("Failed to start arecord for %s: %s", device, e)
        out_queue.put(None)
        return
    first_chunk = True
    try:
        while not stop_event.is_set() and proc.poll() is None:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                try:
                    err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                    if err:
                        logger.warning("ALSA capture read empty (device %s). arecord stderr: %s", device, err)
                    else:
                        logger.warning("ALSA capture read returned empty (device %s); check device/sample rate", device)
                except Exception:
                    logger.warning("ALSA capture read returned empty (device %s)", device)
                break
            if first_chunk:
                first_chunk = False
                logger.info("ALSA first PCM chunk received from %s (%d bytes); pipeline will get amplitude", device, len(chunk))
            out_queue.put(chunk)
    except Exception as e:
        logger.warning("ALSA capture read error for %s: %s", device, e)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass
        out_queue.put(None)
        if proc_holder is not None and proc_holder and proc_holder[0] is proc:
            try:
                proc_holder.clear()
            except Exception:
                pass
        if first_chunk:
            try:
                err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                if err:
                    logger.warning("ALSA capture ended with no chunks (device %s). arecord stderr: %s", device, err)
                else:
                    logger.warning("ALSA capture ended without sending any chunks (device %s); check arecord -D %s", device, dev)
            except Exception:
                logger.warning("ALSA capture ended without sending any chunks (device %s)", device)


def _capture_pyaudio(
    device_index_str: str,
    out_queue: "queue.Queue[Optional[bytes]]",
    stop_event: threading.Event,
) -> None:
    """Capture from PyAudio device by index; put PCM chunks in out_queue. Runs in thread."""
    try:
        import pyaudio
    except ImportError:
        logger.warning("PyAudio not installed; cannot capture from USB device %s", device_index_str)
        out_queue.put(None)
        return
    try:
        device_index = int(device_index_str)
    except ValueError:
        logger.warning("Invalid pyaudio device index: %s", device_index_str)
        out_queue.put(None)
        return
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
        logger.warning("Failed to open PyAudio device %s: %s", device_index, e)
        pa.terminate()
        out_queue.put(None)
        return
    first_chunk = True
    try:
        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                if not data:
                    break
                if first_chunk:
                    first_chunk = False
                    logger.info("PyAudio first PCM chunk from device %s (%d bytes); pipeline will get amplitude", device_index_str, len(data))
                out_queue.put(data)
            except Exception as e:
                if not stop_event.is_set():
                    logger.warning("PyAudio read error for device %s: %s", device_index_str, e)
                break
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
        out_queue.put(None)
        if first_chunk:
            logger.warning("PyAudio capture ended without sending any chunks (device %s)", device_index_str)


def start_server_mic_capture(
    source: str,
    device: Optional[str],
    out_queue: "queue.Queue[Optional[bytes]]",
    stop_event: threading.Event,
    proc_holder: Optional[list] = None,
) -> Optional[threading.Thread]:
    """
    Start a thread that captures from the given server audio input device and puts
    PCM chunks (bytes) into out_queue. When done or on error, puts None.

    Args:
        source: "alsa" or "usb" (pyaudio)
        device: ALSA device (e.g. "hw:3,0") or PyAudio device index string (e.g. "2")
        out_queue: queue to put chunks into; None is sentinel when capture ends
        stop_event: when set, capture thread should exit
        proc_holder: optional list; for ALSA, the arecord subprocess is appended so the caller can terminate it to release the device quickly

    Returns:
        The started thread, or None if capture could not be started.
    """
    if not device:
        return None
    if source == "alsa":
        target = _capture_alsa
        args = (device, out_queue, stop_event, proc_holder)
    elif source == "usb":
        target = _capture_pyaudio
        args = (device, out_queue, stop_event)
    else:
        return None
    t = threading.Thread(target=target, args=args, name="server-mic-capture", daemon=True)
    t.start()
    logger.info("Server mic capture started: %s device %s", source, device)
    return t
