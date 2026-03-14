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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
# Chunk size matching browser: 2048 samples = 4096 bytes @ 16-bit
CHUNK_SAMPLES = 2048
CHUNK_BYTES = CHUNK_SAMPLES * 2


MAX_CAPTURE_RETRIES = 8
RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each attempt up to a cap
RETRY_BACKOFF_MAX = 5.0

# ANSI escape codes for colored terminal output
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

# Sentinel dict placed in the queue to signal capture health events.
# Pipeline code should check `isinstance(item, dict)` before treating as PCM bytes.
CAPTURE_EVENT_TYPE = "__capture_event__"


def _make_capture_event(event: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a capture health event dict for the queue."""
    d: Dict[str, Any] = {"__type__": CAPTURE_EVENT_TYPE, "event": event, "ts": time.time()}
    d.update(kwargs)
    return d


def is_capture_event(item: Any) -> bool:
    """Return True if item is a capture health event (not PCM bytes)."""
    return isinstance(item, dict) and item.get("__type__") == CAPTURE_EVENT_TYPE


@dataclass
class CaptureHealth:
    """Accumulated capture health metrics (thread-safe reads after capture ends)."""
    device: str = ""
    total_drops: int = 0
    total_recoveries: int = 0
    outages: List[Dict[str, float]] = field(default_factory=list)
    gave_up: bool = False

    def to_dict(self) -> Dict[str, Any]:
        total_downtime = sum(o.get("duration_s", 0) for o in self.outages)
        return {
            "device": self.device,
            "total_drops": self.total_drops,
            "total_recoveries": self.total_recoveries,
            "total_downtime_s": round(total_downtime, 3),
            "outages": self.outages,
            "gave_up": self.gave_up,
        }


def _capture_alsa(
    device: str,
    out_queue: "queue.Queue[Optional[bytes]]",
    stop_event: threading.Event,
    proc_holder: Optional[list] = None,
    health: Optional[CaptureHealth] = None,
) -> None:
    """Capture from ALSA device via arecord; put PCM chunks in out_queue. Runs in thread.
    Uses plughw when device is hw:X,Y so ALSA can do sample-rate conversion (many USB mics only support 48kHz).
    If proc_holder is a list, the subprocess is stored as proc_holder[0] so the caller can terminate it to release the device quickly.

    Auto-restarts arecord up to MAX_CAPTURE_RETRIES times when the device
    disappears transiently (e.g. USB bus contention with a camera).
    Sends capture health events through out_queue so the pipeline can track outages.
    """
    if health is not None:
        health.device = device

    dev = (device or "default").strip()
    if dev.startswith("hw:") and not dev.startswith("plughw:"):
        dev = "plug" + dev
        logger.debug("ALSA using %s for rate conversion (requested 16kHz)", dev)
    cmd = ["arecord", "-D", dev, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(CHANNELS), "-t", "raw"]

    retries = 0
    ever_produced_chunk = False
    drop_time: Optional[float] = None

    while not stop_event.is_set():
        logger.info("ALSA capture starting: %s (device=%s)", " ".join(cmd), device)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=CHUNK_BYTES,
            )
            if proc_holder is not None:
                if proc_holder:
                    proc_holder.clear()
                proc_holder.append(proc)
        except FileNotFoundError:
            logger.warning("arecord not found; cannot capture from ALSA device %s", device)
            out_queue.put(None)
            return
        except Exception as e:
            logger.warning("Failed to start arecord for %s: %s", device, e)
            if retries >= MAX_CAPTURE_RETRIES:
                logger.error("ALSA capture giving up after %d retries for %s", retries, device)
                if health is not None:
                    health.gave_up = True
                out_queue.put(None)
                return
            retries += 1
            delay = min(RETRY_BACKOFF_BASE * (2 ** (retries - 1)), RETRY_BACKOFF_MAX)
            logger.info("ALSA capture retry %d/%d in %.1fs for %s", retries, MAX_CAPTURE_RETRIES, delay, device)
            stop_event.wait(delay)
            continue

        first_chunk_this_run = True
        died_unexpectedly = False
        try:
            while not stop_event.is_set() and proc.poll() is None:
                chunk = proc.stdout.read(CHUNK_BYTES)
                if not chunk:
                    try:
                        err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                        if err:
                            logger.error("%sALSA capture read empty (device %s). arecord stderr: %s%s", _RED, device, err, _RESET)
                        else:
                            logger.error("%sALSA capture read returned empty (device %s); check device/sample rate%s", _RED, device, _RESET)
                    except Exception:
                        logger.error("%sALSA capture read returned empty (device %s)%s", _RED, device, _RESET)
                    died_unexpectedly = True
                    break
                if first_chunk_this_run:
                    first_chunk_this_run = False
                    if not ever_produced_chunk:
                        logger.info("ALSA first PCM chunk received from %s (%d bytes); pipeline will get amplitude", device, len(chunk))
                    else:
                        recovery_dur = time.time() - drop_time if drop_time else 0
                        logger.warning(
                            "%s[capture_health] RECOVERED device %s after %.2fs outage (retry %d)%s",
                            _GREEN, device, recovery_dur, retries, _RESET,
                        )
                        if health is not None:
                            health.total_recoveries += 1
                            if health.outages:
                                health.outages[-1]["duration_s"] = round(recovery_dur, 3)
                        out_queue.put(_make_capture_event(
                            "recovered", device=device, outage_s=round(recovery_dur, 3), retry=retries,
                        ))
                        drop_time = None
                    retries = 0
                    ever_produced_chunk = True
                out_queue.put(chunk)

            # arecord exited on its own (proc.poll() != None) while we didn't ask it to stop
            if not died_unexpectedly and not stop_event.is_set() and proc.poll() is not None:
                rc = proc.returncode
                try:
                    err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                except Exception:
                    err = ""
                logger.error(
                    "%sarecord exited unexpectedly for %s (rc=%s): %s%s",
                    _RED, device, rc, err or "(no stderr)", _RESET,
                )
                died_unexpectedly = True
        except Exception as e:
            logger.error("%sALSA capture read error for %s: %s%s", _RED, device, e, _RESET)
            died_unexpectedly = True
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                pass
            if proc_holder is not None and proc_holder and proc_holder[0] is proc:
                try:
                    proc_holder.clear()
                except Exception:
                    pass

        if stop_event.is_set():
            break

        if died_unexpectedly and retries < MAX_CAPTURE_RETRIES:
            if drop_time is None:
                drop_time = time.time()
            retries += 1
            if health is not None:
                health.total_drops += 1
                health.outages.append({"drop_ts": round(drop_time, 3), "retry": retries, "duration_s": 0})
            delay = min(RETRY_BACKOFF_BASE * (2 ** (retries - 1)), RETRY_BACKOFF_MAX)
            logger.error(
                "%s[capture_health] DROPPED device %s; retry %d/%d in %.1fs%s",
                _RED, device, retries, MAX_CAPTURE_RETRIES, delay, _RESET,
            )
            out_queue.put(_make_capture_event(
                "dropped", device=device, retry=retries, max_retries=MAX_CAPTURE_RETRIES,
            ))
            stop_event.wait(delay)
            continue

        if died_unexpectedly:
            logger.error("%s[capture_health] GAVE UP on device %s after %d retries%s", _RED, device, retries, _RESET)
            if health is not None:
                health.gave_up = True
                if health.outages:
                    health.outages[-1]["duration_s"] = round(time.time() - drop_time, 3) if drop_time else 0
            out_queue.put(_make_capture_event("gave_up", device=device, retries=retries))
        elif first_chunk_this_run and not ever_produced_chunk:
            try:
                err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                if err:
                    logger.error("%sALSA capture ended with no chunks (device %s). arecord stderr: %s%s", _RED, device, err, _RESET)
                else:
                    logger.error("%sALSA capture ended without sending any chunks (device %s); check arecord -D %s%s", _RED, device, dev, _RESET)
            except Exception:
                logger.error("%sALSA capture ended without sending any chunks (device %s)%s", _RED, device, _RESET)
        break

    # Log summary if any drops occurred
    if health is not None and health.total_drops > 0:
        summary = health.to_dict()
        logger.warning(
            "%s[capture_health] SESSION SUMMARY for %s: drops=%d recoveries=%d downtime=%.2fs gave_up=%s%s",
            _YELLOW, device, summary["total_drops"], summary["total_recoveries"],
            summary["total_downtime_s"], summary["gave_up"], _RESET,
        )

    out_queue.put(None)


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
    health_out: Optional[list] = None,
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
        health_out: optional list; if provided, CaptureHealth is appended as health_out[0] for retrieval after thread exits

    Returns:
        The started thread, or None if capture could not be started.
    """
    if not device:
        return None
    health = CaptureHealth(device=device or "")
    if source == "alsa":
        target = _capture_alsa
        args = (device, out_queue, stop_event, proc_holder, health)
    elif source == "usb":
        target = _capture_pyaudio
        args = (device, out_queue, stop_event)
    else:
        return None
    if health_out is not None:
        health_out.append(health)
    t = threading.Thread(target=target, args=args, name="server-mic-capture", daemon=True)
    t.start()
    logger.info("Server mic capture started: %s device %s", source, device)
    return t
