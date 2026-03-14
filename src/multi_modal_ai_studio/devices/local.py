# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Local (server-side) USB webcam and audio device enumeration.

Used for devices attached to the server via USB (e.g. when the app runs on the same machine as the devices).
- Cameras: /dev/video* (V4L2), with optional v4l2-ctl for human-readable labels.
- Audio: PyAudio when available (ALSA on Linux); fallback to arecord/aplay -L when not.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _can_capture_video(device_path: str) -> bool:
    """Check if a V4L2 device can actually capture video frames.
    
    Some cameras create multiple /dev/video* nodes where only one is the actual
    capture device (others are metadata/control devices).
    
    Uses V4L2 ioctl to check capabilities without opening the device exclusively,
    so it doesn't conflict with active camera streams.
    """
    import fcntl
    import struct

    VIDIOC_QUERYCAP = 0x80685600
    V4L2_CAP_VIDEO_CAPTURE = 0x00000001

    try:
        fd = open(device_path, "rb")
        try:
            buf = bytearray(104)
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
            capabilities = struct.unpack_from("<I", buf, 84)[0]
            device_caps_field = struct.unpack_from("<I", buf, 88)[0]
            caps = device_caps_field if device_caps_field else capabilities
            return bool(caps & V4L2_CAP_VIDEO_CAPTURE)
        finally:
            fd.close()
    except (OSError, IOError) as e:
        logger.debug("V4L2 capability check for %s: %s", device_path, e)
        # If ioctl fails (e.g. device busy), fall back to checking sysfs
        try:
            dev_name = Path(device_path).name
            index_path = Path(f"/sys/class/video4linux/{dev_name}/index")
            if index_path.exists():
                return index_path.read_text().strip() == "0"
        except Exception:
            pass
        return True
    except Exception as e:
        logger.debug("Check capture capability for %s: %s", device_path, e)
        return True


def list_local_cameras() -> List[Dict[str, str]]:
    """List V4L2 video devices on this machine (e.g. /dev/video0).

    Only includes devices that can actually capture video frames (filters out
    metadata/control devices that some cameras create).

    Returns:
        List of {"id": "/dev/video0", "label": "UVC Camera (046d:0825) (Server USB)"}.
    """
    cameras = []
    if sys.platform != "linux":
        return cameras
    try:
        dev = Path("/dev")
        for p in sorted(dev.glob("video*")):
            if (
                p.name.startswith("video")
                and p.name[-1].isdigit()
                and "_" not in p.name
            ):
                device_id = str(p)
                
                # Skip devices that can't actually capture video
                if not _can_capture_video(device_id):
                    logger.debug("Skipping %s: not a capture device", device_id)
                    continue
                
                label = device_id
                try:
                    out = subprocess.run(
                        ["v4l2-ctl", "--device=" + device_id, "--info"],
                        capture_output=True,
                        text=True,
                        timeout=1,
                    )
                    if out.returncode == 0 and out.stdout:
                        for line in out.stdout.splitlines():
                            if "Card type" in line:
                                parts = line.split(":", 1)
                                if len(parts) == 2 and parts[1].strip():
                                    label = parts[1].strip() + " (Server USB)"
                                    break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                if label == device_id:
                    label = device_id + " (Server USB)"
                cameras.append({"id": device_id, "label": label})
    except Exception as e:
        logger.debug("List local cameras: %s", e)
    return cameras


def _list_audio_pyaudio() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Use PyAudio to list input and output devices. Returns (inputs, outputs)."""
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info is None:
                    continue
                name = (info.get("name") or "Device %d" % i).strip()
                max_in = info.get("maxInputChannels") or 0
                max_out = info.get("maxOutputChannels") or 0
                rate = int(info.get("defaultSampleRate") or 16000)
                dev_id = info.get("index")
                # Prefer a stable id: hostapi + index, or name if unique enough
                device_id = "pyaudio:%d" % dev_id
                entry = {"id": device_id, "label": name, "sample_rate": rate}
                if max_in > 0:
                    inputs.append(entry)
                if max_out > 0:
                    outputs.append(entry)
        finally:
            pa.terminate()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyAudio list devices: %s", e)
    return inputs, outputs


def _parse_alsa_cards() -> Dict[int, str]:
    """Parse /proc/asound/cards to get card index -> human-readable name (e.g. 'EMEET OfficeCore M0 Plus')."""
    names: Dict[int, str] = {}
    if sys.platform != "linux":
        return names
    try:
        path = Path("/proc/asound/cards")
        if not path.exists():
            return names
        text = path.read_text()
        # Format: " 0 [ID]: driver - Name" e.g. " 1 [M0Plus]: USB-Audio - EMEET OfficeCore M0 Plus"
        card_line_re = re.compile(r"^\s*(\d+)\s*\[[^\]]+\]:\s*\S+\s+-\s+(.+)$")
        for line in text.splitlines():
            m = card_line_re.match(line.strip())
            if m:
                card_num = int(m.group(1))
                name = m.group(2).strip()
                if name:
                    names[card_num] = name
    except Exception as e:
        logger.debug("Parse /proc/asound/cards: %s", e)
    return names


def _list_audio_alsa_devices() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Enumerate ALSA capture and playback devices using arecord -l / aplay -l and /proc/asound/cards for names."""
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    if sys.platform != "linux":
        return inputs, outputs
    card_names = _parse_alsa_cards()
    # Add default so there's always at least one option
    seen_in: set = set()
    seen_out: set = set()

    for cmd, dest, seen in [
        ("arecord", inputs, seen_in),
        ("aplay", outputs, seen_out),
    ]:
        try:
            out = subprocess.run(
                [cmd, "-l"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode != 0 or not out.stdout:
                continue
            # Parse "card N: ShortName [FullName], device M: ..." — one entry per card (device 0 only)
            # to avoid flooding the dropdown with many subdevices (e.g. some cards expose 0..31).
            card_re = re.compile(r"card (\d+):\s*\S+\s+\[([^\]]+)\],\s*device (\d+):")
            cards_added: set = set()
            for line in out.stdout.splitlines():
                m = card_re.search(line)
                if m:
                    card_num = int(m.group(1))
                    full_name = m.group(2).strip()
                    dev_num = int(m.group(3))
                    if card_num in cards_added:
                        continue
                    cards_added.add(card_num)
                    hw_id = "hw:%d,%d" % (card_num, dev_num)
                    device_id = "alsa:" + hw_id
                    if device_id in seen:
                        continue
                    seen.add(device_id)
                    label = card_names.get(card_num) or full_name
                    if not label:
                        label = "Card %d" % card_num
                    dest.append({"id": device_id, "label": label})
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.debug("%s -l parse: %s", cmd, e)

    # Ensure at least default is present
    if not inputs:
        inputs.append({"id": "alsa:default", "label": "Default (Server USB)"})
    if not outputs:
        outputs.append({"id": "alsa:default", "label": "Default (Server USB)"})
    return inputs, outputs


def _list_audio_alsa_fallback() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """On Linux when PyAudio is unavailable, enumerate ALSA devices with real names (from /proc/asound/cards and arecord/aplay -l)."""
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    if sys.platform != "linux":
        return inputs, outputs
    inputs, outputs = _list_audio_alsa_devices()
    # Ensure (Server USB) suffix for UI
    for d in inputs:
        if d.get("label") and "(Server USB)" not in d["label"]:
            d["label"] = d["label"] + " (Server USB)"
    for d in outputs:
        if d.get("label") and "(Server USB)" not in d["label"]:
            d["label"] = d["label"] + " (Server USB)"
    return inputs, outputs


def list_local_audio_inputs() -> List[Dict[str, Any]]:
    """List local audio input devices (microphones).

    On Linux: use ALSA enumeration (arecord -l + /proc/asound/cards) so USB devices like
    EMEET OfficeCore M0 Plus show with real names. Otherwise use PyAudio if available.
    Returns list of {"id": "alsa:hw:1,0" or "pyaudio:0", "label": "...", ...}.
    """
    if sys.platform == "linux":
        inputs, _ = _list_audio_alsa_fallback()
    else:
        inputs, _ = _list_audio_pyaudio()
        if not inputs:
            inputs, _ = _list_audio_alsa_fallback()
    for d in inputs:
        if "label" not in d or not d["label"]:
            d["label"] = d.get("id", "Unknown") + " (Server USB)"
        elif "(Server USB)" not in d["label"] and ("alsa:" in str(d.get("id", "")) or "pyaudio:" in str(d.get("id", ""))):
            d["label"] = d["label"] + " (Server USB)"
    return inputs


def list_local_audio_outputs() -> List[Dict[str, Any]]:
    """List local audio output devices (speakers).

    On Linux: use ALSA enumeration (aplay -l + /proc/asound/cards) for real device names.
    Otherwise use PyAudio if available.
    """
    if sys.platform == "linux":
        _, outputs = _list_audio_alsa_fallback()
    else:
        _, outputs = _list_audio_pyaudio()
        if not outputs:
            _, outputs = _list_audio_alsa_fallback()
    for d in outputs:
        if "label" not in d or not d["label"]:
            d["label"] = d.get("id", "Unknown") + " (Server USB)"
        elif "(Server USB)" not in d["label"] and ("alsa:" in str(d.get("id", "")) or "pyaudio:" in str(d.get("id", ""))):
            d["label"] = d["label"] + " (Server USB)"
    return outputs
