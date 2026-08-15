# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CPU/GPU stats for timeline system lane.

The voice pipeline samples at 10 Hz during a live session, sends each sample
over the voice WebSocket (type=system_stats), and appends it to the session.
Orin reads the nvgpu sysfs load counter directly; Thor/desktop uses a
persistent nvidia-smi stream. A one-shot nvidia-smi process is only the final,
lower-rate fallback.
"""

import os
import platform
import socket
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

# Cache updated by the 10 Hz voice-pipeline sampler.
_cache: Dict[str, Any] = {}
_cache_time: float = 0.0
_nvidia_smi_gpu_supported: Optional[bool] = None
SYSTEM_STATS_INTERVAL_MS = 100
GPU_SUBPROCESS_FALLBACK_INTERVAL_MS = 250
STREAM_SAMPLE_MIN_INTERVAL_RATIO = 0.8


def _read_first_text(paths: list[Path]) -> Optional[str]:
    """Read the first non-empty system identity string from *paths*."""
    for path in paths:
        try:
            value = path.read_bytes().replace(b"\x00", b"").decode(
                "utf-8", errors="replace"
            ).strip()
        except OSError:
            continue
        if value:
            return value
    return None


def _network_ipv4_addresses() -> list[str]:
    """Return useful host IPv4 addresses without loopback/container bridges."""
    addresses: list[str] = []
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        candidates = result.stdout.split() if result.returncode == 0 else []
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        candidates = []

    if not candidates:
        try:
            candidates = [
                info[4][0]
                for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
            ]
        except OSError:
            candidates = []

    for address in candidates:
        if address.startswith("127."):
            continue
        parts = address.split(".")
        if len(parts) != 4:
            continue
        if address.startswith("172.") and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
            continue
        if address not in addresses:
            addresses.append(address)
    return addresses[:8]


def _gpu_identity() -> Dict[str, Any]:
    """Read a lightweight GPU identity without importing CUDA frameworks."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    gpu: Dict[str, Any] = {}
    if fields and fields[0] and fields[0] != "[N/A]":
        gpu["name"] = fields[0]
    if len(fields) > 1 and fields[1] and fields[1] != "[N/A]":
        gpu["compute_capability"] = fields[1]
    return gpu


def _l4t_release() -> Optional[str]:
    """Return a compact L4T release such as R39.2.1 on Jetson."""
    value = _read_first_text([Path("/etc/nv_tegra_release")])
    if not value:
        return None
    first_line = value.splitlines()[0]
    try:
        release = first_line.split("# R", 1)[1].split(" ", 1)[0]
        revision = first_line.split("REVISION:", 1)[1].split(",", 1)[0].strip()
    except (IndexError, ValueError):
        return None
    return f"R{release}.{revision}"


@lru_cache(maxsize=1)
def gather_system_info() -> Dict[str, Any]:
    """Capture stable host/hardware metadata for a reproducible session record."""
    memory_total = None
    if psutil is not None:
        try:
            memory_total = int(psutil.virtual_memory().total)
        except Exception:
            pass
    if memory_total is None:
        try:
            memory_total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            pass

    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "ip_addresses": _network_ipv4_addresses(),
        "device_model": _read_first_text(
            [
                Path("/proc/device-tree/model"),
                Path("/sys/firmware/devicetree/base/model"),
                Path("/sys/class/dmi/id/product_name"),
            ]
        ),
        "cpu": {
            "architecture": platform.machine() or None,
            "logical_cores": os.cpu_count(),
        },
        "gpu": _gpu_identity(),
        "memory_total_bytes": memory_total,
        "os": {
            "name": platform.system() or None,
            "kernel": platform.release() or None,
            "l4t": _l4t_release(),
        },
    }
    return info


def stream_sample_is_due(
    now: float,
    last_emit: float,
    interval: float,
) -> bool:
    """Reject buffered telemetry lines that arrive much faster than requested."""
    return (
        not last_emit
        or now - last_emit >= interval * STREAM_SAMPLE_MIN_INTERVAL_RATIO
    )


def next_periodic_deadline(deadline: float, now: float, interval: float) -> float:
    """Advance a monotonic deadline without drift or catch-up bursts."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    next_deadline = deadline + interval
    if next_deadline <= now:
        missed = int((now - next_deadline) // interval) + 1
        next_deadline += missed * interval
    return next_deadline


def read_nvidia_smi_gpu_percent_after_stream_failure() -> Optional[float]:
    """Retry one one-shot nvidia-smi query after its stream has failed."""
    global _nvidia_smi_gpu_supported
    _nvidia_smi_gpu_supported = None
    return _read_nvidia_smi_gpu_percent()


def nvidia_smi_loop_command(interval_ms: int = SYSTEM_STATS_INTERVAL_MS) -> list[str]:
    """Build the persistent GPU-utilization stream command."""
    interval = max(20, int(interval_ms))
    return [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        f"--loop-ms={interval}",
    ]


def parse_nvidia_smi_gpu_percent(value: Any) -> Optional[float]:
    """Parse one nvidia-smi utilization line."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value or "").strip()
    raw = text.splitlines()[0].replace("%", "").strip() if text else ""
    try:
        percent = float(raw)
    except ValueError:
        return None
    return percent if 0.0 <= percent <= 100.0 else None


def read_cpu_percent_nonblocking() -> Optional[float]:
    """Return CPU utilization without delaying the 10 Hz sampler."""
    if psutil is None:
        return None
    try:
        return round(psutil.cpu_percent(interval=None), 1)
    except Exception:
        return None


def _read_nvidia_smi_gpu_percent() -> Optional[float]:
    """Return utilization from nvidia-smi, or None when the driver reports N/A."""
    global _nvidia_smi_gpu_supported
    if _nvidia_smi_gpu_supported is False:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        _nvidia_smi_gpu_supported = False
        return None
    if out.returncode != 0 or not out.stdout.strip():
        _nvidia_smi_gpu_supported = False
        return None
    percent = parse_nvidia_smi_gpu_percent(out.stdout)
    if percent is None:
        # Jetson's nvidia-smi can exit successfully while returning "[N/A]".
        _nvidia_smi_gpu_supported = False
        return None
    _nvidia_smi_gpu_supported = True
    return percent


def find_jetson_sysfs_gpu_load_path() -> Optional[Path]:
    """Return the nvgpu load counter used by Orin, if this host exposes it."""
    candidates = sorted(Path("/sys/class/devfreq").glob("*gpu*/device/load"))
    candidates += sorted(Path("/sys/class/devfreq").glob("*gpu*/load"))
    for path in candidates:
        if read_jetson_sysfs_gpu_percent(path) is not None:
            return path
    return None


def read_jetson_sysfs_gpu_percent(path: Optional[Path] = None) -> Optional[float]:
    """Read the nvgpu devfreq load counter (0-1000 permille) used on Jetson."""
    candidates = [path] if path is not None else []
    if not candidates:
        candidates = sorted(Path("/sys/class/devfreq").glob("*gpu*/device/load"))
        candidates += sorted(Path("/sys/class/devfreq").glob("*gpu*/load"))
    for path in candidates:
        try:
            raw = float(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if raw < 0:
            continue
        return round(min(raw / 10.0, 100.0), 1)
    return None


def set_system_stats_cache(stats: Dict[str, Any]) -> None:
    """Update the shared cache after each live telemetry sample."""
    global _cache, _cache_time
    _cache = dict(stats)
    _cache_time = time.time()


def get_system_stats_cached(max_age_sec: float = 0.2) -> Dict[str, Any]:
    """Return cached stats if fresh; otherwise gather once and cache."""
    global _cache, _cache_time
    now = time.time()
    if _cache and (now - _cache_time) <= max_age_sec:
        return _cache
    stats = gather_system_stats()
    _cache = stats
    _cache_time = now
    return stats


def gather_system_stats() -> Dict[str, Any]:
    """Gather CPU and GPU utilization. Returns cpu_percent and gpu_percent (0-100 or None).

    CPU: system-wide average over a short interval (psutil.cpu_percent(interval=0.05))
    so the first call returns a real value and readings are smoothed.
    GPU: prefer the cheap nvgpu sysfs counter on Orin. Otherwise query
    nvidia-smi once (desktop/Thor).
    """
    cpu_percent = None
    if psutil is not None:
        try:
            cpu_percent = round(psutil.cpu_percent(interval=0.05), 1)
        except Exception:
            pass
    gpu_percent = read_jetson_sysfs_gpu_percent()
    if gpu_percent is None:
        gpu_percent = _read_nvidia_smi_gpu_percent()
    return {"cpu_percent": cpu_percent, "gpu_percent": gpu_percent}
