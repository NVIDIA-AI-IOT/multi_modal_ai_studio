# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CPU/GPU stats for timeline system lane.

The voice pipeline consumes a persistent nvidia-smi stream at 20 Hz during a
live session, sends each sample over the voice WebSocket (type=system_stats),
and appends to session.system_stats for save. No HTTP polling; client renders
from WS messages.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

# Cache updated by voice pipeline at 20 Hz; API returns this when fresh to avoid duplicate gather.
_cache: Dict[str, Any] = {}
_cache_time: float = 0.0
_nvidia_smi_gpu_supported: Optional[bool] = None
GPU_SAMPLE_INTERVAL_MS = 50


def nvidia_smi_loop_command(interval_ms: int = GPU_SAMPLE_INTERVAL_MS) -> list[str]:
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
    """Return CPU utilization without delaying the 20 Hz GPU sampler."""
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


def _read_jetson_sysfs_gpu_percent() -> Optional[float]:
    """Read the nvgpu devfreq load counter (0-1000 permille) used on Jetson."""
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
    """Return cached stats if fresh; otherwise gather once and cache. API uses this to avoid 10 Hz + 10 Hz = 20 Hz."""
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
    GPU: first GPU utilization from nvidia-smi (percentage 0-100). Jetson
    drivers can return N/A for that query, so fall back to the nvgpu devfreq
    load counter exposed as 0-1000 permille in sysfs.
    """
    cpu_percent = None
    if psutil is not None:
        try:
            cpu_percent = round(psutil.cpu_percent(interval=0.05), 1)
        except Exception:
            pass
    gpu_percent = _read_nvidia_smi_gpu_percent()
    if gpu_percent is None:
        gpu_percent = _read_jetson_sysfs_gpu_percent()
    return {"cpu_percent": cpu_percent, "gpu_percent": gpu_percent}
