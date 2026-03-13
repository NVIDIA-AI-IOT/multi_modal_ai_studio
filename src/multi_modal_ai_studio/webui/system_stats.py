# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CPU/GPU stats for timeline system lane.

The voice pipeline calls gather_system_stats() at 10 Hz during a live session,
sends each sample over the voice WebSocket (type=system_stats), and appends to
session.system_stats for save. No HTTP polling; client renders from WS messages.
"""

import subprocess
import time
from typing import Any, Dict

try:
    import psutil
except ImportError:
    psutil = None

# Cache updated by voice pipeline at 10 Hz; API returns this when fresh to avoid duplicate gather.
_cache: Dict[str, Any] = {}
_cache_time: float = 0.0


def set_system_stats_cache(stats: Dict[str, Any]) -> None:
    """Update the shared cache (call from voice pipeline after each gather at 10 Hz)."""
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
    GPU: first GPU utilization from nvidia-smi (percentage 0-100).
    """
    cpu_percent = None
    if psutil is not None:
        try:
            cpu_percent = round(psutil.cpu_percent(interval=0.05), 1)
        except Exception:
            pass
    gpu_percent = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            val = out.stdout.strip().split("\n")[0].strip().replace("%", "").strip()
            try:
                gpu_percent = float(val)
            except ValueError:
                gpu_percent = None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return {"cpu_percent": cpu_percent, "gpu_percent": gpu_percent}
