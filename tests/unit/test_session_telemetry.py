"""Shared telemetry records samples even without browser polling."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.webui import telemetry


@pytest.mark.asyncio
async def test_shared_sampler_persists_and_streams_samples(monkeypatch, tmp_path):
    stopped = asyncio.Event()
    session = SimpleNamespace(timeline=SimpleNamespace(start_time=time.time()), system_stats=[])
    messages = []

    class Socket:
        async def send_str(self, payload):
            messages.append(json.loads(payload))
            if len(messages) == 3:
                stopped.set()

    stats = telemetry.system_stats_module
    monkeypatch.setattr(stats, "find_jetson_sysfs_gpu_load_path", lambda: tmp_path / "load")
    monkeypatch.setattr(stats, "read_jetson_sysfs_gpu_percent", lambda _: 62.5)
    monkeypatch.setattr(stats, "read_cpu_percent_nonblocking", lambda: 25.0)
    monkeypatch.setattr(stats, "SYSTEM_STATS_INTERVAL_MS", 10)
    await asyncio.wait_for(telemetry.stream_system_stats(Socket(), session, stopped), timeout=2)

    assert len(session.system_stats) >= 3
    assert all(s["cpu"] == 25 and s["gpu"] == 62.5 for s in session.system_stats)
    assert [s["t"] for s in session.system_stats] == sorted(s["t"] for s in session.system_stats)
    assert all(m["type"] == "system_stats" for m in messages)
    count = len(session.system_stats)
    await asyncio.sleep(.03)
    assert len(session.system_stats) == count  # Worker stops with the session.
