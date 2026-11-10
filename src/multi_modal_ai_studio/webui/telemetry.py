# SPDX-License-Identifier: Apache-2.0
"""Shared CPU/GPU telemetry lifecycle for cascade and full-duplex sessions."""
import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from multi_modal_ai_studio.webui import system_stats as system_stats_module

logger = logging.getLogger(__name__)

async def stream_system_stats(ws, session, stopped) -> None:
    """Send CPU/GPU samples from the cheapest supported 10 Hz source."""
    loop = asyncio.get_event_loop()
    interval = system_stats_module.SYSTEM_STATS_INTERVAL_MS / 1000.0
    fallback_interval = (
        system_stats_module.GPU_SUBPROCESS_FALLBACK_INTERVAL_MS / 1000.0
    )

    async def emit_sample(
        cpu: Optional[float],
        gpu: Optional[float],
        *,
        timestamp: Optional[float] = None,
        record: bool = True,
    ) -> None:
        t = (
            time.time() - session.timeline.start_time
            if timestamp is None
            else timestamp
        )
        stats = {"cpu_percent": cpu, "gpu_percent": gpu}
        system_stats_module.set_system_stats_cache(stats)
        if record:
            session.system_stats.append({"t": t, "cpu": cpu, "gpu": gpu})
        try:
            await ws.send_str(json.dumps({
                "type": "system_stats",
                "timestamp": t,
                "cpu_percent": cpu,
                "gpu_percent": gpu,
            }))
        except Exception as e:
            logger.debug("Send system_stats failed: %s", e)

    async def sleep_until(deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - loop.time()))

    async def run_sysfs_sampler(path: Path) -> bool:
        """Sample Orin in a thread so inference cannot starve telemetry."""
        samples: queue.Queue = queue.Queue(maxsize=3600)
        sampler_stop = threading.Event()
        unreadable = threading.Event()

        def collect() -> None:
            deadline = time.monotonic()
            try:
                while not sampler_stop.is_set() and not stopped.is_set():
                    gpu = system_stats_module.read_jetson_sysfs_gpu_percent(
                        path
                    )
                    if gpu is None:
                        unreadable.set()
                        logger.warning(
                            "Jetson GPU load counter became unreadable: %s",
                            path,
                        )
                        break
                    cpu = system_stats_module.read_cpu_percent_nonblocking()
                    timestamp = time.time() - session.timeline.start_time
                    system_stats_module.set_system_stats_cache({
                        "cpu_percent": cpu,
                        "gpu_percent": gpu,
                    })
                    # Persist from the sampler thread. WebSocket delivery
                    # may lag under unified-memory pressure, but replay data
                    # must retain the real 10 Hz measurement grid.
                    session.system_stats.append({
                        "t": timestamp,
                        "cpu": cpu,
                        "gpu": gpu,
                    })
                    try:
                        samples.put_nowait((timestamp, cpu, gpu))
                    except queue.Full:
                        logger.warning(
                            "Dropping live telemetry delivery; persisted "
                            "samples remain complete"
                        )
                    deadline = (
                        system_stats_module.next_periodic_deadline(
                            deadline,
                            time.monotonic(),
                            interval,
                        )
                    )
                    sampler_stop.wait(
                        max(0.0, deadline - time.monotonic())
                    )
            finally:
                try:
                    samples.put_nowait(None)
                except queue.Full:
                    pass

        sampler = threading.Thread(
            target=collect,
            name="mmas-system-stats",
            daemon=True,
        )
        sampler.start()
        try:
            while not stopped.is_set():
                sample = await loop.run_in_executor(None, samples.get)
                if sample is None:
                    break
                timestamp, cpu, gpu = sample
                await emit_sample(
                    cpu,
                    gpu,
                    timestamp=timestamp,
                    record=False,
                )
        finally:
            sampler_stop.set()
            sampler.join(timeout=1.0)
        return not unreadable.is_set()

    async def run_nvidia_smi_stream() -> bool:
        """Consume one persistent nvidia-smi process; return False on failure."""
        process: Optional[asyncio.subprocess.Process] = None
        stream_failed = False
        last_stream_emit = 0.0
        pending_gpu_peak: Optional[float] = None
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *system_stats_module.nvidia_smi_loop_command(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                while not stopped.is_set():
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=max(0.5, interval * 4),
                    )
                    if not line:
                        stream_failed = True
                        break
                    gpu = system_stats_module.parse_nvidia_smi_gpu_percent(line)
                    if gpu is None:
                        stream_failed = True
                        break
                    pending_gpu_peak = (
                        gpu
                        if pending_gpu_peak is None
                        else max(pending_gpu_peak, gpu)
                    )
                    now = loop.time()
                    # Some drivers flush several buffered lines together.
                    # Coalesce such bursts while retaining their GPU peak.
                    if not system_stats_module.stream_sample_is_due(
                        now,
                        last_stream_emit,
                        interval,
                    ):
                        continue
                    await emit_sample(
                        system_stats_module.read_cpu_percent_nonblocking(),
                        pending_gpu_peak,
                    )
                    pending_gpu_peak = None
                    last_stream_emit = now
            except (FileNotFoundError, OSError, asyncio.TimeoutError) as e:
                logger.debug("Persistent nvidia-smi telemetry unavailable: %s", e)
                stream_failed = True
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        return not stream_failed

    async def run_subprocess_fallback() -> None:
        """Keep CPU at 10 Hz and query GPU at 4 Hz without timer drift."""
        emit_deadline = loop.time()
        gpu_deadline = loop.time()
        while not stopped.is_set():
            now = loop.time()
            gpu_sample = None
            if now >= gpu_deadline:
                gpu_sample = await loop.run_in_executor(
                    None,
                    system_stats_module.read_nvidia_smi_gpu_percent_after_stream_failure,
                )
                gpu_deadline = system_stats_module.next_periodic_deadline(
                    gpu_deadline,
                    loop.time(),
                    fallback_interval,
                )
            await emit_sample(
                system_stats_module.read_cpu_percent_nonblocking(),
                gpu_sample,
            )
            emit_deadline = system_stats_module.next_periodic_deadline(
                emit_deadline,
                loop.time(),
                interval,
            )
            await sleep_until(emit_deadline)

    sysfs_path = system_stats_module.find_jetson_sysfs_gpu_load_path()
    if sysfs_path is not None:
        logger.info("GPU telemetry source: sysfs %s at 10 Hz", sysfs_path)
        completed = await run_sysfs_sampler(sysfs_path)
        if completed or stopped.is_set():
            return

    logger.info("GPU telemetry source: persistent nvidia-smi at 10 Hz")
    completed = await run_nvidia_smi_stream()
    if completed or stopped.is_set():
        return

    logger.warning("GPU telemetry source: one-shot nvidia-smi fallback at 4 Hz")
    await run_subprocess_fallback()
