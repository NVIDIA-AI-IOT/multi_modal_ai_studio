# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for desktop and Jetson GPU telemetry."""

import subprocess
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.webui import system_stats


@pytest.fixture(autouse=True)
def reset_nvidia_smi_support_cache():
    system_stats._nvidia_smi_gpu_supported = None
    yield
    system_stats._nvidia_smi_gpu_supported = None


def test_nvidia_smi_numeric_utilization_is_used_and_cached(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="37\n", stderr="")

    monkeypatch.setattr(system_stats.subprocess, "run", fake_run)

    assert system_stats._read_nvidia_smi_gpu_percent() == 37.0
    assert system_stats._nvidia_smi_gpu_supported is True
    assert system_stats._read_nvidia_smi_gpu_percent() == 37.0
    assert len(calls) == 2
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["timeout"] == 2


def test_nvidia_smi_loop_uses_100ms_stream_and_parses_samples():
    assert system_stats.nvidia_smi_loop_command() == [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        "--loop-ms=100",
    ]
    assert system_stats.parse_nvidia_smi_gpu_percent(b"37\n") == 37.0
    assert system_stats.parse_nvidia_smi_gpu_percent("12 %") == 12.0
    assert system_stats.parse_nvidia_smi_gpu_percent("[N/A]") is None
    assert system_stats.parse_nvidia_smi_gpu_percent("101") is None


def test_cpu_sampler_is_nonblocking(monkeypatch):
    calls = []
    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(cpu_percent=lambda interval: calls.append(interval) or 12.34),
    )

    assert system_stats.read_cpu_percent_nonblocking() == 12.3
    assert calls == [None]


def test_stream_sample_cadence_rejects_buffered_bursts():
    interval = 0.1

    assert system_stats.stream_sample_is_due(10.0, 0.0, interval)
    assert not system_stats.stream_sample_is_due(10.001, 10.0, interval)
    assert not system_stats.stream_sample_is_due(10.079, 10.0, interval)
    assert system_stats.stream_sample_is_due(10.081, 10.0, interval)
    assert system_stats.stream_sample_is_due(10.1, 10.0, interval)


def test_periodic_deadline_stays_on_grid_and_skips_missed_slots():
    assert system_stats.next_periodic_deadline(10.0, 10.03, 0.1) == pytest.approx(10.1)
    assert system_stats.next_periodic_deadline(10.0, 10.1, 0.1) == pytest.approx(10.2)
    assert system_stats.next_periodic_deadline(10.0, 10.35, 0.1) == pytest.approx(10.4)
    with pytest.raises(ValueError):
        system_stats.next_periodic_deadline(10.0, 10.0, 0.0)


def test_telemetry_intervals_are_10hz_and_4hz_fallback():
    assert system_stats.SYSTEM_STATS_INTERVAL_MS == 100
    assert system_stats.GPU_SUBPROCESS_FALLBACK_INTERVAL_MS == 250


@pytest.mark.parametrize(
    "result",
    [
        subprocess.CompletedProcess([], 0, stdout="[N/A]\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        subprocess.CompletedProcess([], 1, stdout="42\n", stderr="failed"),
        subprocess.CompletedProcess([], 0, stdout="101\n", stderr=""),
    ],
)
def test_nvidia_smi_unsupported_results_are_disabled_after_one_probe(
    monkeypatch,
    result,
):
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return result

    monkeypatch.setattr(system_stats.subprocess, "run", fake_run)

    assert system_stats._read_nvidia_smi_gpu_percent() is None
    assert system_stats._read_nvidia_smi_gpu_percent() is None
    assert system_stats._nvidia_smi_gpu_supported is False
    assert calls == 1


def test_nvidia_smi_missing_binary_is_disabled_after_one_probe(monkeypatch):
    calls = 0

    def missing_binary(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise FileNotFoundError

    monkeypatch.setattr(system_stats.subprocess, "run", missing_binary)

    assert system_stats._read_nvidia_smi_gpu_percent() is None
    assert system_stats._read_nvidia_smi_gpu_percent() is None
    assert calls == 1


def test_jetson_sysfs_load_is_scaled_and_invalid_candidates_are_skipped(
    monkeypatch,
    tmp_path,
):
    invalid = tmp_path / "invalid-load"
    valid = tmp_path / "valid-load"
    invalid.write_text("[N/A]\n")
    valid.write_text("735\n")

    class FakeDevfreqRoot:
        def glob(self, pattern):
            if pattern == "*gpu*/device/load":
                return [invalid, valid]
            return []

    monkeypatch.setattr(system_stats, "Path", lambda path: FakeDevfreqRoot())

    assert system_stats.find_jetson_sysfs_gpu_load_path() == valid
    assert system_stats.read_jetson_sysfs_gpu_percent() == 73.5


def test_gather_system_stats_prefers_jetson_sysfs(monkeypatch):
    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(cpu_percent=lambda interval: 12.34),
    )
    def unexpected_nvidia_smi():
        raise AssertionError("nvidia-smi should not run when sysfs is available")

    monkeypatch.setattr(system_stats, "_read_nvidia_smi_gpu_percent", unexpected_nvidia_smi)
    monkeypatch.setattr(
        system_stats,
        "read_jetson_sysfs_gpu_percent",
        lambda: 56.7,
    )

    assert system_stats.gather_system_stats() == {
        "cpu_percent": 12.3,
        "gpu_percent": 56.7,
    }


def test_gather_system_stats_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr(system_stats, "psutil", None)
    monkeypatch.setattr(
        system_stats,
        "read_jetson_sysfs_gpu_percent",
        lambda: None,
    )
    monkeypatch.setattr(
        system_stats,
        "_read_nvidia_smi_gpu_percent",
        lambda: 41.0,
    )

    assert system_stats.gather_system_stats() == {
        "cpu_percent": None,
        "gpu_percent": 41.0,
    }


def test_stream_failure_retries_nvidia_smi_when_thor_has_no_sysfs_load(
    monkeypatch,
):
    system_stats._nvidia_smi_gpu_supported = False
    monkeypatch.setattr(
        system_stats,
        "read_jetson_sysfs_gpu_percent",
        lambda: None,
    )
    monkeypatch.setattr(
        system_stats,
        "_read_nvidia_smi_gpu_percent",
        lambda: 28.0,
    )

    assert system_stats.read_nvidia_smi_gpu_percent_after_stream_failure() == 28.0
    assert system_stats._nvidia_smi_gpu_supported is None
