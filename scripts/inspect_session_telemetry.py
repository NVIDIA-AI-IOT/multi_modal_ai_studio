#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Report measured telemetry cadence from a saved session, not a nominal rate."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(samples):
    times = [float(s["t"]) for s in samples]
    intervals = sorted((b - a) * 1000 for a, b in zip(times, times[1:]))
    if not intervals or intervals[0] <= 0:
        raise ValueError("Need at least two observations with increasing timestamps")
    return {
        "samples": len(times),
        "observed_hz": (len(times) - 1) / (times[-1] - times[0]),
        "median_interval_ms": statistics.median(intervals),
        "p95_interval_ms": intervals[min(len(intervals) - 1, int(len(intervals) * 0.95))],
        "max_interval_ms": intervals[-1],
        "note": "Polling cadence is not the GPU driver's hardware measurement resolution.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    try:
        result = summarize(session.get("system_stats") or [])
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
