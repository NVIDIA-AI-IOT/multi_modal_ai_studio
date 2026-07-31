#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Add Nemotron transcript CER/WER to existing faster-qwen3-tts result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_qwen3_tts_benchmark import optional_asr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--asr-url",
        default="http://127.0.0.1:8081/v1/audio/transcriptions",
    )
    parser.add_argument(
        "--asr-model",
        default="nvidia/nemotron-3.5-asr-streaming-0.6b",
    )
    args = parser.parse_args()

    for result_path in args.results:
        payload = json.loads(result_path.read_text())
        if payload.get("status") == "skipped":
            print(f"{result_path}: skipped backend; no WAVs to score")
            continue
        for item in payload.get("results", []):
            wav_path = Path(item["wav"])
            item["asr"] = optional_asr(
                wav_path,
                item["text"],
                item["language"],
                args.asr_url,
                args.asr_model,
            )
            print(
                f"{result_path.name} {item['language']} chunk={item['chunk_size']}: "
                f"{item['asr']}"
            )
        payload["asr_quality_endpoint"] = args.asr_url
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
