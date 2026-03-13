#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert Live RIVA WebUI timeline export (JSON array of events) into a
Multi-Modal AI Studio session JSON file.

Usage:
  # Export from Live RIVA WebUI console (exportTimelineForMultiModalStudio()),
  # save to file, then:
  python scripts/convert_live_riva_timeline.py --input /tmp/events.json --output mock_sessions/captured.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description="Convert Live RIVA timeline export to session JSON")
    ap.add_argument("--input", "-i", required=True, help="Input JSON file (array of timeline events)")
    ap.add_argument("--output", "-o", required=True, help="Output session JSON file")
    ap.add_argument("--name", default="Captured from Live RIVA", help="Session name")
    args = ap.parse_args()

    with open(args.input) as f:
        timeline = json.load(f)

    if not isinstance(timeline, list):
        timeline = timeline.get("timeline", timeline)

    session_id = f"captured_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Build chat from asr_final and llm/tts (simplified: we don't have assistant text in events)
    chat = []
    for e in timeline:
        if e.get("event_type") == "asr_final":
            text = (e.get("data") or {}).get("text", "").strip()
            if text:
                chat.append({"role": "user", "content": text, "timestamp": e["timestamp"]})
        # We could add assistant messages if we had tts segment text in data

    # Metrics
    user_ends = [e["timestamp"] for e in timeline if e.get("event_type") == "user_speech_end"]
    tts_first = [e["timestamp"] for e in timeline if e.get("event_type") == "tts_first_audio"]
    ttls = []
    for i, ue in enumerate(user_ends):
        if i < len(tts_first) and tts_first[i] > ue:
            ttls.append((tts_first[i] - ue) * 1000)
    avg_ttl = sum(ttls) / len(ttls) if ttls else 0
    max_time = max((e.get("timestamp", 0) for e in timeline), default=0)

    session = {
        "session_id": session_id,
        "name": args.name,
        "created_at": created_at,
        "config": {
            "name": "Captured",
            "asr": {"scheme": "riva", "server": "localhost:50051", "language": "en-US"},
            "llm": {"scheme": "openai", "model": "llama3.2:3b"},
            "tts": {"scheme": "riva", "server": "localhost:50051"},
            "devices": {},
            "app": {},
        },
        "chat": chat,
        "timeline": timeline,
        "metrics": {
            "total_turns": len(user_ends),
            "avg_ttl": round(avg_ttl, 1),
            "min_ttl": round(min(ttls), 1) if ttls else 0,
            "max_ttl": round(max(ttls), 1) if ttls else 0,
            "session_duration": round(max_time, 2),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(session, f, indent=2)

    print(f"Wrote {len(timeline)} events to {out}")
    print(f"  Turns: {session['metrics']['total_turns']}, Avg TTL: {session['metrics']['avg_ttl']} ms")


if __name__ == "__main__":
    main()
