#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Create a realistic session JSON from the Voice Timeline screenshot (0:00–0:14).

Represents actual data: three turns with TTL 1984 ms, 871 ms, 600 ms.
No need for realistic phrases; event types and timings match the screenshot.

Usage:
  python scripts/create_session_from_screenshot.py
  # Writes mock_sessions/voice_timeline_screenshot__YYYYMMDD.json
"""

import json
from pathlib import Path
from datetime import datetime, timezone


def sec(*args):
    return round(sum(args), 3)


def event(ts, event_type, lane, data=None):
    return {"timestamp": ts, "event_type": event_type, "lane": lane, "data": data or {}}


def main():
    timeline = []

    # Session start
    timeline.append(event(0.0, "session_start", "system"))

    # ----- Turn 1: "Tell me a joke" (0–2.5s), TTS 3.5–5.5s, TTL 1984 ms -----
    # User speech ~0–2.2s; first partial ~0.3s; final at 2.25s
    timeline.append(event(0.0, "user_speech_start", "audio"))
    for t, text in [(0.30, "Tell"), (0.65, "Tell me"), (1.05, "Tell me a"), (1.50, "Tell me a joke")]:
        timeline.append(event(t, "asr_partial", "speech", {"text": text, "is_final": False}))
    timeline.append(event(2.20, "user_speech_end", "audio"))
    timeline.append(event(2.25, "asr_final", "speech", {"text": "Tell me a joke.", "confidence": 0.96}))

    timeline.append(event(2.30, "llm_start", "llm"))
    timeline.append(event(3.20, "llm_complete", "llm", {"text": "Why don't some couples go to the gym?"}))
    timeline.append(event(3.25, "tts_start", "tts"))
    # TTL 1984 ms: user_speech_end 2.20 → tts_first_audio 4.184
    timeline.append(event(4.184, "tts_first_audio", "tts"))
    timeline.append(event(5.50, "tts_complete", "tts", {"text": "Why don't some couples go to the gym?"}))

    # ----- Turn 2: "Any joke about computer" (8–10.5s), TTS 11–13.5s, TTL 871 ms -----
    timeline.append(event(8.00, "user_speech_start", "audio"))
    for t, text in [(8.25, "Any"), (8.60, "Any joke"), (8.95, "Any joke about"), (9.30, "Any joke about computer")]:
        timeline.append(event(t, "asr_partial", "speech", {"text": text, "is_final": False}))
    timeline.append(event(9.50, "user_speech_end", "audio"))
    timeline.append(event(9.55, "asr_final", "speech", {"text": "Any joke about computer.", "confidence": 0.94}))

    timeline.append(event(9.60, "llm_start", "llm"))
    timeline.append(event(9.95, "llm_complete", "llm", {"text": "Why do programmers prefer dark mode?"}))
    timeline.append(event(10.00, "tts_start", "tts"))
    # TTL 871 ms: user_speech_end 9.50 → tts_first_audio 10.371
    timeline.append(event(10.371, "tts_first_audio", "tts"))
    timeline.append(event(13.50, "tts_complete", "tts", {"text": "Why do programmers prefer dark mode?"}))

    # ----- Turn 3: "Wall" (13.5–14.5s), TTL 600 ms -----
    timeline.append(event(13.50, "user_speech_start", "audio"))
    timeline.append(event(13.70, "asr_partial", "speech", {"text": "Wall", "is_final": False}))
    timeline.append(event(13.80, "user_speech_end", "audio"))
    timeline.append(event(13.85, "asr_final", "speech", {"text": "Wall.", "confidence": 0.98}))

    timeline.append(event(13.90, "llm_start", "llm"))
    timeline.append(event(14.00, "llm_complete", "llm", {"text": "..."}))
    timeline.append(event(14.05, "tts_start", "tts"))
    # TTL 600 ms: user_speech_end 13.80 → tts_first_audio 14.40
    timeline.append(event(14.40, "tts_first_audio", "tts"))
    timeline.append(event(14.90, "tts_complete", "tts", {"text": "..."}))

    # Optional: sparse audio amplitude samples (user + AI) for waveform
    for t in [0.2, 0.6, 1.0, 1.4, 1.8, 2.1]:
        e = event(t, "audio_amplitude", "audio")
        e["amplitude"] = 50 + (int(t * 10) % 40)
        e["source"] = "user"
        timeline.append(e)
    for t in [4.2, 4.6, 5.0, 5.3, 10.5, 11.0, 12.0, 12.8]:
        e = event(t, "audio_amplitude", "audio")
        e["amplitude"] = 40 + (int(t * 5) % 35)
        e["source"] = "tts"
        timeline.append(e)

    # Sort by timestamp
    timeline.sort(key=lambda e: e["timestamp"])

    # Metrics
    user_ends = [e["timestamp"] for e in timeline if e["event_type"] == "user_speech_end"]
    tts_firsts = [e["timestamp"] for e in timeline if e["event_type"] == "tts_first_audio"]
    ttls_ms = []
    for i, ue in enumerate(user_ends):
        if i < len(tts_firsts) and tts_firsts[i] > ue:
            ttls_ms.append(round((tts_firsts[i] - ue) * 1000))
    max_ts = max(e["timestamp"] for e in timeline)
    session_id = f"voice_timeline_screenshot__{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    chat = [
        {"role": "user", "content": "Tell me a joke.", "timestamp": 2.25},
        {"role": "assistant", "content": "Why don't some couples go to the gym?", "timestamp": 5.50},
        {"role": "user", "content": "Any joke about computer.", "timestamp": 9.55},
        {"role": "assistant", "content": "Why do programmers prefer dark mode?", "timestamp": 13.50},
        {"role": "user", "content": "Wall.", "timestamp": 13.85},
        {"role": "assistant", "content": "...", "timestamp": 14.90},
    ]

    session = {
        "session_id": session_id,
        "name": "Voice Timeline (from screenshot)",
        "created_at": created_at,
        "config": {
            "name": "Screenshot",
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
            "avg_ttl": round(sum(ttls_ms) / len(ttls_ms), 1) if ttls_ms else 0,
            "min_ttl": min(ttls_ms) if ttls_ms else 0,
            "max_ttl": max(ttls_ms) if ttls_ms else 0,
            "session_duration": round(max_ts, 2),
        },
    }

    out_dir = Path(__file__).parent.parent / "mock_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.json"
    with open(out_path, "w") as f:
        json.dump(session, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  Events: {len(timeline)}, Turns: {session['metrics']['total_turns']}")
    print(f"  TTLs (ms): {ttls_ms}, Avg: {session['metrics']['avg_ttl']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
