#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal test for the OpenAI Realtime WebSocket client (Phase 2).

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/test_realtime_client.py

Connects to wss://api.openai.com/v1/realtime, sends session.update, optionally
sends a short silence, and prints the first few server events (session.updated,
etc.). Exits after a few seconds or after receiving response.done / error.
"""

import asyncio
import os
import sys

# Add project root so we can import multi_modal_ai_studio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_modal_ai_studio.backends.realtime import OpenAIRealtimeClient, RealtimeEvent


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Set OPENAI_API_KEY to run this test.", file=sys.stderr)
        sys.exit(1)

    url = "wss://api.openai.com/v1/realtime"
    client = OpenAIRealtimeClient(
        url=url,
        api_key=api_key,
        model="gpt-realtime",
        instructions="You are a helpful assistant. Reply briefly.",
    )

    print("Connecting to", url, "...")
    await client.connect()
    print("Connected. Waiting for session.updated and events (timeout 15s)...")

    count = 0

    async def consume() -> None:
        nonlocal count
        async for ev in client.events():
            nonlocal count
            if ev is None:
                print("Stream ended.")
                return
            count += 1
            if ev.kind == "error":
                print("Error:", ev.message)
                return
            if ev.kind == "session_ready":
                print("  [session.updated] session ready")
            elif ev.kind == "audio":
                print(f"  [audio] {len(ev.audio or b'')} bytes @ {ev.sample_rate} Hz")
            elif ev.kind == "transcript_delta":
                print(f"  [transcript delta] {repr((ev.text or '')[:60])}")
            elif ev.kind == "transcript_completed":
                print(f"  [transcript completed] {repr((ev.text or '')[:80])}")
            elif ev.kind == "response_done":
                print("  [response.done]")
                return
            else:
                print(f"  [{ev.kind}]")
            if count >= 20:
                print("  (stopping after 20 events)")
                return

    try:
        await asyncio.wait_for(consume(), timeout=15.0)
    except asyncio.TimeoutError:
        print("  (15s timeout; no more events)")
    finally:
        await client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
