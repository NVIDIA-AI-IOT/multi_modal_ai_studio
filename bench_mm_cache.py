#!/usr/bin/env python3
"""
Benchmark: Text-only history vs Multi-modal history (UUID cached inputs).

CASE 1 — Current approach (text-only history):
  Each turn sends [system, *text_history, user(video_N + prompt)]
  History = text only. Model cannot compare across turns.

CASE 2 — Full multi-modal history with UUID hints:
  Each turn sends [system, *mm_history, user(video_N + prompt)]
  History contains previous video blobs + text + uuid hints.
  vLLM's mm_processor_cache + encoder_cache + prefix_caching skip
  re-processing/re-encoding identical videos from history.
  UUIDs help vLLM identify cached items without content hashing.

Uses 3 real debug videos captured from the live pipeline.
Standalone — no dependency on multi_modal_ai_studio.

Usage:
    python bench_mm_cache.py                       # run both cases
    python bench_mm_cache.py --case 1              # only CASE 1
    python bench_mm_cache.py --case 2              # only CASE 2
    python bench_mm_cache.py --runs 3              # average over 3 runs
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import asyncio

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

NUM_TURNS = 3  # Use first 3 videos/prompts

VIDEO_DIR = Path(__file__).parent / "src" / "debug_videos"
VIDEO_FILES = [
    "cosmos_20260220_014518_9f_3fps.mp4",
    "cosmos_20260220_014528_8f_3fps.mp4",
    "cosmos_20260220_014535_6f_3fps.mp4",
]

PROMPTS = [
    "What do you see?",
    "What changed from before?",
    "How many fingers am I showing?",
]

DEFAULT_API_BASE = "http://localhost:8003/v1"
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/7d6a645088b550bbd45daaf782e2430bba9c82bb"

SYSTEM_PROMPT = (
    "You are a vision assistant observing the user through a live camera. "
    "Describe what you see accurately and concisely. Focus on actions, "
    "gestures, objects, and changes. One short sentence only."
)

TEMPERATURE = 0.3
MAX_TOKENS = 256


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_video_b64(path: Path) -> str:
    """Read an MP4 file and return a data-URL string."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def make_video_content_full(video_url: str, uuid: str) -> dict:
    """Video content block with actual bytes + uuid (first time)."""
    return {"type": "video_url", "video_url": {"url": video_url}, "uuid": uuid}


def make_video_content_with_uuid(video_url: str, uuid: str) -> dict:
    """Video content block with bytes + uuid (for history, uuid aids cache lookup)."""
    return {"type": "video_url", "video_url": {"url": video_url}, "uuid": uuid}


def make_user_mm(video_content: dict, prompt: str) -> dict:
    """User message with video content block + text."""
    return {
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": prompt},
        ],
    }


def make_user_text(prompt: str) -> dict:
    """User message with text only (for CASE 1 history)."""
    return {"role": "user", "content": prompt}


def make_assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


async def call_vlm(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    messages: list[dict],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, float, float, dict]:
    """Send a streaming chat completion request and measure timing."""
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.monotonic()
    ttft: Optional[float] = None
    chunks: list[str] = []
    usage_info: dict = {}

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"API error {resp.status}: {body[:500]}")

        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "usage" in data and data["usage"]:
                usage_info = data["usage"]

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                if ttft is None:
                    ttft = time.monotonic() - t0
                chunks.append(token)

    total = time.monotonic() - t0
    response_text = "".join(chunks)
    if ttft is None:
        ttft = total

    return response_text, ttft, total, usage_info


def estimate_payload_kb(messages: list[dict]) -> float:
    """Rough estimate of the JSON payload size in KB."""
    return len(json.dumps(messages)) / 1024


# ──────────────────────────────────────────────────────────────────────
# CASE 1: Text-only history (current approach)
# ──────────────────────────────────────────────────────────────────────

async def run_case1(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    video_urls: list[str],
    prompts: list[str],
) -> list[dict]:
    """
    Turn N:
      [system] + [text_history_1..N-1] + [user(video_N + prompt_N)]
    History is text-only — no video blobs in previous turns.
    """
    results = []
    text_history: list[dict] = []

    for i, (video_url, prompt) in enumerate(zip(video_urls, prompts)):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(text_history)

        # Current turn: full video + text
        video_content = {"type": "video_url", "video_url": {"url": video_url}}
        messages.append(make_user_mm(video_content, prompt))

        payload_kb = estimate_payload_kb(messages)
        n_hist = len(text_history)

        response, ttft, total, usage = await call_vlm(
            session, api_base, model, messages
        )

        result = {
            "turn": i + 1,
            "prompt": prompt,
            "video": VIDEO_FILES[i],
            "history_msgs": n_hist,
            "payload_kb": round(payload_kb, 1),
            "ttft_ms": round(ttft * 1000),
            "total_ms": round(total * 1000),
            "response": response.strip()[:120],
            "prompt_tokens": usage.get("prompt_tokens", "?"),
            "completion_tokens": usage.get("completion_tokens", "?"),
        }
        results.append(result)
        print(f"  Turn {i+1}: TTFT={result['ttft_ms']:>5}ms  Total={result['total_ms']:>5}ms  "
              f"Prompt_tok={str(result['prompt_tokens']):>5}  "
              f"Payload={result['payload_kb']:>6.1f}KB  Hist={n_hist}")

        # Append text-only history
        text_history.append(make_user_text(prompt))
        text_history.append(make_assistant(response.strip()))

    return results


# ──────────────────────────────────────────────────────────────────────
# CASE 2: Multi-modal history with UUID cached inputs
# ──────────────────────────────────────────────────────────────────────

async def run_case2(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    video_urls: list[str],
    prompts: list[str],
) -> list[dict]:
    """
    Turn N:
      [system]
      + [user(vid_1+uuid + prompt_1), asst_1]   ← full bytes + uuid (cached by vLLM)
      + ...
      + [user(vid_N-1+uuid + prompt_N-1), asst_N-1]
      + [user(vid_N+uuid + prompt_N)]            ← new video

    History sends full bytes + UUID. vLLM's mm_processor_cache, encoder_cache,
    and prefix_caching avoid re-processing identical videos.
    """
    results = []
    mm_history: list[dict] = []
    # Store video_urls alongside their UUIDs for re-sending in history
    history_videos: list[tuple[str, str, str]] = []  # (video_url, prompt, uuid)

    for i, (video_url, prompt) in enumerate(zip(video_urls, prompts)):
        uuid = f"vid-turn-{i+1}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(mm_history)

        # Current turn: full video + uuid
        video_content = make_video_content_full(video_url, uuid)
        messages.append(make_user_mm(video_content, prompt))

        payload_kb = estimate_payload_kb(messages)
        n_hist = len(mm_history)

        response, ttft, total, usage = await call_vlm(
            session, api_base, model, messages
        )

        result = {
            "turn": i + 1,
            "prompt": prompt,
            "video": VIDEO_FILES[i],
            "history_msgs": n_hist,
            "payload_kb": round(payload_kb, 1),
            "ttft_ms": round(ttft * 1000),
            "total_ms": round(total * 1000),
            "response": response.strip()[:120],
            "prompt_tokens": usage.get("prompt_tokens", "?"),
            "completion_tokens": usage.get("completion_tokens", "?"),
            "videos_in_hist": i,
        }
        results.append(result)
        print(f"  Turn {i+1}: TTFT={result['ttft_ms']:>5}ms  Total={result['total_ms']:>5}ms  "
              f"Prompt_tok={str(result['prompt_tokens']):>5}  "
              f"Payload={result['payload_kb']:>6.1f}KB  Hist={n_hist}  "
              f"Hist_vids={i}")

        # Add to history with full bytes + uuid (vLLM caches by uuid/hash)
        hist_video_content = make_video_content_with_uuid(video_url, uuid)
        mm_history.append(make_user_mm(hist_video_content, prompt))
        mm_history.append(make_assistant(response.strip()))

    return results


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def print_report(case_name: str, all_runs: list[list[dict]]) -> tuple[float, float]:
    """Print a summary table for a case, averaging across runs."""
    n_turns = len(all_runs[0])
    n_runs = len(all_runs)

    print(f"\n{'='*90}")
    print(f"  {case_name}  ({n_runs} run{'s' if n_runs > 1 else ''})")
    print(f"{'='*90}")
    print(f"{'Turn':>5} {'Video':>32} {'Hist':>5} {'Payload':>9} {'TTFT(ms)':>10} {'Total(ms)':>10} {'Prmt_tok':>9} {'Response'}")
    print(f"{'-'*5:>5} {'-'*32:>32} {'-'*5:>5} {'-'*9:>9} {'-'*10:>10} {'-'*10:>10} {'-'*9:>9} {'-'*35}")

    total_ttft = 0
    total_time = 0

    for t in range(n_turns):
        avg_ttft = sum(r[t]["ttft_ms"] for r in all_runs) / n_runs
        avg_total = sum(r[t]["total_ms"] for r in all_runs) / n_runs
        avg_payload = sum(r[t]["payload_kb"] for r in all_runs) / n_runs
        last = all_runs[-1][t]
        total_ttft += avg_ttft
        total_time += avg_total
        print(f"{last['turn']:>5} {last['video']:>32} {last['history_msgs']:>5} "
              f"{avg_payload:>7.1f}KB {avg_ttft:>10.0f} {avg_total:>10.0f} "
              f"{str(last['prompt_tokens']):>9} {last['response'][:35]}")

    print(f"{'-'*5:>5} {'-'*32:>32} {'-'*5:>5} {'-'*9:>9} {'-'*10:>10} {'-'*10:>10}")
    print(f"{'SUM':>5} {'':>32} {'':>5} {'':>9} {total_ttft:>10.0f} {total_time:>10.0f}")
    print(f"{'AVG':>5} {'':>32} {'':>5} {'':>9} {total_ttft/n_turns:>10.0f} {total_time/n_turns:>10.0f}")
    print()

    return total_ttft, total_time


def print_comparison(c1_ttft, c1_total, c2_ttft, c2_total, n_turns):
    """Print side-by-side comparison."""
    print(f"\n{'='*90}")
    print(f"  COMPARISON (lower is better)")
    print(f"{'='*90}")
    print(f"{'Metric':>20} {'CASE1 (text hist)':>22} {'CASE2 (mm+uuid)':>22} {'Δ':>12} {'Δ%':>8}")
    print(f"{'-'*20:>20} {'-'*22:>22} {'-'*22:>22} {'-'*12:>12} {'-'*8:>8}")

    def row(label, v1, v2):
        delta = v2 - v1
        pct = (delta / v1 * 100) if v1 else 0
        sign = "+" if delta > 0 else ""
        print(f"{label:>20} {v1:>19.0f}ms {v2:>19.0f}ms {sign}{delta:>9.0f}ms {sign}{pct:>5.1f}%")

    row("Sum TTFT", c1_ttft, c2_ttft)
    row("Sum Total", c1_total, c2_total)
    row("Avg TTFT/turn", c1_ttft / n_turns, c2_ttft / n_turns)
    row("Avg Total/turn", c1_total / n_turns, c2_total / n_turns)
    print()
    if c2_ttft < c1_ttft:
        saving = (1 - c2_ttft / c1_ttft) * 100
        print(f"  ✅ CASE 2 (mm+uuid) is {saving:.1f}% faster TTFT than CASE 1")
    elif c2_ttft > c1_ttft:
        overhead = (c2_ttft / c1_ttft - 1) * 100
        print(f"  ⚠️  CASE 2 (mm+uuid) is {overhead:.1f}% slower TTFT than CASE 1")
    else:
        print(f"  ≈ Both cases have similar TTFT")
    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark text-only vs multi-modal (UUID cached) history for Cosmos VLM"
    )
    parser.add_argument("--api", default=DEFAULT_API_BASE,
                        help=f"vLLM API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path")
    parser.add_argument("--case", type=int, choices=[1, 2], default=None,
                        help="Run only CASE 1 or CASE 2")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs to average (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens per response")
    args = parser.parse_args()

    # Verify videos exist
    video_paths = []
    for vf in VIDEO_FILES[:NUM_TURNS]:
        p = VIDEO_DIR / vf
        if not p.exists():
            print(f"❌ Video not found: {p}")
            sys.exit(1)
        video_paths.append(p)

    # Pre-load videos
    print(f"Loading {NUM_TURNS} videos...")
    video_urls = []
    for p in video_paths:
        url = load_video_b64(p)
        size_kb = len(base64.b64decode(url.split(",", 1)[1])) / 1024
        print(f"  {p.name}: {size_kb:.0f} KB")
        video_urls.append(url)

    prompts = PROMPTS[:NUM_TURNS]

    print(f"\nAPI: {args.api}")
    print(f"Model: {args.model}")
    print(f"Turns: {NUM_TURNS}")
    print(f"Runs: {args.runs}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {TEMPERATURE}")

    # Verify API
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(f"{args.api}/models") as resp:
                if resp.status != 200:
                    print(f"❌ API not reachable: {resp.status}")
                    sys.exit(1)
                print("✅ API reachable\n")
        except Exception as e:
            print(f"❌ Cannot reach API: {e}")
            sys.exit(1)

        c1_results_all = []
        c2_results_all = []

        for run_idx in range(args.runs):
            if args.runs > 1:
                print(f"\n{'─'*40} Run {run_idx+1}/{args.runs} {'─'*40}")

            if args.case is None or args.case == 1:
                print(f"\n▶ CASE 1: Text-only history (current approach)")
                c1 = await run_case1(session, args.api, args.model, video_urls, prompts)
                c1_results_all.append(c1)

            if args.case is None:
                print("\n  (pausing 2s between cases...)")
                await asyncio.sleep(2)

            if args.case is None or args.case == 2:
                print(f"\n▶ CASE 2: Full multi-modal history + UUID hints")
                c2 = await run_case2(session, args.api, args.model, video_urls, prompts)
                c2_results_all.append(c2)

        # ── Reports ──
        if c1_results_all:
            c1_ttft, c1_total = print_report(
                "CASE 1: Text-only history (current)", c1_results_all
            )
        if c2_results_all:
            c2_ttft, c2_total = print_report(
                "CASE 2: Full MM history + UUID hints", c2_results_all
            )

        if c1_results_all and c2_results_all:
            print_comparison(c1_ttft, c1_total, c2_ttft, c2_total, NUM_TURNS)

        # Save raw results
        output = {
            "config": {
                "api_base": args.api,
                "model": args.model,
                "temperature": TEMPERATURE,
                "max_tokens": args.max_tokens,
                "n_turns": NUM_TURNS,
                "n_runs": args.runs,
                "videos": VIDEO_FILES[:NUM_TURNS],
                "prompts": prompts,
                "system_prompt": SYSTEM_PROMPT,
                "case2_uses_uuid_hints": True,
            },
        }
        if c1_results_all:
            output["case1_text_history"] = c1_results_all
        if c2_results_all:
            output["case2_mm_uuid_hints"] = c2_results_all

        out_path = Path(__file__).parent / "bench_mm_cache_results.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"📄 Raw results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
