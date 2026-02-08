# Why the phantom partial at tts_complete appears in MMAI Studio but not in Live RIVA WebUI

## What you see in MMAI Studio

A late **asr_partial** (same text as the last final) sometimes appears on the timeline **exactly at tts_complete** time — i.e. a phantom partial that looks like it happened when TTS finished, not when you spoke.

## Why Live RIVA WebUI doesn’t show it

The difference is **pipeline architecture**, not Riva itself.

### Live RIVA WebUI

- **Riva runs in a separate thread.** The thread has a synchronous loop: `for response in responses:` → for each result it **immediately** sends `{"type": "transcript", "text": ..., "is_final": ...}` to the client via the event loop.
- **LLM/TTS do not block that thread.** On a final, it schedules `_process_conversation()` (LLM → TTS) on the main event loop and **keeps iterating** in the Riva thread. So the thread keeps reading the next response from Riva right away.
- If Riva sends a **late partial** (same utterance, after the final), the thread sees it in the **next** `response` and sends it **as soon as Riva produces it** — typically a few hundred ms after the final. The client receives it at that time, not “at tts_complete.” So even if the client draws it on a timeline, it appears shortly after the final, not 10+ seconds later.

### MMAI Studio

- **One async loop** does both: `async for result in asr.receive_results()` **and** the full turn (LLM then TTS). So: read one result → if final, run LLM, then TTS → **then** read the next result.
- Any result Riva put in the queue **after** the final (e.g. a late partial) is only read **after** TTS completes. When we process it we assign `timestamp = now_ts` (current time), which is **tts_complete** time.
- So the same “late partial” from Riva is **stored and drawn at tts_complete** on the timeline → visible phantom.

## Summary

| App              | ASR result handling                    | When late partial is processed | Where it appears on timeline   |
|------------------|----------------------------------------|---------------------------------|---------------------------------|
| Live RIVA WebUI  | Thread sends each result immediately   | As soon as Riva sends it        | Shortly after final             |
| MMAI Studio      | One loop: result → LLM → TTS → next   | After TTS finishes              | At tts_complete (phantom)       |

So the phantom is a **consequence of our pipeline design** (single loop that blocks on LLM+TTS), not of Riva behaving differently. We fix it by **filtering** late partials that match the last final (same-utterance check in `voice_pipeline.py` results_loop).

---

## Barge-in: same design makes it impossible

With the **single loop** (read result → if final, LLM → TTS → then next result), we **never read the next ASR result until the current turn is done**. So we cannot see “user started speaking again” until after TTS has finished. That makes **barge-in** (interrupt AI speech) effectively impossible: we’d need to see a new final (or strong partial) *while* TTS is playing and then stop playback and start a new turn — but we don’t consume ASR at all during LLM/TTS.

So the config/schema option `barge_in_enabled` is not implemented in the pipeline today; the architecture doesn’t support it.

---

## Should we make ASR independent (like Live RIVA WebUI)?

**Idea:** Run ASR in its own task/thread that only drains `asr.receive_results()` and either sends events to the client and/or pushes finals into a queue. A separate “turn” task waits for a final from that queue, runs LLM → TTS, but can be **cancelled** when a new final arrives (user interrupted). Then we’d process the new final and start a new turn.

**Pros**

- **Barge-in becomes possible:** we see a new final while TTS is playing → cancel TTS task, stop audio, start new turn.
- **No phantom partial at tts_complete:** late partials are processed and sent as soon as they’re produced.
- **Closer to Live RIVA WebUI:** same mental model and behavior.

**Cons / considerations**

- **Complexity:** Two logical flows (ASR consumer vs turn executor), shared state (current TTS task, “in turn” flag), and cancellation semantics. Need a clear contract: e.g. “only one turn running; new final cancels it.”
- **Ordering and duplicates:** Need to avoid starting two turns for the same final, or processing finals out of order. A single “finals” queue and one consumer keeps order.
- **Thread vs asyncio:** Live RIVA runs Riva in a **thread** (blocking `streaming_response_generator`) and uses `run_coroutine_threadsafe` to send to the client and schedule work. We could do the same, or we could use two asyncio tasks: one that only `async for result in asr.receive_results()` and enqueues/sends, and one that `await finals_queue.get()` and runs LLM+TTS (cancellable). The asyncio-only approach avoids threads and fits our current stack; the “ASR in thread” approach matches Live RIVA and keeps Riva’s blocking API in one place.
- **TTS cancellation:** We must actually stop sending audio and flush or stop playback when we cancel. Doable with a shared “cancelled” or “abort” flag that the TTS send loop checks.

**Summary:** Making ASR independent (separate consumer that never blocks on LLM/TTS) is the right direction if we want barge-in and to remove the phantom at the source. The main cost is refactor and a bit more state/cancellation logic; there’s no fundamental con that makes it a bad idea.
