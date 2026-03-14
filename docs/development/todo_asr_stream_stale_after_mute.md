# ASR stream goes stale / dies prematurely

The Riva ASR gRPC stream can die mid-session — silently stopping result production or terminating entirely. This happens in multiple scenarios: after mute/unmute, after long idle periods, or even during normal operation with 0 results.

## Observed behavior

### Scenario 1: Stale after long LLM block + mute/unmute
- Session `c87be1b2` (2026-03-14): 9 turns completed successfully.
- Turn 8 triggered a degenerate LLM reasoning loop (10,101 chars, **91.89 s** wall-clock).
- During that wait, the user muted and later unmuted the mic.
- After turn 9 completed, no further `asr_final` events appeared for ~1.5 min despite the green **user_amplitude** waveform being visible on the timeline.
- Terminal showed no ASR errors; the stream ended normally at session close.

### Scenario 2: Stream dies after ~2 min (normal operation, no mute)
- Session on jat-4cbb47141bb7 (2026-03-14): 3 turns completed in ~2 min.
- After turn 3, Riva ASR stream ended with 16 results total.
- `_feed_pcm_to_pipeline` continued sending PCM but `send_audio()` raised `RuntimeError: Stream not started` — **crashing the entire pipeline**.

### Scenario 3: Stream dies with 0 results (~23s)
- Session `f9748641` on same device: ASR stream started, 0 results received, stream ended after ~23s.
- Same `RuntimeError` crash.

### Scenario 4: USB contention with Brio 4K camera
- On Jetsons with Brio 4K (USB 3.0) + USB audio, severe bus contention causes:
  - Camera `VIDIOC_REQBUFS: errno=19 (No such device)` — camera disappears from bus.
  - `arecord: audio open error: Device or resource busy` — audio device locked by previous pipeline.
  - ASR stream dies with 0 results; pipeline crashes before user even speaks.

## Why amplitude shows but ASR does not

In `_feed_pcm_to_pipeline`, amplitude is always computed and sent to the client regardless of `mic_muted`. The ASR send is gated:

```python
if not mic_muted:
    accepted = await asr.send_audio(pcm_bytes)
```

So the timeline waveform looks alive, but if the Riva gRPC stream has internally timed out or died, newly sent audio is silently dropped (or previously, would crash).

## Root causes

Riva Streaming ASR has internal session limits:
- **gRPC keepalive / idle timeout**: if no audio is sent for an extended period the server may silently close the stream.
- **VAD state**: after a long silence gap, the VAD model may reset or require a fresh trigger.
- **Maximum session duration**: Riva may cap single-stream duration (~2 min observed); after that, the stream yields no more results.
- **USB bus contention**: on Jetson devices with multiple USB peripherals (especially high-bandwidth cameras like Brio 4K), the audio device can become temporarily unavailable, preventing `arecord` from opening.

## Implemented fixes (2026-03-14)

### Fix 1: Graceful `send_audio` (riva.py)
`send_audio()` now returns `bool` instead of raising `RuntimeError` when the stream is dead. Returns `False` if `_sync_audio_queue` is `None`, allowing the PCM feeder to continue without crashing.

### Fix 2: Log-once warning (_feed_pcm_to_pipeline)
When `send_audio` returns `False`, a warning is logged once per dead-stream episode: `[asr] send_audio dropped — ASR stream not active (waiting for auto-restart)`. The flag resets on stream restart.

### Fix 3: Auto-restart in asr_consumer (voice_pipeline.py)
`asr_consumer` now wraps the `async for result in asr.receive_results()` loop in a `while not stopped.is_set()` loop. When the inner iterator ends (stream died) and the pipeline is still running:
1. Increments restart counter (max 10).
2. Logs a WARNING with result count and restart number.
3. Emits `asr_stream_restart` timeline event.
4. Calls `asr.stop_stream()` → sleep with exponential backoff (2s, 4s, ..., max 10s) → `asr.start_stream()`.
5. Resets the `send_audio` log-once flag and result counter.
6. Re-enters the `async for` loop on the fresh stream.

## Remaining work

### Option A: Keep-alive noise during mute
While `mic_muted` is True, send very low amplitude white noise (e.g., ±10 out of ±32768) at normal cadence. This keeps the gRPC stream active and the VAD model warm.

Pros: Prevents idle timeout during mute. \
Cons: Does not help if stream has a hard session-duration cap (but auto-restart covers that now).

### Device contention mitigations
- Investigate separating camera and audio onto different USB host controllers.
- Consider CSI camera instead of USB to free USB bandwidth entirely.
- Current `arecord` retry logic (8 attempts with backoff) helps, but persistent `Device or resource busy` across all retries indicates the previous pipeline's `arecord` process was not killed before the new one started.

## Diagnosis checklist

- [x] Add auto-restart when gRPC stream ends unexpectedly.
- [x] Make `send_audio` graceful (no crash on dead stream).
- [x] Emit timeline events for stream restarts.
- [ ] Confirm Riva session limits: check `riva_asr` config for `max_duration_seconds`, keepalive settings, or gRPC deadline.
- [ ] Implement keep-alive noise during mute (Option A).
- [ ] Investigate cleanup of old `arecord` processes on WebSocket reconnect.

## Effort

**Done**: Auto-restart (Fix 3) + graceful send_audio (Fix 1). \
**Remaining**: Keep-alive noise ~30 min. Device cleanup investigation ~1–2 hours.
