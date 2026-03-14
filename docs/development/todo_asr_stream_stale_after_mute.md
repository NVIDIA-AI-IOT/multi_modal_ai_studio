# ASR stream goes stale after mute/unmute or long silence

After muting then unmuting the mic (or after a prolonged period where no speech reaches Riva), the ASR stream silently stops producing results even though PCM audio is still flowing.

## Observed behavior

- Session `c87be1b2` (2026-03-14): 9 turns completed successfully.
- Turn 8 triggered a degenerate LLM reasoning loop (10,101 chars, **91.89 s** wall-clock).
- During that wait, the user muted and later unmuted the mic.
- After turn 9 completed, no further `asr_final` events appeared for ~1.5 min despite the green **user_amplitude** waveform being visible on the timeline (PCM capture was healthy).
- Terminal showed no ASR errors; the stream ended normally at session close with `Stream task timeout, cancelling`.

## Why amplitude shows but ASR does not

In `_feed_pcm_to_pipeline`, amplitude is always computed and sent to the client (lines 1005-1024) regardless of `mic_muted`. The ASR send is gated:

```python
if not mic_muted:
    await asr.send_audio(pcm_bytes)
```

So the timeline waveform looks alive, but if the Riva gRPC stream has internally timed out (or VAD state has gone stale after 90+ seconds of silence/mute), newly sent audio produces no results.

## Probable root cause (needs confirmation)

Riva Streaming ASR has internal session limits:
- **gRPC keepalive / idle timeout**: if no audio is sent for an extended period the server may silently close the stream.
- **VAD state**: after a long silence gap, the VAD model may reset or require a fresh trigger to start detecting speech again.
- **Maximum session duration**: Riva may cap single-stream duration; after that, the stream yields no more results even though it stays open.

The exact Riva behavior here is unconfirmed — the stream appeared open (no error logged) but stopped producing finals.

## What is already in place

- `mic_muted` gates `asr.send_audio()` in the classic pipeline (line 1003).
- On mute, 0.5 s of silence is injected (`b"\x00" * int(16000 * 2 * 0.5)`) to flush any pending VAD partial (line 1041-1044).
- On unmute, `mic_muted = False` resumes sending PCM to ASR.
- No stream-health monitoring or automatic restart exists today.

## Proposed solutions (pick one or combine)

### Option A: Keep-alive noise during mute

While `mic_muted` is True, instead of sending nothing, send **very low amplitude white noise** (e.g., ±10 out of ±32768) at normal cadence. This keeps the gRPC stream active and the VAD model warm without triggering false speech detection.

Pros: Simplest change; no stream lifecycle management. \
Cons: Assumes the Riva stream itself is still healthy; does not help if the stream has a hard session-duration cap.

### Option B: Restart ASR stream after stale timeout

Monitor elapsed time since the last `asr_final`. If no final arrives within a configurable window (e.g., 60 s while unmuted), tear down the current `RivaASRBackend` stream and create a fresh one.

1. Track `_last_asr_final_time` in the turn executor; update it on every `asr_final`.
2. In `server_capture_consumer` (or a watchdog task), check `time.time() - _last_asr_final_time > ASR_STALE_TIMEOUT`.
3. If stale and `not mic_muted`: call `asr.stop()`, then `asr.start()` to open a fresh streaming session.
4. Log `[asr] Stream restarted after stale timeout` at WARNING level.

Pros: Covers all root causes (idle timeout, VAD reset, session-duration cap). \
Cons: Slightly more complex; brief gap in ASR coverage during restart (~200 ms).

### Option C: Proactive stream rotation

After every turn (or every N turns), close and re-open the ASR stream. This preempts any session-duration limit and keeps the stream fresh.

Pros: Eliminates stale state entirely. \
Cons: Adds latency at turn boundaries; may lose a partial if speech is ongoing during rotation.

## Recommendation

**Option A + B combined**: send keep-alive noise during mute (A) to prevent idle timeout, and add a stale-timeout watchdog (B) as a safety net for unexpected stream failures. Option C is heavier and only needed if Riva has a hard session cap that A+B cannot address.

## Diagnosis checklist (before implementing)

- [ ] Confirm Riva Streaming ASR session limits: check `riva_asr` service config for `max_duration_seconds`, keepalive settings, or gRPC deadline.
- [ ] Add a log line in `RivaASRBackend` when the gRPC response iterator ends (to distinguish "server closed stream" from "no results but stream open").
- [ ] Reproduce by muting for 60+ s mid-session and verifying ASR stops producing results on unmute.

## Effort

**Small–Medium**: Option A is ~30 min (noise generator in `_feed_pcm_to_pipeline`). Option B is ~1–2 hours (watchdog task + stream restart plumbing + tests).
