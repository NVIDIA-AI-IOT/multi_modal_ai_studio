# Investigating near-full user amplitude with no ASR

## False green only on replay (live was correct)

If **live** recording showed correct user amplitude (e.g. server logs `amp_0_100=0.00` every second and the live green waveform was flat), but **replay** of the same session shows high green bars (e.g. around 26 s):

- The bug is **not** in the live pipeline display (live uses client `liveAudioAmplitudeHistory`).
- Replay draws from the **saved timeline** (server-recorded). So the high values are in the **stored** timeline: the server did call `add_audio_amplitude(..., source="user")` with high `amp` for some 50 ms samples. The 1 s throttled log only shows one sample per second, so it can miss those.
- **Tracing:** Server now logs **every** user amplitude ≥ 20 as `[user_amplitude_high] session_t=... amp_0_100=...`. Client logs every buffer with amp ≥ 20 as `[user_amplitude_high] client: session_t=...`. Run a session; if replay later shows false green, check server logs and client console for those lines. If only the server logged high at that time → high value introduced on server (e.g. wrong chunk or corruption). If both log at similar `session_t` → client sent a loud buffer (same data the server stored).

## Problem

In some recorded sessions, the timeline shows **user (green) amplitude at or near 100%** during periods when:

- No TTS is playing, and
- There is **no** `asr_final` (or `asr_partial`) nearby — i.e. the user was not speaking.

So the stored data has high "user" amplitude with no corresponding speech. That is a **data/recording bug**, not just a display issue. We do **not** hide these bars in the UI so the artifact remains visible for debugging.

## Where the data comes from

User amplitude in the **timeline** is recorded only on the **server**:

1. **Server** (`voice_pipeline.py`): For each binary (mic) WebSocket message, every 50 ms it calls:
   - `_pcm_rms_to_amplitude(msg.data)` → RMS of the PCM chunk, scaled 0–100.
   - `session.timeline.add_audio_amplitude(amplitude=amp, source="user")`.

So high "user" amplitude with no ASR means one or more of:

- **Client is sending loud/wrong data** — The PCM chunks the browser sends (from the mic) have high RMS even when the user is silent. Possible causes:
  - Mic gain / system volume / device (e.g. some USB/Bluetooth mics send constant high level).
  - Bug in the ScriptProcessor / AudioWorklet path: wrong buffer, wrong channel, or scaling.
  - Another tab or app feeding the same mic and sending different data.
- **Server RMS or scaling** — `_pcm_rms_to_amplitude()` or the way chunks are received could be wrong (e.g. endianness, chunk size, or scaling).
- **Clock/ordering** — Unlikely to produce *constant* high amplitude; more likely to cause timing drift.

## How to trace

**Logging is in place:**

1. **Server-side** (`voice_pipeline.py`, receive_loop)
   - **Every high amplitude** (amp ≥ 20): `[user_amplitude_high] session_t=... amp_0_100=... raw_rms=... chunk_len=...` (so we can see exactly when high values get into the timeline and appear as false green on replay).
   - **Throttled ~1 s:** `[user_amplitude] session_t=... chunk_len=... raw_rms=... amp_0_100=...`
   - `raw_rms` = RMS of the 16-bit PCM chunk (before scaling to 0–100). Silence ≈ 0; loud speech might be in the hundreds to low thousands.
   - `amp_0_100` = value stored in the timeline (same as `_pcm_rms_to_amplitude`). If you see `amp_0_100` near 100 but no one is speaking, compare with `raw_rms` to confirm the server is receiving loud PCM.

2. **Client-side** (`app.js`, `processor.onaudioprocess`)
   - **Every high amplitude** (amp ≥ 20): `[user_amplitude_high] client: session_t=... amp_0_100=... (same buffer we send to server)`.
   - **Throttled ~1 s:** `[user_amplitude] client: buffer_len=... float_rms=... amp_0_100=...`
   - `float_rms` = RMS of the float32 channel buffer (before int16 conversion and send). If this is high when the room is silent, the artifact is on the client (mic/device or capture path).
   - Compare client and server `[user_amplitude_high]` timestamps: if only server logs at a given `session_t`, the high value was introduced on the server; if both log at similar time, the client sent a loud buffer.

3. **Compare**
   - Run a session that previously showed the artifact. Watch server logs and browser console. Note whether high amplitude appears first on client or only on server.
   - Test with different mics (e.g. built-in vs AirPods vs USB) and browsers to see if the artifact is device- or browser-specific.

3. **Session JSON**
   - Open a session that shows the artifact. Inspect `timeline.events` for `event_type === "audio_amplitude"` and `source === "user"` in the suspicious time range. Check whether values are consistently high (e.g. > 90) and whether they align with any other events (TTS, ASR).

## Files to check

| Role   | File / area |
|--------|-------------|
| Server | `src/multi_modal_ai_studio/webui/voice_pipeline.py`: receive loop, `_pcm_rms_to_amplitude`, `add_audio_amplitude(..., source="user")` |
| Client | `src/multi_modal_ai_studio/webui/static/app.js`: mic capture (ScriptProcessor / AudioWorklet), `state.liveAudioAmplitudeHistory`, and the code that sends binary PCM over the WebSocket |
| Model  | `src/multi_modal_ai_studio/core/timeline.py`: `add_audio_amplitude` (just stores the value; no logic that would inflate it) |

Once the source (client vs server, and which path) is identified, we can fix the bug at the origin instead of hiding the symptom in the UI.
