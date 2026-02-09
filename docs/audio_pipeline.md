# Multi-modal AI Studio — Audio Pipeline

This document describes the voice pipeline from browser microphone through ASR → LLM → TTS and back to the browser. For comparison with Live RIVA WebUI, see [Difference from Live RIVA WebUI](#difference-from-live-riva-webui) below.

---

## 1. High-Level Flow

```
[Browser]  ←—— WebSocket /ws/voice (JSON config + binary PCM) ——→  [Server]
   │                                                                     │
   │  Mic → ScriptProcessor @ 16 kHz → PCM (no resampling)                │
   │         │                    │                                       │
   │         │                    └—— binary WS (after config) ————————┘
   │         │                                                             │
   │         │                    [voice_pipeline.py]                      │
   │         │                      First message: { type: 'config', config }
   │         │                      Then: binary PCM → asr.send_audio()
   │         │                                                             ▼
   │         │                    [RivaASRBackend] streaming gRPC (executor)
   │         │                              │
   │         │                              ▼
   │         │                    results_loop: asr_final → LLM → TTS
   │         │                              │
   │         │                    [OpenAILLMBackend] generate_stream()
   │         │                              │
   │         │                    [RivaTTSBackend] synthesize_stream()
   │         │                              │
   │         │                    Events: type 'event', event: { event_type, lane, data, timestamp }
   │         │                    TTS:    type 'tts_audio', data (base64), sample_rate
   │         │                              │
   │         ▼                              ▼
   │  [Playback] playTtsChunk() — gapless (nextStartTime), session_saved on close
   └─────────────────────────────────────────────────────────────────────
```

---

## 2. Browser → Server (Capture & Upload)

**Location:** `webui/static/app.js` — `startVoiceMicStream()`, `connectPcmToWs()`, WebSocket `/ws/voice`.

| Step | What happens |
|------|----------------|
| 1 | User starts live session → WebSocket opens to `wss://host/ws/voice`. |
| 2 | **First message must be JSON config:** `{ type: 'config', config: { asr, llm, tts, devices, app } }`. Server waits up to 30s for it; then runs the pipeline. |
| 3 | Mic: `getUserMedia({ audio: true })` (or reuse preview stream). `AudioContext` created with **sampleRate: 16000** so capture is 16 kHz directly (browser resamples device to context rate if needed). |
| 4 | `createScriptProcessor(2048, 1, 1)` — runs every 2048/16000 = **128 ms**. Float32 → Int16 PCM → `ws.send(pcmData.buffer)` (binary). No JS resampling. |
| 5 | Control: `{ type: 'stop' }` (TEXT) stops the session; server sets `stopped` and exits. |

**Formats:**
- **Upload:** 16 kHz, mono, 16-bit PCM (binary WebSocket).
- **Config:** Full session config (ASR, LLM, TTS, devices) sent once at start; no per-chunk options like barge-in/voice in the stream.

---

## 3. Server: WebSocket and Voice Pipeline

**Location:** `webui/voice_pipeline.py` — `handle_voice_ws`, `_run_voice_pipeline`.

| Step | What happens |
|------|----------------|
| 1 | `handle_voice_ws`: prepare WebSocket, **receive first TEXT message** (must be `type: 'config'` with `config`). Normalize keys (e.g. `asr.backend` → `asr.scheme`, `asr.riva_server` → `asr.server`), build `SessionConfig.from_dict()`. |
| 2 | `_run_voice_pipeline(ws, session_config, session_dir)`: creates `Session` (with timeline), instantiates **backends** from config: `RivaASRBackend`, `OpenAILLMBackend`, `RivaTTSBackend`. Requires ASR/TTS scheme `riva` and LLM `api_base`. |
| 3 | **receive_loop** (task): async for msg in ws. TEXT with `type: 'stop'` → set `stopped`. BINARY → `await asr.send_audio(msg.data)`. |
| 4 | **results_loop** (task): `async for result in asr.receive_results()`. For each result: push to **timeline** (e.g. `asr_partial` / `asr_final`), send `type: 'event'` with `event_type`, `lane`, `data`, `timestamp`. On **final**: `session.start_turn()`, send `llm_start`, stream LLM via `llm.generate_stream()`, send `llm_complete`, then `tts.synthesize_stream()`, send `tts_audio` (base64) and `tts_first_audio` / `tts_complete` events, `session.end_turn()`. |
| 5 | On `stopped`: cancel recv/results tasks, `asr.stop_stream()`, compute metrics, **save session** to `session_dir / {session_id}.json`, return session_id. Then send `{ type: 'session_saved', session_id }` and close WS. |

**Backends:**
- ASR: `backends/asr/riva.py` — `RivaASRBackend`. Sync audio queue + executor for Riva streaming; results put on async queue, consumed by results_loop.
- LLM: `backends/llm/openai.py` — `OpenAILLMBackend`. OpenAI-compatible API, streaming tokens.
- TTS: `backends/tts/riva.py` — `RivaTTSBackend`. Streams PCM chunks (e.g. 22050 Hz).

---

## 4. ASR (Riva)

**Location:** `backends/asr/riva.py`.

- `start_stream()`: creates sync queue and async results queue; starts `_stream_to_riva()` as asyncio task.
- `send_audio(bytes)`: put chunk on sync queue (consumed by executor thread that runs Riva’s blocking generator).
- `_stream_to_riva()`: runs in executor; pulls from sync queue, feeds `asr_service.streaming_response_generator()`, pushes `ASRResult` (text, is_final, confidence) to results queue.
- Config: VAD via `EndpointingConfig` (speech_pad_ms, speech_timeout_ms, vad_start/stop_threshold), LINEAR_PCM 16 kHz, interim_results True.

---

## 5. LLM and TTS

- **LLM:** `OpenAILLMBackend.generate_stream(prompt, history, system_prompt)` — yields tokens; conversation_history (user/assistant) is kept in voice_pipeline and passed per turn.
- **TTS:** `RivaTTSBackend.synthesize_stream(text)` — yields audio chunks (PCM, sample_rate). Pipeline base64-encodes and sends `{ type: 'tts_audio', data, sample_rate, is_final }`. Timeline events: `tts_first_audio`, `tts_complete`.

---

## 6. Browser: Events and TTS Playback

- **Events:** Every server event is sent as `{ type: 'event', event: { event_type, lane, data, timestamp } }`. Frontend pushes to `state.liveTimelineEvents`, updates timeline and chat (e.g. `event_type === 'chat'` → append to `liveChatTurns`).
- **TTS:** On `msg.type === 'tts_audio'`, `playTtsChunk(msg.data, msg.sample_rate || 24000)`. Decode base64 → Int16 → Float32 → AudioBuffer; **gapless** scheduling via `state.ttsNextStartTime` (same pattern as Live RIVA).
- **Session end:** On `session_saved`, store `lastSavedSessionId` and refresh session list. On WS close, stop mic stream and set session state to stopped.

---

## 7. Data Formats Summary

| Stage | Format | Sample rate | Notes |
|-------|--------|-------------|--------|
| Browser → Server (mic) | PCM Int16, binary WS | 16 kHz | 16 kHz AudioContext, 2048-frame chunks; no resampling |
| Server → Riva ASR | PCM (from queue) | 16 kHz | LINEAR_PCM |
| Server → Browser (events) | JSON | — | type: 'event', event: { event_type, lane, data, timestamp } |
| Server → Browser (TTS) | JSON with base64 PCM | e.g. 22050 Hz | type: 'tts_audio', data, sample_rate, is_final |
| Browser playback | PCM Int16 → Float32, AudioBuffer | From chunk | Gapless via nextStartTime |

---

## 8. Key Files

| Component | File | Notes |
|-----------|------|--------|
| WS handler & pipeline | `webui/voice_pipeline.py` | handle_voice_ws, _run_voice_pipeline, receive_loop, results_loop |
| ASR | `backends/asr/riva.py` | RivaASRBackend, send_audio, receive_results, _stream_to_riva (executor) |
| LLM | `backends/llm/openai.py` | OpenAILLMBackend, generate_stream |
| TTS | `backends/tts/riva.py` | RivaTTSBackend, synthesize_stream |
| Session & timeline | `core/session.py`, `core/timeline.py` | Session.start_turn, end_turn, timeline.add_event |
| Browser | `webui/static/app.js` | /ws/voice, config send, connectPcmToWs, playTtsChunk, event handling |

---

## Difference from Live RIVA WebUI

| Aspect | Multi-modal AI Studio | Live RIVA WebUI |
|--------|------------------------|------------------|
| **WebSocket** | `/ws/voice`; **first message = config** (full ASR/LLM/TTS/session). Then binary PCM + TEXT `stop`. | `/ws`; **control messages** `start` / `stop` / `tts_playback_complete` with **inline options** (barge-in, trigger, partial_count, tts_voice). Audio is binary PCM after `start`. |
| **Config** | Session-scoped: one `SessionConfig` at connect; backends created once per connection. | Per-`start`: barge-in, trigger, TTS voice, etc. No full session schema over WS. |
| **Backends** | Pluggable: `RivaASRBackend`, `OpenAILLMBackend`, `RivaTTSBackend` (from schema). Pipeline is backend-agnostic after config. | Monolithic: `RivaBridge` owns ASR + LLM client + TTS; single class handles WS, Riva gRPC, and conversation flow. |
| **Timeline / session** | **Session** and **Timeline** objects; every ASR/LLM/TTS step pushes events to timeline; session saved to JSON on disconnect. | No persistent session object; timeline/metrics are frontend-driven (and server sends transcript/llm_response/tts_audio). |
| **Events** | Unified **event** envelope: `{ type: 'event', event: { event_type, lane, data, timestamp } }` for ASR, LLM, TTS, chat, session_start. | Separate message types: `transcript`, `llm_thinking`, `llm_response`, `tts_start`, `tts_audio`, `tts_complete`, `tts_interrupted`, `recording_started`, `recording_stopped`. |
| **Barge-in** | Not implemented in the voice pipeline (single turn flow; no “interrupt TTS” on new speech). | Implemented: while TTS is playing, ASR can trigger barge-in; server cancels TTS task and sends `tts_interrupted`; frontend sends `tts_playback_complete` when playback ends. |
| **Browser capture** | 16 kHz AudioContext, 2048 samples = 128 ms chunks; no resampling. | 16 kHz AudioContext, 4096 samples ≈ 256 ms chunks; no resampling. |
| **TTS playback** | Gapless via `ttsNextStartTime`; no explicit “playback complete” message to server. | Gapless queue (8-chunk buffer, then play); frontend sends `tts_playback_complete` so server can clear `tts_playing` for barge-in. |
| **Session persistence** | Session saved to disk on WS close (`session_dir / {session_id}.json`); `session_saved` sent before close. | No server-side session save in the voice path. |

In short: **Multi-modal** is config-first, backend-pluggable, and session/timeline-centric with a unified event stream; **Live RIVA** is connection-centric with inline control (barge-in, TTS voice) and no shared session/timeline model on the server.
