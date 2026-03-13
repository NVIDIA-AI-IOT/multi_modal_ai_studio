# Barge-in with server speaker (Server USB)

What it takes to support barge-in when **audio output** is **Server USB** (server speaker) instead of the browser.

## Current behavior

- **Browser speaker**: Frontend barge-in stops playback by calling `stopTtsPlayback()` (stops scheduled BufferSource nodes and ignores further `tts_audio` until next `tts_start`). Backend keeps sending chunks; only the client stops playing.
- **Server speaker**: TTS audio is played on the server via an **aplay** subprocess (`server_speaker_proc`). The backend writes PCM to `server_speaker_proc.stdin` in `_send_tts_audio()`. The **frontend** does not play any TTS when server speaker is selected (it uses `recordTtsSegmentOnly` for waveform only). So:
  - Stopping playback for server speaker must happen **on the backend**: stop writing to the aplay process and call `stop_server_speaker_playback(server_speaker_proc)` so the subprocess is terminated and audio stops.

## What is already in place

- **Barge-in during LLM** (just implemented): When we abort a turn because a new final is in the queue, we already call `stop_server_speaker_playback(server_speaker_proc)` in the abort path. So if the user speaks **before** or **during** the LLM phase, we never start (or we abort) TTS and we stop the server speaker. That already covers "barge-in before TTS" and "barge-in during LLM" for server speaker.
- **Barge-in during TTS** (not yet): When the user speaks **while TTS is already playing** (we're in the TTS phase, sending chunks to `server_speaker_proc`), we do **not** today stop the server speaker. The frontend can't stop it (it doesn't control aplay). So we need the **backend** to stop the current TTS and the aplay process when a new asr_final (or N partials, if we support that) arrives.

## What's needed for "barge-in during TTS" with server speaker

Goal: when barge-in is enabled and a **new asr_final** (or partial trigger) arrives **while we're in the TTS phase** (streaming to server speaker), stop sending more audio and stop the aplay process so playback stops immediately.

### Option A: TTS consumer checks a barge-in signal

1. **Shared signal**: Introduce an `asyncio.Event` (e.g. `barge_in_requested`) that **asr_consumer** sets when it does `finals_queue.put_nowait(result)` and `config.app.barge_in_enabled` is True. (So any new final sets the event.)
2. **TTS consumer**: In `_tts_consumer` (and in the non-stream TTS loop that writes to `server_speaker_proc`), **check** the event (or `not finals_queue.empty()`) before/after sending each chunk (or each text chunk). If set, exit the consumer early (and in the non-stream path, break out of the loop).
3. **Turn executor**: When `tts_task` completes (or the sequential TTS loop ends), check whether the completion was due to barge-in (event is set or queue non-empty). If so: call `stop_server_speaker_playback(server_speaker_proc)`, do **not** append to conversation_history for the **interrupted** turn (we already sent some audio; the "turn" is abandoned), call `session.end_turn()`, and `continue` so the next `finals_queue.get()` takes the new final.
4. **Cleanup**: In the barge-in path, ensure `stop_server_speaker_playback(server_speaker_proc)` is always called so aplay is terminated.

**Caveat**: Distinguishing "TTS finished normally" from "TTS exited due to barge-in" can be done by checking the event or queue when the task completes. We must not append the partial turn to history when we barge-in during TTS (the new final will be processed as the next turn).

### Option B: Cancel TTS task from outside

- **turn_executor** could `tts_task.cancel()` when it notices a new final in the queue. But the executor is blocked on `await tts_task` (or on the sequential TTS loop), so it can't "notice" until we use a different wait: e.g. `await asyncio.wait({tts_task, barge_in_event.wait()}, return_when=asyncio.FIRST_COMPLETED)`. If barge_in_event wins, cancel tts_task, stop server speaker, end_turn, continue. This avoids passing the event into _tts_consumer but requires restructuring how we wait for TTS completion.

### Recommendation

- **Option A** is more consistent with the existing "check queue" approach used for barge-in during LLM: re-use the same idea (new final in queue or a dedicated event set by asr_consumer) and have the TTS path check it and exit early. Then in the turn_executor, when the TTS task returns (or the sequential loop ends), check the event/queue and treat as barge-in: stop server speaker, skip history append for this turn, end_turn, continue.

### Summary table

| Scenario                         | Browser speaker                    | Server speaker                                      |
|----------------------------------|------------------------------------|-----------------------------------------------------|
| Barge-in **before/during LLM**   | Backend aborts turn; no TTS sent.  | Backend aborts turn; `stop_server_speaker_playback` (done). |
| Barge-in **during TTS**         | Frontend stops playback (done).    | Backend must stop aplay and stop sending (Option A above). |

### Effort (server speaker during TTS)

- **Medium**: Add barge-in event or queue check in `_tts_consumer` and in the sequential TTS loop; when they exit early, turn_executor detects barge-in, stops server speaker, skips history for that turn, ends turn, continues. Roughly 1–2 hours plus tests.
