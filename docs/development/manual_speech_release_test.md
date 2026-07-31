# Manual Speech Release Test Plan

This checklist covers behavior that unit and integration tests cannot fully
validate: real microphone capture, browser playback, ASR endpointing, TTS
continuity, barge-in, saved timelines, and local USB devices.

Use it before a release candidate and after changes to the voice pipeline,
audio devices, ASR/TTS adapters, session timing, or browser UI. Mark unavailable
hardware as `N/A` with a reason rather than silently skipping it.

## Release record

Record the following with the test results:

```text
Date:
Tester:
Git commit:
MMAS version:
Platform: Thor / Orin / other
Jetson Linux / BSP:
Power mode:
Browser and version:
Microphone / speaker / camera:
Preset:
ASR model and API:
LLM model and API:
TTS model and API:
```

Keep the resulting session JSON and note its filename. Screenshots and short
audio recordings are useful evidence, but generated WAV files should normally
be stored as test artifacts rather than committed to Git.

## Minimum release matrix

| Priority | Platform | Speech stack | Devices | Required coverage |
|---|---|---|---|---|
| P0 | Jetson Thor | Public OpenAI-compatible Nemotron ASR + LLM + Magpie | Browser mic and speaker | Normal turns, endpointing, TTS, barge-in, session timeline |
| P0 | Jetson Thor | Riva ASR + LLM + Riva TTS | Browser mic and speaker | Existing-path regression, partial/final ASR, TTS, barge-in |
| P1 | Jetson AGX Orin | Public open-model stack | Browser mic and speaker | One complete three-turn smoke test |
| P1 | Jetson Thor | Riva with Silero VAD | Browser mic and speaker | Initial-word pickup and response-rate comparison |
| P1 | Available Jetson | Release-default stack | Server USB mic and speaker | Capture, playback, barge-in, reconnect |
| P2 | Available Jetson | Multilingual-capable stack | Browser devices | Japanese ASR, LLM, and TTS smoke test |
| P2 | Available platform | VLM configuration | Browser or server camera | Permission, preview, and one vision turn |

When `Start speaking before LLM finishes` is enabled by default, its test is
P0. Otherwise it can be P1 with any known same-GPU contention limitation
documented in the release notes.

## Automated preflight

Run the normal CI suite first:

```bash
pytest tests/unit -v -m "not slow"
```

For changes involving TTS scheduling, resource contention, or MIG, also run the
Speech Contention Lab described in
[`speech_contention_lab.md`](speech_contention_lab.md). It does not replace the
listening tests below.

Before opening the UI:

- Confirm ASR, LLM, and TTS health checks succeed.
- Confirm the pipeline ribbon shows the configured model names.
- Record container tags or service revisions.
- Hard-refresh the page after updating MMAS so stale JavaScript is not tested.
- Use English for the P0 public demonstration; run Japanese separately at P2.

## Core browser test

Use Browser WebRTC microphone and Browser speaker. Keep
`Start speaking before LLM finishes` **off** for the baseline.

### 1. Startup and permissions

- Open the MMAS HTTPS URL.
- Grant microphone access and, when applicable, camera access.
- Confirm the microphone preview responds before starting the session.
- Confirm the local camera preview appears when a browser camera is selected.
- Select **Check servers** and verify all configured backends are healthy.
- Start a session and confirm there are no browser-console or server errors.

### 2. Normal three-turn conversation

Say:

1. `Hi, can you hear me?`
2. `What is one plus one?`
3. `Tell me one short safety rule for a home robot.`

Pass criteria:

- Every utterance produces one final transcript and one appropriate response.
- The ASR, LLM, and TTS lanes appear in the correct order.
- Generated speech plays once, without a missing first response.
- Speech is continuous; there is no loud noise, clipping, or unexpected replay.
- Model names remain correct in the pipeline ribbon.

### 3. Endpointing and initial-word pickup

From silence, say `What is one plus one?` five times as separate turns. Begin
speaking naturally; do not add an artificial leading word.

Record:

- responses received out of 5;
- how often `What` is missing;
- partial and final transcripts when the ASR supports partials;
- visible VAD/ASR ribbon behavior.

P0 expectation is 5/5 responses without a systematic missing initial word.
When comparing VAD configurations, keep microphone, position, level, phrases,
and browser unchanged.

### 4. Barge-in and cancellation

Ask:

`Explain three safety rules for a home robot.`

While the AI is audibly speaking, interrupt with:

`Stop. What is one plus one?`

Then ask:

`Tell me one short safety rule.`

Pass criteria:

- The interruption receives a final transcript.
- Old browser audio stops at the configured barge-in trigger.
- The old purple AI waveform ends at the actual playback stop.
- A red Barge-in marker identifies the playback stop, and its tooltip reports
  the trigger, cancelled turn, and cancellation timing.
- Already-scheduled discarded audio is shown only as a pale purple hatched
  region; backend work after playback stop is hatched on the TTS lane.
- The cancelled TTS does not block the next LLM response.
- No old audio resumes or becomes attached to the next turn.
- The next two TTS responses both play.
- TTL values are present where audio played and are never negative.

Also perform one early interruption before AI audio starts. A cancelled turn
with no played audio must not borrow the next turn's TTS or TTL.

### 5. Streaming-before-completion

Repeat the normal and barge-in tests with
`Start speaking before LLM finishes` **on**.

Record:

- configured units before first speech;
- LLM generation start and completion;
- TTS start, TTFA, and first browser playback;
- audible gaps, stutter, or prosody resets;
- playback underruns if measured.

Pass criteria:

- TTS begins only after LLM generation supplies the configured trigger;
- audio begins before LLM completion;
- speech remains intelligible and acceptably continuous;
- barge-in still cancels the active turn.

If same-GPU LLM/TTS contention is an accepted limitation, link the contention
benchmark result and state whether the feature is disabled by default.

### 6. Session save and replay

- Stop the session normally.
- Open its saved session entry.
- Confirm the turn count and transcript text.
- Confirm user and AI waveforms match what was heard.
- Confirm ASR, LLM, TTS, and TTL lanes remain associated with the correct turn.
- Switch timeline zoom and confirm no lane disappears.
- Record the session JSON filename in the release record.

### 7. Lifecycle regression

- Start and stop two sessions without reloading the page.
- Mute and unmute the microphone during a session.
- Leave the session idle for at least two minutes, then speak again.
- Reload the page and start another session.

Pass criteria:

- No duplicate playback or duplicated WebSocket events.
- ASR resumes after mute, idle, and reload.
- Device permission state remains understandable to the user.
- The server does not retain a stale ASR stream or TTS task.

## Server USB audio test

Connect the physical devices before starting MMAS.

1. Select the server USB microphone and speaker.
2. Confirm microphone preview levels before starting the session.
3. Run the normal three-turn test.
4. Run the barge-in test while audio is playing on the server speaker.
5. Stop the session and confirm the saved user and AI waveforms.
6. Stop MMAS, disconnect and reconnect the devices, restart MMAS, and confirm
   that device selection and capture recover cleanly.

Pass criteria:

- Browser PCM is not accidentally sent while the server microphone is active.
- Server capture produces the same ASR and timeline semantics as browser input.
- Server playback is cancelled as well as browser playback.
- Sample-rate conversion does not change pitch or playback speed.
- No capture task, subprocess, or audio device remains busy after session stop.

## Result template

Copy this table into the PR or release qualification note:

| Test | Configuration | Result | Session/evidence | Notes |
|---|---|---|---|---|
| Startup and permissions | | PASS / FAIL / N/A | | |
| Normal three turns | | PASS / FAIL / N/A | | |
| Initial-word pickup, 5 trials | | PASS / FAIL / N/A | | |
| Barge-in during playback | | PASS / FAIL / N/A | | |
| Barge-in before playback | | PASS / FAIL / N/A | | |
| Streaming before completion | | PASS / FAIL / N/A | | |
| Session save and replay | | PASS / FAIL / N/A | | |
| Mute, idle, reload | | PASS / FAIL / N/A | | |
| Server USB audio | | PASS / FAIL / N/A | | |
| Orin smoke test | | PASS / FAIL / N/A | | |
| Japanese smoke test | | PASS / FAIL / N/A | | |

For every failure, distinguish among:

- an MMAS regression;
- a model-quality result;
- an external service or credential failure;
- a documented hardware/resource limitation.

P0 failures block a normal release unless explicitly waived and documented.
