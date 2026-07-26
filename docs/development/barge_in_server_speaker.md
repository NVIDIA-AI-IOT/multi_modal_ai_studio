# Barge-in with Server Speaker

Server Speaker playback is implemented with an `aplay` subprocess, so browser-only
`stopTtsPlayback()` cannot interrupt it. MMAS therefore enforces barge-in at both
ends:

- Browser Speaker: the frontend stops scheduled Web Audio sources and ignores
  remaining `tts_audio` until the next `tts_start`.
- Server Speaker: `BargeInController` observes ASR events while TTS is active.
  A final transcript, or the configured number of partial transcripts, sets the
  backend interruption signal.
- The classic cascade cancels the active TTS consumer, stops `aplay`, emits a
  cancelled `tts_complete`, skips conversation-history insertion for the
  interrupted response, and advances to the queued user utterance.
- Realtime speech-to-speech stops `aplay` and suppresses subsequent server
  playback chunks for that response. It still consumes upstream response events
  because provider-side response cancellation is backend-specific.

The trigger is configured by:

```yaml
app:
  barge_in_enabled: true
  barge_in_trigger: final   # final or partial
  barge_in_partial_count: 3
```

Physical Server Speaker validation should cover both triggers and confirm that
device teardown leaves no `aplay` process after STOP, browser disconnect, or a
broken playback pipe.
