# REST ASR latency diagnostics

Use this only for an intentional test, with the microphone user's consent. Raw microphone recordings and transcripts are private data. It is disabled by default, is not a session audio recorder, and does not change VAD or inference settings.

## Capture the actual request

Set `MMAS_ASR_CAPTURE_DIR` in the MMAS server process environment before starting it:

```bash
MMAS_ASR_CAPTURE_DIR=/absolute/private/asr-capture \
  multi-modal-ai-studio --preset speaches-jetson --port 8092
```

Each REST ASR request saves the exact mono 16 kHz PCM16 WAV sent to the provider plus a JSON sidecar. The sidecar includes the model, normalized language, audio duration, session clock anchor, speech/request times, HTTP status, transcript, elapsed request time, and WAV SHA256. It excludes API keys, headers, full configuration, and endpoint URLs. Request latency excludes capture disk I/O; writes run outside the asyncio event loop. Errors or full capture storage do not stop ASR.

The directory must be private (mode 700); new files have mode 600. There is a fixed cap of 32 recordings, 64 MiB total WAV data, and 4 MiB per WAV. The cap persists across restarts. Use a separate directory per MMAS process. It does not rotate/delete existing recordings or capture after the cap; check the log. Short diagnostics are intended, not continuous recording. An interrupted process may leave a pending sidecar. Stop the server or unset the variable and restart to disable capture. Keep recordings outside the Git checkout, do not commit/share them without review, and delete them after diagnosis when no longer needed.

## Identical-input replay

From a checkout matching the installed wheel, using a Python environment with `aiohttp`:

```bash
python scripts/replay_rest_asr.py /absolute/private/asr-capture/asr-ID.wav \
  --language ja --repeats 5 --output /absolute/private/replay-results.json
```

This uses existing services only. It never downloads models, changes configuration, or plays sound. It compares identical WAV bytes in rotating mode order: ASR alone, ASR during a Kokoro synthesis request, and ASR immediately after synthesis completes. Actual HTTP request overlap is recorded rather than inferred from audio playback. Override model/voice/endpoints for your services; `SPEECH_API_KEY` supplies a shared speech-provider key without placing it on the command line. The output refuses overwriting evidence and contains transcripts; keep its directory private. A synthesis request may still finish on the server after client cancellation.

Inspect cold and warm runs separately. The measured ASR HTTP latency includes provider preprocessing, queueing, inference, and serialization. A TTS request interval is not proof of GPU occupancy throughout it. Pair results with provider logs and hardware telemetry. Do not assume the browser's buffered audio playback means the GPU is still synthesizing, or that a slow final result is a model reload. Preserve failed/empty results; compare transcript quality as well as latency. Use original microphone WAVs to diagnose input-dependent failures; synthetic speech is only a separate execution baseline.
