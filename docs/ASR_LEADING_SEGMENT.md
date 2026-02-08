# ASR: Capturing the Start of Utterances

## Problem

Sometimes the first word or two of what you said are missing in the transcript. For example you say **"Tell me a joke"** but the app shows **"a joke"** (Live RIVA WebUI may show the full phrase with the same Riva server).

## Cause

Riva’s streaming ASR uses **Speech Pad** (mapped to `start_history` in `EndpointingConfig`). That value is the pre-speech window (in ms) used when detecting “speech start.” If it’s too small, the very beginning of the utterance can fall outside the window and be dropped.

Pipeline and chunking differences (e.g. when we start sending PCM, chunk size) can make this show up more in Multi-Modal AI Studio than in Live RIVA WebUI even with the same Riva/VAD settings.

## Fix

1. **Increase Speech Pad**  
   In **Configuration → ASR → VAD Tuning**, set **Speech Pad** to **500** (or 400–600). The default is now 500 ms so more of the start of speech is included. If you still lose the beginning, try 600.

2. **Presets**  
   The **Balanced** preset uses 300 ms. If you use that preset and notice clipped beginnings, either raise Speech Pad manually after applying Balanced or use a higher Speech Pad default.

## Technical

- **Schema:** `ASRConfig.speech_pad_ms` (default 500).
- **Riva:** Passed as `EndpointingConfig.start_history` in `_create_streaming_config()` in `backends/asr/riva.py`.
- **UI:** Slider 0–1000 ms; tooltip on Speech Pad explains that higher values help capture the start of speech.
