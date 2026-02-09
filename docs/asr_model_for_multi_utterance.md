# ASR Model Choice: Silero VAD Required for Multi-Utterance / Second Turn

## Summary

For **multi-turn voice conversations** (user speaks → assistant replies → user speaks again), use the **Silero VAD** ASR model:

- **Recommended**: `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer`
- **Avoid as default**: `parakeet-1.1b-en-US-asr-streaming` (base Parakeet, no Silero VAD)

If you use the base Parakeet model, the **second (and later) user utterances are often not recognized** — you may only see the first turn. The Silero VAD model handles continuous multi-utterance streaming correctly.

## Where it's configured: Riva server (config.sh)

**Which ASR models are available is determined by the Riva server**, not by this app. The app only **queries** the server for the list of model names and lets you choose one.

For the Riva quickstart (e.g. in Live RIVA WebUI), that means **`config.sh`** in the quickstart directory (e.g. `riva_quickstart_arm64_v2.24.0/config.sh`):

- **`asr_acoustic_model=("parakeet_1.1b")`** — which acoustic model is deployed.
- **`asr_accessory_model=("silero_diarizer")`** — deploy with **Silero VAD** (and diarization). This is what produces the model name `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer` on the server.

If you deploy **without** the Silero accessory (or with a different config), the server may expose only the base model name (`parakeet-1.1b-en-US-asr-streaming`), and multi-utterance / second turn will often fail. So for multi-turn dialogue, set `asr_accessory_model=("silero_diarizer")` in config.sh, rebuild/redeploy Riva, then use the Silero VAD model in the app.

## Why

- The **base Parakeet** streaming model (`parakeet-1.1b-en-US-asr-streaming`) is tuned for single-utterance or different VAD behavior; in practice it often stops producing finals after the first turn or misses follow-up speech.
- The **Silero VAD + SortFormer** variant (`...-silero-vad-sortformer`) uses Silero VAD for segment boundaries and matches the behavior expected by Live RIVA WebUI and this app for back-and-forth dialogue.

## What the app does (it only uses what Riva exposes)

1. **Backend default**  
   `DEFAULT_ASR_MODEL` in `src/multi_modal_ai_studio/backends/asr/riva.py` is set to  
   `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer`. This is used when the server returns no models or when we recommend which model to select.

2. **API**  
   `GET /api/asr/models?server=...` **queries the Riva server** for the list of available model names (from `GetRivaSpeechRecognitionConfig`). The app does not configure the server — it only reads the list. The response uses the Silero VAD model as `default_model` when it appears in that list (so the UI suggests it); otherwise it uses the first model in the list or the backend default.

3. **UI**  
   When you open the Voice tab or load config, the frontend calls `/api/asr/models`, sets the ASR model dropdown to the returned `default_model`, and saves the chosen model in session config (and in `config.asr.model` / `asr_model_name` in saved sessions).

4. **Session files**  
   The chosen model is stored in each session's `config.asr.model` and `config.asr_model_name`. If your Riva server was deployed without Silero VAD, the list will only contain the base model and sessions will use that; fix the server config (config.sh) and redeploy, then new sessions will see and use the Silero VAD model.

## Checking your setup

- **Riva server**: Ensure the Silero VAD model is deployed and listed by `GetRivaSpeechRecognitionConfig` (the same list returned to the app via `/api/asr/models`).
- **UI**: After connecting to Riva, the ASR model dropdown should show "Default model from RIVA server." when the selected value is the API's `default_model` (the Silero VAD model when available).
- **Saved session**: In the session JSON, `config.asr.model` (and `asr_model_name`) should be  
  `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer` for multi-turn use.

## See also

- [ASR_MISSING_UTTERANCES_VAD.md](ASR_MISSING_UTTERANCES_VAD.md) — VAD and missed first utterances
- [ASR_PHANTOM_PARTIAL_WHY_LIVE_RIVA_DIFFERS.md](ASR_PHANTOM_PARTIAL_WHY_LIVE_RIVA_DIFFERS.md) — partial vs final behavior
