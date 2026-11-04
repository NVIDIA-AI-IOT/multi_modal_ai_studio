# Speech model selection and installation

OpenAI-compatible model discovery is not a standard model-installation API. MMAS lists models reported by the configured provider; registry candidates are not a promise that every model has been tested on the current device.

In the REST ASR tab, **Reload** refreshes the available/downloaded model list. It does not download weights, restart a container, or load a model into GPU memory. **Downloaded** means the provider reports model files in its server-side cache, not that GPU inference is already warm.

When the provider identifies itself as Speaches and advertises the model download operation in its OpenAPI schema, selecting an undownloaded ASR model exposes a **Download** button. Downloading requires an explicit confirmation, consumes server-side storage and network bandwidth, and may take several minutes. Completion refreshes the list. A failed or timed-out request is not reported as a successful installation; use Reload to check the resulting cache state. Providers without this capability keep model selection and Reload, but no Download button.

Speaches 0.9.0-rc.3 checks that model metadata is installed locally before transcription. Do not assume that Faster-Whisper's underlying download behavior makes an uninstalled model usable through Speaches. Install it before starting a conversation, either with the button or the provider's API:

```bash
curl --fail --show-error -X POST \
  'http://localhost:18080/v1/models/Systran%2Ffaster-whisper-small'
```

The files are stored by Speaches, not by MMAS. The standard launcher persists them in its Docker model volume. First inference can still take longer while weights are loaded into memory. For an offline demonstration, download and warm up all selected models while network access is available.

For regression testing, choose a small model that is not installed, verify it is absent from `GET /v1/models`, select it without downloading, and confirm no download starts. Explicitly download it, verify it appears in the local list, and transcribe a fixed short WAV. Preserve existing model caches rather than deleting them to manufacture a cold start. Also test an unsupported provider, download failure, and repeated clicks.
