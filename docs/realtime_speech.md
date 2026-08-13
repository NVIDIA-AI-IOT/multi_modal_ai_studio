# Realtime speech backends

Multi-modal AI Studio (MMAS) separates the Realtime wire protocol from the
speech model provider. A Realtime transcription session can therefore replace
the ASR stage in the classic `ASR -> LLM -> TTS` cascade without coupling MMAS
to OpenAI, Speaches, or a future NVIDIA Open Speech service.

## Supported paths

| MMAS mode | Input | Output | Use |
| --- | --- | --- | --- |
| Realtime transcription | streaming PCM over WebSocket | partial/final transcripts and speech boundaries | ASR stage in the classic cascade |
| Full Realtime voice | streaming PCM/text over WebSocket | streaming response audio and transcripts | provider-hosted speech-to-speech |
| REST speech | WAV/text HTTP requests | final transcript or synthesized audio | file-style ASR and exact-text TTS |

The Realtime client also exposes generic text-item, response-audio, and cancel
operations. Realtime response audio is not the same contract as exact-text
TTS. The latter continues to use `POST /v1/audio/speech`.

Two wire formats are selectable:

- `openai-ga` uses the current typed session and nested `audio.input` schema.
- `openai-beta` supports preview-schema providers such as Speaches 0.8.x.

## Speaches reference setup

Speaches is a conformance provider for this path; it is not hard-coded in the
MMAS backend. Speaches 0.8.x must be able to call its own REST transcription
endpoint from the Realtime server. Set its loopback URL when starting the
container:

```bash
docker run --rm --runtime nvidia --ipc host \
  -p 18080:8000 \
  -v "$HOME/.cache/mmas-speaches:/data" \
  -e LOOPBACK_HOST_URL=http://127.0.0.1:8000 \
  -e WHISPER__INFERENCE_DEVICE=cuda \
  -e WHISPER__COMPUTE_TYPE=float16 \
  ghcr.io/nvidia-ai-iot/speaches:0.8.3-cu130-sm87-sm110-auto
```

Use `presets/speaches-realtime-asr-jetson.yaml` after downloading the ASR and
TTS models listed in that preset. The preset uses:

- Realtime WebSocket transcription for Faster-Whisper ASR;
- the existing OpenAI-compatible LLM stage;
- REST `/v1/audio/speech` for Kokoro TTS.

Speaches 0.8.x emits speech start/end and final transcripts, but may not emit
incremental transcript deltas. It can also report `prefix_padding_ms` as
read-only after requiring that field in the session request; MMAS treats that
specific compatibility notice as a warning. MMAS will display partials whenever
a compatible provider sends them.

## Provider contract smoke test

The same command tests Speaches or another compatible provider with a PCM16
mono WAV:

```bash
python scripts/test_realtime_transcription.py sample.wav \
  --url 'ws://127.0.0.1:18080/v1/realtime?intent=transcription' \
  --api-style openai-beta \
  --model Systran/faster-whisper-small \
  --language en
```

For an OpenAI GA-compatible service, select `--api-style openai-ga` and put its
credential in `OPENAI_API_KEY`.

## NVIDIA Open Speech Realtime ASR

The bundled NVIDIA Open Speech service now presents this Realtime transcription
contract for `nvidia/nemotron-3.5-asr-streaming-0.6b`:

```bash
python scripts/test_realtime_transcription.py sample.wav \
  --url 'ws://127.0.0.1:8081/v1/realtime' \
  --api-style openai-ga \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --language en-US
```

Unlike a cumulative-window adapter, this service uses the model's native
cache-aware feature generator and RNNT token streamer. It emits
`speech_started`, transcript `delta`, `speech_stopped`, and transcription
`completed` events. The service accepts 16 kHz or 24 kHz mono PCM16 input and
uses its lightweight energy VAD for server-side utterance boundaries.

Load `presets/nvidia-open-models-realtime-speech-jetson.yaml` to use this ASR
with the existing OpenAI Chat Completions LLM and Magpie REST TTS stages.
Realtime response audio and exact-text streaming TTS are separate follow-up
contracts; this ASR milestone does not claim either one.
