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
- `openai-beta` supports preview-schema providers such as Speaches 0.9.0-rc.3.

## Speaches release-candidate setup

Speaches is a conformance provider for this path; it is not hard-coded in the
MMAS backend. Start the pinned dual-SM Jetson image and its default models with:

```bash
./scripts/speaches_speech.sh start
./scripts/speaches_speech.sh verify
```

The launcher sets the required loopback URL and pins
`ghcr.io/nvidia-ai-iot/speaches:0.9.0-rc.3-cu130-sm87-sm110-auto`. See
[Speaches on Jetson](setup_speaches_jetson.md) for the complete recommended
quick start. The Realtime preset uses:

- Realtime WebSocket chunk transcription for Faster-Whisper ASR;
- a separately managed OpenAI-compatible LLM stage;
- REST `/v1/audio/speech` for Kokoro TTS.

Speaches 0.9.0-rc.3 emits speech start/end and final transcripts, but
Faster-Whisper is invoked on completed VAD chunks rather than decoding the
entire stream token by token. It may therefore not emit incremental transcript
deltas. MMAS displays partials whenever a compatible provider sends them.

## Provider contract smoke test

The same command tests Speaches or another compatible provider with a PCM16
mono WAV:

```bash
python scripts/test_realtime_transcription.py sample.wav \
  --url 'ws://127.0.0.1:18080/v1/realtime?intent=transcription' \
  --api-style openai-beta \
  --model Systran/faster-whisper-tiny.en \
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
with a separately managed OpenAI Chat Completions LLM and the Magpie exact-text
Realtime adapter. The adapter returns completed Magpie text chunks as PCM
deltas and supports delivery cancellation; the underlying `do_tts()` call is
still phrase-level generation rather than model-native incremental waveform
synthesis.
