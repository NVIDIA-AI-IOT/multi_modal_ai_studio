<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Speech API backends

Multi-modal AI Studio can mix speech providers independently. The classic voice
pipeline is:

```text
ASR -> OpenAI-compatible LLM -> TTS
```

ASR can use Riva gRPC, an OpenAI-compatible REST service, or a Realtime
transcription WebSocket. TTS can use Riva gRPC or OpenAI-compatible REST. The
LLM uses an OpenAI-compatible Chat Completions endpoint.

| Component | MMAS scheme | Required endpoint | Streaming behavior |
|---|---|---|---|
| ASR | `openai-rest` | `POST /v1/audio/transcriptions` | MMAS performs local VAD and sends one WAV per utterance |
| ASR | `openai-realtime` (`transcription`) | Realtime WebSocket | Streaming PCM, speech boundaries, partial/final transcripts when provided |
| LLM | `openai` | `POST /v1/chat/completions` | Server-sent streaming supported |
| TTS | `openai-rest` | `POST /v1/audio/speech` | Response body is read incrementally when the provider streams it |
| Full voice | `openai-realtime` (`full`) | OpenAI Realtime WebSocket | Bidirectional audio and events |
| ASR/TTS | `riva` | Riva gRPC | Native streaming |

## Local OpenAI-compatible speech

For a service on the same machine, no API key is required:

```yaml
asr:
  scheme: openai-rest
  api_base: http://localhost:8081/v1
  model: nvidia/nemotron-3.5-asr-streaming-0.6b
  language: en-US

tts:
  scheme: openai-rest
  api_base: http://localhost:8082/v1
  model: nvidia/magpie_tts_multilingual_357m
  language: en-US
  voice: Sofia
  sample_rate: 22050
  response_format: pcm
```

The bundled service adds a `language` request field to the standard speech
request so multilingual local checkpoints can be selected explicitly. Standard
clients that omit it continue to work and default to English.

The public Jetson quickstart and primary qualification use English. Japanese
and other supported languages are a second-stage multilingual qualification;
set both ASR and TTS `language` fields together when running those tests.

### TTS streaming boundary

MMAS interleaves LLM generation and TTS by sending short text chunks before
the LLM completes. It also forwards PCM response chunks to the browser as soon
as they arrive. These are separate from model-native audio streaming.

The bundled public Magpie v2607 service currently uses
`MagpieTTSModel.do_tts()`. That path generates all audio codes for one text
chunk and calls `codes_to_audio()` once, so its HTTP time-to-first-byte is
effectively the completion time for that chunk. Smaller text chunks reduce
time to first audio, but this is phrase-level interleaving rather than
incremental waveform generation.

## Interoperability boundary

The REST endpoints make model replacement easy, but file transcription is not
a native streaming protocol. MMAS buffers microphone audio until local VAD
detects end-of-utterance, then calls `/audio/transcriptions`. This is a good
portable baseline for evaluation. Use a Realtime API or a native streaming
adapter when partial transcripts and sub-utterance latency are required.

See [Open models on Jetson](setup_open_models_jetson.md) for the public
Nemotron 3.5 ASR and Magpie deployment. See [Realtime speech](realtime_speech.md)
for wire formats, Speaches setup, and the provider contract smoke test.
