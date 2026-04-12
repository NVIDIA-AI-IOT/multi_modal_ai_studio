---
name: riva-speech
description: >
  Set up and configure NVIDIA Riva for speech-to-text (ASR) and text-to-speech (TTS)
  in Multi-modal AI Studio. Covers Riva installation, model discovery, health checks,
  ASR streaming with VAD tuning, TTS voice selection, and chunked audio streaming.
  Use when the user needs real-time speech recognition or synthesis on Jetson or GPU platforms.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
---

# Riva Speech (ASR + TTS)

Real-time speech-to-text and text-to-speech using NVIDIA Riva.

## Overview

Multi-modal AI Studio uses NVIDIA Riva for both directions of the voice pipeline:

- **ASR (Speech-to-Text)**: Streaming speech recognition with Parakeet models, VAD, partial/final transcripts
- **TTS (Text-to-Speech)**: Streaming synthesis with chunked output — the user hears the response before it is fully generated

Both services run on the same Riva gRPC endpoint (default `localhost:50051`).

## Prerequisites

- NVIDIA Riva installed and running on `localhost:50051` with ASR and TTS enabled
  - **First-time setup**: Follow [setup_riva.md](../../../docs/setup_riva.md) for full installation (NGC CLI, model download, Docker setup)
  - Verify: `docker ps | grep riva-speech`
- Recommended ASR model: `parakeet-1.1b-en-US-asr-streaming` (Riva 2.24.0 ARM64)

## Instructions

### Check Riva health

```bash
curl -s "http://localhost:8092/api/health/riva?server=localhost:50051"
# {"status": "ok"}
```

### Discover ASR models

```bash
curl -s "http://localhost:8092/api/asr/models?server=localhost:50051" | python3 -m json.tool
```

Response:
```json
{
  "models": ["parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"],
  "default_model": "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"
}
```

### Discover TTS voices

```bash
curl -s "http://localhost:8092/api/tts/voices?server=localhost:50051" | python3 -m json.tool
```

Response (truncated):
```json
[
  {"voice_name": "English-US.Female-1", "language": "en-US"},
  {"voice_name": "English-US.Male-1", "language": "en-US"}
]
```

### Configure via CLI

```bash
python -m multi_modal_ai_studio \
  --asr-server localhost:50051 \
  --asr-model parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer \
  --asr-language en-US \
  --tts-server localhost:50051 \
  --tts-voice "English-US.Female-1"
```

### Configure via session config

```json
{
  "asr": {
    "scheme": "riva",
    "server": "localhost:50051",
    "model": "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
    "language": "en-US",
    "vad_start_threshold": 0.5,
    "vad_stop_threshold": 0.3,
    "speech_pad_ms": 600,
    "speech_timeout_ms": 1200
  },
  "tts": {
    "scheme": "riva",
    "server": "localhost:50051",
    "voice": "",
    "sample_rate": 22050,
    "stream_tts": true,
    "tts_chunk_words": 10
  }
}
```

### Audio format (ASR input)

The voice WebSocket (`/ws/voice`) expects:
- **Format**: 16-bit signed integer, little-endian (S16_LE)
- **Sample rate**: 16,000 Hz
- **Channels**: 1 (mono)
- **Delivery**: Raw PCM as binary WebSocket frames

### Chunked TTS streaming

MMAS splits LLM output into chunks of `tts_chunk_words` words (default: 10) and sends each chunk to Riva TTS immediately. The user starts hearing the response after the first ~10 words, rather than waiting for the full response.

### Server USB microphone and speaker

For server-attached USB devices (e.g., on Jetson headless deployments):

```bash
# List server microphones
curl -s http://localhost:8092/api/devices/audio-inputs | python3 -m json.tool

# List server speakers
curl -s http://localhost:8092/api/devices/audio-outputs | python3 -m json.tool
```

MMAS uses `plughw:` ALSA devices for automatic sample rate conversion (e.g., USB mics at 48kHz → 16kHz for Riva).

## Input Schema

### ASR configuration

- `scheme` (string, required): `"riva"`
- `server` (string, required): Riva gRPC address (e.g., `"localhost:50051"`)
- `model` (string, optional): ASR model name
- `language` (string, optional, default: `"en-US"`): Language code
- `vad_start_threshold` (float, optional, default: 0.5): VAD sensitivity for speech start (0.0-1.0)
- `vad_stop_threshold` (float, optional, default: 0.3): VAD sensitivity for speech end
- `speech_pad_ms` (integer, optional, default: 600): Padding after speech detection (ms)
- `speech_timeout_ms` (integer, optional, default: 1200): Silence duration to end utterance (ms)

### TTS configuration

- `scheme` (string, required): `"riva"`
- `server` (string, required): Riva gRPC address (e.g., `"localhost:50051"`)
- `voice` (string, optional): Voice name (empty string for default)
- `sample_rate` (integer, optional, default: 22050): Audio sample rate in Hz
- `stream_tts` (boolean, optional, default: true): Enable streaming synthesis
- `tts_chunk_words` (integer, optional, default: 10): Words per TTS chunk

## Output Schema

### ASR events (via `/ws/voice` WebSocket)

```json
{
  "type": "event",
  "data": {
    "type": "asr_partial",
    "text": "what is the color of",
    "timestamp": 2.34
  }
}
```

```json
{
  "type": "event",
  "data": {
    "type": "asr_final",
    "text": "What is the color of my shirt?",
    "confidence": 0.92,
    "timestamp": 3.12
  }
}
```

### TTS events (via `/ws/voice` WebSocket)

```json
{
  "type": "tts_start",
  "text": "The worker is wearing a hard hat."
}
```

```json
{
  "type": "tts_audio",
  "data": "<base64 PCM audio, 22050Hz mono 16-bit LE>"
}
```

## Guidelines

- **Riva must be initialized** with `riva_init.sh` before first use (downloads models, 15-45 min)
- First startup after `riva_start.sh` takes 2-5 minutes to load models into GPU memory
- Lower `vad_start_threshold` to detect softer speech; raise it to reduce false triggers
- Increase `speech_timeout_ms` for users who pause mid-sentence
- Leave `voice` empty to use the Riva default voice
- Lower `tts_chunk_words` (e.g., 5) for faster time-to-first-audio but more synthesis calls
- Riva TTS supports multilingual voices with `language: "multi"` in Riva config
