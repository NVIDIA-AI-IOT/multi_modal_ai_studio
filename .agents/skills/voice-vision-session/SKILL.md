---
name: voice-vision-session
description: >
  Deploy and run a live voice + vision AI session using Multi-modal AI Studio.
  Real-time pipeline: Riva ASR (speech-to-text) → VLM/LLM (reasoning) → Riva TTS (text-to-speech)
  with optional camera or video input. Runs on NVIDIA Jetson or desktop GPU.
  Use when the user wants a conversational AI assistant that can see and speak.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
---

# Voice + Vision Session

Launch a real-time conversational AI session with speech and optional vision input.

## Overview

Multi-modal AI Studio provides a full voice+vision AI pipeline:

```
Microphone → Riva ASR → LLM/VLM → Riva TTS → Speaker
                           ↑
                    Camera / Video frames
```

The server exposes a WebUI at `https://<host>:8092` and a WebSocket-based voice pipeline at `/ws/voice`.

## Prerequisites

- Python 3.12+ with `multi_modal_ai_studio` installed (`pip install -e .`)
- NVIDIA Riva running on `localhost:50051` (ASR + TTS)
- An OpenAI-compatible LLM/VLM backend (vLLM, Ollama, TensorRT Edge LLM, or OpenAI API)
- GPU with sufficient VRAM for the chosen model

## Instructions

### Step 1: Start the server with a preset

```bash
# Cosmos-Reason2 on vLLM (vision + video input)
python -m multi_modal_ai_studio --preset cosmos-reason --host 0.0.0.0

# Cosmos-Reason2 on TensorRT Edge LLM (optimized edge inference)
python -m multi_modal_ai_studio --preset tensorrt-edge-cosmos --host 0.0.0.0

# Text-only with Ollama
python -m multi_modal_ai_studio --preset text-only --host 0.0.0.0

# Custom configuration via CLI
python -m multi_modal_ai_studio \
  --asr-server localhost:50051 \
  --llm-api-base http://localhost:8010/v1 \
  --llm-model nvidia/cosmos-reason2-8b-fp8 \
  --host 0.0.0.0 --port 8092
```

### Step 2: Open the WebUI

Navigate to `https://<host>:8092` in a browser. The UI provides:
- Configuration panel (ASR, LLM, TTS, device settings)
- Camera preview (browser or server USB camera)
- Microphone waveform indicator
- START/STOP session controls
- Live transcript and timeline visualization

### Step 3: Start a session

Click START in the WebUI. The voice pipeline activates:
1. Browser or server microphone captures audio
2. Audio streams to Riva ASR via WebSocket (`/ws/voice`)
3. Transcripts route to the configured LLM/VLM
4. If vision is enabled, camera frames are captured and sent with the prompt
5. LLM response streams back through Riva TTS
6. Synthesized audio plays in the browser or server speaker

### Step 4: Stop and review

Click STOP. The session is auto-saved as a JSON file in `./sessions/`. The timeline shows all ASR, LLM, and TTS events with latency metrics.

## WebSocket Protocol (`/ws/voice`)

For programmatic integration, connect to the voice WebSocket directly:

```python
import asyncio
import websockets
import json

async def run_session():
    uri = "wss://localhost:8092/ws/voice"
    async with websockets.connect(uri, ssl=True) as ws:
        # Send config as first message
        config = {
            "type": "config",
            "config": {
                "asr": {"scheme": "riva", "server": "localhost:50051"},
                "llm": {
                    "api_base": "http://localhost:8010/v1",
                    "model": "nvidia/cosmos-reason2-8b-fp8",
                    "enable_vision": True,
                    "vision_video_encode": True
                },
                "tts": {"scheme": "riva", "server": "localhost:50051"}
            }
        }
        await ws.send(json.dumps(config))

        # Send start_session
        await ws.send(json.dumps({"type": "start_session"}))

        # Stream 16kHz mono 16-bit LE PCM audio as binary frames
        # Receive JSON events: transcripts, TTS audio, timeline events
        async for msg in ws:
            if isinstance(msg, str):
                event = json.loads(msg)
                print(event.get("type"), event.get("data", {}).get("text", ""))
```

## Input Schema

- `preset` (string, optional): Preset name (`cosmos-reason`, `tensorrt-edge-cosmos`, `low-latency`, `text-only`)
- `host` (string, default: `0.0.0.0`): Server bind address
- `port` (integer, default: `8092`): Server port
- `config` (object, optional): Full `SessionConfig` object (asr, llm, tts, devices, app sections)

## Output Schema

Session JSON saved to `./sessions/<uuid>.json`:
- `session_id` (string): UUID
- `config` (object): Full config used
- `turns` (array): Each turn contains `user_transcript`, `ai_response`, `latencies`
- `timeline` (array): Timestamped ASR/LLM/TTS events
- `metrics` (object): Aggregated latency statistics

## Guidelines

- Ensure Riva is running before starting (`docker ps | grep riva-speech`)
- Ensure the LLM backend is healthy: `curl http://localhost:8010/v1/models`
- For vision, the LLM must support image/video input (e.g., Cosmos-Reason2, LLaVA, GPT-4V)
- Set `vision_video_encode: true` for models that accept MP4 video (Cosmos-Reason2)
- Default port is 8092 (avoids conflict with other services)
- Sessions are auto-saved on stop; find them in `./sessions/`
