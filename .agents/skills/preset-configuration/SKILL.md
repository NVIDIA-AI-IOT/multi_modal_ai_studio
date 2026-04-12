---
name: preset-configuration
description: >
  Create, load, and manage YAML configuration presets for Multi-modal AI Studio.
  Presets define the full ASR + LLM + TTS + device stack in a single file.
  Use when the user wants to save, share, or switch between different deployment configurations.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
---

# Preset Configuration

Manage YAML presets that define the full Multi-modal AI Studio stack.

## Overview

A preset is a YAML file that configures all components (ASR, LLM/VLM, TTS, devices, app settings) in one place. Presets can be loaded at startup via CLI or at runtime via the API.

## Built-in Presets

| Preset | LLM Backend | Vision | Use Case |
|--------|------------|--------|----------|
| `cosmos-reason` | vLLM (Cosmos-Reason2 FP8) | Video encode | Production VLM on Jetson |
| `tensorrt-edge-cosmos` | TensorRT Edge LLM | Video encode | Optimized edge inference |
| `default` | Ollama (llama3.2:3b) | Off | General text conversation |
| `low-latency` | Ollama | Off | Fast responses |
| `high-accuracy` | Ollama | Off | Detailed responses |
| `text-only` | Ollama | Off | No voice, text chat only |
| `openai-realtime` | OpenAI Realtime API | Off | Cloud-based voice |
| `llm-router` | Remote MoM model | Video encode | Hybrid edge+cloud routing |

## Instructions

### Load a preset at startup

```bash
python -m multi_modal_ai_studio --preset cosmos-reason --host 0.0.0.0
```

### List presets via API

```bash
curl -s http://localhost:8092/api/presets | python3 -m json.tool
```

### Get a specific preset

```bash
curl -s http://localhost:8092/api/presets/cosmos-reason | python3 -m json.tool
```

### Create a new preset via API

```bash
curl -s -X POST http://localhost:8092/api/presets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-custom-preset",
    "config": {
      "asr": {
        "scheme": "riva",
        "server": "localhost:50051"
      },
      "llm": {
        "api_base": "http://localhost:8010/v1",
        "model": "nvidia/cosmos-reason2-8b-fp8",
        "enable_vision": true,
        "vision_video_encode": true,
        "vision_frames": 30,
        "vision_max_width": 768,
        "vision_buffer_fps": 5.0,
        "system_prompt": "You are a helpful voice AI assistant.",
        "vision_system_prompt": "You are a helpful voice and vision assistant. Give ONE short sentence answers only."
      },
      "tts": {
        "scheme": "riva",
        "server": "localhost:50051"
      }
    }
  }'
```

### Delete a preset

```bash
curl -s -X DELETE http://localhost:8092/api/presets/my-custom-preset
```

### Preset YAML format

```yaml
name: "My Custom Preset"
description: "Description shown in the UI"

asr:
  scheme: riva
  server: localhost:50051
  model: parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer
  language: en-US

llm:
  scheme: openai
  api_base: http://localhost:8010/v1
  model: nvidia/cosmos-reason2-8b-fp8
  temperature: 0.3
  max_tokens: 512
  system_prompt: "You are a helpful voice AI assistant."
  enable_vision: true
  vision_system_prompt: "You are a helpful voice and vision assistant."
  vision_frames: 30
  vision_video_encode: true

tts:
  scheme: riva
  server: localhost:50051
  voice: ""
  sample_rate: 22050
  stream_tts: true

devices:
  audio_input_source: browser
  audio_output_source: browser

app:
  barge_in_enabled: true
  session_auto_save: true
  session_output_dir: ./sessions
  theme: dark
```

## Input Schema

- `name` (string, required): Preset display name
- `config` (object, required): Full or partial SessionConfig with `asr`, `llm`, `tts`, `devices`, `app` sections

## Output Schema

Preset list:
```json
[
  {"name": "cosmos-reason", "description": "Cosmos-Reason2 on vLLM..."},
  {"name": "tensorrt-edge-cosmos", "description": "Cosmos-Reason2 on TensorRT Edge LLM..."}
]
```

## Guidelines

- Preset files are stored in the `presets/` directory as `<slug>.yaml`
- CLI `--preset` takes the filename without `.yaml` extension
- Presets can be partial — unspecified fields use defaults from `SessionConfig`
- The `extra_request_body` field accepts a JSON string for backend-specific options
- Use `vision_video_encode: true` only for models that accept MP4 video input
