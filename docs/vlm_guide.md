# VLM (Vision-Language Model) Guide

This guide covers how Vision-Language Models work within Multi-modal AI Studio — from backend setup and frame capture to video/image input modes, conversation history, performance tuning, and the API formats used under the hood.

The examples use **Cosmos-Reason2-8B** as the reference model, but the application works with any OpenAI-compatible VLM backend (vLLM, Ollama, OpenAI, etc.).

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Browser                             │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐│
│  │   Mic    │  │  Camera  │  │   Speaker (TTS audio)  ││
│  └────┬─────┘  └────┬─────┘  └───────────▲────────────┘│
│       │ PCM audio   │ JPEG frames        │              │
└───────┼─────────────┼────────────────────┼──────────────┘
        │ WebSocket   │                    │
        ▼             ▼                    │
┌───────────────────────────────────────────┐
│        Multi-modal AI Studio              │
│  ┌────────┐  ┌────────┐  ┌────────┐      │
│  │  ASR   │→│  VLM   │→│  TTS   │──────┘│
│  │ (Riva) │  │(vLLM)  │  │(Riva) │       │
│  └────────┘  └────────┘  └────────┘      │
│      ↑          ↑                        │
│    :50051    :8003                       │
└───────────────────────────────────────────┘
        ↓          ↓
┌────────────┐ ┌────────────────────────────┐
│   Riva     │ │   VLM Backend (vLLM,      │
│ Container  │ │   Ollama, OpenAI, etc.)    │
└────────────┘ └────────────────────────────┘
```

## Prerequisites

- **Hardware**: NVIDIA GPU (Jetson Thor / AGX Orin for edge, or discrete GPU)
- **Containers**: NVIDIA Riva (for ASR/TTS), a VLM inference backend
- **Camera**: Browser webcam or USB camera
- **Memory**: 16GB+ GPU memory for 8B models

---

## Step 1: Start a VLM Backend

Multi-modal AI Studio works with any backend that exposes an OpenAI-compatible `/v1/chat/completions` endpoint with image or video support.

### Option A: vLLM with Cosmos-Reason2 (Recommended)

For full setup instructions including model download, Docker images, and platform-specific commands, see:

**[Cosmos-Reason2-8B on Jetson AI Lab](https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/)**

Quick reference for Jetson Thor (after downloading the FP8 model per the link above):

```bash
export MODEL_PATH="${HOME}/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8"

sudo docker run -it --rm --runtime=nvidia --network host \
  -v $MODEL_PATH:/models/cosmos-reason2-8b:ro \
  ghcr.io/nvidia-ai-iot/vllm:0.14.0-r38.3-arm64-sbsa-cu130-24.04 \
  vllm serve /models/cosmos-reason2-8b \
    --served-model-name nvidia/cosmos-reason2-8b-fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --reasoning-parser qwen3 \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --enable-prefix-caching \
    --port 8000
```

> **`--reasoning-parser qwen3`**: Enables server-side parsing of `<think>...</think>` chain-of-thought tokens, separating reasoning from the final answer in the API response.
>
> **`--media-io-kwargs '{"video": {"num_frames": -1}}'`**: Processes all video frames instead of sampling a subset.
>
> **`--enable-prefix-caching`**: Reuses the KV cache for identical request prefixes, reducing first-token latency on repeated video queries.
>
> **Memory tuning**: On shared-memory systems (Jetson), lower `--gpu-memory-utilization` to leave room for the OS, Riva, and the application. On discrete GPUs with dedicated VRAM, `0.8` is safe. If you hit OOM, lower `--max-model-len` to `4096`.

> **Note (Desktop GPU / x86_64)**: Use `vllm/vllm-openai:latest` or `nvcr.io/nvidia/vllm:latest` instead of the Jetson image. See [vLLM documentation](https://docs.vllm.ai/) for details.

### Option B: Ollama (Easiest Setup)

Ollama supports vision models with minimal configuration:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a vision model
ollama pull llava-llama3    # ~5GB, good quality
# or
ollama pull gemma3:4b       # ~3GB, multimodal
```

Configure in the UI:

| Setting | Value |
|---------|-------|
| **API Base** | `http://localhost:11434/v1` |
| **Model** | `llava-llama3` |
| **Enable Vision** | Checked |

### Option C: OpenAI API

No local container needed — uses OpenAI's API directly:

```yaml
llm:
  scheme: openai
  api_base: https://api.openai.com/v1
  api_key: sk-your-openai-key
  model: gpt-4o
  enable_vision: true
```

### Verify Your Backend

```bash
# Check the API responds
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Health check (vLLM)
curl -s http://localhost:8000/health && echo "READY" || echo "NOT READY"
```

---

## Step 2: Start NVIDIA Riva (ASR + TTS)

Riva provides speech recognition and text-to-speech. See [setup_riva.md](setup_riva.md) for full installation and configuration instructions.

Verify Riva is running on port `50051` before proceeding:
```bash
docker ps | grep riva
```

---

## Step 3: Run Multi-modal AI Studio

### Install (first time only):

```bash
cd multi_modal_ai_studio
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run the Application:

```bash
# With the Cosmos-Reason preset
python -m multi_modal_ai_studio --preset cosmos-reason --host 0.0.0.0

# Or with explicit CLI options
python -m multi_modal_ai_studio \
  --host 0.0.0.0 \
  --llm-api-base http://localhost:8000/v1 \
  --llm-model nvidia/cosmos-reason2-8b-fp8

# Debug mode: save encoded videos to disk for inspection
MMAS_DEBUG_VIDEOS=1 python -m multi_modal_ai_studio --preset cosmos-reason --host 0.0.0.0
```

> **`--host 0.0.0.0`** is required to access the UI from another machine (not just localhost).
>
> **`MMAS_DEBUG_VIDEOS=1`** saves every MP4 video sent to the VLM for offline inspection.

Open your browser: **https://localhost:8092**

> **HTTPS Note**: The app uses HTTPS with a self-signed certificate (required for browser camera/mic access). On first visit, click **"Advanced" -> "Proceed"** to accept the certificate.

---

## Step 4: Configure Vision in the UI

1. **Open Config Panel** (gear icon or "Config" tab)

2. **LLM Settings**:
   - Set **API Base** to your backend URL (e.g., `http://localhost:8000/v1`)
   - Select your **Model** from the dropdown

3. **Enable Vision**:
   - Check **"Enable Vision (VLM)"**
   - Choose **Vision Input Mode**: "Video Input" (for Cosmos) or "N-Image Input" (for other VLMs)

4. **Adjust Settings** (optional):

   | Setting | Default | Description |
   |---------|---------|-------------|
   | Frames per Turn | 4 | Number of frames captured per speech turn |
   | Quality | 0.7 | JPEG compression (0.3-1.0) |
   | Max Width | 640 | Frame width in pixels |

5. **Start Session**: Click **"Start Live"** and allow camera/microphone access

---

## How Frame Capture Works

When you speak, the system:

1. **Continuously captures** JPEG frames from your camera into a ring buffer
2. **On speech end** (ASR final), retrieves frames from the speech time window
3. Depending on the configured **Vision Input Mode**:
   - **Video Input**: encodes frames into an H.264 MP4 video, sent as a single `video_url`
   - **N-Image Input**: selects N evenly-spaced frames, sent as individual `image_url` entries

### Video Input Mode

Models like Cosmos-Reason2 are optimized for video input and can decode temporal frame deltas, using fewer tokens than equivalent individual images. The FPS is calculated dynamically from the speech duration:

```
Speech: "What did I just do?"
        |<-------- 3 seconds -------->|
        t_start                    t_end

Frames retrieved from ring buffer (e.g. 30 frames @ 10fps)
  -> Encoded into H.264 MP4 @ fps = 30/3 = 10fps
  -> Sent as single video_url
  -> Encoding overhead: ~100-200ms
```

### N-Image Input Mode

```
Speech: "What am I holding?"
        |<-------- 2 seconds -------->|
        Start                        End

Frames per Turn = 4:
        |-------|-------|-------|------|
       Frame1  Frame2  Frame3  Frame4
        @0.5s   @1.0s   @1.5s   @2.0s
```

---

## Default VLM Settings

VLM settings are configured in the UI or via presets. Defaults are defined in `config/schema.py`:

```python
enable_vision: bool = False
vision_frames: int = 4           # Frames per turn
vision_quality: float = 0.7      # JPEG quality (0.3-1.0)
vision_max_width: int = 640      # Max frame width
vision_buffer_fps: float = 3.0   # Ring buffer capture rate
vision_video_encode: bool = False # Video input mode
```

### Configuration Methods

| Method | Use Case |
|--------|----------|
| **UI Config Panel** | Interactive adjustment, experimentation |
| **CLI arguments** | Scripting, automation |
| **Preset YAML** | Reproducible deployments |

### Creating a Custom Preset

Create a preset file (e.g., `presets/my-vlm.yaml`):

```yaml
name: "My VLM Setup"
description: "Custom VLM configuration"

asr:
  scheme: riva
  server: localhost:50051

llm:
  scheme: openai
  api_base: http://localhost:8000/v1
  model: nvidia/cosmos-reason2-8b-fp8
  enable_vision: true
  vision_video_encode: true
  temperature: 0.7
  max_tokens: 512

tts:
  scheme: riva
  server: localhost:50051
```

Run with: `python -m multi_modal_ai_studio --preset my-vlm`

---

## Conversation History

By default, conversation history is **not passed** for VLM turns. Each turn is independent — the VLM sees only the current video/frames and text.

> **Why?** When text-only history from prior turns is included (without their original images), models tend to anchor on old answers instead of analyzing the current visual input. Without history, each turn gets a fresh analysis.

This behavior can be toggled in the UI if your use case benefits from multi-turn follow-ups (e.g., "How about this?").

---

## Performance Tuning

| Goal | Settings |
|------|----------|
| **Fastest response** | `vision_frames: 1`, `max_tokens: 64` |
| **Better understanding** | `vision_frames: 4-6`, `max_tokens: 128` |
| **Motion analysis** | `vision_frames: 8-10`, `vision_buffer_fps: 5.0` |

---

## Troubleshooting

### Backend Issues

| Issue | Solution |
|-------|----------|
| Container won't start | Check GPU memory: `nvidia-smi` |
| CUDA out of memory | Reduce `--max-model-len` or lower `--gpu-memory-utilization` |
| Model download fails | Check disk space, HuggingFace token, network |
| GPU memory not released | Stop container fully, wait, verify with `nvidia-smi`, then restart |

### Camera Issues

| Issue | Solution |
|-------|----------|
| Camera not detected | Use HTTPS or localhost (not HTTP + IP) |
| Permission denied | Check browser camera permissions |
| Black frames | Ensure camera isn't used by another app |

### VLM Response Issues

| Issue | Solution |
|-------|----------|
| Slow responses | Reduce `vision_frames`, lower `max_tokens` |
| Generic answers | Lower `temperature` (try 0.3), improve system prompt |
| "I can't see" errors | Check `enable_vision: true` and camera is working |
| VLM hallucinates scenes | VLMs generate visual descriptions even without images — ensure camera is active |

### ASR Issues

See [setup_riva.md](setup_riva.md) for ASR troubleshooting.

---

## API Format Reference

### Video Input (Cosmos-Reason2, etc.)

```json
{
  "model": "nvidia/cosmos-reason2-8b-fp8",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIG..."}},
        {"type": "text", "text": "What am I doing?"}
      ]
    }
  ],
  "temperature": 0.3,
  "max_tokens": 256
}
```

### Multi-Image Input (LLaVA, Ollama VLMs, etc.)

```json
{
  "model": "llava-llama3",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "What am I doing?"}
      ]
    }
  ],
  "max_tokens": 128
}
```

---

## Further Reading

- [Cosmos-Reason2-8B on Jetson AI Lab](https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/)
- [NVIDIA Cosmos-Reason2 Model Card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- [vLLM Documentation](https://docs.vllm.ai/)
