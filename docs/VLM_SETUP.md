# VLM (Vision-Language Model) Setup Guide

This guide explains how to set up and run Vision-Language Models (VLMs) with Multi-modal AI Studio. VLMs can process both images and text, enabling visual understanding in voice conversations.

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
│   Riva     │ │   vLLM + Cosmos-Reason2    │
│ Container  │ │        Container           │
└────────────┘ └────────────────────────────┘
```

## Supported VLM Models

| Model | Provider | Use Case |
|-------|----------|----------|
| **Cosmos-Reason2-8B** | NVIDIA | Physical world reasoning, spatial understanding |
| **LLaVA** | Open Source | General vision-language tasks |
| **Qwen-VL** | Alibaba | Multi-modal understanding |
| **GPT-4V** | OpenAI | General vision (requires API key) |

## Prerequisites

- **Hardware**: NVIDIA GPU (Jetson AGX Orin recommended for edge, or dGPU)
- **Containers**: NVIDIA Riva (for ASR/TTS), vLLM (for VLM inference)
- **Camera**: Browser webcam or USB camera
- **Memory**: 16GB+ GPU memory for 8B models

---

## Step 1: Get the vLLM Docker Image

### Option A: Jetson (ARM64)

```bash
# Pull Jetson-optimized vLLM image
docker pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
```

### Option B: Desktop GPU (x86_64)

```bash
# Pull standard vLLM image
docker pull vllm/vllm-openai:latest
```

### Option C: From NVIDIA NGC

```bash
# Login to NGC (requires NGC API key)
docker login nvcr.io
docker pull nvcr.io/nvidia/vllm:latest
```

---

## Step 2: Start vLLM with Cosmos-Reason2

### For Jetson:

```bash
docker run -d --gpus all \
  --name vllm-cosmos \
  -p 8003:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Cosmos-Reason2-8B \
    --max-model-len 16384 \
    --port 8000
```

### For Desktop GPU:

```bash
docker run -d --gpus all \
  --name vllm-cosmos \
  -p 8003:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model nvidia/Cosmos-Reason2-8B \
  --max-model-len 16384 \
  --port 8000
```

> **Note**: First run downloads the model (~16GB). This may take several minutes.

### Verify vLLM is Running:

```bash
# Check container status
docker ps | grep vllm

# Test the API
curl http://localhost:8003/v1/models
```

Expected output:
```json
{"data":[{"id":"nvidia/Cosmos-Reason2-8B","object":"model",...}]}
```

---

## Step 3: Start NVIDIA Riva (ASR + TTS)

Riva provides speech recognition and text-to-speech. Follow the [Riva Quick Start](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html) or use your existing Riva deployment.

```bash
# Example: Start Riva (adjust path to your installation)
cd /path/to/riva-quickstart
bash riva_start.sh
```

Verify Riva is running on port 50051:
```bash
docker ps | grep riva
```

---

## Step 4: Run Multi-modal AI Studio

### Install (first time only):

```bash
cd multi_modal_ai_studio
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run the Application:

```bash
# Start with defaults
python -m multi_modal_ai_studio

# Or with CLI options
python -m multi_modal_ai_studio \
  --llm-api-base http://localhost:8003/v1 \
  --llm-model nvidia/Cosmos-Reason2-8B
```

Then enable vision in the UI (see Step 5).

Open your browser: **http://localhost:8000**

---

## Step 5: Configure VLM in the UI

1. **Open Config Panel** (click the gear icon or "Config" tab)

2. **LLM Settings**:
   - Verify **API Base**: `http://localhost:8003/v1`
   - Verify **Model**: `nvidia/Cosmos-Reason2-8B`

3. **Enable Vision**:
   - Check ✅ **"Enable Vision (VLM)"**
   
4. **Adjust VLM Settings** (optional):
   | Setting | Default | Description |
   |---------|---------|-------------|
   | Frames per Turn | 4 | Number of frames sent per request (1-10) |
   | Quality | 0.7 | JPEG compression (0.3-1.0) |
   | Max Width | 640 | Frame width in pixels |

5. **Start Session**:
   - Click **"Start Live"**
   - Allow camera and microphone access when prompted

---

## How VLM Frame Capture Works

When you speak, the system:

1. **Continuously captures** frames from your camera into a ring buffer
2. **On speech end** (ASR final), selects N frames evenly spaced across your speech duration
3. **Sends frames + text** to the VLM in OpenAI-compatible format
4. **VLM responds** with visual understanding

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

VLM settings are configured in the UI or via CLI. Defaults are defined in `schema.py`:

```python
# LLMConfig defaults (in config/schema.py)

# Text LLM system prompt (used when vision is disabled)
system_prompt: str = "You are a helpful voice assistant."

# VLM settings (used when enable_vision=True)
enable_vision: bool = False
vision_system_prompt: str = "You are a vision assistant. Give ONE short sentence answers only. Be direct. No explanations."
vision_frames: int = 4           # Frames per turn (1-10)
vision_quality: float = 0.7      # JPEG quality (0.3-1.0)
vision_max_width: int = 640      # Max frame width
vision_buffer_fps: float = 3.0   # Ring buffer capture rate
vision_detail: str = "auto"      # OpenAI vision detail level
```

### Configuration Methods

| Method | Use Case |
|--------|----------|
| **UI Config Panel** | Interactive adjustment, experimentation |
| **CLI arguments** | Scripting, automation |
| **Custom preset YAML** | Reproducible deployments |

### Creating a Custom Preset

Create your own preset file (e.g., `presets/my-vlm.yaml`):

```yaml
name: "My VLM Setup"
description: "Custom VLM configuration"

asr:
  scheme: riva
  server: localhost:50051

llm:
  scheme: openai
  api_base: http://localhost:8003/v1
  model: nvidia/Cosmos-Reason2-8B
  enable_vision: true
  vision_frames: 4
  system_prompt: "You are a vision assistant. Be concise."

tts:
  scheme: riva
  server: localhost:50051
```

Run with: `python -m multi_modal_ai_studio --preset presets/my-vlm.yaml`

---

## Performance Tuning

| Goal | Settings |
|------|----------|
| **Fastest response** | `vision_frames: 1`, `max_tokens: 64` |
| **Better understanding** | `vision_frames: 4-6`, `max_tokens: 128` |
| **Motion analysis** | `vision_frames: 8-10`, `vision_buffer_fps: 5.0` |

### Memory Considerations

| Model | GPU Memory | Frames |
|-------|------------|--------|
| Cosmos-Reason2-8B | ~16GB | 4-6 frames comfortable |
| Cosmos-Reason2-2B | ~8GB | Up to 10 frames |

---

## Troubleshooting

### VLM Container Issues

| Issue | Solution |
|-------|----------|
| Container won't start | Check GPU memory: `nvidia-smi` |
| Model download fails | Check disk space, HuggingFace access |
| CUDA out of memory | Reduce `--max-model-len` to 8192 |

### Camera Issues

| Issue | Solution |
|-------|----------|
| Camera not detected | Use HTTPS or localhost (not HTTP + IP) |
| Permission denied | Check browser camera permissions |
| Black frames | Ensure camera isn't used by another app |

### ASR Issues

| Issue | Solution |
|-------|----------|
| No speech detected | Lower VAD threshold to 0.2-0.3 |
| Cuts off early | Increase `speech_timeout_ms` to 1000 |
| Echo/feedback | Use headphones or reduce speaker volume |

### VLM Response Issues

| Issue | Solution |
|-------|----------|
| Slow responses | Reduce `vision_frames`, lower `max_tokens` |
| Generic answers | Improve system prompt, increase frames |
| "I can't see" errors | Check `enable_vision: true`, camera working |

---

## Using Other VLM Models

### LLaVA

```bash
docker run -d --gpus all \
  --name vllm-llava \
  -p 8003:8000 \
  vllm/vllm-openai:latest \
  --model liuhaotian/llava-v1.6-vicuna-7b \
  --max-model-len 4096
```

Update preset:
```yaml
llm:
  model: liuhaotian/llava-v1.6-vicuna-7b
  enable_vision: true
```

### OpenAI GPT-4V

No vLLM container needed - uses OpenAI API directly:

```yaml
llm:
  scheme: openai
  api_base: https://api.openai.com/v1
  api_key: sk-your-openai-key
  model: gpt-4-vision-preview
  enable_vision: true
```

---

## API Format Reference

VLM requests use OpenAI-compatible multi-modal format:

```json
{
  "model": "nvidia/Cosmos-Reason2-8B",
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

- [NVIDIA Cosmos-Reason2 Model Card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Riva Documentation](https://docs.nvidia.com/deeplearning/riva/)

