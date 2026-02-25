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

| Model | Provider | Backend | Use Case |
|-------|----------|---------|----------|
| **Cosmos-Reason2-8B** | NVIDIA | vLLM | Physical world reasoning, spatial understanding |


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

### Hugging Face Access (Required for Gated Models)

Cosmos-Reason2 is a **gated model**. You must:
1. Accept the license at [huggingface.co/nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
2. Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Pass the token as an environment variable (see below)

### For Jetson / Thor (ARM64):

```bash
docker run -d --gpus all \
  --name vllm-cosmos \
  -p 8003:8000 \
  -e HUGGING_FACE_HUB_TOKEN=hf_your_token_here \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  python3 -m vllm.entrypoints.openai.api_server \
    --model nvidia/Cosmos-Reason2-8B \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6 \
    --port 8000
```

### For Desktop GPU (x86_64):

```bash
docker run -d --gpus all \
  --name vllm-cosmos \
  -p 8003:8000 \
  -e HUGGING_FACE_HUB_TOKEN=hf_your_token_here \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model nvidia/Cosmos-Reason2-8B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --port 8000
```

> **Note**: First run downloads the model (~16GB). This may take several minutes.
>
> **Memory tuning**: On shared/unified-memory systems (Jetson Thor), use `--gpu-memory-utilization 0.6` to leave room for the OS, Riva, and the AI Studio app. On discrete GPUs with dedicated VRAM, `0.8` is safe. If you hit OOM, lower `--max-model-len` to `4096`.

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

The Riva container (`riva-speech`) should expose these ports:

| Port | Service |
|------|---------|
| `50051` | gRPC (ASR/TTS — this is what AI Studio connects to) |
| `8888` | Riva HTTP API |

Verify Riva is running:
```bash
docker ps | grep riva
# Should show riva-speech container with port 50051 mapped
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
# Recommended: use the Cosmos-Reason preset (sets temperature, prompts, vision, etc.)
python -m multi_modal_ai_studio --preset cosmos-reason --host 0.0.0.0

# Or with explicit CLI options
python -m multi_modal_ai_studio \
  --host 0.0.0.0 \
  --llm-api-base http://localhost:8003/v1 \
  --llm-model nvidia/Cosmos-Reason2-8B

# Debug mode: save encoded videos to disk for inspection
MMAS_DEBUG_VIDEOS=1 python -m multi_modal_ai_studio --preset cosmos-reason --host 0.0.0.0
```

> **`--host 0.0.0.0`** is required if you want to access the UI from another machine (not just localhost).
>
> **`MMAS_DEBUG_VIDEOS=1`** saves every MP4 video sent to the VLM into `src/debug_videos/` for offline inspection.

Then enable vision in the UI (see Step 5).

Open your browser: **https://localhost:8092**

> **HTTPS Note**: The app defaults to HTTPS with a self-signed certificate (required for browser camera/mic access via WebRTC). On first visit, your browser will show a security warning — click **"Advanced" → "Proceed"** to accept the self-signed cert.
>
> To disable HTTPS (not recommended): `python -m multi_modal_ai_studio --no-ssl`

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

1. **Continuously captures** JPEG frames from your camera into a ring buffer (~10fps)
2. **On speech end** (ASR final), retrieves all frames from the speech time window
3. **For Cosmos models**: encodes frames into an **H.264 MP4 video** with dynamic FPS, sent as a single `video_url`
4. **For other VLMs** (LLaVA, GPT-4V): selects N evenly-spaced frames, sent as individual `image_url` entries
5. **VLM responds** with visual understanding

### Cosmos Video Encoding (Temporal)

Cosmos-Reason2 is optimized for video input and can decode frame deltas, using far fewer tokens than equivalent individual images (~2x reduction). The system dynamically calculates FPS from your speech duration:

```
Speech: "What did I just do?"
        |<-------- 3 seconds -------->|
        t_start                    t_end

All frames in window retrieved from ring buffer (e.g. 30 frames @ 10fps)
  → Encoded into H.264 MP4 @ fps = 30/3 = 10fps
  → Sent as single video_url to Cosmos
  → Encoding overhead: ~100-200ms (ultrafast preset)
```

### Standard VLM Frame Selection (Non-Cosmos)

```
Speech: "What am I holding?"
        |<-------- 2 seconds -------->|
        Start                        End

Frames per Turn = 4:
        |-------|-------|-------|------|
       Frame1  Frame2  Frame3  Frame4
        @0.5s   @1.0s   @1.5s   @2.0s
```

### Per-Component Encoding Summary

| Encode | When | Rate | Purpose |
|--------|------|------|---------|
| JPEG (FrameBroker) | Every frame | ~10fps continuous | VLM frame storage |
| VP8/H.264 (WebRTC) | Every frame | ~30fps continuous | UI live camera display |
| H.264 MP4 (Cosmos) | Per speech turn | ~1 call / 3-10s | VLM inference |

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

### Built-in Cosmos Preset

The `cosmos-reason` preset (`presets/cosmos-reason.yaml`) is pre-configured with optimized settings:

| Setting | Value | Why |
|---------|-------|-----|
| `temperature` | 0.3 | Low temp for precise, consistent vision responses |
| `max_tokens` | 256 | Short spoken answers (voice assistant use case) |
| `enable_vision` | true | Camera frames sent to VLM |
| `vision_frames` | 4 | Overridden at runtime — Cosmos receives all frames as MP4 video |
| `system_prompt` | "You are a vision assistant..." | Tuned for concise visual descriptions |

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
  temperature: 0.3
  max_tokens: 256
  system_prompt: "You are a vision assistant. Be concise."

tts:
  scheme: riva
  server: localhost:50051
```

Run with a preset name: `python -m multi_modal_ai_studio --preset cosmos-reason`

Or load a custom config file: `python -m multi_modal_ai_studio --config presets/my-vlm.yaml`

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
| Model download fails | Check disk space, HuggingFace token, network |
| CUDA out of memory | Reduce `--max-model-len` to `4096` or lower `--gpu-memory-utilization` |
| Container crashes on restart | See "GPU Memory Not Released" below |

#### GPU Memory Not Released After Restart

If `docker restart vllm-cosmos` fails with:
```
ValueError: Free memory on device (27.92/122.82 GiB) is less than desired GPU memory utilization
```

The previous vLLM process didn't fully release GPU memory. Fix:

```bash
# 1. Stop the container completely
docker stop vllm-cosmos

# 2. Verify GPU memory is freed
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# Only Riva processes should remain (~7 GiB)

# 3. Wait a few seconds, then start
sleep 5
docker start vllm-cosmos

# 4. Monitor startup (model loading takes 1-2 minutes)
docker logs -f vllm-cosmos
# Wait for: "INFO: Application startup complete"
```

#### Verifying vLLM Health

```bash
# Health check (returns empty 200 when ready)
curl -s http://localhost:8003/health && echo "READY" || echo "NOT READY"

# List available models
curl -s http://localhost:8003/v1/models | python3 -m json.tool
```

### Camera Issues

| Issue | Solution |
|-------|----------|
| Camera not detected | Use HTTPS or localhost (not HTTP + IP) |
| Permission denied | Check browser camera permissions |
| Black frames | Ensure camera isn't used by another app |
| USB camera not listed | Only capture devices are shown (metadata devices like /dev/video1 are filtered) |

#### Camera Source Options

| Source | Setting | Use Case |
|--------|---------|----------|
| **Browser** | `video_source: browser` | Use client webcam via WebRTC |
| **USB (Server)** | `video_source: usb` + device path | Use server-attached USB camera (e.g., `/dev/video0`) |
| **None** | `video_source: none` | Disable camera for text-only mode |

> **Note**: When using USB camera, frames are captured via OpenCV and streamed to both UI preview and VLM.

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
| Generic answers | Lower `temperature` (try 0.3), improve system prompt |
| Repeats old answers | History is disabled by default; verify no custom history logic |
| "I can't see" errors | Check `enable_vision: true`, camera working |
| VLM hallucinates scenes | See "VLM vs LLM Behavior" below |
| Short/empty video | Check that speech duration is >0.5s; ghost ASR partials can inflate the capture window |

#### VLM vs LLM Behavior (Important!)

**VLMs hallucinate when asked visual questions without images:**

| Model Type | "Describe the scene" (no camera) |
|------------|----------------------------------|
| **VLM** (Cosmos-Reason, LLaVA) | Hallucinates: "A winter scene with snow..." |
| **Regular LLM** (Llama, GPT-3.5) | "I can't see anything" |

This is because VLMs are trained on image-text pairs and will generate visual descriptions even without input images.

**When camera is set to "none"**, the system adds a prompt note to help prevent hallucination, but results depend on the model.

---

## Using Other VLM Models

### Ollama (Easiest Setup)

Ollama supports vision models with no Docker configuration needed:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a vision model
ollama pull llava-llama3    # ~5GB, good quality
# or
ollama pull llava-phi3      # ~3GB, faster
# or
ollama pull moondream       # ~1.7GB, smallest
```

Configure in UI:
| Setting | Value |
|---------|-------|
| **API Base** | `http://localhost:11434/v1` |
| **Model** | `llava-llama3` |
| **Enable Vision** | ✅ Checked |

Or in preset:
```yaml
llm:
  scheme: openai
  api_base: http://localhost:11434/v1
  model: llava-llama3
  enable_vision: true
```

### LLaVA via vLLM

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

### Cosmos-Reason2 (Video Input)

Cosmos models receive a single MP4 video per turn:

```json
{
  "model": "nvidia/Cosmos-Reason2-8B",
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

### Standard VLMs (Multi-Image Input)

Other VLMs (LLaVA, GPT-4V) receive individual JPEG frames:

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

### Conversation History

Conversation history is **not passed** for VLM turns. Each turn is independent — the VLM sees only the current video/frames and text. This prevents "answer anchoring" where the model repeats previous (potentially wrong) answers instead of analyzing the current visual input.

> **Why no history?** Testing showed that when text history from prior turns was included, Cosmos would anchor on old answers (e.g., repeating "5 to 2" when the score had changed). Without history, each turn gets a fresh analysis of the current video.

---

## Further Reading

- [NVIDIA Cosmos-Reason2 Model Card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Riva Documentation](https://docs.nvidia.com/deeplearning/riva/)

