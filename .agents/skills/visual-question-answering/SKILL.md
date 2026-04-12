---
name: visual-question-answering
description: >
  Send images or video to a Vision Language Model and get text answers about the visual content.
  Supports vLLM, Ollama, and OpenAI-compatible backends with speech-synchronized video encoding.
  Includes reasoning (chain-of-thought) control and UI configuration for frame capture settings.
  Use when the user wants to analyze images or video with natural language questions.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
---

# Visual Question Answering

Ask natural language questions about images or video using a Vision Language Model.

## Overview

Multi-modal AI Studio sends visual content to any OpenAI-compatible VLM endpoint via `POST /v1/chat/completions`. It supports:
- **N-Image input**: Multiple JPEG frames as `image_url` with base64 data URLs
- **Video input**: Frames encoded as MP4 and sent as `video_url` with base64 data URL
- **File path input**: `file://` URLs for videos accessible inside backend containers

## Prerequisites

- An OpenAI-compatible VLM backend running and accessible:
  - **vLLM**: `http://localhost:8010/v1` with a vision model (e.g., Cosmos-Reason2-8B)
  - **Ollama**: `http://localhost:11434/v1` with a vision model (e.g., `gemma3:4b`)
  - **OpenAI**: `https://api.openai.com/v1` with GPT-4V or similar

## Instructions

### Image input (base64 JPEG)

```python
import base64
import requests

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "nvidia/cosmos-reason2-8b-fp8",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful vision assistant. Give ONE short sentence answers only."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 256,
        "stream": False
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Video input (base64 MP4)

For models with temporal understanding (e.g., Cosmos-Reason2):

```python
import base64
import requests

with open("video.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "nvidia/cosmos-reason2-8b-fp8",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful vision assistant. Give ONE short sentence answers only."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is happening in this video?"},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 256,
        "stream": False
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### With reasoning (chain-of-thought)

For vLLM with `--reasoning-parser qwen3`, reasoning appears in a separate `reasoning_content` field:

```python
response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "nvidia/cosmos-reason2-8b-fp8",
        "messages": [
            {"role": "system", "content": "You are a vision assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this worker wearing safety equipment?\n\nThink step-by-step inside <think> tags, then write your final answer immediately after </think>."},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 512,
        "stream": False
    }
)

result = response.json()
msg = result["choices"][0]["message"]
print("Reasoning:", msg.get("reasoning_content", ""))
print("Answer:", msg["content"])
```

### Check video encode capability

```bash
curl -s http://localhost:8092/api/vision/video-encode-available
# {"available": true}  — PyAV and PIL are installed
```

## Reasoning: When to Enable and When Not To

| Scenario | Reasoning | Why |
|----------|-----------|-----|
| Real-time voice assistant | **Off** | Reasoning adds 2-5x latency (thinking tokens consume `max_tokens`). Users waiting for spoken answers need fast responses. |
| Safety/compliance analysis | **On** | Accuracy matters more than speed. Chain-of-thought helps the model notice details it might miss. |
| Simple visual questions ("what color?", "how many?") | **Off** | Direct answers are fast and accurate enough without reasoning. |
| Complex scene understanding ("is this a violation?") | **On** | Multi-step reasoning improves accuracy on judgment calls. |
| Benchmarking / evaluation | **On** | Reasoning output helps verify the model's thought process and debug wrong answers. |

**Key trade-off**: Reasoning tokens consume from `max_tokens`. With `max_tokens: 256` and reasoning enabled, the thinking might use 200 tokens leaving only 56 for the actual answer — which can result in empty or truncated responses. Set `max_tokens: 512` or higher when reasoning is enabled.

In MMAS, reasoning text is automatically **stripped before TTS** — the user only hears the final answer, not the chain-of-thought.

## Speech-Synchronized Video Encoding

When running in a live voice session, MMAS captures camera frames during the user's speech and encodes them as a single MP4 video aligned to the speech duration:

```
User speaks (3.2 seconds)
    ↓
Camera captures frames at vision_buffer_fps (e.g., 5 fps → 16 frames)
    ↓
Frames encoded to MP4 at fps = n_frames / speech_duration (16/3.2 = 5 fps)
    ↓
Video duration ≈ speech duration (3.2s)
    ↓
Sent to VLM as video_url with base64 data URL
```

This ensures the video the model sees **temporally matches** what happened while the user was speaking. The `speech_duration` parameter is passed to the video encoder so the MP4 fps is calculated to match real-world timing. This is critical for Cosmos-Reason2 models that use temporal token compression.

The speech window includes a 2-second lookback before speech start (to capture context like a hand gesture before the question) and is capped at 10 seconds maximum.

## UI Configuration

In the MMAS WebUI, vision settings are under the **LLM** configuration tab:

| UI Setting | Config Field | Default | Description |
|-----------|-------------|---------|-------------|
| Enable Vision | `enable_vision` | `false` | Master toggle for VLM mode |
| Video Encode | `vision_video_encode` | `false` | Encode frames as MP4 (required for Cosmos-Reason2) |
| Vision Frames | `vision_frames` | `4` | Number of frames captured per turn |
| Frame Quality | `vision_quality` | `0.7` | JPEG quality (0.3 = fast/small, 1.0 = high quality) |
| Max Width | `vision_max_width` | `640` | Max frame width in pixels (768 recommended for Cosmos) |
| Buffer FPS | `vision_buffer_fps` | `3.0` | Frame capture rate (5.0 recommended for video encode) |
| Vision System Prompt | `vision_system_prompt` | (see below) | Overrides `system_prompt` when vision is active |
| Enable Reasoning | `enable_reasoning` | `false` | Append reasoning prompt to user message |

**Recommended settings for Cosmos-Reason2:**

```json
{
  "enable_vision": true,
  "vision_video_encode": true,
  "vision_frames": 30,
  "vision_max_width": 768,
  "vision_buffer_fps": 5.0,
  "vision_quality": 0.8,
  "vision_system_prompt": "You are a helpful voice and vision assistant. Give ONE short sentence answers only. Be direct. Plain text only, no markdown, no bullet points, no emojis."
}
```

**Recommended settings for Ollama (gemma3:4b):**

```json
{
  "enable_vision": true,
  "vision_video_encode": false,
  "vision_frames": 1,
  "vision_max_width": 640,
  "vision_buffer_fps": 3.0
}
```

## Input Schema

- `messages` (array, required): OpenAI-format messages with `image_url` or `video_url` content parts
- `model` (string, required): Model name on the backend
- `temperature` (float, optional, default: 0.7): Sampling temperature
- `max_tokens` (integer, optional, default: 512): Maximum response tokens
- `stream` (boolean, optional, default: false): Stream response tokens
- `chat_template_kwargs` (object, optional): Backend-specific options (e.g., `{"enable_thinking": true}`)

## Output Schema

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The worker is not wearing a hard hat.",
      "reasoning_content": "Looking at the video frames..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "completion_tokens": 12
  }
}
```

- `content` (string): The final answer (used for TTS in MMAS)
- `reasoning_content` (string, optional): Chain-of-thought reasoning (only present when reasoning is enabled)

## Guidelines

- **Ollama does not support `video_url`** — use N-Image (multiple `image_url`) instead; set `vision_video_encode: false`
- Keep video files small (< 2MB base64) to avoid exceeding `max_model_len` on the backend
- For vLLM, use `--media-io-kwargs '{"video": {"num_frames": -1}}'` to accept variable frame counts
- Reduce `vision_max_width` (e.g., 768) and `vision_quality` (e.g., 0.8) to keep payloads manageable
- When reasoning is on, increase `max_tokens` to 512+ to avoid the answer being truncated after thinking
- Set `vision_buffer_fps` to at least 5.0 when using `vision_video_encode: true` — lower values produce choppy video with too few temporal cues
- The speech window captures frames from 2 seconds before speech start to speech end, capped at 10 seconds; ghost ASR partials may inflate this window
- Debug video output: set env var `MMAS_DEBUG_VIDEOS=1` to save encoded MP4s to `src/debug_videos/` for inspection
