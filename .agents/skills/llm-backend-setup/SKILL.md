---
name: llm-backend-setup
description: >
  Configure and connect to an OpenAI-compatible LLM or VLM backend for Multi-modal AI Studio.
  Covers vLLM, Ollama, and OpenAI API setup including Docker commands,
  model listing, health checks, and warmup. Use when setting up the language model backend
  before running voice or vision sessions.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
---

# LLM Backend Setup

Configure an OpenAI-compatible LLM/VLM backend for Multi-modal AI Studio.

## Overview

MMAS connects to any OpenAI-compatible API (`/v1/chat/completions`). Supported backends:

| Backend | Default Port | Vision Support | Best For |
|---------|-------------|---------------|----------|
| vLLM | 8010 | Yes (image + video) | Production VLM serving |
| Ollama | 11434 | Yes (images only) | Easy local setup |
| OpenAI API | N/A | Yes (GPT-4V) | Cloud-based |

## Prerequisites

- Docker with NVIDIA runtime configured (for containerized backends)
- GPU with sufficient VRAM for the chosen model
- For vLLM with Cosmos-Reason2: download the model first — see [INSTALL.md](../../../INSTALL.md) or [vlm_guide.md](../../../docs/vlm_guide.md) for model setup instructions

## Instructions

### Option A: Ollama (easiest)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull nemotron-3-nano:4b     # Text-only LLM
ollama pull gemma3:4b            # Vision LLM (images only, no video)

# Ollama auto-starts on port 11434
curl -s http://localhost:11434/v1/models | python3 -m json.tool
```

### Option B: vLLM (recommended for VLMs)

> **Model setup**: Download the Cosmos-Reason2 model before running vLLM.
> See [INSTALL.md](../../../INSTALL.md) or [vlm_guide.md](../../../docs/vlm_guide.md) for download instructions and Jetson AI Lab links.

```bash
export MODEL_PATH=/path/to/cosmos-reason2-8b-fp8

sudo docker run -d --network host --runtime=nvidia \
  --name vllm-cosmos \
  -v $MODEL_PATH:/models/cosmos-reason2-8b:ro \
  ghcr.io/nvidia-ai-iot/vllm:latest \
  vllm serve /models/cosmos-reason2-8b \
    --served-model-name nvidia/cosmos-reason2-8b-fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --reasoning-parser qwen3 \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --enable-prefix-caching \
    --port 8010
```

> **GPU memory cleanup**: If vLLM fails with OOM after stopping another GPU container:
> ```bash
> sudo sysctl -w vm.drop_caches=3
> ```

### Option C: OpenAI API

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
python -m multi_modal_ai_studio \
  --llm-api-base https://api.openai.com/v1 \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-model gpt-4o
```

### Verify the backend

```bash
# List available models
curl -s http://localhost:8092/api/llm/models?api_base=http://localhost:8010/v1 | python3 -m json.tool

# Health check
curl -s http://localhost:8092/api/health/llm?api_base=http://localhost:8010/v1
# {"status": "ok"}

# Warm up the model (first request is slow due to CUDA graph capture)
curl -s -X POST http://localhost:8092/api/llm/warmup \
  -H "Content-Type: application/json" \
  -d '{"api_base": "http://localhost:8010/v1", "model": "nvidia/cosmos-reason2-8b-fp8"}'
```

### Direct API test

```bash
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/cosmos-reason2-8b-fp8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "stream": false
  }' | python3 -m json.tool
```

## Input Schema

MMAS LLM configuration fields:

- `api_base` (string, required): Backend URL (e.g., `"http://localhost:8010/v1"`)
- `model` (string, required): Model name as served by the backend
- `api_key` (string, optional): API key for authenticated endpoints
- `temperature` (float, optional, default: 0.7): Sampling temperature
- `max_tokens` (integer, optional, default: 512): Maximum response tokens
- `extra_request_body` (string, optional): JSON merged into every request (e.g., `'{"chat_template_kwargs": {"enable_thinking": false}}'`)

## Output Schema

Health check:
```json
{"status": "ok"}
```

Model list:
```json
{
  "models": ["nvidia/cosmos-reason2-8b-fp8"],
  "default_model": "nvidia/cosmos-reason2-8b-fp8"
}
```

## Guidelines

- Only one GPU-heavy container should run at a time on Jetson (shared memory)
- Stop and remove old containers before starting new ones: `docker stop <name> && docker rm <name>`
- Ollama does not support `video_url` — use image-only vision or vLLM for video
- vLLM first request is slow (CUDA graph compilation); use the warmup endpoint
