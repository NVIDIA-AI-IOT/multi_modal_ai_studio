# Installation & Setup Guide

This guide covers installing the app, setting up LLM/VLM backends, NVIDIA Riva for voice (ASR/TTS), and running the WebUI.

## Quick Start

The fastest way to get started:

```bash
cd multi-modal-ai-studio

# Run automated setup
./scripts/setup_dev.sh

# You'll be asked if you want USB audio support:
# - Answer Y if you plan to use headless mode or local USB devices
# - Answer N if you only need WebUI with browser audio (simpler)
```

This script will:
1. ✅ Create virtual environment (`.venv/`)
2. ✅ Install Python package in development mode
3. ✅ Install all core dependencies
4. ⚙️ Optionally install USB audio support (pyaudio - needs portaudio)

## Run the app

**Device support:** Voice and video use **browser devices (WebRTC)** by default. A **server USB microphone** (ALSA, e.g. EMEET) is supported—select it in the Devices tab; no PyAudio install needed on Linux. USB speaker and USB webcam on the server are not supported yet.

**View sessions only** (no Riva or LLM needed):

```bash
source .venv/bin/activate
multi-modal-ai-studio --port 8092
```

Open **https://localhost:8092** (accept the self-signed cert). Sessions in `sessions/` (or `--session-dir mock_sessions`) appear in the sidebar.

**Voice with Riva + LLM:** Set up Riva and an LLM (e.g. Ollama) as in [NVIDIA Riva Setup](#nvidia-riva-setup-for-voice-asrtts) below, then:

```bash
source .venv/bin/activate
multi-modal-ai-studio --port 8092 \
  --asr-server localhost:50051 \
  --tts-server localhost:50051 \
  --llm-api-base http://localhost:11434/v1 \
  --llm-model llama3.2:3b
```

## Manual Installation

If you prefer manual setup:

### 1. System Dependencies (optional, for PyAudio only)

**Server USB microphone (ALSA, e.g. EMEET):** On Linux, the app uses `arecord` for ALSA devices. No extra install needed—just select the device in the UI.

**PyAudio (for `pyaudio:N` device list / headless):** If you install the `[audio]` extra, install PortAudio first:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
# then: pip install -e ".[audio]"
```

### 2. Virtual Environment

> **Note:** If `python3 -m venv` fails with "No module named venv", install it first:
> ```bash
> sudo apt install python3.12-venv
> ```

```bash
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Package

```bash
# Install in development mode (editable)
pip install -e .

# This installs:
# - aiohttp (async HTTP)
# - grpcio (Riva gRPC)
# - nvidia-riva-client (Riva Python SDK)
# - openai (OpenAI API client)
# - pyyaml (config files)
# - numpy (audio processing)
# - websockets (WebSocket support)
# - ... and other dependencies
```

## Verify Installation

```bash
# Activate venv
source .venv/bin/activate

# Test imports
python3 -c "import multi_modal_ai_studio; print('✓ Package installed')"

# Test config system
python3 -c "
from multi_modal_ai_studio.config import SessionConfig
cfg = SessionConfig.from_yaml('presets/default.yaml')
print(f'✓ Config loaded: {cfg.name}')
print(f'  Required services: {cfg.get_required_services()}')
"

# Test backends (initialization only)
python3 scripts/test_backends.py

# Test CLI
multi-modal-ai-studio --help
```

## NVIDIA Riva Setup (for voice ASR/TTS)

To use **voice input/output** with the Riva backend, you need a running Riva server. This app does not install or start Riva; it connects to an existing Riva gRPC endpoint (typically `localhost:50051`).

### Platform support (as of 2025)

- **x86 data center**: Riva SDK is **no longer supported**. Use [Riva ASR NIM](https://docs.nvidia.com/nim/nims/riva/) for x86.
- **ARM64 / Jetson**: **Supported**. Use the ARM64 quickstart (e.g. version 2.24.0). This section focuses on Jetson.

### Prerequisites

- **Jetson**: Orin, Thor, AGX Xavier, or newer (ARM64/L4T)
- **JetPack**: 6.0+ recommended. Docker and NVIDIA Container Toolkit are pre-installed.
- **NGC account with Riva access**: Required to download the quickstart and models. Use an account that has Riva entitlements (company or personal). If one account fails with 403, try another.

---

### Part 1: Install NGC CLI

NGC CLI is required to download Riva’s quickstart bundle.

1. **Download page**: [https://org.ngc.nvidia.com/setup/installers/cli](https://org.ngc.nvidia.com/setup/installers/cli) — select **ARM64 Linux** for Jetson.

2. **Install (command line)**:

```bash
mkdir -p ~/.local/share/ngc-cli
cd ~/.local/share/ngc-cli

# Replace version with latest from the download page if needed
wget --content-disposition \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.11.1/files/ngccli_arm64.zip \
  -O ngccli_arm64.zip

unzip ngccli_arm64.zip
chmod +x ngc-cli/ngc

mkdir -p ~/.local/bin
ln -s ~/.local/share/ngc-cli/ngc-cli/ngc ~/.local/bin/ngc
```

3. **Ensure `ngc` is in PATH**:

```bash
# If ngc not found, add ~/.local/bin to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

ngc --version   # Should print e.g. NGC CLI 4.11.1
```

**Alternative (system-wide):** `sudo cp ~/.local/share/ngc-cli/ngc-cli/ngc /usr/local/bin/`

---

### Part 2: Configure NGC CLI

1. **Create an NGC API key**: Log in at [NGC](https://ngc.nvidia.com) → **Setup** → **Generate API Key** (at least **NGC Catalog** permissions). Copy the key.

2. **Configure**:

```bash
ngc config set
```

When prompted:

| Prompt           | Value / Notes                                      |
|------------------|----------------------------------------------------|
| **API key**      | Your NGC API key (long alphanumeric string)        |
| **CLI output**   | `ascii` (default)                                  |
| **Org**          | Choose the org that has Riva access                |
| **Team**         | Team with Riva entitlement (e.g. `swteg-jarvis-jetson` for some NVIDIA orgs) |
| **ACE**          | `no-ace` (default)                                 |

NGC API keys are different from NVIDIA API keys (`nvapi-...`). If `riva_init.sh` later fails with 403, try a different NGC account or team.

3. **Verify access**:

```bash
ngc registry resource list nvidia/riva
```

You should see resources listed. Empty table usually means no Riva entitlement for that org/team.

---

### Part 3: Download Riva quickstart (ARM64)

```bash
ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0

cd riva_quickstart_arm64_v2.24.0
```

Use the version that matches your NGC catalog (e.g. 2.24.0 or newer for Jetson).

---

### Part 4: Configure Riva (`config.sh`)

Edit `config.sh` in the quickstart directory:

```bash
cd riva_quickstart_arm64_v2.24.0
vi config.sh   # or your editor
```

**Important options for this app**:

```bash
riva_target_gpu_family="tegra"
# Set to your platform, e.g. thor, orin, etc.
riva_tegra_platform="thor"

# Enable only what you need (ASR + TTS for voice)
service_enabled_asr=true
service_enabled_nlp=false
service_enabled_tts=true
service_enabled_nmt=false

# Language
asr_language_code="en-US"
tts_language_code="en-US"

# Low latency for real-time voice (recommended)
use_asr_streaming_throughput_mode=false

# For multi-turn dialogue, use Silero VAD (recommended)
# This exposes the model: parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer
asr_acoustic_model=("parakeet_1.1b")
asr_accessory_model=("silero_diarizer")
```

Without `asr_accessory_model=("silero_diarizer")`, the server may only expose the base Parakeet model and second/later user turns can be missed. See `docs/asr_model_for_multi_utterance.md` for details.

---

### Part 5: Initialize Riva (once)

Downloads Docker images and models (~15–45 minutes on Jetson):

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_init.sh
```

This pulls ARM64 Riva images and pre-optimized ASR/TTS models; no separate model build step. If you get **403 errors**, fix NGC credentials (`ngc config set`) or try another account. Ensure enough disk space (e.g. 64GB+ free); check with `df -h`.

---

### Part 6: Start Riva

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_start.sh
```

Riva gRPC server listens on **port 50051**. First startup can take 2–5 minutes to load models into GPU.

**Verify**:

```bash
docker ps
# Expect: riva-speech container Up

docker logs -f riva-speech
# Look for: "Riva server listening on 0.0.0.0:50051" / "All models loaded successfully"
```

If using USB mic/speaker, connect it **before** running `riva_start.sh`.

---

### Part 7: Stop Riva

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_stop.sh
```

Models remain in the Riva model volume; next `riva_start.sh` is fast.

---

### Run this app with Riva

After Riva is running:

```bash
source .venv/bin/activate
multi-modal-ai-studio --port 8092 \
  --asr-server localhost:50051 \
  --tts-server localhost:50051 \
  --llm-api-base http://localhost:11434/v1 \
  --llm-model llama3.2:3b
```

Open https://localhost:8092 (accept the self-signed cert).

---

### Riva troubleshooting

| Issue | What to do |
|-------|------------|
| **403 when downloading** | NGC account/team lacks Riva entitlement. Run `ngc config set` and pick a team with access, or try another NGC account. |
| **NGC CLI not found after install** | Add `~/.local/bin` to PATH (see Part 1). Use symlink, not moving the binary. |
| **Riva container won’t start / GPU errors** | Run `nvidia-smi`; test Docker GPU: `docker run --rm --gpus all ubuntu nvidia-smi`. Check `docker logs riva-speech`. |
| **Very slow model download** | First pull is large (10–30 GB). Use a good network; ensure enough disk space. |

**Resources**: [NVIDIA Riva User Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/), [Riva Quick Start](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/), [NGC CLI](https://docs.nvidia.com/ngc/ngc-cli/index.html).

## LLM / VLM Backend Setup

The application connects to any **OpenAI-compatible** `/v1/chat/completions` endpoint for language and vision inference. You need at least one backend running before starting a conversation.

### Option A: Ollama (Easiest)

[Ollama](https://ollama.com/) runs models locally with no Docker or GPU configuration needed.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a text-only model
ollama pull llama3.2:3b        # ~2GB, fast general-purpose

# Or pull a vision model (for camera/video input)
ollama pull llava-llama3       # ~5GB, image understanding
ollama pull gemma3:4b          # ~3GB, multimodal
```

Ollama serves on `http://localhost:11434/v1` by default. In the UI, set:

| Setting | Value |
|---------|-------|
| **API Base** | `http://localhost:11434/v1` |
| **Model** | `llama3.2:3b` (text) or `llava-llama3` (vision) |

### Option B: vLLM (Recommended for VLMs / Production)

vLLM provides high-throughput serving with GPU acceleration. Example with Cosmos-Reason2 for vision:

**[Cosmos-Reason2-8B on Jetson AI Lab](https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/)** — full setup including model download and platform-specific Docker images. The FP8 model is downloaded from NGC; you need an NGC account with access to the **nim** org (and often the **nvidia** team). If NGC download fails, see [NGC Cosmos model download fails](#ngc-cosmos-model-download-fails-completed-0-failed-n) below.

Quick reference for Jetson Thor (after downloading the FP8 model per the link above):

```bash
export MODEL_PATH="${HOME}/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8"

mkdir -p ~/.cache/vllm
sudo sysctl -w vm.drop_caches=3
sudo docker run -it --rm --runtime=nvidia --network host \
  -v $MODEL_PATH:/models/cosmos-reason2-8b:ro \
  -v ${HOME}/.cache/vllm:/root/.cache/vllm \
  ghcr.io/nvidia-ai-iot/vllm:0.14.0-r38.3-arm64-sbsa-cu130-24.04 \
  vllm serve /models/cosmos-reason2-8b \
    --served-model-name nvidia/cosmos-reason2-8b-fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --reasoning-parser qwen3 \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --enable-prefix-caching \
    --port 8010
```

The second volume `-v ${HOME}/.cache/vllm:/root/.cache/vllm` persists vLLM’s **torch.compile cache** on the host. The first run compiles kernels and writes them there; later runs reuse the cache and start faster. Create `~/.cache/vllm` **before** the first run (as in the example above) so it is owned by your user; otherwise the container may create it as root and you can hit permission issues later.

`vm.drop_caches=3` frees **system (CPU) memory** (page cache, etc.); it does **not** free **GPU VRAM**. If you start vLLM a second time while the first container is still running, the GPU has no free VRAM and vLLM will fail with "Free memory on device cuda:0 (...) is less than desired". **Stop the first vLLM container** (e.g. Ctrl+C or `docker stop`) so the driver releases GPU memory, then start again.

> **Port conflict with Riva**: The **Riva container** exposes ports **8000–8002** (and 8888, 50051). If you run both Riva and vLLM on the same machine, use a different vLLM port so they don't clash. The example above uses `--port 8010`; in the app set **LLM API Base** to `http://localhost:8010/v1`. If Riva is not running, `--port 8000` is fine.
>
> **Memory tuning**: On shared-memory systems (Jetson), lower `--gpu-memory-utilization` to leave room for the OS, Riva, and the application. On discrete GPUs with dedicated VRAM, `0.8` is safe.
>
> **GPU memory cleanup**: If vLLM fails to start with an OOM error after stopping another GPU container, free cached memory first:
> ```bash
> sudo sysctl -w vm.drop_caches=3
> ```
>
> **Desktop GPU / x86_64**: Use `vllm/vllm-openai:latest` or `nvcr.io/nvidia/vllm:latest` instead of the Jetson image.

### vLLM troubleshooting

#### `OSError: [Errno 98] Address already in use`

vLLM fails at startup with `sock.bind(addr) OSError: [Errno 98] Address already in use` when the API port (default **8000**) is already taken—for example by a previous vLLM run, another container, or another service.

**1. Find what is using the port**

```bash
# Default vLLM port is 8000; use your --port if different
lsof -i :8000
# or
ss -tlnp | grep 8000
# or
fuser 8000/tcp
```

If **`ss` shows port 8000 in LISTEN but `lsof` and `fuser` show no PID**, the process is usually **inside a Docker container**. List containers and look for one that has port 8000:

```bash
docker ps -a
# Look for a container with 0.0.0.0:8000->8000/tcp or similar in PORTS
```

**2. Free the port or use another**

- **Riva is using 8000** (container `riva-speech` exposes 8000–8002): Don't stop Riva. Start vLLM on a different port and point the app to it:
  ```bash
  # In the vllm serve command, use e.g.:
  --port 8010

  # In Multi-modal AI Studio, set LLM API Base to:
  # http://localhost:8010/v1
  ```
- **Another Docker container** (e.g. leftover vLLM): Stop and remove it if you don't need it:
  ```bash
  docker ps -a
  docker stop <container_id_or_name>
  docker rm <container_id_or_name>
  # or: docker rm -f <container_id_or_name>
  ```
- **Process on the host** (when lsof/fuser show a PID): Kill it:
  ```bash
  kill <PID>
  # or: fuser -k 8000/tcp
  ```

**3. Use a different port**

If you need to keep whatever is on 8000, start vLLM with `--port 8010` (or another free port) and set the app's **LLM API Base** to `http://localhost:8010/v1`.

#### `ValueError: Free memory on device cuda:0 (...) is less than desired GPU memory utilization`

Another process (often a **previous vLLM container**) is still using the GPU, so there isn’t enough free VRAM. Stop the other process: if the first vLLM was started in another terminal, press **Ctrl+C** there, or run `docker ps` and `docker stop <container_id>`. The driver may take **30–60 seconds** to release VRAM after the container exits; run `nvidia-smi` and wait until free memory is back to normal before starting vLLM again. `vm.drop_caches=3` only frees system RAM, not GPU VRAM.

#### `ValidationError: Invalid repository ID or local directory specified: '/models/...'`

vLLM fails during startup with a message like **Invalid repository ID or local directory specified: '/models/cosmos-reason2-8b'** when the model path inside the container is missing, wrong, or doesn't contain the expected config files.

**1. Check the model directory on the host**

Ensure `MODEL_PATH` points to the directory that contains the model files (e.g. `config.json` for Hugging Face–style models):

```bash
echo $MODEL_PATH
ls -la "$MODEL_PATH"
# Must contain at least: config.json (and usually model weights, tokenizer files, etc.)
```

If the directory is missing or empty, download the model first (see [Cosmos-Reason2-8B on Jetson AI Lab](https://www.jetson-ai-lab.com/models/cosmos-reason2-8b/) or your model’s instructions).

**2. Check the volume mount**

The `docker run` command must mount that host path into the container path vLLM uses:

```bash
# Example: host path -> container path /models/cosmos-reason2-8b
-v $MODEL_PATH:/models/cosmos-reason2-8b:ro
```

- Use an **absolute path** for `MODEL_PATH` (e.g. `$HOME/.cache/huggingface/hub/...`), not a relative one, so the mount is correct from any working directory.
- The path after the colon must match the path you pass to `vllm serve` (e.g. `vllm serve /models/cosmos-reason2-8b`).

**3. Verify the container sees the files**

Run a quick check that the mounted directory exists and has a config inside the container:

```bash
docker run --rm -v "$MODEL_PATH:/models/cosmos-reason2-8b:ro" \
  ghcr.io/nvidia-ai-iot/vllm:0.14.0-r38.3-arm64-sbsa-cu130-24.04 \
  ls -la /models/cosmos-reason2-8b
```

You should see `config.json` and other model files. If the list is empty or "No such file or directory", fix `MODEL_PATH` or the mount path and try again.

#### Fix Hugging Face cache permissions (root-owned)

If `~/.cache/huggingface` or `~/.cache/huggingface/hub` is owned by **root** (e.g. created by [jetson-containers](https://github.com/dusty-nv/jetson-containers) or another tool running with `sudo`), commands run as your user (NGC CLI, Python, Hugging Face libraries) will get **Permission denied** when writing there.

**Fix:** make the cache tree owned by the current user:

```bash
sudo chown -R $USER:$USER ~/.cache/huggingface
```

Then retry the download or command that was failing. To avoid the issue in the future, create the directory as your user before any tool that might run as root: `mkdir -p ~/.cache/huggingface/hub`.

#### NGC Cosmos model download fails (Completed: 0, Failed: N)

If `ngc registry model download-version "nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8" --dest ~/.cache/huggingface/hub` fails with **Completed: 0, Failed: 14**:

- **Permission denied when writing files:** If the debug log shows `[Errno 13] Permission denied` for paths under `~/.cache/huggingface/hub/`, the destination is likely **root-owned**. See [Fix Hugging Face cache permissions (root-owned)](#fix-hugging-face-cache-permissions-root-owned) above: run `sudo chown -R $USER:$USER ~/.cache/huggingface`, then retry. If you only need to fix the model subdirectory: `sudo chown -R $USER:$USER ~/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8` (and ensure the parent `hub` is writable). Alternatively remove the partial dir and re-download: `rm -rf ~/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8` then run the same `ngc registry model download-version ...` again, or use a different `--dest` you can write to.

- **403 or auth/org errors:** If the debug log shows **403** or org/entitlement errors (rather than Permission denied), try setting the effective org to **nvidia**. The NGC CLI uses **`NGC_CLI_ORG`** from the environment. Example for `~/.bashrc`:
  ```bash
  export NGC_CLI_ORG=nvidia
  # optional: export NGC_CLI_API_KEY=<your-key>
  ```
  Then in the same shell (or a new terminal after `source ~/.bashrc`):
  ```bash
  ngc config current   # check effective org
  ngc registry model download-version "nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8" --dest ~/.cache/huggingface/hub
  export MODEL_PATH="${HOME}/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8"
  ```
  If the env var is not picked up, use `--org nvidia` on the command (next bullet). The download often succeeds with the default org (e.g. with `NGC_CLI_ORG` unset); only try `nvidia` if you see 403 or org/entitlement errors.

- **Explicit org/team:** `ngc registry model download-version "nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8" --org nim --team nvidia --dest ~/.cache/huggingface/hub`

- **Browser:** If you can download from the [catalog page](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-8b?version=1208-fp8-static-kv8) in the browser, save the files into `~/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8/` and set `MODEL_PATH` to that directory.

- **Different machine:** If the same API key and `NGC_CLI_ORG=nvidia` work on one host but not another, the failing host may differ by network, NGC CLI version, or backend. Run with **`--debug`** to see the underlying error: `ngc --debug registry model download-version "nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8" --dest ~/.cache/huggingface/hub`. Reliable workaround: **copy the model from the working machine** (e.g. from jat03): `rsync -avz jetson@jat03-iso384:~/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8/ ~/.cache/huggingface/hub/cosmos-reason2-8b_v1208-fp8-static-kv8/` then set `MODEL_PATH` to that directory.


### Option C: OpenAI API

No local setup needed. Set **API Base** to `https://api.openai.com/v1`, provide your API key, and choose a model (`gpt-4o` for vision, `gpt-4o-mini` for text).

### Verify Your Backend

```bash
# Ollama
curl -s http://localhost:11434/v1/models | python3 -m json.tool

# vLLM (use your port if different, e.g. 8010 when Riva uses 8000)
curl -s http://localhost:8010/v1/models | python3 -m json.tool
curl -s http://localhost:8010/health && echo "READY" || echo "NOT READY"
```

### Using Vision

To use camera/video input, your backend must support image or video content. Enable **"Enable Vision (VLM)"** in the UI config panel. For details on input modes, frame capture, and tuning, see the [VLM Guide](docs/vlm_guide.md).

---

## Virtual Environment Usage

### Activate

Every time you work on the project:

```bash
cd /home/jetson/multi-modal-ai-studio
source .venv/bin/activate
```

You'll see `(.venv)` in your prompt.

### Deactivate

When done:

```bash
deactivate
```

### IDE Integration

**VS Code / Cursor:**
- The IDE should auto-detect `.venv/`
- If not, select Python interpreter: `.venv/bin/python`

## Dependencies

### Python Packages (Installed Automatically)

**Core:**
- `aiohttp>=3.8.0` - Async HTTP client/server
- `psutil>=5.9.0` - CPU/GPU stats for timeline (WebUI)
- `grpcio>=1.50.0` - gRPC support
- `nvidia-riva-client>=2.14.0` - NVIDIA Riva SDK
- `openai>=1.0.0` - OpenAI API client
- `pyyaml>=6.0` - YAML config files
- `python-dotenv>=1.0.0` - Environment variables
- `numpy>=1.21.0` - Array operations
- `websockets>=11.0` - WebSocket support

**Audio:**
- `pyaudio>=0.2.13` - Audio I/O (requires portaudio)

**Development (Optional):**
- `pytest>=7.0.0` - Testing
- `pytest-asyncio>=0.21.0` - Async testing
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Linting
- `mypy>=1.0.0` - Type checking

### System Dependencies

**Ubuntu/Debian (Jetson):**
```bash
sudo apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    python3-dev \
    build-essential
```

## Troubleshooting

### PyAudio Installation Fails

**Error**: `portaudio.h: No such file or directory`

PyAudio needs the PortAudio development headers. Install them before `pip install -e ".[audio]"`:

```bash
sudo apt-get install -y portaudio19-dev
pip install -e ".[audio]"
```

**Note:** If you only use a Server USB mic that appears as ALSA (e.g. EMEET OfficeCore M0 Plus), you don't need PyAudio—the app uses `arecord` on Linux. Skip the `.[audio]` extra.

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'multi_modal_ai_studio'`

**Fix**: Make sure venv is activated and package is installed:
```bash
source .venv/bin/activate
pip install -e .
```

### Riva Client Errors

**Error**: `ModuleNotFoundError: No module named 'riva'`

**Fix**: Riva client should be installed automatically. If not:
```bash
pip install nvidia-riva-client
```

### Permission Errors

If you get permission errors, make sure you're NOT using sudo with pip inside venv:
```bash
# ✓ Good (inside venv)
source .venv/bin/activate
pip install -e .

# ✗ Bad (inside venv)
sudo pip install -e .  # Don't do this!
```

## Development Workflow

### Typical Session

```bash
# 1. Navigate to project
cd /home/jetson/multi-modal-ai-studio

# 2. Activate venv
source .venv/bin/activate

# 3. Work on code...
# (Edit files, run tests, etc.)

# 4. Test changes
python3 scripts/test_backends.py
python3 -m pytest  # When tests are added

# 5. Deactivate when done
deactivate
```

### Adding Dependencies

To add a new dependency:

```bash
# 1. Add to requirements.txt or pyproject.toml

# 2. Reinstall
source .venv/bin/activate
pip install -e .

# Or install directly
pip install new-package
```

## Next Steps

Once installed:

1. **Set up a backend**: See [LLM / VLM Backend Setup](#llm--vlm-backend-setup) (Ollama is the fastest way to get started)
2. **Add voice**: Complete [NVIDIA Riva Setup](#nvidia-riva-setup-for-voice-asrtts)
3. **Run the app**: See [Run the app](#run-the-app) above
4. **Add vision**: Enable in the UI and see the [VLM Guide](docs/vlm_guide.md) for details

## Notes

- ✅ Virtual environment keeps dependencies isolated
- ✅ `.venv/` is in `.gitignore` (not committed)
- ✅ Development mode (`-e`) means code changes are immediately effective
- ✅ Each developer has their own `.venv/`
