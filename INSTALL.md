# Installation & Setup Guide

This guide covers installing the app, optional NVIDIA Riva setup for voice ASR/TTS, and running the WebUI.

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

**Device support:** Voice and video use **browser devices (WebRTC)**—microphone, speaker, and camera are accessed through the browser. Local USB microphone, USB speaker, or USB webcam on the server machine are **not supported yet**.

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

### 1. System Dependencies

```bash
# Install portaudio (required for pyaudio)
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

### 2. Virtual Environment

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

**Fix**:
```bash
sudo apt-get install -y portaudio19-dev
pip install --upgrade --force-reinstall pyaudio
```

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

1. **Test backends**: `python3 scripts/test_backends.py`
2. **Run the app**: See [Run the app](#run-the-app) above; for voice, complete [NVIDIA Riva Setup](#nvidia-riva-setup-for-voice-asrtts).

## Notes

- ✅ Virtual environment keeps dependencies isolated
- ✅ `.venv/` is in `.gitignore` (not committed)
- ✅ Development mode (`-e`) means code changes are immediately effective
- ✅ Each developer has their own `.venv/`
