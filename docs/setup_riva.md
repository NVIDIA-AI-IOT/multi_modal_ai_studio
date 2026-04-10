# NVIDIA Riva Setup Guide

This guide walks through setting up NVIDIA Riva locally for voice (ASR/TTS). It is shared by **Multi-modal AI Studio** and **Live RIVA WebUI**.

## ⚠️ IMPORTANT CHANGE (January 2025)

**NVIDIA Riva SDK platform support has changed significantly:**

- **x86 platforms (data center)**: Riva SDK is **NO LONGER SUPPORTED**
  - For x86 deployments, use [Riva ASR NIM](https://docs.nvidia.com/nim/nims/riva/) instead
  - This guide does NOT cover NIM deployment

- **ARM64/Embedded platforms (Jetson)**: **STILL SUPPORTED** ✅
  - Latest version: `2.24.0` (as of December 2025)
  - Fully compatible with Jetson Orin, Thor, AGX, etc.
  - This guide focuses on Jetson deployment

**Since these apps were originally developed on PC (x86), this guide has been updated to focus on Jetson Thor deployment.**

## Prerequisites

- **Jetson Platform**: Jetson Orin, Thor, AGX Xavier, or newer (ARM64/L4T)
- **JetPack**: Recent JetPack version (6.0+ recommended)
- **Docker + NVIDIA Container Toolkit**: Pre-installed on JetPack. The ARM64 quickstart uses **plain Docker only** (`docker run`, `docker exec`, etc.) — **Docker Compose is not required**.
- **NGC account with Riva access**: Required for downloading Riva resources
  - Try your account that has Riva entitlements (company or personal)
  - NVIDIA employees: Internal access may require specific team membership
  - External users: May need AI Enterprise trial or proper entitlements
  - **Tip**: If one account doesn't work, try another you have access to

## Configure Docker for GPU (Jetson) — do this first

Riva runs in a container that needs GPU access. On Jetson, Docker must use the **NVIDIA Container Runtime** and be **restarted** after any config change. Doing this once at the start avoids the "container stays Created" / "use --runtime=nvidia instead" errors.

1. **Ensure `/etc/docker/daemon.json` has the NVIDIA runtime and default**

   If the file doesn't exist or is empty, create it. Otherwise merge the `runtimes` and `default-runtime` into your existing config:

   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia"
   }
   ```

   Example (create or edit with sudo):

   ```bash
   sudo nano /etc/docker/daemon.json
   ```

2. **Restart Docker so the config is applied**

   **This step is required.** Changes to `daemon.json` do not apply until Docker is restarted.

   ```bash
   sudo systemctl restart docker
   ```

3. **Optional: verify GPU access in a container**

   ```bash
   docker run --rm --runtime=nvidia nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi
   ```

   You should see your GPU; if not, check NVIDIA Container Toolkit and JetPack install.

Then continue with Part 1 (NGC CLI) below.

## Part 1: Install NGC CLI

The NGC CLI is required to download Riva's quickstart bundle from NVIDIA's catalog.

### Download NGC CLI for ARM64 Linux (Jetson)

1. **Go to the official NGC CLI download page**: [https://org.ngc.nvidia.com/setup/installers/cli](https://org.ngc.nvidia.com/setup/installers/cli)

2. **Select "ARM64 Linux"** from the installer options

3. **Download and install** (or use command line):

```bash
# Create local tools directory
mkdir -p ~/.local/share/ngc-cli
cd ~/.local/share/ngc-cli

# Download NGC CLI for ARM64 Linux (version 4.11.1, released 01/22/2026)
# Note: Replace the version number with the latest from the download page
wget --content-disposition \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.11.1/files/ngccli_arm64.zip \
  -O ngccli_arm64.zip

# Extract
unzip ngccli_arm64.zip
chmod +x ngc-cli/ngc

# Symlink to ~/.local/bin (should already be in PATH)
mkdir -p ~/.local/bin
ln -s ~/.local/share/ngc-cli/ngc-cli/ngc ~/.local/bin/ngc

# Check if NGC CLI is accessible (tests if ~/.local/bin is in PATH)
ngc --version 2>/dev/null || echo "⚠️  NGC not in PATH yet - see PATH setup below"
# Should output: NGC CLI 4.11.1 (or your downloaded version)
```

**⚠️ If NGC CLI is not found:** Your PATH likely doesn't include `~/.local/bin`. Add it now:
```bash
# Add ~/.local/bin to PATH (permanent)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Apply immediately
source ~/.bashrc

# Test again
ngc --version
```

**Alternative: System-wide installation**

If you prefer system-wide installation:
```bash
cd ~/.local/share/ngc-cli
sudo cp ngc-cli/ngc /usr/local/bin/
```

### Verify PATH

If the `ngc --version` command above didn't work, you need to add `~/.local/bin` to your PATH:

```bash
# Check if ~/.local/bin is in PATH
echo $PATH | grep -q "$HOME/.local/bin" && \
  echo "✓ ~/.local/bin is in PATH" || \
  echo "⚠️  ~/.local/bin is NOT in PATH - needs to be added"

# View your current PATH
echo $PATH
```

**If `~/.local/bin` is missing from PATH**, add it to `~/.bashrc`:

```bash
# Add to PATH (permanent, takes effect in new terminals)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Apply immediately in current terminal
source ~/.bashrc

# OR reload bash entirely
exec bash

# Verify it worked
ngc --version
# Should now output: NGC CLI 4.11.1
```

**Quick test to confirm NGC CLI is working:**
```bash
which ngc
# Should output: /home/jetson/.local/bin/ngc (or your username)

ngc --version
# Should output: NGC CLI 4.11.1
```

### Resources

- **NGC CLI Documentation**: [https://docs.ngc.nvidia.com/cli/cmd.html](https://docs.ngc.nvidia.com/cli/cmd.html)
- **Download Page**: [https://org.ngc.nvidia.com/setup/installers/cli](https://org.ngc.nvidia.com/setup/installers/cli)

## Part 2: Configure NGC CLI

### Generate NGC API Key

1. Log in to [NGC](https://ngc.nvidia.com)
2. Navigate to **Setup** → **Generate API Key**
3. Create a new key with at least **NGC Catalog** permissions
4. Copy the generated key (long alphanumeric string, may appear as base64-encoded)

**Note**: NGC API keys are different from NVIDIA API keys (`nvapi-...`) used for build.nvidia.com services.

### Configure CLI

```bash
ngc config set
```

You'll be prompted for:

| Prompt | Value | Notes |
|--------|-------|-------|
| **API key** | Your NGC API key | Long alphanumeric/base64 string from NGC |
| **CLI output format** | `ascii` | Press Enter for default |
| **Org** | Depends on your account | Choose the org shown for your account |
| **Team** | Varies by org | May need specific team for Riva access |
| **ACE** | `no-ace` | Press Enter for default |

**Account Access Notes**:
- **Use whichever NGC account works**: Company or personal, whichever has Riva entitlements
- **NVIDIA employees**: Your company account may or may not have access to internal Riva resources
  - Internal org: `nvidian` (with "n"), team: `swteg-jarvis-jetson`
  - If your company account doesn't work, try a personal NGC account with entitlements
- **External users**: Org options will vary; you may need AI Enterprise trial entitlement
- **Tip**: If configuration succeeds but `riva_init.sh` fails with 403 errors, try a different NGC account

### Verify Access

```bash
# List available Riva resources
ngc registry resource list nvidia/riva/*

# Should show empty table if not entitled, or list resources if access granted
```

If the table is empty after selecting the correct team, contact NGC admin or NVIDIA support for entitlement.

## Part 3: Download Riva Quick Start (ARM64 for Jetson)

Once NGC CLI is configured with proper access:

```bash
# Download the latest ARM64 quickstart bundle for Jetson
ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0

# This extracts to: riva_quickstart_arm64_v2.24.0/
cd riva_quickstart_arm64_v2.24.0
```

**Platform-specific versions**:
- ❌ **x86 data center**: `nvidia/riva/riva_quickstart:<version>` - **NO LONGER SUPPORTED**
  - Use [Riva ASR NIM](https://docs.nvidia.com/nim/nims/riva/) instead
- ✅ **Jetson/ARM64**: `nvidia/riva/riva_quickstart_arm64:<version>` - **SUPPORTED**
  - Current version: `2.24.0`
  - Includes pre-optimized models for Jetson GPUs

## Part 4: Configure Riva Deployment (Jetson)

Edit `config.sh` to customize your deployment:

```bash
cd riva_quickstart_arm64_v2.24.0
vi config.sh  # or your preferred editor
```

### Key Configuration Options for Jetson Thor

```bash
riva_target_gpu_family="tegra"

# Name of tegra platform that is being used. Supported tegra platforms: thor
riva_tegra_platform="thor"

# Services to enable (true/false)
service_enabled_asr=true      # Automatic Speech Recognition
service_enabled_nlp=false     # Natural Language Processing (not needed for voice ASR/TTS)
service_enabled_tts=true      # Text-to-Speech
service_enabled_nmt=false     # Neural Machine Translation (not needed)

# Model storage
riva_model_loc="riva-model-repo"  # Docker volume (default)
# Or use local path: riva_model_loc="/home/jetson/riva-models"

# Language/model selection
asr_acoustic_model="parakeet_1.1b"  # Default for ARM64 v2.24.0
asr_language_code="en-US"           # ASR language
asr_accessory_model="silero_diarizer"  # Adds Silero VAD + speaker diarization
use_asr_streaming_throughput_mode=false  # false=low latency (recommended)

tts_language_code=("multi")           # TTS language
```

### Model Selection Notes for Jetson

**Riva 2.24.0 ARM64 defaults to Parakeet 1.1b:**
- **Parakeet 1.1b**: Newer model, optimized for low latency, excellent quality
- Language codes available: `en-US`, `multi` (multilingual)
- Pre-optimized for Jetson GPUs (no build step required)

**ASR accessory model** (`asr_accessory_model`):
- Set to `"silero_diarizer"` to deploy with **Silero VAD** and speaker diarization
- This makes the `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer` model available alongside the base `parakeet-1.1b-en-US-asr-streaming`
- The Silero VAD variant provides better voice activity detection — without it, the base model often clips the beginning of utterances (e.g., "How many monitors do you see?" becomes "monitors do you see") because its default VAD reacts too late to speech onset
- Only available when `asr_acoustic_model` is `"parakeet_1.1b"`
- After changing this setting, re-run `riva_init.sh` and `riva_start.sh`

**For Multi-modal AI Studio and Live RIVA WebUI**, recommended settings:
- Enable ASR + TTS only (NLP/NMT not needed)
- Use default `parakeet_1.1b` for ASR (best quality/latency balance)
- Set `asr_accessory_model="silero_diarizer"` for Silero VAD support
- Keep `use_asr_streaming_throughput_mode=false` for real-time voice apps
- SSL/TLS can be added later for production deployments

## Part 5: Initialize Riva (Jetson)

This step downloads Docker images and pre-optimized ASR/TTS models for Jetson (~15-45 minutes):

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_init.sh
```

**When you run it**, the script will prompt: `Please enter API key for ngc.nvidia.com:` — use the same NGC API key you configured in Part 2 (from NGC → Setup → Generate API Key).

**What happens on Jetson**:
1. Pulls ARM64-specific `nvcr.io/nvidia/riva/riva-speech` Docker images
2. Downloads **pre-optimized** ASR/TTS models from NGC (optimized for Jetson GPUs)
3. Prepares model repository and configs
4. ✅ **No model optimization step required** (unlike x86 deployments)

**Expected output**:
```
Pulling Docker images...
Downloading models from NGC...
- Parakeet 1.1b English (US) ASR model
- HiFiGAN English (US) TTS model
Preparing model repository...
✓ Initialization complete!
```

**Troubleshooting**:
- **403 errors**: NGC credentials expired or no entitlement
  - Verify: `ngc registry resource list nvidia/riva/*`
  - Fix: Reconfigure with correct team: `ngc config set`
- **Out of disk space**: Models are 5-20 GB; ensure sufficient space on Jetson
  - Check: `df -h`
  - Recommended: 64GB+ available storage
- **GPU not detected**: Verify `nvidia-smi` works (should be pre-installed with JetPack)
  - Test: `sudo docker run --rm --gpus all ubuntu nvidia-smi`

**Time estimate for Jetson Thor**:
- Fast network: ~15-20 minutes
- Slow network: ~30-45 minutes

## Part 6: Start Riva Services (Jetson)

Run from inside the quickstart directory (so `config.sh` is found):

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_start.sh
```

This launches the Riva server via **Docker** (the script uses `docker run`; no Docker Compose). One container is started:
- **riva-speech**: gRPC server on port `50051` (ASR/TTS). A client shell or sample scripts are available separately via `riva_start_client.sh` (see Part 7).

**Note for USB audio**: If using USB microphone/speaker, connect it **before** running `riva_start.sh`. The script will automatically mount it into the container.

### Verify Deployment

```bash
# Check that the riva-speech container is running (name comes from config.sh)
docker ps -f "name=riva-speech"

# Check logs if anything looks wrong
docker logs riva-speech
# Follow logs in real time:
docker logs -f riva-speech
```

Look for successful startup message:
```
Riva server listening on 0.0.0.0:50051
All models loaded successfully
```

**First-time startup on Jetson**: May take 2-5 minutes to load models into GPU memory.

## Part 7: Test ASR with Sample Client (Jetson)

### Start client container (if not already running)

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_start_client.sh
```

This drops you into the `riva-client` container shell at `/opt/riva/`.

### Check available models

```bash
# List streaming ASR models (primary mode for Parakeet 1.1b)
riva_streaming_asr_client --list_models

# Expected output for Riva 2.24.0 ARM64:
# 'en-US': 'parakeet-1.1b-en-us-asr-streaming'
```

**Note**: Riva 2.24.0 on Jetson defaults to **Parakeet 1.1b**, which is optimized for low-latency streaming ASR. This is the recommended model for real-time voice applications (e.g. Multi-modal AI Studio, Live RIVA WebUI).

### Test Streaming ASR (Primary mode for Parakeet 1.1b)

Streaming ASR processes audio in chunks, emitting interim results (lower latency, suitable for real-time):

```bash
riva_streaming_asr_client --audio_file=/opt/riva/wav/en-US_sample.wav
```

**Expected output** (shows interim + final transcripts):
```
Loading eval dataset...
filename: /opt/riva/wav/en-US_sample.wav
Done loading 1 files
what
what is
what is natural
what is natural language
what is natural language processing
What is natural language processing?
-----------------------------------------------------------
File: /opt/riva/wav/en-US_sample.wav

Final transcripts:
0 : What is natural language processing?

Timestamps:
Word                                    Start (ms)      End (ms)        Confidence
What                                    920             960             1.9195e-01
is                                      1200            1240            5.4836e-01
natural                                 1720            2080            1.0869e-01
language                                2240            2600            6.7237e-01
processing?                             2720            3200            1.0000e+00

Audio processed: 4.0000e+00 sec.
-----------------------------------------------------------

Throughput: 8.3569e+00 RTFX
```

**Streaming ASR is the primary mode for Riva 2.24.0 on Jetson**:
- Low latency (~100-200ms)
- Real-time interim results
- Optimized for conversational AI applications (e.g. Multi-modal AI Studio, Live RIVA WebUI)

### Test with Opus file (WebRTC codec)

Riva includes Opus sample files, which is the codec WebRTC uses:

```bash
# List available audio formats
ls /opt/riva/wav/en-US_*

# Output:
# en-US_sample.ogg
# en-US_sample.opus  ← WebRTC audio codec
# en-US_sample.wav
# en-US_wordboosting_sample.wav

# Test streaming ASR with Opus
riva_streaming_asr_client --audio_file=/opt/riva/wav/en-US_sample.opus
```

### Advanced: Python API examples

Alternative Python scripts are available in `/opt/riva/examples/`:

```bash
cd /opt/riva/examples

# List available scripts
ls *.py
# Output:
# nmt.py
# punctuation_client.py
# riva_streaming_asr_client.py
# talk.py
# transcribe_file.py
# transcribe_file_offline.py
# transcribe_mic.py

# Example: streaming ASR with Python
python3 riva_streaming_asr_client.py \
  --input-file /opt/riva/wav/en-US_sample.wav \
  --server riva-speech:50051 \
  --automatic-punctuation
```

### Test TTS

```bash
# From inside riva-client container
riva_tts_client \
  --text="Hello from Riva text to speech" \
  --output=/tmp/tts_output.wav

# Play the generated audio (if you have audio output)
aplay /tmp/tts_output.wav
```

## Part 8: Stop Riva (Jetson)

When finished testing:

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_stop.sh
```

This stops and removes containers while preserving downloaded models in the `riva-model-repo` volume.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│       Voice apps (Multi-modal AI Studio / Live RIVA WebUI)  │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Browser    │◄───────►│  WebUI       │                  │
│  │  (WebRTC)    │  WS/RTC │  Server      │                  │
│  └──────────────┘         └───────┬──────┘                  │
│                                   │ gRPC                    │
├───────────────────────────────────┼─────────────────────────┤
│                    Docker         │                         │
│  ┌────────────────────────────────▼───────────────────┐     │
│  │         riva-speech (port 50051)                   │     │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │     │
│  │  │    ASR     │  │    TTS     │  │    NMT     │    │     │
│  │  │ StreamingR │  │ Synthesize │  │ Translate  │    │     │
│  │  │ ecognize   │  │   Online   │  │            │    │     │
│  │  └────────────┘  └────────────┘  └────────────┘    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  Model Repository (riva-model-repo volume or local path)    │
│  ├── asr/                                                   │
│  │   ├── conformer-en-US/                                   │
│  │   └── parakeet-1.1b-en-US/                               │
│  └── tts/                                                   │
│      └── hifigan-en-US/                                     │
└─────────────────────────────────────────────────────────────┘
```

## Part 9: WebRTC and Opus Audio

### Why Opus matters for voice applications

**Opus is WebRTC's standard audio codec** — all modern browsers encode microphone audio as Opus by default. Riva's inclusion of Opus sample files (`/opt/riva/wav/en-US_sample.opus`) confirms it can handle this codec natively.

For Multi-modal AI Studio and Live RIVA WebUI, the audio flow is:
```
Browser (WebRTC) → Opus audio → WebSocket → Bridge → PCM → Riva gRPC → Transcripts
```

### NVIDIA's WebSocket Bridge

NVIDIA provides an **open-source WebSocket ↔ Riva bridge**: [nvidia-riva/websocket-bridge](https://github.com/nvidia-riva/websocket-bridge)

**Features**:
- Accepts WebSocket connections from browsers
- Decodes Opus audio to PCM
- Forwards to Riva's gRPC `StreamingRecognize`
- Returns transcripts over WebSocket
- Compatible with AudioCodes VoiceGateway (SIP/WebRTC)

**Implementation**: JavaScript/Node.js

For Multi-modal AI Studio and Live RIVA WebUI, options include:
1. Use the nvidia-riva/websocket-bridge as-is
2. Build a Python version integrated into the existing async server (e.g. reusing Live VLM WebUI's WebRTC scaffolding)

## Next Steps (Multi-modal AI Studio / Live RIVA WebUI)

1. **Audio Bridge**: WebSocket/WebRTC → gRPC adapter
   - Accept Opus audio from browser
   - Decode Opus → PCM (or use Riva's native Opus support)
   - Stream to `riva_asr.StreamingRecognize` gRPC
   - Emit transcripts back via WebSocket

2. **LLM Integration**: Route transcripts to configurable LLM endpoint
   - Support OpenAI-compatible APIs (local or cloud)
   - Handle streaming responses

3. **TTS Loop**: Stream LLM response through Riva TTS
   - `riva_tts.SynthesizeOnline` gRPC
   - Send audio back to browser via WebRTC

4. **UI**: Web frontend (e.g. React/TypeScript)
   - Mic capture (WebRTC audio)
   - Live captions overlay
   - Chat transcript panel
   - Settings: LLM endpoint, Riva host, models

5. **Production**: SSL/TLS, resource limits, monitoring, multi-user support

## Troubleshooting

### "Waiting for Riva server to load all models... retrying in 10 seconds" (never finishes)

The Riva server container is not becoming healthy within the timeout. The quickstart uses **plain Docker** (no Compose); troubleshoot as follows.

1. **Run from the quickstart directory** (so `config.sh` is loaded)
   ```bash
   cd /path/to/riva_quickstart_arm64_v2.24.0
   bash riva_start.sh
   ```

2. **Check container status**
   ```bash
   docker ps -a -f "name=riva-speech"
   ```
   - If **riva-speech** is missing or status is **Exited**: the container failed. Check logs (step 3).
   - If it is **Up** but the script still retries: health check may be slow (first load can take several minutes), or the server may be failing internally — check logs.

3. **Inspect riva-speech logs**
   The script suggests: `docker logs riva-speech`. If that shows nothing, see [Health ready check failed and empty logs](#health-ready-check-failed-and-empty-docker-logs-riva-speech).
   ```bash
   docker logs riva-speech
   docker logs --tail=200 riva-speech
   ```
   Look for:
   - **GPU / CUDA errors**: Ensure `nvidia-smi` works and NVIDIA Container Toolkit is installed.
   - **Model not found / path errors**: Re-run `bash riva_init.sh` and ensure it completed without errors.
   - **Out of memory**: Jetson may need more swap or fewer models; disable TTS or NLP in `config.sh` to reduce memory.

4. **Restart cleanly**
   ```bash
   bash riva_stop.sh
   bash riva_start.sh
   ```
   In another terminal, run `docker logs -f riva-speech` to watch startup output.

### "Health ready check failed" and empty `docker logs riva-speech`

The script suggests `docker logs riva-speech` (the container name is set in `config.sh` as `riva_daemon_speech="riva-speech"`). If that command prints **nothing**, check the container **STATUS** with `docker ps -a -f "name=riva-speech"`:

- **STATUS = Created** → The container was created but **never started** (main process never ran). See [Container stuck in Created](#container-stuck-in-created-never-started) below.
- **STATUS = Exited** → The process ran then exited; see step 2 below.
- **STATUS = Up** → Container is running; logs may appear after a short delay, or try `docker logs -f riva-speech`.

1. **Confirm the container exists and its name**
   ```bash
   docker ps -a | grep -i riva
   ```
   The quickstart creates a container named **riva-speech**. If you see a different name (e.g. from an older run or custom config), use that:
   ```bash
   docker logs <container_name_or_id>
   ```

2. **Container exited immediately**
   If the container is **Exited**, it may have crashed before writing much. You can still try:
   ```bash
   docker logs riva-speech
   docker logs --tail=200 riva-speech
   ```
   Exited containers often keep stdout/stderr; if logs are still empty, the process may have died before any output. Run again and watch in real time:
   ```bash
   bash riva_stop.sh
   bash riva_start.sh
   ```
   In a second terminal, as soon as the container starts:
   ```bash
   docker logs -f riva-speech
   ```
   Look for GPU/CUDA, model path, or OOM errors in the first lines.

### Container stuck in **Created** (never started)

If `docker ps -a -f "name=riva-speech"` shows **STATUS = Created** (and no "Up" time), the container was created by `docker run -d` but the main process never started. There are no logs because the entrypoint hasn't run. Common causes: missing or inaccessible device (e.g. GPU, USB/sound), volume mount failure, or Docker/runtime blocking start.

**Do this:**

1. **Remove the stuck container and try again from the quickstart directory**
   ```bash
   docker rm -f riva-speech
   cd /path/to/riva_quickstart_arm64_v2.24.0
   bash riva_start.sh
   ```
   In a second terminal, watch for the container to go from Created → Up and then stream logs:
   ```bash
   watch -n 1 'docker ps -a -f "name=riva-speech"'
   # When STATUS becomes "Up", run:
   docker logs -f riva-speech
   ```

2. **If it stays in Created again**, try starting it manually to see the error:
   ```bash
   docker start riva-speech
   docker logs -f riva-speech
   ```
   If `docker start` fails or logs show nothing, inspect the container:
   ```bash
   docker inspect riva-speech
   ```
   Check **`State.Error`** for the exact message. A very common one on Jetson is below.

4. **If you see: "invoking the NVIDIA Container Runtime Hook directly ... use the NVIDIA Container Runtime (--runtime=nvidia) instead"**
   See [Riva container stays "Created": use NVIDIA Container Runtime](#riva-container-stays-created-use-nvidia-container-runtime) below.

5. **Verify GPU and devices**
   The Riva start script mounts `--gpus` and on Tegra also `--device /dev/bus/usb --device /dev/snd`. Ensure:
   - `nvidia-smi` works and NVIDIA Container Toolkit is installed.
   - No security profile (e.g. AppArmor) is blocking device access.
   - If you don't need USB/sound for the server, you could temporarily comment out the extra `--device` flags in `riva_start.sh` to see if the container then starts (for debugging only).

### Container starts then exits immediately (or stays "Created")

If the container goes **Created** and never shows **Up**, or it exits so quickly that `docker logs riva-speech` is empty, the script is hiding the error: it runs `docker run -d ... &> /dev/null`, so all output is discarded. Run the same container **in the foreground** so you see the real error (CUDA, model path, OOM, etc.):

```bash
cd /path/to/riva_quickstart_arm64_v2.24.0
source config.sh

# Remove any existing container so we can use the same name
docker rm -f riva-speech 2>/dev/null

# Same as riva_start.sh but -it (foreground) and no -d; output goes to your terminal
docker run -it --rm \
  --init --ipc=host \
  --gpus "$gpus_to_use" \
  -p $riva_speech_api_port:$riva_speech_api_port \
  -p $riva_speech_api_http_port:$riva_speech_api_http_port \
  -e RIVA_SERVER_HTTP_PORT=$riva_speech_api_http_port \
  -e "LD_PRELOAD=$ld_preload" \
  -e "RIVA_API_KEY=$RIVA_API_KEY" \
  -e "RIVA_API_NGC_ORG=$RIVA_API_NGC_ORG" \
  -e "RIVA_EULA=$RIVA_EULA" \
  -v $riva_model_loc:/data \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 \
  $image_speech_api \
  start-riva --riva-uri=0.0.0.0:$riva_speech_api_port \
  --asr_service=$service_enabled_asr \
  --tts_service=$service_enabled_tts \
  --nlp_service=$service_enabled_nlp
```

(On Tegra the script also adds `--device /dev/bus/usb --device /dev/snd`; if the command above runs and you need those, add them before `$image_speech_api`.)

- **What you see** is the real failure (e.g. "could not load model", "CUDA error", "No such file", OOM). Fix that and then use `bash riva_start.sh` again.
- **If it stays in Created** even with this foreground run, the failure is before the process starts (e.g. device or runtime); check `docker events` in another terminal and run the `docker run` above to see the event error.

### Riva container stays "Created": use NVIDIA Container Runtime

If `docker inspect riva-speech` shows in **State.Error** something like:

```text
invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the docker --gpus flag) is not supported.
Please use the NVIDIA Container Runtime (e.g. specify the --runtime=nvidia flag) instead: unknown
```

then Docker on this host is set up to use the **NVIDIA Container Runtime** (full runtime), not the hook used by `--gpus`. The container never starts because the runtime rejects the `--gpus`-based GPU setup.

**Fix A — Configure Docker to use the NVIDIA runtime by default (recommended)**
Ensure `/etc/docker/daemon.json` has the nvidia runtime and set it as default:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

If the file already has `"runtimes": { "nvidia": ... }` but no `"default-runtime": "nvidia"`, add that. Then:

```bash
sudo systemctl restart docker
```

After that, run `bash riva_start.sh` again.

**Fix B — Workaround: use `--runtime=nvidia` in the start script**
If you prefer not to change the default runtime, patch `riva_start.sh` so the container uses the nvidia runtime on Tegra instead of `--gpus`:

1. Open `riva_start.sh` in your quickstart directory.
2. Find the line: `--gpus '"'$gpus_to_use'"' \`
3. Replace it with: `--runtime=nvidia \`
   (This is safe for Jetson/Tegra; the nvidia runtime gives the container GPU access.)

Then run `bash riva_start.sh` again.

### "403 Forbidden" when downloading quickstart

- **Cause**: NGC account lacks Riva entitlement
- **Fix**:
  - NVIDIA employees: Select `swteg-jarvis-jetson` team in `ngc config set`
  - External users: Request AI Enterprise trial or contact NVIDIA sales

### "No such file or directory: libpython3.11.so.1.0"

- **Cause**: NGC CLI moved without dependencies
- **Fix**: Symlink instead of moving: `ln -s ~/.local/share/ngc-cli/ngc-cli/ngc ~/.local/bin/ngc`

### Riva server won't start / GPU errors

- **Verify GPU**: `nvidia-smi` should show your GPU
- **Check toolkit**: `docker run --rm --gpus all ubuntu nvidia-smi`
- **Review logs**: `docker logs riva-speech` (container name from `config.sh`: `riva_daemon_speech`)

### Models downloading very slowly

- **Expected**: Initial download is 10-30 GB depending on models
- **Workaround**: Use Jetson pre-optimized models if on ARM, or download once and reuse `riva_model_loc`

## Resources

- [NVIDIA Riva Documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/)
- [Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/)
- [NGC CLI Documentation](https://docs.nvidia.com/ngc/ngc-cli/index.html)
- [Riva GitHub Org](https://github.com/nvidia-riva) (sample apps, tutorials)
- [NVIDIA LaunchPad](https://www.nvidia.com/en-us/launchpad/) (pre-configured labs)

---

**Document Status**: Updated for Jetson ARM64 deployment (x86 support discontinued)
**Last Updated**: March 2025
**Riva Version**: 2.24.0 (ARM64)
**Platform**: NVIDIA Jetson Thor (JAT03)

