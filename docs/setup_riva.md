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
- **Docker + NVIDIA Container Toolkit**: Pre-installed on JetPack
- **NGC account with Riva access**: Required for downloading Riva resources
  - Try your account that has Riva entitlements (company or personal)
  - NVIDIA employees: Internal access may require specific team membership
  - External users: May need AI Enterprise trial or proper entitlements
  - **Tip**: If one account doesn't work, try another you have access to

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
ngc registry resource list nvidia/riva

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
use_asr_streaming_throughput_mode=false  # false=low latency (recommended)

tts_language_code=("multi")           # TTS language
```

### Model Selection Notes for Jetson

**Riva 2.24.0 ARM64 defaults to Parakeet 1.1b:**
- **Parakeet 1.1b**: Newer model, optimized for low latency, excellent quality
- Language codes available: `en-US`, `multi` (multilingual)
- Pre-optimized for Jetson GPUs (no build step required)

**For Multi-modal AI Studio and Live RIVA WebUI**, recommended settings:
- Enable ASR + TTS only (NLP/NMT not needed)
- Use default `parakeet_1.1b` for ASR (best quality/latency balance)
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
  - Verify: `ngc registry resource list nvidia/riva`
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

```bash
cd riva_quickstart_arm64_v2.24.0
bash riva_start.sh
```

This launches the Riva server via Docker Compose. Services:
- **riva-speech**: gRPC server on port `50051` (ASR/TTS)
- **riva-client**: Client container with sample scripts and test files

**Note for USB audio**: If using USB microphone/speaker, connect it **before** running `riva_start.sh`. The script will automatically mount it into the container.

### Verify Deployment

```bash
# Check container status
docker compose ps

# Expected output:
# NAME                  STATUS
# riva-speech           Up X minutes
# riva-client           Up X minutes

# Check logs
docker compose logs -f riva-speech
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

**Note**: Riva 2.24.0 on Jetson defaults to **Parakeet 1.1b**, which is optimized for low-latency streaming ASR. This is the recommended model for real-time voice applications like Live RIVA WebUI.

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
- Optimized for conversational AI applications like Live RIVA WebUI

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
│                     Live RIVA WebUI                         │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Browser    │◄───────►│  WebUI       │                  │
│  │  (WebRTC)    │  WS/RTC │  Server      │                  │
│  └──────────────┘         └───────┬──────┘                  │
│                                   │ gRPC                    │
├───────────────────────────────────┼─────────────────────────┤
│                    Docker         │                         │
│  ┌────────────────────────────────▼───────────────────┐     │
│  │         riva-speech-api (port 50051)               │     │
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

### Why Opus matters for Live RIVA WebUI

**Opus is WebRTC's standard audio codec** - all modern browsers encode microphone audio as Opus by default. Riva's inclusion of Opus sample files (`/opt/riva/wav/en-US_sample.opus`) confirms it can handle this codec natively.

For Live RIVA WebUI, the audio flow will be:
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

For Live RIVA WebUI, we can either:
1. Use the nvidia-riva/websocket-bridge as-is
2. Build a Python version integrated into our existing async server (reusing Live VLM WebUI's WebRTC scaffolding)

## Next Steps for Live RIVA WebUI

1. **Audio Bridge**: Build WebSocket/WebRTC → gRPC adapter
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

4. **UI**: React/TypeScript frontend
   - Mic capture (WebRTC audio)
   - Live captions overlay
   - Chat transcript panel
   - Settings: LLM endpoint, Riva host, models

5. **Production**: SSL/TLS, resource limits, monitoring, multi-user support

## Troubleshooting

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
- **Review logs**: `docker compose logs riva-speech-api`

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
**Last Updated**: January 2025
**Riva Version**: 2.24.0 (ARM64)
**Platform**: NVIDIA Jetson Thor (JAT03)

