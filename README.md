# Multi-modal AI Studio

![](./docs/images/screenshot_example_2.png)

**Voice, text, and video conversational AI with session analysis and latency metrics**

Multi-modal AI Studio is a conversational AI interface for building and tuning voice AI systems. It supports NVIDIA Riva, OpenAI, and other backends; records sessions with full config snapshots; and provides a real-time timeline and latency analysis (TTFA, turn-taking) to compare and optimize setups.

## 🌟 Key Features

### Multi-modal Support
- **Voice**: Streaming ASR and TTS (Riva, OpenAI, or other backends)
- **Text**: Chat-only mode or combined with voice
- **Video**: Camera feed for vision-language models (VLM); browser WebRTC or server USB webcam
- **Mixed modes**: Voice-to-text, text-to-voice, voice-to-voice, or text-only

### Multi-backend Architecture
- Speech
  - **NVIDIA Riva**: gRPC streaming ASR/TTS (Jetson/ARM64)
  - **OpenAI-compatible Realtime API**: Realtime API
- LLM: **OpenAI-compatible** REST API, to works with many inference engines for various LLM/VLM models
- **Extensible**: Plugin-style backends; Azure Speech and others can be added

### Session Management
- **Config snapshots**: Every session stores ASR/LLM/TTS and device settings
- **Timeline recording**: Performance data for offline analysis
- **Presets**: Save and load configuration presets

### Performance Analysis
- **Real-time timeline**: Multi-lane view (Audio, Speech, LLM, TTS)
- **Latency metrics**: TTFA (Time to First Audio), turn-taking

### UI & Devices
- **Chat-style UI**: Familiar layout, video full-screen mode, keyboard shortcuts. Most settings are exposed in the UI (ASR/LLM/TTS, models, devices) so you can tweak and switch backends without editing config files or code.
- **Devices**: Client-side (browser WebRTC) and server-side (Linux USB mic, USB speaker, USB webcam); choose in the Devices tab.
- **Headless** (experimental, not well tested): CLI with config file or args; see [INSTALL.md](INSTALL.md).

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Audio/video**: Browser (WebRTC) for mic, speaker, and camera. On Linux, server **USB microphone**, **USB speaker**, and **USB webcam** are also supported; see [INSTALL.md](INSTALL.md).
- **Backends (as needed)**: [NVIDIA Riva](INSTALL.md#nvidia-riva-setup-for-voice-asrtts) for ASR/TTS; OpenAI API key for OpenAI/Realtime backends (optional).
- **Optional**: `jq` for pretty-printed LLM logs in the console (`apt install jq` or `brew install jq`).

### Installation

Use a virtual environment (e.g. `.venv`) so dependencies stay isolated. Recommended:

> **Note:** If `python3 -m venv` fails with "No module named venv", install the venv package for your Python version:
> ```bash
> sudo apt install python3-venv   # or python3.X-venv, e.g. python3.10-venv, python3.12-venv
> ```

```bash
# Clone repository
git clone https://github.com/NVIDIA-AI-IOT/multi_modal_ai_studio.git
cd multi_modal_ai_studio

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

One-line setup (creates `.venv`, installs deps): `./scripts/setup_dev.sh`
Full steps and troubleshooting: [INSTALL.md](INSTALL.md)

### Run WebUI

```bash
# View sessions and timeline (no backend required)
python -m multi_modal_ai_studio --port 8092
```

Open **https://localhost:8092** in your browser. For voice (Riva, OpenAI, etc.) and other options, see [INSTALL.md](INSTALL.md).

### Kill a Running Server

If the server is running in the background or the port is stuck with `address already in use`:

```bash
# Find and kill the process on port 8092
fuser -k 8092/tcp

# Or find the PID manually
lsof -i :8092
kill <PID>
```

### Sessions and sample data

Sessions are stored in `sessions/` by default. To try sample timelines, run with `--session-dir mock_sessions` and open a session from the sidebar.

### Run headless (experimental)

CLI-only mode for automation or local audio devices. Requires the `[audio]` extra and device setup; see [INSTALL.md](INSTALL.md).

```bash
python -m multi_modal_ai_studio --mode headless --config my-config.yaml

# Or with CLI args (e.g. ALSA devices)
python -m multi_modal_ai_studio --mode headless \
  --audio-input alsa:hw:0,0 --audio-output alsa:hw:1,0 \
  --asr-scheme riva --llm-model llama3.2:3b
```

## 📖 Documentation

| Doc | Description |
|-----|-------------|
| [INSTALL.md](INSTALL.md) | Installation, backends, and troubleshooting |
| [Riva Setup](docs/setup_riva.md) | NVIDIA Riva ASR/TTS (Jetson/ARM64) |
| [VLM Guide](docs/vlm_guide.md) | Vision-language models, frame capture, tuning |
| [Architecture](docs/architecture.md) | System design and components |

## 🤝 Contributing

This project is under active development. Issues, pull requests, and feedback are welcome!

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

