# Multi-modal AI Studio

**Voice, Text, and Video AI Interface with Advanced Performance Analysis**

Multi-modal AI Studio is a next-generation conversational AI interface designed for analyzing and optimizing voice AI systems. Built on NVIDIA Riva, OpenAI APIs, and other backends, it features sophisticated session management, real-time timeline visualization, and comprehensive latency metrics.

## 🌟 Key Features

### Multi-modal Support
- **Voice Input/Output**: Streaming ASR and TTS via Riva or OpenAI
- **Text Chat**: Traditional text-based conversation
- **Video**: Camera feed for vision-enabled models (future)
- **Mixed Modes**: Voice-to-text, text-to-voice, or text-only

### Multi-backend Architecture
- **NVIDIA Riva**: gRPC streaming ASR/TTS
- **OpenAI**: REST API (Whisper, TTS) and Realtime API
- **Azure Speech**: Coming soon
- **Custom backends**: Extensible plugin system

### Session Management
- **Configuration Snapshots**: Every session saves ASR/LLM/TTS configs
- **Timeline Recording**: Store performance data for offline analysis
- **Preset System**: Save and load configuration presets
- **Export/Import**: Generate CLI commands or YAML configs from WebUI

### Performance Analysis
- **Real-time Timeline**: Multi-lane visualization (Audio, Speech, LLM, TTS)
- **Latency Metrics**: TTFA (Time to First Audio), turn-taking analysis
- **Comparison Mode**: Compare multiple sessions to optimize configs
- **Session Replay**: Analyze recorded timeline data

### Flexible Deployment
- **WebUI Mode**: Rich browser interface (default)
- **Headless Mode**: CLI-only for production/automation
- **Device Flexibility**: Browser WebRTC or local USB devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA Riva (for Riva backend) - see [docs/setup.md](docs/setup.md)
- OpenAI API key (for OpenAI backend) - optional

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/multi-modal-ai-studio.git
cd multi-modal-ai-studio

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Run WebUI

```bash
# With Riva backend
multi-modal-ai-studio \
  --port 8091 \
  --riva-server localhost:50051 \
  --llm-api-base http://localhost:11434/v1 \
  --llm-model llama3.2:3b

# With OpenAI Realtime API
multi-modal-ai-studio \
  --port 8091 \
  --asr-scheme openai-realtime \
  --tts-scheme openai-realtime \
  --openai-api-key sk-...

# With preset
multi-modal-ai-studio --preset low-latency
```

Open browser to `https://localhost:8091`

### Run Headless

```bash
# From config file
multi-modal-ai-studio --mode headless --config my-config.yaml

# From CLI args
multi-modal-ai-studio \
  --mode headless \
  --audio-input alsa:hw:0,0 \
  --audio-output alsa:hw:1,0 \
  --asr-scheme riva \
  --llm-model llama3.2:3b
```

## 📖 Documentation

- [Setup Guide](docs/setup.md)
- [Configuration Reference](docs/configuration.md)
- [API Backends](docs/api-backends.md)
- [CLI Reference](docs/cli-reference.md)
- [Presets](docs/presets.md)

## 🏗️ Project Status

**Current Phase**: Foundation (Phase 1)

- [x] Project structure
- [x] Configuration system design
- [x] Cursor rules and documentation
- [ ] Riva backend implementation
- [ ] Basic WebUI
- [ ] Session storage
- [ ] CLI interface

See [docs/cursor/IMPLEMENTATION_PHASES.md](docs/cursor/IMPLEMENTATION_PHASES.md) for roadmap.

## 🤝 Contributing

This project is under active development. Issues, pull requests, and feedback are welcome!

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on top of proven concepts from [Live RIVA WebUI](https://github.com/yourusername/live-riva-webui).
