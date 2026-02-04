# Audio Modes Explained

Multi-modal AI Studio supports two audio routing modes:

## 1. Browser Audio (WebRTC) - Default

**What**: Audio captured/played through user's browser  
**Requires**: Nothing extra (just web browser)  
**Use case**: WebUI mode, remote access, demos

### How It Works
```
[Browser Microphone] 
    ↓ WebRTC/WebSocket
[Server] → Riva ASR → LLM → Riva TTS
    ↓ WebSocket
[Browser Speaker]
```

### Setup
```bash
# Install without audio support
pip install -e .

# Run WebUI
multi-modal-ai-studio --port 8091
```

### Pros & Cons
- ✅ No hardware setup needed
- ✅ Works remotely (access from any device)
- ✅ Cross-platform (Windows, Mac, Linux, mobile)
- ✅ Easier to get started
- ⚠️ Requires HTTPS (for microphone permission)
- ⚠️ Network latency (audio over WebSocket)

---

## 2. Local USB Audio (pyaudio/portaudio)

**What**: Direct access to USB/ALSA audio devices  
**Requires**: pyaudio + portaudio system library  
**Use case**: Headless mode, low latency, production

### How It Works
```
[USB Microphone]
    ↓ pyaudio (ALSA/USB)
Riva ASR → LLM → Riva TTS
    ↓ pyaudio (ALSA/USB)
[USB Speaker]
```

### Setup
```bash
# Install system dependencies
sudo apt-get install -y portaudio19-dev

# Install with audio support
pip install -e ".[audio]"

# Or use setup script and answer Y
./scripts/setup_dev.sh
```

### Run Headless Mode
```bash
# List available devices
python3 -c "import pyaudio; \
    p = pyaudio.PyAudio(); \
    [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') \
     for i in range(p.get_device_count())]"

# Run with specific devices
multi-modal-ai-studio \
  --mode headless \
  --audio-input alsa:hw:0,0 \
  --audio-output alsa:hw:1,0
```

### Pros & Cons
- ✅ Lower latency (no network round-trip)
- ✅ Better audio quality
- ✅ Works offline
- ✅ Production-ready (headless servers)
- ⚠️ Requires hardware setup
- ⚠️ More complex dependencies (portaudio)
- ⚠️ Platform-specific (ALSA on Linux)

---

## When to Use Each

### Use Browser Audio (Default) When:
- 🌐 **Just starting out** - Easier setup
- 🌐 **WebUI development** - No hardware needed
- 🌐 **Remote demos** - Show to others over network
- 🌐 **Testing/prototyping** - Quick iterations
- 🌐 **Multiple users** - Different people accessing server

### Use Local USB Audio When:
- 🎤 **Headless deployment** - Running as service without browser
- 🎤 **Production systems** - Kiosks, robots, embedded devices
- 🎤 **Low latency critical** - Real-time conversations
- 🎤 **High audio quality** - Professional microphones/speakers
- 🎤 **Offline operation** - No network required

---

## Mixed Mode (Advanced)

You can also **mix** both modes in WebUI:

### Example: USB Microphone + Browser Output
```yaml
devices:
  audio_input_source: usb          # High-quality USB mic
  audio_input_device: "hw:0,0"
  audio_output_source: browser     # Play through browser
```

**Use case**: Better input quality but want browser playback

### Example: Browser Input + USB Speaker
```yaml
devices:
  audio_input_source: browser      # User's laptop mic
  audio_output_source: usb         # Local high-quality speaker
  audio_output_device: "hw:1,0"
```

**Use case**: Remote user but local playback

---

## Recommendation for Phase 1

**Start with Browser Audio Only**:

1. ✅ Simpler setup (no pyaudio/portaudio)
2. ✅ Faster to test WebUI
3. ✅ Cross-platform development
4. ✅ Can add USB support later (Phase 5)

**Add USB Audio Later** (Phase 5):
- After WebUI works
- For headless mode
- For production deployments

---

## Installation Summary

### Browser Audio Only (Phase 1 & 2)
```bash
pip install -e .  # No [audio] extra
```

**Installed packages**:
- aiohttp, grpcio, nvidia-riva-client
- openai, pyyaml, numpy, websockets
- (No pyaudio)

### With USB Audio Support (Phase 5)
```bash
sudo apt-get install -y portaudio19-dev
pip install -e ".[audio]"
```

**Additional packages**:
- pyaudio (requires portaudio system library)

---

## Architecture Implications

### Code Structure
Our `DeviceRouter` class handles both:

```python
class DeviceRouter:
    def setup_audio_input(self, config: DeviceConfig):
        if config.audio_input_source == "browser":
            return WebRTCAudioBridge()  # WebSocket
        elif config.audio_input_source == "usb":
            return USBAudioCapture()     # pyaudio (optional)
```

### Graceful Degradation
If pyaudio not installed:
- ✅ Browser mode works fine
- ⚠️ USB mode shows helpful error: "Install pyaudio for USB support"

---

## Testing Without USB Devices

You can develop/test everything with browser audio:
- ✅ Backend logic (ASR, LLM, TTS) is device-agnostic
- ✅ WebUI works with browser audio
- ✅ Session recording works the same
- ✅ Timeline visualization same

USB audio is just an **I/O adapter** - the core pipeline is identical.
