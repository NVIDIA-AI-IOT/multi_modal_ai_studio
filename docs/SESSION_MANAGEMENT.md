# Session Management

Session management is the core of Multi-modal AI Studio's analysis capabilities. Each session records complete configuration, timeline events, and calculated metrics for offline analysis.

## Overview

A **Session** represents one complete conversational interaction with:

- **Configuration Snapshot**: ASR, LLM, TTS, device, and app settings
- **Timeline Events**: All pipeline events with microsecond precision
- **Conversation Turns**: User inputs and AI responses
- **Calculated Metrics**: TTL, component latencies, confidence scores

Sessions can be saved to JSON and loaded later for analysis, comparison, and visualization.

## Core Concepts

### Timeline

The `Timeline` records all events that occur during a voice AI session:

```python
from multi_modal_ai_studio.core import Timeline, Lane, EventType

timeline = Timeline()
timeline.start()  # Sets time origin

# Add events as they occur
timeline.add_event(EventType.USER_SPEECH_END, Lane.AUDIO)
timeline.add_event(EventType.ASR_FINAL, Lane.SPEECH, {"text": "Hello!", "confidence": 0.98})
timeline.add_event(EventType.TTS_FIRST_AUDIO, Lane.TTS)
```

**Timeline Lanes:**
- `SYSTEM`: Session start/stop, pauses
- `AUDIO`: Raw audio events, levels
- `SPEECH`: ASR (transcription) events
- `LLM`: Language model processing
- `TTS`: Text-to-speech synthesis

### Session

The `Session` class manages the complete conversational state:

```python
from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core import Session

# Load configuration
config = SessionConfig.from_yaml("presets/default.yaml")

# Create session
session = Session(config, name="Latency Test 1")
session.start()

# Record a turn
session.start_turn()
session.add_event(EventType.USER_SPEECH_START, Lane.AUDIO)
# ... more events ...
session.update_turn_transcript("Hello, how are you?", confidence=0.98)
session.update_turn_response("I'm doing well, thank you!")
session.end_turn()

# Calculate metrics
metrics = session.calculate_metrics()
print(f"Average TTL: {metrics.avg_ttl * 1000:.0f}ms")

# Save for later analysis
session.save("sessions/test1.json")
```

### Key Metrics

#### Turn-Taking Latency (TTL) - PRIMARY METRIC

**TTL measures the time from when the user stops speaking to when the AI starts speaking.**

```
TTL = tts_first_audio - user_speech_end
```

This is the critical metric that users actually perceive. It includes:
- ASR finalization time (after VAD detects silence)
- LLM generation time (prefill + first token)
- TTS synthesis time (up to first audio chunk)

**Example:**
```python
# Get TTL for the last turn
ttl = session.timeline.calculate_ttl()
print(f"TTL: {ttl * 1000:.0f}ms")

# Get component breakdown
latencies = session.timeline.calculate_component_latencies()
print(f"ASR: {latencies['asr_latency'] * 1000:.0f}ms")
print(f"LLM: {latencies['llm_latency'] * 1000:.0f}ms")
print(f"TTS: {latencies['tts_latency'] * 1000:.0f}ms")
print(f"Total TTL: {latencies['ttl'] * 1000:.0f}ms")
```

#### Session Metrics

The `SessionMetrics` dataclass provides aggregate statistics:

```python
@dataclass
class SessionMetrics:
    total_turns: int          # Number of conversation turns
    avg_ttl: float           # Average Turn-Taking Latency
    min_ttl: float           # Fastest turn
    max_ttl: float           # Slowest turn
    avg_asr_latency: float   # Average ASR processing time
    avg_llm_latency: float   # Average LLM generation time
    avg_tts_latency: float   # Average TTS synthesis time
    session_duration: float  # Total session time
    active_duration: float   # Active conversation time (excludes pauses)
```

## Event Types

Standard event types are defined in `EventType`:

### System Events
- `SESSION_START`: Session begins
- `SESSION_END`: Session ends
- `SESSION_PAUSE`: Session paused
- `SESSION_RESUME`: Session resumed

### Audio Events
- `AUDIO_START`: Audio stream started
- `AUDIO_LEVEL`: Audio level measurement
- `AUDIO_END`: Audio stream stopped

### ASR Events
- `USER_SPEECH_START`: VAD detected speech
- `USER_SPEECH_END`: **VAD detected silence (KEY for TTL!)**
- `ASR_PARTIAL`: Partial transcript
- `ASR_FINAL`: Final transcript

### LLM Events
- `LLM_START`: LLM processing begins
- `LLM_FIRST_TOKEN`: Prefill complete, first token generated
- `LLM_TOKEN`: Token generated
- `LLM_COMPLETE`: Response complete

### TTS Events
- `TTS_START`: TTS synthesis begins
- `TTS_FIRST_AUDIO`: **First audio chunk ready (KEY for TTL!)**
- `TTS_AUDIO`: Audio chunk generated
- `TTS_COMPLETE`: Synthesis complete

### Barge-in Events
- `BARGE_IN`: User interrupted AI

## Usage Patterns

### Backend Integration

Backends should emit timeline events as they process:

```python
class MyASRBackend(ASRBackend):
    async def process_audio(self, audio_data, session):
        # Emit events to session timeline
        if vad_detected_speech_start:
            session.add_event(EventType.USER_SPEECH_START, Lane.AUDIO)
        
        if partial_transcript:
            session.add_event(
                EventType.ASR_PARTIAL,
                Lane.SPEECH,
                {"text": partial_transcript, "is_final": False}
            )
        
        if vad_detected_silence:
            session.add_event(EventType.USER_SPEECH_END, Lane.AUDIO)
            
        if final_transcript:
            session.add_event(
                EventType.ASR_FINAL,
                Lane.SPEECH,
                {"text": final_transcript, "confidence": confidence}
            )
            session.update_turn_transcript(final_transcript, confidence)
```

### Session Comparison

Compare multiple sessions to analyze configuration impact:

```python
# Load multiple sessions
sessions = [
    Session.load("sessions/test1.json"),
    Session.load("sessions/test2.json"),
    Session.load("sessions/test3.json"),
]

# Compare TTL
for session in sessions:
    metrics = session.calculate_metrics()
    print(f"{session.name}: {metrics.avg_ttl * 1000:.0f}ms TTL")
```

### Headless Mode

In headless mode, sessions auto-save after completion:

```bash
# Run with auto-save
multi-modal-ai-studio --no-webui \
  --config test_config.yaml \
  --session-dir ./sessions
```

## File Format

Sessions are saved as JSON with the following structure:

```json
{
  "session_id": "c8521434-...",
  "name": "Test Session",
  "created_at": "2026-02-03T10:30:00",
  "config": {
    "asr": {...},
    "llm": {...},
    "tts": {...},
    "device": {...},
    "app": {...}
  },
  "timeline": [
    {
      "timestamp": 0.0,
      "event_type": "session_start",
      "lane": "system",
      "data": {}
    },
    {
      "timestamp": 0.523,
      "event_type": "user_speech_end",
      "lane": "audio",
      "data": {}
    },
    {
      "timestamp": 0.684,
      "event_type": "tts_first_audio",
      "lane": "tts",
      "data": {"chunk_size": 2048}
    }
  ],
  "turns": [
    {
      "turn_id": 1,
      "user_transcript": "Hello, how are you?",
      "ai_response": "I'm doing well, thank you!",
      "user_confidence": 0.98,
      "start_time": 0.1,
      "end_time": 2.3,
      "latencies": {
        "asr_latency": 0.05,
        "llm_latency": 0.08,
        "tts_latency": 0.03,
        "ttl": 0.161
      }
    }
  ],
  "metrics": {
    "total_turns": 3,
    "avg_ttl": 0.161,
    "min_ttl": 0.158,
    "max_ttl": 0.165,
    "avg_asr_latency": 0.05,
    "avg_llm_latency": 0.08,
    "avg_tts_latency": 0.03,
    "session_duration": 8.5,
    "active_duration": 8.5
  }
}
```

## Best Practices

1. **Start timeline early**: Call `session.start()` before any audio processing
2. **Emit events liberally**: More events = better visualization and debugging
3. **Use standard event types**: Prefer `EventType` constants over strings
4. **Calculate metrics often**: Metrics are cheap to compute, refresh frequently
5. **Save sessions regularly**: Auto-save after each turn or on timeout
6. **Include event data**: Add context (confidence, chunk size, etc.) to events

## Next Steps

- Implement backend timeline integration (ASR, LLM, TTS emit events)
- Build WebUI timeline visualization
- Add session comparison view
- Export sessions to CSV for external analysis
