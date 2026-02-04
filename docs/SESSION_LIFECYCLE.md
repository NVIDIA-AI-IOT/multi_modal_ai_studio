# Session Lifecycle & Workflow

## Overview

Multi-modal AI Studio has a deliberate workflow that separates **configuration** from **recording**. This allows users to test devices, preview video, and adjust settings before beginning the analyzed session.

## Session States

### 1. Setup Mode (Configuration & Preview)

**Entry**: User clicks "+ New Voice Chat" button

**Purpose**: Configure and test everything before recording

**Features Available**:
- ✅ Edit all ASR/LLM/TTS configuration
- ✅ Select video/mic/speaker devices
- ✅ Preview camera feed (WebRTC active)
- ✅ Test microphone (see audio levels)
- ✅ Verify speaker output
- ❌ NO timeline recording
- ❌ NO conversation storage
- ❌ NO performance analysis

**UI Elements**:
- Config panel: Fully editable
- Video container: Shows camera preview
- Device controls: Dropdowns active, audio level indicators visible
- Session button: Green "▶ START Session" button
- Status: Gray indicator "Setup Mode"

**User Actions**:
1. Adjust ASR settings (VAD thresholds, language, etc.)
2. Select LLM model and parameters
3. Configure TTS voice and speed
4. Pick camera/mic/speaker devices
5. See video preview to confirm correct camera
6. Speak to see mic level indicator move
7. **When ready**: Click "▶ START Session"

---

### 2. Live Mode (Recording & Analysis)

**Entry**: User clicks "▶ START Session" from Setup Mode

**Purpose**: Record conversation with full timeline tracking and metrics

**Features Available**:
- ✅ All ASR/LLM/TTS backends active
- ✅ Timeline recording every event
- ✅ Turn-by-turn conversation tracking
- ✅ Real-time metrics calculation (TTL, latencies)
- ✅ Live timeline visualization updates
- ⚠️ Configuration read-only (changes require restart)
- ✅ Conversation history accumulates

**UI Elements**:
- Config panel: Read-only (collapsed recommended)
- Video container: Live video feed
- Device controls: Active, showing real-time levels
- Session button: Red "⏸ STOP Session" button
- Status: Red pulsing indicator "🔴 Recording & Analyzing"
- Chat panel: User/AI messages appear in real-time
- Timeline: Updates live as events occur

**User Actions**:
1. Have voice conversation with AI
2. Watch timeline populate with events
3. Monitor TTL and component latencies
4. See ASR transcripts appear
5. Hear TTS responses
6. **When done**: Click "⏸ STOP Session"

**Backend Activity**:
```
User speaks → ASR streaming → VAD detects end
  → Timeline: user_speech_start, user_speech_end
  → ASR final transcript
  → Timeline: asr_final
  → LLM generation starts
  → Timeline: llm_start, llm_first_token, llm_tokens, llm_complete
  → TTS synthesis starts
  → Timeline: tts_start, tts_first_audio, tts_complete
  → Calculate TTL: user_speech_end → tts_first_audio
  → Display in chat + metrics
```

---

### 3. Stopped Mode (Session Complete)

**Entry**: User clicks "⏸ STOP Session" from Live Mode

**Purpose**: Finalize and save the recorded session

**Actions**:
1. Stop all WebRTC streams
2. Close ASR/LLM/TTS connections
3. Finalize timeline data
4. Calculate final session metrics:
   - Total turns
   - Avg/min/max TTL
   - Component latencies (ASR, LLM, TTS)
   - Session duration
5. Save session JSON to disk
6. Add to session list
7. Generate session ID and timestamp

**Result**: New session appears in sidebar, ready for analysis

---

### 4. Historical Mode (Playback & Analysis)

**Entry**: User clicks on a completed session from the sidebar

**Purpose**: Review and analyze past sessions

**Features Available**:
- ✅ View complete conversation history
- ✅ See all configuration used
- ✅ Timeline visualization (playback)
- ✅ Metrics comparison
- ✅ Export session data
- ❌ NO live recording
- ❌ NO device access
- ❌ Configuration read-only

**UI Elements**:
- Config panel: Read-only, shows settings used
- Video container: Hidden (no video recording saved)
- Device controls: Hidden
- Chat panel: Full conversation history
- Timeline: Rendered from recorded events, scrubber available
- Metrics: Final calculated values displayed

**User Actions**:
1. Review conversation quality
2. Analyze TTL and latencies
3. Compare with other sessions
4. Identify configuration improvements
5. Export data for further analysis

---

## UI Layout Adaptations

### Config Panel Open (Setup/Analysis Mode)

```
┌──────────┬──────────────┬──────────────────┐
│ Sessions │ Config Panel │ Chat Panel       │
│  List    │              │ ┌──────────────┐ │
│          │ ASR  LLM     │ │ 📹 Video     │ │
│ [+ New]  │ TTS Devices  │ └──────────────┘ │
│          │              │ [🎤 🔊 Devices]  │
│  S1      │ [Config...]  │ [▶ START]        │
│  S2      │              │                  │
│  S3      │              │ 👤 User: ...     │
│          │              │ 🤖 AI: ...       │
└──────────┴──────────────┴──────────────────┘
```

**Use case**: Configuration and testing

### Config Panel Collapsed (Focus Mode)

```
┌──────────┬────────────────────────────────┐
│ Sessions │    Chat Panel (Full Width)     │
│  List    │                                │
│          │ ┌──────────┐  [🎤 🔊 Devices] │
│ [+ New]  │ │ 📹 Video │  [▶ START]       │
│          │ └──────────┘  [Status: Setup]  │
│  S1      │                                │
│  S2      │  👤 User: ...                 │
│  S3      │  🤖 AI: ...                   │
└──────────┴────────────────────────────────┘
```

**Use case**: Live conversation with minimal distractions

---

## Device Preview vs Recording

### Device Preview (Setup Mode)

- **Purpose**: Verify correct devices selected
- **WebRTC**: Active and streaming
- **Timeline**: NOT recording
- **Storage**: NO data saved
- **Latency**: Not measured
- **User sees**: Live camera feed, mic levels

### Device Recording (Live Mode)

- **Purpose**: Capture conversation for analysis
- **WebRTC**: Active and streaming
- **Timeline**: Recording all events
- **Storage**: Everything saved to session JSON
- **Latency**: Measured and tracked
- **User sees**: Same video/audio + metrics + timeline

**Key Point**: Devices can be active BEFORE recording starts. This allows users to:
1. Confirm they selected the right camera
2. Check microphone is picking up audio
3. Test speaker output
4. Make adjustments without polluting the analyzed session data

---

## Workflow Example

### Scenario: Testing Low-Latency Config

1. **Load Preset**:
   - User selects "Low Latency" preset
   - Config auto-fills with aggressive VAD settings

2. **Create Session**:
   - Click "+ New Voice Chat"
   - UI switches to Setup Mode
   - Config panel shows low-latency settings

3. **Test Devices**:
   - Select "Browser WebRTC" for all devices
   - Camera preview appears
   - User sees themselves on screen
   - Speak into mic, see level indicator move

4. **Adjust Config**:
   - Lower VAD stop threshold to 0.4
   - Change LLM to llama3.2:3b (faster)
   - Set max_tokens to 256 (shorter responses)

5. **Start Recording**:
   - Click "▶ START Session"
   - Status changes to "🔴 Recording"
   - Timeline begins capturing events

6. **Have Conversation**:
   - Ask: "What's the weather?"
   - Watch timeline populate:
     - User speech starts/ends
     - ASR processing
     - LLM generation
     - TTS synthesis
   - Response plays: "It's sunny and 72 degrees"
   - TTL displayed: 145ms ✅

7. **Stop Session**:
   - Click "⏸ STOP Session"
   - Session saved with name "Low Latency Test"
   - Appears in sidebar

8. **Review**:
   - Click session in sidebar
   - View conversation history
   - Check timeline visualization
   - See metrics: Avg TTL 145ms
   - Compare with other configs

---

## Future Enhancements

### Pause/Resume (Not Implemented Yet)

Add ability to pause recording mid-session without stopping:
- Useful for demonstrations
- Excludes paused sections from analysis
- Timeline shows pause events

### Auto-Save During Live (Not Implemented Yet)

Periodically save session state during recording:
- Prevents data loss if browser crashes
- Allows recovery of long sessions
- Incremental JSON updates

### Background Device Test (Not Implemented Yet)

Continuously monitor device health in Setup Mode:
- Detect microphone disconnection
- Warn if audio levels too low
- Alert on camera errors
- Recommend fixes before starting

---

## Technical Implementation Notes

### Session State Machine

```javascript
state.sessionState = 'setup' | 'live' | 'stopped'
state.isLiveSession = true | false
```

### State Transitions

```
[Initial State] → "+ New Voice Chat" → Setup Mode
Setup Mode → "▶ START Session" → Live Mode
Live Mode → "⏸ STOP Session" → Stopped → Historical Mode
[Any State] → Click Historical Session → Historical Mode
```

### WebRTC Lifecycle

```
Setup Mode:
  - getUserMedia() → camera/mic streams
  - Display video preview
  - Monitor audio levels
  - NO recording/analysis

Live Mode:
  - Keep existing streams
  - Begin timeline recording
  - Send audio to ASR backend
  - Record all events

Stopped Mode:
  - Stop all streams
  - Release camera/mic access
  - Close WebRTC connections
```

### Configuration Locking

During Live Mode, configuration should be read-only to ensure session integrity. Changes require:
1. Stop current session
2. Modify config
3. Start new session

This prevents inconsistent data (e.g., "which VAD setting caused this TTL spike?").

---

## Best Practices

1. **Always test devices in Setup Mode** before starting recording
2. **Collapse config panel during Live Mode** for focused conversation
3. **Keep sessions short** (5-15 turns) for easier analysis
4. **Name sessions descriptively** when saving
5. **Compare similar configs** to identify optimal settings
6. **Review timeline** after each session to understand latency sources

---

**Related Docs**:
- `docs/SESSION_MANAGEMENT.md` - Session data structure and persistence
- `docs/TIMELINE_DESIGN.md` - Timeline event system
- `docs/AUDIO_MODES.md` - WebRTC vs USB audio devices
