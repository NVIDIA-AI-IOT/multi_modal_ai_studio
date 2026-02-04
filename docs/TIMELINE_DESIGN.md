# Timeline Design & Data Collection

## Core Design Principle

**Timeline always collects ALL events. There is NO buffer limit on data collection.**

This document explains the architectural decision to separate data collection from rendering concerns.

## The Problem (Removed `timeline_buffer_sec`)

Early in development, we had a `timeline_buffer_sec` configuration that suggested limiting the amount of timeline data collected. This was **removed** because:

1. **Misleading**: Suggested we would discard old events (we don't)
2. **Wrong layer**: Data limits don't belong in session configuration
3. **Anti-feature**: Analysis requires complete historical data
4. **Never implemented**: The Timeline class ignored this setting anyway

## Separation of Concerns

### Data Collection Layer (Backend)

**Always unlimited:**
```python
class Timeline:
    def add_event(self, event_type, lane, data=None):
        # NO limit on events - just append to list
        self.events.append(event)
```

**Why unlimited is OK:**
- Timeline events are tiny (~100 bytes each)
- 1-hour conversation ≈ 10,000 events ≈ 1MB
- Storage is cheap, analysis requires complete data
- Sessions serialize to compact JSON (~200 KB for 1-hour session)

### Rendering Layer (Frontend/UI)

**This is where limits belong:**

```javascript
// UI-side rendering optimization
const RENDER_WINDOW_SEC = 60;  // Show last 60 seconds
const viewport = timeline.events.filter(e => 
  e.timestamp > currentTime - RENDER_WINDOW_SEC
);
```

**UI Optimization Strategies:**
1. **Viewport culling**: Only render events in visible time window
2. **Canvas rendering**: Use HTML5 canvas for thousands of events
3. **Progressive loading**: Lazy-load events as user scrolls
4. **LOD (Level of Detail)**: Aggregate events when zoomed out

## Timeline Position

The `timeline_position` setting controls **where** the timeline renders, not **how much** data it collects:

```yaml
app:
  timeline_position: right  # Options: right, bottom, hidden
```

- **`right`**: Timeline beside session list (full height) - **default**
- **`bottom`**: Timeline below config panel (full width)
- **`hidden`**: No timeline visualization (still records events!)

Even with `timeline_position: hidden`, the session still records all timeline events for later analysis.

## Data Volume Examples

### Typical Session (5 minutes, 10 turns)

```
Events per turn:
  USER_SPEECH_START        1 event
  USER_SPEECH_END          1 event
  ASR_PARTIAL             ~5 events
  ASR_FINAL                1 event
  LLM_START                1 event
  LLM_FIRST_TOKEN          1 event
  LLM_TOKEN              ~20 events
  LLM_COMPLETE             1 event
  TTS_START                1 event
  TTS_FIRST_AUDIO          1 event
  TTS_AUDIO               ~8 events
  TTS_COMPLETE             1 event
  --------------------------------
  Total per turn:        ~42 events

10 turns × 42 events = 420 events ≈ 42 KB JSON
```

### Long Session (1 hour, 120 turns)

```
120 turns × 42 events = 5,040 events ≈ 500 KB JSON

With compression: ~150 KB
With gzip: ~50 KB
```

**Conclusion**: Even long sessions are tiny. No need to limit collection.

## Session Analysis Benefits

By recording **all** events, users can:

1. **Compare configurations**: Load two sessions and diff their timelines
2. **Find patterns**: Analyze latency trends over entire conversation
3. **Debug issues**: Pinpoint exact moment something went wrong
4. **Export data**: Convert to CSV for external analysis
5. **Replay sessions**: Visualize past conversations frame-by-frame

## Implementation Status

### ✅ Data Collection
- Timeline class records all events (no limits)
- Session class wraps Timeline
- Save/load preserves complete timeline
- Events are timestamped with microsecond precision

### ✅ Configuration
- Removed `timeline_buffer_sec` from `AppConfig`
- Updated all presets to remove buffer setting
- Changed default `timeline_position` to `right`
- Added documentation explaining design

### 🚧 Rendering (TODO)
- Implement viewport culling in frontend
- Add zoom/pan controls
- Progressive loading for long sessions
- Render optimization benchmarks

## Best Practices

### ✅ DO
- Record every event, no matter how small
- Include detailed event data (confidence, chunk sizes, etc.)
- Timestamp with sub-second precision
- Save complete sessions for analysis

### ❌ DON'T
- Limit events based on time windows
- Discard "old" events during collection
- Skip events to "save space" (negligible savings)
- Mix rendering concerns with data collection

## Future Enhancements

1. **Streaming sessions to disk**: For multi-hour sessions, stream events to disk instead of memory
2. **Event indexing**: Add time-based indexing for fast range queries
3. **Compression**: Compress timeline data when saving (gzip)
4. **Sampling**: For visualization, allow downsampling without affecting stored data

## References

- `src/multi_modal_ai_studio/core/timeline.py` - Timeline implementation
- `src/multi_modal_ai_studio/core/session.py` - Session management
- `docs/SESSION_MANAGEMENT.md` - Complete session docs
- `scripts/test_session.py` - Example showing 61 events for 3 turns
