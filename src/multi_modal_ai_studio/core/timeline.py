"""
Timeline event system for Multi-modal AI Studio.

Records all events in the voice AI pipeline for performance analysis
and visualization. Enables both live rendering and playback from recorded data.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal
from enum import Enum


class Lane(str, Enum):
    """Timeline lanes for different pipeline stages."""
    SYSTEM = "system"
    AUDIO = "audio"
    SPEECH = "speech"  # ASR
    LLM = "llm"
    TTS = "tts"


@dataclass
class TimelineEvent:
    """Single event in the timeline.
    
    Attributes:
        timestamp: Seconds since session start (float for precision)
        event_type: Type of event (e.g., "asr_start", "llm_token", "tts_audio")
        lane: Which lane this event belongs to (audio, speech, llm, tts)
        data: Event-specific data (flexible dict)
        start_time: Optional start time for duration events (rectangles)
        end_time: Optional end time for duration events (rectangles)
        amplitude: Optional amplitude for waveform rendering (0-100)
        source: Optional source identifier ('user', 'tts', 'ai')
        render_type: Optional rendering hint ('point', 'rectangle', 'waveform')
    """
    timestamp: float
    event_type: str
    lane: Lane
    data: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    amplitude: Optional[float] = None
    source: Optional[str] = None
    render_type: Optional[Literal['point', 'rectangle', 'waveform']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "lane": self.lane.value,
            "data": self.data,
        }
        # Include optional fields only if set
        if self.start_time is not None:
            result["start_time"] = self.start_time
        if self.end_time is not None:
            result["end_time"] = self.end_time
        if self.amplitude is not None:
            result["amplitude"] = self.amplitude
        if self.source is not None:
            result["source"] = self.source
        # render_type is a UI preference, not persisted in session data
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimelineEvent':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            lane=Lane(data["lane"]),
            data=data.get("data", {}),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            amplitude=data.get("amplitude"),
            source=data.get("source"),
            render_type=data.get("render_type"),
        )


class Timeline:
    """Timeline manager for recording and analyzing events.
    
    Records all events in the voice AI pipeline and provides
    methods for metric calculation and visualization.
    """
    
    def __init__(self):
        """Initialize empty timeline."""
        self.events: List[TimelineEvent] = []
        self.start_time: Optional[float] = None
        self._last_event_time: float = 0
    
    def start(self) -> None:
        """Start the timeline (sets time origin)."""
        self.start_time = time.time()
        self._last_event_time = self.start_time
        self.add_event("session_start", Lane.SYSTEM)
    
    def add_event(
        self,
        event_type: str,
        lane: Lane,
        data: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        amplitude: Optional[float] = None,
        source: Optional[str] = None,
        render_type: Optional[Literal['point', 'rectangle', 'waveform']] = None
    ) -> TimelineEvent:
        """Add event to timeline.
        
        Args:
            event_type: Event identifier (e.g., "asr_final", "llm_token")
            lane: Timeline lane
            data: Optional event-specific data
            start_time: Optional start time for duration events (rectangles)
            end_time: Optional end time for duration events (rectangles)
            amplitude: Optional amplitude for waveform rendering (0-100)
            source: Optional source identifier ('user', 'tts', 'ai')
            render_type: Optional rendering hint ('point', 'rectangle', 'waveform')
        
        Returns:
            The created TimelineEvent
        """
        if self.start_time is None:
            self.start()
        
        timestamp = time.time() - self.start_time
        event = TimelineEvent(
            timestamp=timestamp,
            event_type=event_type,
            lane=lane,
            data=data or {},
            start_time=start_time,
            end_time=end_time,
            amplitude=amplitude,
            source=source,
            render_type=render_type
        )
        
        self.events.append(event)
        self._last_event_time = time.time()
        
        return event
    
    def get_events_by_lane(self, lane: Lane) -> List[TimelineEvent]:
        """Get all events for a specific lane."""
        return [e for e in self.events if e.lane == lane]
    
    def get_events_by_type(self, event_type: str) -> List[TimelineEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[TimelineEvent]:
        """Get events within a time range."""
        return [
            e for e in self.events
            if start_time <= e.timestamp <= end_time
        ]
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert timeline to dictionary for JSON serialization."""
        return [event.to_dict() for event in self.events]
    
    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> 'Timeline':
        """Create timeline from dictionary."""
        timeline = cls()
        timeline.events = [TimelineEvent.from_dict(e) for e in data]
        if timeline.events:
            timeline.start_time = 0  # Relative time already stored
        return timeline
    
    def calculate_ttl(self, turn_id: int = -1) -> Optional[float]:
        """Calculate Turn-Taking Latency (TTL) for a turn.
        
        TTL = Time from user_speech_end to tts_first_audio
        This is the critical metric for voice AI responsiveness.
        
        Args:
            turn_id: Which turn to calculate (default: -1 for last turn)
        
        Returns:
            TTL in seconds, or None if incomplete
        """
        # Find user_speech_end events
        speech_end_events = self.get_events_by_type("user_speech_end")
        
        if not speech_end_events:
            return None
        
        # Get the specified turn (or last)
        if turn_id < 0:
            turn_id = len(speech_end_events) + turn_id
        
        if turn_id < 0 or turn_id >= len(speech_end_events):
            return None
        
        speech_end = speech_end_events[turn_id]
        
        # Find next tts_first_audio after this speech_end
        tts_events = [
            e for e in self.get_events_by_type("tts_first_audio")
            if e.timestamp > speech_end.timestamp
        ]
        
        if not tts_events:
            return None
        
        tts_first = tts_events[0]
        
        # TTL = time from user stopped speaking to AI started speaking
        ttl = tts_first.timestamp - speech_end.timestamp
        
        return ttl
    
    def calculate_component_latencies(self, turn_id: int = -1) -> Dict[str, float]:
        """Calculate individual component latencies for a turn.
        
        Returns breakdown:
        - asr_latency: user_speech_end to asr_final
        - llm_latency: asr_final to llm_complete
        - tts_latency: llm_complete to tts_first_audio
        - ttl: Total turn-taking latency (user_speech_end to tts_first_audio)
        
        Args:
            turn_id: Which turn to analyze (default: -1 for last turn)
        
        Returns:
            Dictionary of latencies in seconds
        """
        latencies = {}
        
        # Find events for this turn
        speech_end_events = self.get_events_by_type("user_speech_end")
        if not speech_end_events:
            return latencies
        
        if turn_id < 0:
            turn_id = len(speech_end_events) + turn_id
        
        if turn_id < 0 or turn_id >= len(speech_end_events):
            return latencies
        
        speech_end = speech_end_events[turn_id]
        
        # Find subsequent events
        asr_final = self._find_next_event("asr_final", speech_end.timestamp)
        llm_complete = self._find_next_event("llm_complete", speech_end.timestamp)
        tts_first = self._find_next_event("tts_first_audio", speech_end.timestamp)
        
        # Calculate latencies
        if asr_final:
            latencies["asr_latency"] = asr_final.timestamp - speech_end.timestamp
        
        if asr_final and llm_complete:
            latencies["llm_latency"] = llm_complete.timestamp - asr_final.timestamp
        
        if llm_complete and tts_first:
            latencies["tts_latency"] = tts_first.timestamp - llm_complete.timestamp
        
        if tts_first:
            latencies["ttl"] = tts_first.timestamp - speech_end.timestamp
        
        return latencies
    
    def _find_next_event(
        self,
        event_type: str,
        after_timestamp: float
    ) -> Optional[TimelineEvent]:
        """Find next event of given type after timestamp."""
        matching = [
            e for e in self.events
            if e.event_type == event_type and e.timestamp > after_timestamp
        ]
        return matching[0] if matching else None
    
    def add_vad_segment(
        self,
        start_time: float,
        end_time: float,
        data: Optional[Dict[str, Any]] = None
    ) -> TimelineEvent:
        """Add a VAD (Voice Activity Detection) segment as a rectangle.
        
        Args:
            start_time: Start time relative to session start
            end_time: End time relative to session start
            data: Optional metadata
        
        Returns:
            The created TimelineEvent
        """
        return self.add_event(
            event_type="vad_segment",
            lane=Lane.AUDIO,
            data=data,
            start_time=start_time,
            end_time=end_time,
        )
    
    def add_asr_segment(
        self,
        start_time: float,
        end_time: float,
        text: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> TimelineEvent:
        """Add an ASR segment as a rectangle.
        
        Args:
            start_time: Start time relative to session start
            end_time: End time relative to session start
            text: Transcribed text
            data: Optional metadata
        
        Returns:
            The created TimelineEvent
        """
        event_data = data or {}
        event_data['text'] = text
        return self.add_event(
            event_type="asr_segment",
            lane=Lane.SPEECH,
            data=event_data,
            start_time=start_time,
            end_time=end_time,
        )
    
    def add_tts_segment(
        self,
        start_time: float,
        end_time: float,
        text: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> TimelineEvent:
        """Add a TTS segment as a rectangle.
        
        Args:
            start_time: Start time relative to session start
            end_time: End time relative to session start
            text: Text being synthesized
            data: Optional metadata
        
        Returns:
            The created TimelineEvent
        """
        event_data = data or {}
        event_data['text'] = text
        return self.add_event(
            event_type="tts_segment",
            lane=Lane.TTS,
            data=event_data,
            start_time=start_time,
            end_time=end_time,
        )
    
    def add_audio_amplitude(
        self,
        amplitude: float,
        source: str = 'user',
        data: Optional[Dict[str, Any]] = None
    ) -> TimelineEvent:
        """Add an audio amplitude sample for waveform visualization.
        
        Args:
            amplitude: Audio amplitude (0-100 scale)
            source: Audio source ('user' or 'tts'/'ai')
            data: Optional metadata
        
        Returns:
            The created TimelineEvent
        """
        return self.add_event(
            event_type="audio_amplitude",
            lane=Lane.AUDIO,
            data=data,
            amplitude=amplitude,
            source=source,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timeline summary statistics.
        
        Returns:
            Dictionary with event counts, duration, etc.
        """
        if not self.events:
            return {"event_count": 0, "duration": 0}
        
        # Count events by lane
        lane_counts = {}
        for lane in Lane:
            lane_counts[lane.value] = len(self.get_events_by_lane(lane))
        
        # Calculate duration
        duration = self.events[-1].timestamp if self.events else 0
        
        # Count turns (user_speech_end events)
        turn_count = len(self.get_events_by_type("user_speech_end"))
        
        return {
            "event_count": len(self.events),
            "duration": duration,
            "turn_count": turn_count,
            "events_by_lane": lane_counts,
        }


# Standard event types for consistency
class EventType:
    """Standard event type constants."""
    
    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_PAUSE = "session_pause"
    SESSION_RESUME = "session_resume"
    
    # Audio events
    AUDIO_START = "audio_start"
    AUDIO_LEVEL = "audio_level"
    AUDIO_END = "audio_end"
    
    # ASR events
    USER_SPEECH_START = "user_speech_start"  # VAD detected speech
    USER_SPEECH_END = "user_speech_end"      # VAD detected silence (KEY for TTL!)
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    
    # LLM events
    LLM_START = "llm_start"
    LLM_FIRST_TOKEN = "llm_first_token"  # Prefill complete
    LLM_TOKEN = "llm_token"
    LLM_COMPLETE = "llm_complete"
    
    # TTS events
    TTS_START = "tts_start"
    TTS_FIRST_AUDIO = "tts_first_audio"  # First audio chunk ready (KEY for TTL!)
    TTS_AUDIO = "tts_audio"
    TTS_COMPLETE = "tts_complete"
    
    # Barge-in events
    BARGE_IN = "barge_in"  # User interrupted AI
