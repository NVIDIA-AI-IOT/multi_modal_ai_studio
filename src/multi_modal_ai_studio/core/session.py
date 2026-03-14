# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Session management for Multi-modal AI Studio.

A Session represents a complete conversational interaction with:
- Configuration snapshot (ASR, LLM, TTS, device settings)
- Timeline of all events
- Conversation turns (user inputs and AI responses)
- Calculated metrics (TTL, component latencies)

Sessions can be saved to JSON and loaded for later analysis.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from multi_modal_ai_studio import __version__
from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core.timeline import Timeline, TimelineEvent, Lane, EventType


@dataclass
class Turn:
    """Single conversation turn (user input + AI response).

    Attributes:
        turn_id: Turn number (1-indexed)
        user_transcript: What the user said
        ai_response: What the AI responded
        user_confidence: ASR confidence score (0.0-1.0)
        start_time: When user started speaking (seconds since session start)
        end_time: When AI finished responding (seconds since session start)
        latencies: Component latencies (asr, llm, tts, ttl)
    """
    turn_id: int
    user_transcript: str
    ai_response: str
    user_confidence: float = 1.0
    start_time: float = 0.0
    end_time: float = 0.0
    latencies: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionMetrics:
    """Calculated metrics for a session.

    Attributes:
        total_turns: Number of conversation turns
        avg_ttl: Average Turn-Taking Latency across all turns
        min_ttl: Fastest TTL
        max_ttl: Slowest TTL
        avg_asr_latency: Average ASR latency
        avg_llm_latency: Average LLM latency
        avg_tts_latency: Average TTS latency
        session_duration: Total session time in seconds
        active_duration: Active conversation time (excludes pauses)
    """
    total_turns: int = 0
    avg_ttl: float = 0.0
    min_ttl: float = 0.0
    max_ttl: float = 0.0
    avg_asr_latency: float = 0.0
    avg_llm_latency: float = 0.0
    avg_tts_latency: float = 0.0
    session_duration: float = 0.0
    active_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetrics':
        """Create from dictionary."""
        return cls(**data)


class Session:
    """A complete conversational session with configuration, timeline, and metrics.

    The Session class is the core data structure that:
    - Records timeline events as they occur
    - Manages conversation turns
    - Calculates performance metrics (especially TTL)
    - Saves/loads from JSON for offline analysis
    """

    def __init__(
        self,
        config: SessionConfig,
        session_id: Optional[str] = None,
        name: Optional[str] = None
    ):
        """Initialize a new session.

        Args:
            config: Complete session configuration
            session_id: Optional custom session ID (generates UUID if not provided)
            name: Optional session name (uses config.name if not provided)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.name = name or config.name
        self.created_at = datetime.utcnow()
        self.config = config

        self.timeline = Timeline()
        self.turns: List[Turn] = []
        self.system_stats: List[Dict[str, Any]] = []  # [{t, cpu, gpu}, ...] session-relative, from client
        self._current_turn: Optional[Dict[str, Any]] = None
        self._metrics: Optional[SessionMetrics] = None

    def start(self) -> None:
        """Start the session and timeline."""
        self.timeline.start()

    def add_event(
        self,
        event_type: str,
        lane: Lane,
        data: Optional[Dict[str, Any]] = None
    ) -> TimelineEvent:
        """Add event to timeline.

        Convenience method that delegates to Timeline.

        Args:
            event_type: Event identifier
            lane: Timeline lane
            data: Optional event-specific data

        Returns:
            The created TimelineEvent
        """
        return self.timeline.add_event(event_type, lane, data)

    def start_turn(self, user_transcript: str = "") -> None:
        """Start a new conversation turn.

        Args:
            user_transcript: Initial user transcript (may be updated as ASR finalizes)
        """
        turn_id = len(self.turns) + 1

        self._current_turn = {
            "turn_id": turn_id,
            "user_transcript": user_transcript,
            "ai_response": "",
            "user_confidence": 1.0,
            "start_time": self.timeline.events[-1].timestamp if self.timeline.events else 0,
            "latencies": {},
        }

    def update_turn_transcript(self, transcript: str, confidence: float = 1.0) -> None:
        """Update current turn's user transcript.

        Args:
            transcript: Finalized transcript
            confidence: ASR confidence score
        """
        if self._current_turn:
            self._current_turn["user_transcript"] = transcript
            self._current_turn["user_confidence"] = confidence

    def update_turn_response(self, response: str) -> None:
        """Update current turn's AI response.

        Args:
            response: Complete AI response
        """
        if self._current_turn:
            self._current_turn["ai_response"] = response

    def end_turn(self) -> None:
        """End current turn and calculate metrics."""
        if not self._current_turn:
            return

        # Set end time
        self._current_turn["end_time"] = (
            self.timeline.events[-1].timestamp if self.timeline.events else 0
        )

        # Calculate latencies for this turn
        turn_id = self._current_turn["turn_id"] - 1  # 0-indexed for timeline
        latencies = self.timeline.calculate_component_latencies(turn_id)
        self._current_turn["latencies"] = latencies

        # Create Turn object and add to turns list
        turn = Turn.from_dict(self._current_turn)
        self.turns.append(turn)

        # Log TTL
        if "ttl" in latencies:
            ttl_ms = latencies["ttl"] * 1000
            print(f"Turn {turn.turn_id} TTL: {ttl_ms:.0f}ms")

        self._current_turn = None

    def calculate_metrics(self) -> SessionMetrics:
        """Calculate session metrics from timeline and turns.

        Returns:
            SessionMetrics with all calculated values
        """
        metrics = SessionMetrics()

        metrics.total_turns = len(self.turns)

        if not self.turns:
            return metrics

        # Calculate TTL statistics
        ttls = [turn.latencies.get("ttl", 0) for turn in self.turns if "ttl" in turn.latencies]
        if ttls:
            metrics.avg_ttl = sum(ttls) / len(ttls)
            metrics.min_ttl = min(ttls)
            metrics.max_ttl = max(ttls)

        # Calculate component latency averages
        asr_latencies = [
            turn.latencies.get("asr_latency", 0)
            for turn in self.turns if "asr_latency" in turn.latencies
        ]
        if asr_latencies:
            metrics.avg_asr_latency = sum(asr_latencies) / len(asr_latencies)

        llm_latencies = [
            turn.latencies.get("llm_latency", 0)
            for turn in self.turns if "llm_latency" in turn.latencies
        ]
        if llm_latencies:
            metrics.avg_llm_latency = sum(llm_latencies) / len(llm_latencies)

        tts_latencies = [
            turn.latencies.get("tts_latency", 0)
            for turn in self.turns if "tts_latency" in turn.latencies
        ]
        if tts_latencies:
            metrics.avg_tts_latency = sum(tts_latencies) / len(tts_latencies)

        # Session duration
        if self.timeline.events:
            metrics.session_duration = self.timeline.events[-1].timestamp

        # Active duration (exclude paused time)
        # For now, same as session duration (will track pauses later)
        metrics.active_duration = metrics.session_duration

        self._metrics = metrics
        return metrics

    def apply_ttl_bands(self) -> None:
        """Overwrite each turn's TTL from ttl_bands (band-based = first audio_amplitude tts). Single source of truth for TTL."""
        bands = getattr(self, "ttl_bands", None) or []
        if not bands or not self.turns:
            return
        for i, turn in enumerate(self.turns):
            if i >= len(bands):
                break
            band = bands[i]
            if not isinstance(band, dict):
                continue
            ttl_ms = band.get("ttlMs")
            if ttl_ms is not None:
                turn.latencies["ttl"] = float(ttl_ms) / 1000.0
        self._metrics = None  # force recalc so avg_ttl etc. use band-based TTLs

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization.

        Returns:
            Complete session data as dictionary
        """
        # Calculate metrics if not already done
        if self._metrics is None:
            self.calculate_metrics()

        # Emit created_at as UTC with Z so the UI can show local time
        created_at_str = self.created_at.isoformat()
        if self.created_at.tzinfo is None and "Z" not in created_at_str and "+" not in created_at_str:
            created_at_str = created_at_str + "Z"
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": created_at_str,
            "config": self.config.to_dict(),
            "timeline": self.timeline.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "metrics": self._metrics.to_dict() if self._metrics else {},
            "system_stats": getattr(self, "system_stats", None) or [],
            "tts_playback_segments": getattr(self, "tts_playback_segments", None) or [],
            "audio_amplitude_history": getattr(self, "audio_amplitude_history", None) or [],
            "ttl_bands": getattr(self, "ttl_bands", None) or [],
            "app_version": getattr(self, "app_version", None) or __version__,
            "capture_health": getattr(self, "capture_health", None),
        }

    def save(self, path: Path) -> None:
        """Save session to JSON file.

        Args:
            path: Path to save JSON file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✓ Session saved: {path}")

    @classmethod
    def load(cls, path: Path) -> 'Session':
        """Load session from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded Session instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct session
        config = SessionConfig.from_dict(data["config"])
        session = cls(
            config=config,
            session_id=data["session_id"],
            name=data["name"]
        )

        # Restore timeline
        session.timeline = Timeline.from_dict(data["timeline"])

        # Restore turns
        session.turns = [Turn.from_dict(t) for t in data["turns"]]

        # Restore metrics
        if data.get("metrics"):
            session._metrics = SessionMetrics.from_dict(data["metrics"])
        # Restore system_stats (CPU/GPU samples from live session)
        session.system_stats = data.get("system_stats") or []
        session.tts_playback_segments = data.get("tts_playback_segments") or []
        session.audio_amplitude_history = data.get("audio_amplitude_history") or []
        session.ttl_bands = data.get("ttl_bands") or []
        session.app_version = data.get("app_version")
        session.apply_ttl_bands()  # so loaded session metrics match band-based TTL

        # Parse created_at (support Z for UTC from saved JSON); store naive UTC
        raw = data["created_at"]
        if isinstance(raw, str) and raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        session.created_at = dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt

        return session

    def get_metrics_summary(self) -> str:
        """Get human-readable metrics summary.

        Returns:
            Formatted string with key metrics
        """
        metrics = self.calculate_metrics()

        lines = [
            f"Session: {self.name}",
            f"Turns: {metrics.total_turns}",
            f"Duration: {metrics.session_duration:.1f}s",
        ]

        if metrics.avg_ttl > 0:
            lines.append(f"Avg TTL: {metrics.avg_ttl * 1000:.0f}ms")
            lines.append(f"TTL Range: {metrics.min_ttl * 1000:.0f}-{metrics.max_ttl * 1000:.0f}ms")

        if metrics.avg_asr_latency > 0:
            lines.append(f"Avg ASR: {metrics.avg_asr_latency * 1000:.0f}ms")

        if metrics.avg_llm_latency > 0:
            lines.append(f"Avg LLM: {metrics.avg_llm_latency * 1000:.0f}ms")

        if metrics.avg_tts_latency > 0:
            lines.append(f"Avg TTS: {metrics.avg_tts_latency * 1000:.0f}ms")

        return "\n".join(lines)
