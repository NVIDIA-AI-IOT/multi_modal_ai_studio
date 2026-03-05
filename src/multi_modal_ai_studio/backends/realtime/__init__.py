"""OpenAI-compatible Realtime API WebSocket client."""

from multi_modal_ai_studio.backends.realtime.client import (
    DISABLE_TURN_DETECTION,
    REALTIME_SAMPLE_RATE,
    OpenAIRealtimeClient,
    RealtimeEvent,
)

__all__ = [
    "DISABLE_TURN_DETECTION",
    "REALTIME_SAMPLE_RATE",
    "OpenAIRealtimeClient",
    "RealtimeEvent",
]
