# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-modal AI Studio

A voice/text/video AI interface with advanced performance analysis.
"""

__version__ = "0.1.0"
__author__ = "Multi-modal AI Studio Contributors"
__license__ = "Apache-2.0"

from multi_modal_ai_studio.config.schema import (
    ASRConfig,
    LLMConfig,
    TTSConfig,
    DeviceConfig,
    AppConfig,
    SessionConfig,
)

from multi_modal_ai_studio.core import (
    Timeline,
    TimelineEvent,
    Lane,
    EventType,
    Session,
    Turn,
    SessionMetrics,
)

__all__ = [
    # Configuration
    "ASRConfig",
    "LLMConfig",
    "TTSConfig",
    "DeviceConfig",
    "AppConfig",
    "SessionConfig",
    # Session Management
    "Timeline",
    "TimelineEvent",
    "Lane",
    "EventType",
    "Session",
    "Turn",
    "SessionMetrics",
]
