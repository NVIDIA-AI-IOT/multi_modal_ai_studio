# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment-backed service configuration."""

from dataclasses import dataclass
import os
from typing import Literal, Optional

ASR_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
TTS_MODEL = "nvidia/magpie_tts_multilingual_357m"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for one speech service."""

    mode: Literal["asr", "tts"]
    model_id: str
    model_revision: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    eager_load: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate settings from environment variables."""
        mode = os.getenv("SPEECH_SERVICE_MODE", "asr").strip().lower()
        if mode not in {"asr", "tts"}:
            raise ValueError("SPEECH_SERVICE_MODE must be 'asr' or 'tts'")
        default_model = ASR_MODEL if mode == "asr" else TTS_MODEL
        return cls(
            mode=mode,
            model_id=os.getenv("SPEECH_MODEL_ID", default_model).strip(),
            model_revision=os.getenv("SPEECH_MODEL_REVISION") or None,
            device=os.getenv("SPEECH_DEVICE", "auto").strip(),
            dtype=os.getenv("SPEECH_DTYPE", "auto").strip(),
            eager_load=os.getenv("SPEECH_EAGER_LOAD", "0").lower() in {"1", "true", "yes"},
        )
