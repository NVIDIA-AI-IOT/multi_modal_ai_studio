# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight PCM energy VAD used for timeline observation.

This detector does not gate audio or replace a speech backend's endpointing.
It observes the same PCM16 stream so MMAS can show physical speech boundaries
independently from partial/final transcript delivery.
"""

from dataclasses import dataclass
import math
import struct
from typing import List, Optional

from multi_modal_ai_studio.config.schema import ASRConfig


@dataclass(frozen=True)
class EnergyVADEvent:
    """One observed speech boundary on the session timeline."""

    event_type: str
    timestamp: float
    observed_at: float
    normalized_rms: float


class EnergyVADObserver:
    """Observe speech start/end in 16 kHz mono PCM16 without gating audio."""

    def __init__(
        self,
        config: ASRConfig,
        *,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ):
        self.config = config
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.reset()

    def reset(self) -> None:
        self.speech_active = False
        self.speech_start: Optional[float] = None
        self.silence_start: Optional[float] = None
        self.silence_ms = 0.0

    def observe(self, pcm: bytes, *, chunk_end_time: float) -> List[EnergyVADEvent]:
        """Return any boundaries observed in a PCM chunk.

        Event timestamps describe the audio boundary. ``observed_at`` describes
        when enough audio had arrived to make the decision. Consequently,
        ``vad_end.timestamp`` is the beginning of the confirmed trailing
        silence and can precede ``vad_end.observed_at`` by the endpoint timeout.
        """
        if not pcm:
            return []
        byte_count = len(pcm) - (len(pcm) % self.sample_width)
        if byte_count <= 0:
            return []
        pcm = pcm[:byte_count]
        duration = byte_count / (self.sample_rate * self.sample_width)
        chunk_end_time = max(0.0, float(chunk_end_time))
        chunk_start_time = max(0.0, chunk_end_time - duration)
        normalized_rms = self.normalized_rms(pcm)
        threshold_setting = (
            self.config.vad_stop_threshold
            if self.speech_active
            else self.config.vad_start_threshold
        )
        speech = normalized_rms >= self.rms_threshold(threshold_setting)
        events: List[EnergyVADEvent] = []

        if not self.speech_active:
            if speech:
                self.speech_active = True
                self.speech_start = chunk_start_time
                self.silence_start = None
                self.silence_ms = 0.0
                events.append(
                    EnergyVADEvent(
                        event_type="vad_start",
                        timestamp=chunk_start_time,
                        observed_at=chunk_end_time,
                        normalized_rms=normalized_rms,
                    )
                )
            return events

        if speech:
            self.silence_start = None
            self.silence_ms = 0.0
            return events

        if self.silence_start is None:
            self.silence_start = chunk_start_time
        self.silence_ms += duration * 1000.0
        if self.silence_ms < max(0, self.config.speech_timeout_ms):
            return events

        speech_end = max(self.speech_start or 0.0, self.silence_start)
        events.append(
            EnergyVADEvent(
                event_type="vad_end",
                timestamp=speech_end,
                observed_at=chunk_end_time,
                normalized_rms=normalized_rms,
            )
        )
        self.reset()
        return events

    @staticmethod
    def normalized_rms(pcm: bytes) -> float:
        """Return PCM16 RMS normalized to the 0..1 full-scale range."""
        sample_count = len(pcm) // 2
        if sample_count == 0:
            return 0.0
        samples = struct.unpack(f"<{sample_count}h", pcm[: sample_count * 2])
        rms = math.sqrt(sum(sample * sample for sample in samples) / sample_count)
        return rms / 32768.0

    @staticmethod
    def rms_threshold(setting: float) -> float:
        """Map the UI's model-VAD setting to a practical PCM RMS threshold."""
        return 0.005 * (10 ** max(0.0, min(setting, 1.0)))
