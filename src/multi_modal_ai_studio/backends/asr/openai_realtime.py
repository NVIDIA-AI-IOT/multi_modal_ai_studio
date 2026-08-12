# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Realtime transcription adapter for the classic cascade."""

import logging
import time
from typing import AsyncIterator, Dict, Optional

import numpy as np

from multi_modal_ai_studio.backends.base import ASRBackend, ASRResult, ConfigError
from multi_modal_ai_studio.backends.realtime.client import (
    DISABLE_TURN_DETECTION,
    OpenAIRealtimeClient,
    REALTIME_SAMPLE_RATE,
)
from multi_modal_ai_studio.config.schema import ASRConfig
from multi_modal_ai_studio.core.timeline import Timeline

logger = logging.getLogger(__name__)

_PIPELINE_SAMPLE_RATE = 16000


def _language_code(value: Optional[str]) -> Optional[str]:
    """Return an ISO-639-style language code accepted by common providers."""
    value = (value or "").strip()
    if not value or value.lower() == "auto":
        return None
    return value.split("-", 1)[0].lower()


def _resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linearly resample aligned mono PCM16 for Realtime audio input."""
    pcm = pcm[: len(pcm) - (len(pcm) % 2)]
    if not pcm or from_rate <= 0 or to_rate <= 0:
        return b""
    if from_rate == to_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2")
    output_count = int(round(len(samples) * to_rate / from_rate))
    if output_count <= 0:
        return b""
    output = np.interp(
        np.linspace(0, len(samples) - 1, output_count),
        np.arange(len(samples)),
        samples.astype(np.float64),
    ).astype("<i2")
    return output.tobytes()


class OpenAIRealtimeASRBackend(ASRBackend):
    """Use a Realtime transcription session as an MMAS streaming ASR backend."""

    def __init__(self, config: ASRConfig, timeline: Optional[Timeline] = None):
        super().__init__(config)
        if config.scheme != "openai-realtime":
            raise ConfigError(
                f"Expected scheme 'openai-realtime', got '{config.scheme}'"
            )
        if config.realtime_transport != "websocket":
            raise ConfigError("Realtime transcription currently requires WebSocket")
        if config.realtime_session_type != "transcription":
            raise ConfigError(
                "Realtime ASR adapter requires realtime_session_type='transcription'"
            )
        if not (config.realtime_url or config.api_base):
            raise ConfigError("Realtime transcription URL is required")

        self.timeline = timeline
        self._client: Optional[OpenAIRealtimeClient] = None
        self._partials: Dict[str, str] = {}
        self._speech_starts: Dict[str, float] = {}
        self._active_start: Optional[float] = None
        self._last_speech_end: Optional[float] = None

    def _timestamp(self) -> float:
        if self.timeline is not None and self.timeline.start_time is not None:
            return max(0.0, time.time() - self.timeline.start_time)
        return time.monotonic()

    def _make_client(self) -> OpenAIRealtimeClient:
        language = _language_code(self.config.language)
        transcription = {"model": self.config.model}
        if language:
            transcription["language"] = language

        turn_detection = DISABLE_TURN_DETECTION
        if self.config.enable_vad:
            turn_detection = {
                "type": "server_vad",
                "threshold": self.config.vad_start_threshold,
                "prefix_padding_ms": self.config.speech_pad_ms,
                "silence_duration_ms": self.config.speech_timeout_ms,
            }
            if self.config.realtime_api_style == "openai-beta":
                # Preview-schema servers may otherwise synthesize an assistant
                # response, which is not wanted in an ASR-only cascade.
                turn_detection["create_response"] = False

        return OpenAIRealtimeClient(
            url=(self.config.realtime_url or self.config.api_base or "").strip(),
            api_key=(self.config.api_key or "").strip(),
            model=self.config.model,
            session_type="transcription",
            api_style=self.config.realtime_api_style,
            input_audio_transcription=transcription,
            turn_detection=turn_detection,
        )

    async def start_stream(self) -> None:
        if self._client is not None:
            raise RuntimeError("Stream already started")
        self._partials.clear()
        self._speech_starts.clear()
        self._active_start = None
        self._last_speech_end = None
        client = self._make_client()
        try:
            await client.connect()
        except Exception:
            await client.disconnect()
            raise
        self._client = client
        logger.info(
            "[OpenAI Realtime ASR] Connected: model=%s style=%s",
            self.config.model,
            self.config.realtime_api_style,
        )

    async def send_audio(self, audio_chunk: bytes) -> bool:
        if self._client is None:
            return False
        pcm_24k = _resample_pcm16(
            audio_chunk,
            _PIPELINE_SAMPLE_RATE,
            REALTIME_SAMPLE_RATE,
        )
        if pcm_24k:
            await self._client.send_audio(pcm_24k)
        return True

    async def receive_results(self) -> AsyncIterator[ASRResult]:
        client = self._client
        if client is None:
            raise RuntimeError("Stream not started. Call start_stream() first.")

        async for event in client.events():
            if event is None:
                break
            timestamp = self._timestamp()
            item_id = event.item_id or "active"
            common = {
                "backend": "openai-realtime",
                "model": self.config.model,
                "event_timestamp": timestamp,
                "item_id": event.item_id,
            }

            if event.kind == "session_ready":
                continue
            if event.kind == "speech_started":
                self._active_start = timestamp
                self._speech_starts[item_id] = timestamp
                yield ASRResult(
                    text="",
                    is_final=False,
                    start_time=timestamp,
                    metadata={**common, "control_event": "vad_start"},
                )
                continue
            if event.kind == "speech_stopped":
                self._last_speech_end = timestamp
                yield ASRResult(
                    text="",
                    is_final=False,
                    start_time=self._speech_starts.get(item_id, self._active_start),
                    end_time=timestamp,
                    metadata={**common, "control_event": "vad_end"},
                )
                continue
            if event.kind == "transcript_delta":
                delta = event.text or ""
                cumulative = self._partials.get(item_id, "") + delta
                self._partials[item_id] = cumulative
                if cumulative and self.config.interim_results:
                    yield ASRResult(
                        text=cumulative,
                        is_final=False,
                        start_time=self._speech_starts.get(item_id, self._active_start),
                        metadata=common,
                    )
                continue
            if event.kind == "transcript_completed":
                transcript = (event.text or self._partials.get(item_id, "")).strip()
                self._partials.pop(item_id, None)
                if transcript:
                    yield ASRResult(
                        text=transcript,
                        is_final=True,
                        start_time=self._speech_starts.pop(
                            item_id, self._active_start
                        ),
                        end_time=self._last_speech_end or timestamp,
                        metadata=common,
                    )
                continue
            if event.kind == "error":
                yield ASRResult(
                    text="",
                    is_final=True,
                    confidence=0.0,
                    metadata={**common, "error": event.message or "Realtime ASR error"},
                )

    async def stop_stream(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.disconnect()
