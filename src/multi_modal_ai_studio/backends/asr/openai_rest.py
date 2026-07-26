# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible REST ASR with local endpointing for live microphones."""

import asyncio
from collections import deque
import io
import logging
import math
import struct
from typing import AsyncIterator, Deque, Optional
from urllib.parse import urljoin
import wave

import aiohttp

from multi_modal_ai_studio.backends.base import (
    ASRBackend,
    ASRResult,
    ConfigError,
    ConnectionError,
)
from multi_modal_ai_studio.config.schema import ASRConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2


class OpenAIRestASRBackend(ASRBackend):
    """Buffer microphone speech and call `/v1/audio/transcriptions` at EOU.

    OpenAI's file transcription endpoint is not a bidirectional microphone
    protocol. This adapter supplies lightweight local endpointing so it can
    still participate in MMAS's streaming backend interface.
    """

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        if config.scheme != "openai-rest":
            raise ConfigError(f"Expected scheme 'openai-rest', got '{config.scheme}'")
        if not config.api_base:
            raise ConfigError("OpenAI-compatible ASR api_base is required")

        self._session: Optional[aiohttp.ClientSession] = None
        self._results: Optional[asyncio.Queue] = None
        self._pre_roll: Deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._utterance = bytearray()
        self._speech_active = False
        self._silence_ms = 0.0
        self._requests: Optional[asyncio.Queue] = None
        self._request_worker: Optional[asyncio.Task] = None
        self._chunk_count = 0
        self._max_normalized_rms = 0.0
        self._audio_position_ms = 0.0
        self._speech_start_ms: Optional[float] = None

    async def start_stream(self) -> None:
        """Open the HTTP session and reset endpointing state."""
        if self._session is not None:
            raise RuntimeError("Stream already started")
        timeout = aiohttp.ClientTimeout(total=120)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._results = asyncio.Queue()
        self._requests = asyncio.Queue()
        self._request_worker = asyncio.create_task(self._run_request_worker())
        self._reset_audio_state()
        self._chunk_count = 0
        self._max_normalized_rms = 0.0
        self._audio_position_ms = 0.0
        logger.info(
            "[OpenAI REST ASR] Stream started: model=%s vad_start=%.4f "
            "vad_stop=%.4f silence=%dms",
            self.config.model,
            self._rms_threshold(self.config.vad_start_threshold),
            self._rms_threshold(self.config.vad_stop_threshold),
            self.config.speech_timeout_ms,
        )

    async def send_audio(self, audio_chunk: bytes) -> bool:
        """Feed PCM16 mono audio into the local end-of-utterance detector."""
        if self._session is None or self._results is None:
            return False
        if not audio_chunk:
            return True

        duration_ms = len(audio_chunk) / (_SAMPLE_RATE * _SAMPLE_WIDTH) * 1000.0
        chunk_start_ms = self._audio_position_ms
        chunk_end_ms = chunk_start_ms + duration_ms
        self._chunk_count += 1
        normalized_rms = self._normalized_rms(audio_chunk)
        self._max_normalized_rms = max(self._max_normalized_rms, normalized_rms)
        threshold_setting = (
            self.config.vad_stop_threshold
            if self._speech_active
            else self.config.vad_start_threshold
        )
        speech = normalized_rms >= self._rms_threshold(threshold_setting)
        if not self._speech_active:
            self._append_pre_roll(audio_chunk)
            if speech:
                self._speech_active = True
                self._speech_start_ms = chunk_start_ms
                self._utterance.extend(b"".join(self._pre_roll))
                self._pre_roll.clear()
                self._pre_roll_bytes = 0
                self._silence_ms = 0.0
                logger.info(
                    "[OpenAI REST ASR] VAD start: chunk=%d rms=%.4f "
                    "threshold=%.4f pre_roll=%dms",
                    self._chunk_count,
                    normalized_rms,
                    self._rms_threshold(self.config.vad_start_threshold),
                    self.config.speech_pad_ms,
                )
        else:
            self._utterance.extend(audio_chunk)
            if speech:
                self._silence_ms = 0.0
            else:
                self._silence_ms += duration_ms
                if self._silence_ms >= self.config.speech_timeout_ms:
                    # Anchor the UI's light-blue finalization interval at the
                    # beginning of the confirmed silence, not when the full
                    # silence timeout expires. This mirrors the Riva timeline's
                    # last-partial -> final semantics.
                    speech_end_ms = max(
                        self._speech_start_ms or 0.0,
                        chunk_end_ms - self._silence_ms,
                    )
                    utterance_ms = len(self._utterance) / (_SAMPLE_RATE * _SAMPLE_WIDTH) * 1000.0
                    logger.info(
                        "[OpenAI REST ASR] VAD end: utterance=%.0fms "
                        "silence=%.0fms, queueing transcription",
                        utterance_ms,
                        self._silence_ms,
                    )
                    self._queue_transcription(
                        bytes(self._utterance),
                        start_time=(self._speech_start_ms or 0.0) / 1000.0,
                        end_time=speech_end_ms / 1000.0,
                    )
                    self._reset_audio_state()
        self._audio_position_ms = chunk_end_ms
        return True

    async def receive_results(self) -> AsyncIterator[ASRResult]:
        """Yield final transcription results as REST calls complete."""
        if self._results is None:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        while True:
            result = await self._results.get()
            if result is None:
                break
            yield result

    async def stop_stream(self) -> None:
        """Flush pending speech and close the HTTP session."""
        if self._session is None:
            return
        if self._speech_active and self._utterance:
            logger.info(
                "[OpenAI REST ASR] Stream stopped during speech; flushing %.0fms",
                len(self._utterance) / (_SAMPLE_RATE * _SAMPLE_WIDTH) * 1000.0,
            )
            self._queue_transcription(
                bytes(self._utterance),
                start_time=(self._speech_start_ms or 0.0) / 1000.0,
                end_time=self._audio_position_ms / 1000.0,
            )
        elif self._chunk_count:
            logger.info(
                "[OpenAI REST ASR] Stream stopped without VAD start: "
                "chunks=%d max_rms=%.4f threshold=%.4f",
                self._chunk_count,
                self._max_normalized_rms,
                self._rms_threshold(self.config.vad_start_threshold),
            )
        if self._requests is not None:
            await self._requests.put(None)
        if self._request_worker is not None:
            await self._request_worker
        if self._results is not None:
            await self._results.put(None)
        await self._session.close()
        self._session = None
        self._results = None
        self._requests = None
        self._request_worker = None
        self._reset_audio_state()

    def _queue_transcription(
        self,
        pcm: bytes,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """Queue a REST transcription without stalling microphone ingestion."""
        minimum_bytes = int(_SAMPLE_RATE * _SAMPLE_WIDTH * 0.2)
        if len(pcm) < minimum_bytes or self._requests is None:
            return
        logger.info(
            "[OpenAI REST ASR] Queued %.0fms for transcription",
            len(pcm) / (_SAMPLE_RATE * _SAMPLE_WIDTH) * 1000.0,
        )
        self._requests.put_nowait((pcm, start_time, end_time))

    async def _run_request_worker(self) -> None:
        """Transcribe utterances serially so results retain microphone order."""
        if self._requests is None:
            return
        while True:
            request = await self._requests.get()
            if request is None:
                break
            if isinstance(request, tuple):
                pcm, start_time, end_time = request
                await self._transcribe(
                    pcm,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                # Backward-compatible with direct queue users and older tests.
                await self._transcribe(request)

    async def _transcribe(
        self,
        pcm: bytes,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        session = self._session
        results = self._results
        if session is None or results is None:
            return
        form = aiohttp.FormData()
        form.add_field(
            "file",
            self._wav_bytes(pcm),
            filename="utterance.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self.config.model)
        form.add_field("language", self.config.language or "auto")
        form.add_field("response_format", "json")
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = urljoin(self.config.api_base.rstrip("/") + "/", "audio/transcriptions")
        logger.info(
            "[OpenAI REST ASR] POST %s (%.0fms PCM)",
            url,
            len(pcm) / (_SAMPLE_RATE * _SAMPLE_WIDTH) * 1000.0,
        )
        try:
            async with session.post(url, data=form, headers=headers) as response:
                body = await response.text()
                if response.status >= 400:
                    raise ConnectionError(f"ASR request failed ({response.status}): {body[:300]}")
                payload = await response.json()
            text = str(payload.get("text", "")).strip()
            if text:
                logger.info("[OpenAI REST ASR] Transcript: %r", text[:120])
                await results.put(
                    ASRResult(
                        text=text,
                        is_final=True,
                        start_time=start_time,
                        end_time=end_time,
                        metadata={"backend": "openai-rest", "model": self.config.model},
                    )
                )
            else:
                logger.warning("[OpenAI REST ASR] Endpoint returned an empty transcript")
        except Exception as exc:
            logger.exception("OpenAI-compatible ASR request failed")
            await results.put(
                ASRResult(
                    text="",
                    is_final=True,
                    confidence=0.0,
                    start_time=start_time,
                    end_time=end_time,
                    metadata={"error": str(exc), "backend": "openai-rest"},
                )
            )

    def _append_pre_roll(self, chunk: bytes) -> None:
        self._pre_roll.append(chunk)
        self._pre_roll_bytes += len(chunk)
        max_bytes = int(_SAMPLE_RATE * _SAMPLE_WIDTH * max(self.config.speech_pad_ms, 0) / 1000)
        while self._pre_roll and self._pre_roll_bytes > max_bytes:
            removed = self._pre_roll.popleft()
            self._pre_roll_bytes -= len(removed)

    @staticmethod
    def _normalized_rms(pcm: bytes) -> float:
        """Return PCM16 RMS normalized to the 0..1 full-scale range."""
        sample_count = len(pcm) // 2
        if sample_count == 0:
            return 0.0
        samples = struct.unpack(f"<{sample_count}h", pcm[: sample_count * 2])
        rms = math.sqrt(sum(sample * sample for sample in samples) / sample_count)
        return rms / 32768.0

    @staticmethod
    def _rms_threshold(setting: float) -> float:
        """Map the UI's 0..1 VAD setting to a practical PCM RMS threshold."""
        # The UI threshold was designed for model VAD probabilities. For raw
        # audio, map 0..1 to -46..-26 dBFS (roughly 0.005..0.05 linear RMS).
        return 0.005 * (10 ** max(0.0, min(setting, 1.0)))

    def _is_speech(self, pcm: bytes) -> bool:
        """Return whether a PCM chunk crosses the current VAD threshold."""
        setting = (
            self.config.vad_stop_threshold
            if self._speech_active
            else self.config.vad_start_threshold
        )
        return self._normalized_rms(pcm) >= self._rms_threshold(setting)

    @staticmethod
    def _wav_bytes(pcm: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(_SAMPLE_WIDTH)
            wav_file.setframerate(_SAMPLE_RATE)
            wav_file.writeframes(pcm)
        return output.getvalue()

    def _reset_audio_state(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_bytes = 0
        self._utterance = bytearray()
        self._speech_active = False
        self._speech_start_ms = None
        self._silence_ms = 0.0
