# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible REST text-to-speech backend."""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import urljoin

import aiohttp

from multi_modal_ai_studio.backends.base import (
    ConfigError,
    ConnectionError,
    TTSBackend,
    TTSChunk,
    split_tts_text,
)
from multi_modal_ai_studio.config.schema import TTSConfig
from multi_modal_ai_studio.core.voice_library import VoiceLibrary

logger = logging.getLogger(__name__)

# Preserve complete short answers while bounding Magpie's time-to-first-audio.
# Longer responses split at sentence/phrase boundaries rather than the former
# 50-character limit that restarted prosody every few words.
MAX_REST_TTS_CHARS = 400
REST_TTS_CONNECTION_RETRIES = 1

QWEN3_LANGUAGES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian", "auto": "Auto",
}


def build_speech_request(config: TTSConfig, text: str) -> dict:
    """Keep spoken text separate from direction and provider-specific controls.

    The Qwen profiles target vLLM-Omni's audio/speech extension, not the
    WebSocket Realtime API. Its default streaming format is SSE, not PCM.
    """
    errors = config.validate_rest_controls()
    if errors:
        raise ConfigError("; ".join(errors))
    request = {
        "model": config.model or "tts-1", "input": text,
        "voice": config.voice, "response_format": "pcm", "speed": config.speed,
    }
    if config.instructions.strip():
        request["instructions"] = config.instructions.strip()
    if config.request_profile in ("qwen3-customvoice", "qwen3-voicedesign", "qwen3-base"):
        language = (config.language or "auto").strip().lower().replace("_", "-")
        names = {name.lower(): name for name in QWEN3_LANGUAGES.values()}
        language = names.get(language) or QWEN3_LANGUAGES.get(language.split("-")[0])
        if language is None:
            raise ConfigError("Unsupported Qwen3-TTS language; choose one of its ten languages or Auto")
        design = config.request_profile == "qwen3-voicedesign"
        clone = config.request_profile == "qwen3-base"
        request.update(
            task_type="Base" if clone else "VoiceDesign" if design else "CustomVoice",
            language=language, stream=True, stream_format="audio",
        )
        if design or clone:
            request.pop("voice")
        if clone:
            try:
                request.update(VoiceLibrary().speech_fields(config.reference_voice_id, config.reference_voice_mode))
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
    elif (config.model or "").startswith("nvidia/"):
        request["language"] = (config.language or "en").split("-")[0].lower()
    return request


@dataclass
class _RequestDone:
    """Terminal message from an isolated REST request worker."""

    cancelled: bool = False


@dataclass
class _RequestFailure:
    """Propagate a request worker failure without cancelling its caller."""

    error: Exception


class _PCM16FrameAligner:
    """Preserve 16-bit PCM sample boundaries across arbitrary HTTP chunks."""

    def __init__(self) -> None:
        self._carry = b""

    def feed(self, data: bytes) -> bytes:
        """Return complete PCM16 frames and retain a trailing partial frame."""
        combined = self._carry + data
        aligned_length = len(combined) - (len(combined) % 2)
        self._carry = combined[aligned_length:]
        return combined[:aligned_length]

    def finish(self) -> bytes:
        """Return any malformed trailing bytes so callers can report them."""
        trailing = self._carry
        self._carry = b""
        return trailing


class OpenAIRestTTSBackend(TTSBackend):
    """Stream PCM bytes from an OpenAI-compatible `/v1/audio/speech` API."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        if config.scheme != "openai-rest":
            raise ConfigError(f"Expected scheme 'openai-rest', got '{config.scheme}'")
        if not config.api_base:
            raise ConfigError("OpenAI-compatible TTS api_base is required")
        # Fail before opening a connection, including in voice preview.
        build_speech_request(config, "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_tasks = set()

    def cancel_synthesis(self) -> int:
        """Cancel active HTTP workers without cancelling the turn executor."""
        active = [
            task
            for task in self._active_tasks
            if not task.done()
        ]
        for task in active:
            task.cancel()
        if active:
            logger.info("Cancelled %d active REST TTS request(s)", len(active))
        return len(active)

    async def _produce_request_audio(
        self,
        *,
        url: str,
        request: dict,
        headers: dict,
        text_index: int,
        total_text_chunks: int,
        is_last_text: bool,
        output: asyncio.Queue,
    ) -> None:
        """Run one HTTP request in an isolated, cancellable worker task."""
        try:
            for attempt in range(REST_TTS_CONNECTION_RETRIES + 1):
                published_audio = False
                try:
                    async with self._session.post(
                        url,
                        json=request,
                        headers=headers,
                    ) as response:
                        if response.status >= 400:
                            body = await response.text()
                            raise ConnectionError(
                                f"TTS request failed ({response.status}): {body[:300]}"
                            )
                        content_type = getattr(response, "headers", {}).get("Content-Type", "").lower()
                        if any(kind in content_type for kind in ("text/", "json", "wav", "mpeg", "flac", "ogg")):
                            raise ConnectionError(
                                "TTS returned a non-PCM response (" + content_type + "). "
                                "Select the correct request profile; event streams must not be played as PCM."
                            )
                        pending = None
                        aligner = _PCM16FrameAligner()
                        async for audio in response.content.iter_chunked(16384):
                            audio = aligner.feed(audio)
                            if not audio:
                                continue
                            if pending is not None:
                                output.put_nowait(
                                    TTSChunk(
                                        audio=pending,
                                        sample_rate=self.config.sample_rate,
                                        is_final=False,
                                        metadata={
                                            "text_chunk_index": text_index,
                                            "total_text_chunks": total_text_chunks,
                                        },
                                    )
                                )
                                published_audio = True
                            pending = audio
                        trailing = aligner.finish()
                        if trailing:
                            logger.warning(
                                "OpenAI-compatible TTS returned %d trailing byte(s) "
                                "outside a complete PCM16 frame; dropping them",
                                len(trailing),
                            )
                        if pending:
                            output.put_nowait(
                                TTSChunk(
                                    audio=pending,
                                    sample_rate=self.config.sample_rate,
                                    is_final=is_last_text,
                                    metadata={
                                        "text_chunk_index": text_index,
                                        "total_text_chunks": total_text_chunks,
                                    },
                                )
                            )
                            published_audio = True
                    break
                except aiohttp.ClientConnectionError:
                    if (
                        attempt >= REST_TTS_CONNECTION_RETRIES
                        or published_audio
                    ):
                        raise
                    logger.warning(
                        "OpenAI-compatible TTS connection closed before audio; "
                        "retrying request once with a fresh connection"
                    )
        except asyncio.CancelledError:
            output.put_nowait(_RequestDone(cancelled=True))
            raise
        except aiohttp.ClientError as exc:
            output.put_nowait(
                _RequestFailure(
                    ConnectionError(
                        f"OpenAI-compatible TTS request failed: {exc}"
                    )
                )
            )
        except Exception as exc:
            output.put_nowait(_RequestFailure(exc))
        else:
            output.put_nowait(_RequestDone())

    async def synthesize_stream(self, text: str) -> AsyncIterator[TTSChunk]:
        """Generate an utterance and yield response chunks as PCM."""
        if not text.strip():
            return
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = urljoin(self.config.api_base.rstrip("/") + "/", "audio/speech")
        text_chunks = split_tts_text(text, MAX_REST_TTS_CHARS)

        for text_index, text_chunk in enumerate(text_chunks):
            request = build_speech_request(self.config, text_chunk)
            output: asyncio.Queue = asyncio.Queue()
            worker = asyncio.create_task(
                self._produce_request_audio(
                    url=url,
                    request=request,
                    headers=headers,
                    text_index=text_index,
                    total_text_chunks=len(text_chunks),
                    is_last_text=text_index == len(text_chunks) - 1,
                    output=output,
                )
            )
            self._active_tasks.add(worker)
            cancelled = False
            try:
                while True:
                    message = await output.get()
                    if isinstance(message, TTSChunk):
                        yield message
                    elif isinstance(message, _RequestFailure):
                        raise message.error
                    elif isinstance(message, _RequestDone):
                        cancelled = message.cancelled
                        break
            finally:
                if not worker.done():
                    worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
                self._active_tasks.discard(worker)
            if cancelled:
                return

    async def close(self) -> None:
        """Close the reusable HTTP connection pool."""
        if self._session is not None:
            await self._session.close()
            self._session = None
