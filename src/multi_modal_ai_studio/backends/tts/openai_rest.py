# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible REST text-to-speech backend."""

import asyncio
import logging
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

logger = logging.getLogger(__name__)

# The local Magpie endpoint buffers a complete do_tts() call before returning
# HTTP audio. Keep requests bounded even when LLM-to-TTS streaming is disabled.
MAX_REST_TTS_CHARS = 50


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
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_tasks = set()

    def cancel_synthesis(self) -> int:
        """Cancel active REST synthesis tasks immediately."""
        caller = asyncio.current_task()
        active = [
            task
            for task in self._active_tasks
            if not task.done() and task is not caller
        ]
        for task in active:
            task.cancel()
        if active:
            logger.info("Cancelled %d active REST TTS request(s)", len(active))
        return len(active)

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
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_tasks.add(current_task)

        try:
            for text_index, text_chunk in enumerate(text_chunks):
                request = {
                    "model": self.config.model or "tts-1",
                    "input": text_chunk,
                    "voice": self.config.voice,
                    "response_format": "pcm",
                    "speed": self.config.speed,
                }
                if (self.config.model or "").startswith("nvidia/"):
                    request["language"] = (
                        self.config.language or "en"
                    ).split("-")[0].lower()
                is_last_text = text_index == len(text_chunks) - 1
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
                    pending = None
                    aligner = _PCM16FrameAligner()
                    async for audio in response.content.iter_chunked(16384):
                        audio = aligner.feed(audio)
                        if not audio:
                            continue
                        if pending is not None:
                            yield TTSChunk(
                                audio=pending,
                                sample_rate=self.config.sample_rate,
                                is_final=False,
                                metadata={
                                    "text_chunk_index": text_index,
                                    "total_text_chunks": len(text_chunks),
                                },
                            )
                        pending = audio
                    trailing = aligner.finish()
                    if trailing:
                        logger.warning(
                            "OpenAI-compatible TTS returned %d trailing byte(s) "
                            "outside a complete PCM16 frame; dropping them",
                            len(trailing),
                        )
                    if pending:
                        yield TTSChunk(
                            audio=pending,
                            sample_rate=self.config.sample_rate,
                            is_final=is_last_text,
                            metadata={
                                "text_chunk_index": text_index,
                                "total_text_chunks": len(text_chunks),
                            },
                        )
        except aiohttp.ClientError as exc:
            raise ConnectionError(f"OpenAI-compatible TTS request failed: {exc}") from exc
        finally:
            if current_task is not None:
                self._active_tasks.discard(current_task)

    async def close(self) -> None:
        """Close the reusable HTTP connection pool."""
        if self._session is not None:
            await self._session.close()
            self._session = None
