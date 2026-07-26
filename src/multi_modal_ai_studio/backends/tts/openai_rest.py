# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible REST text-to-speech backend."""

import logging
from typing import AsyncIterator, Optional
from urllib.parse import urljoin

import aiohttp

from multi_modal_ai_studio.backends.base import (
    ConfigError,
    ConnectionError,
    TTSBackend,
    TTSChunk,
)
from multi_modal_ai_studio.config.schema import TTSConfig

logger = logging.getLogger(__name__)


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

    async def synthesize_stream(self, text: str) -> AsyncIterator[TTSChunk]:
        """Generate an utterance and yield response chunks as PCM."""
        if not text.strip():
            return
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))

        request = {
            "model": self.config.model or "tts-1",
            "input": text,
            "voice": self.config.voice,
            "response_format": "pcm",
            "speed": self.config.speed,
        }
        if (self.config.model or "").startswith("nvidia/"):
            request["language"] = (self.config.language or "en").split("-")[0].lower()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = urljoin(self.config.api_base.rstrip("/") + "/", "audio/speech")

        try:
            async with self._session.post(url, json=request, headers=headers) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise ConnectionError(f"TTS request failed ({response.status}): {body[:300]}")
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
                        is_final=True,
                    )
        except aiohttp.ClientError as exc:
            raise ConnectionError(f"OpenAI-compatible TTS request failed: {exc}") from exc

    async def close(self) -> None:
        """Close the reusable HTTP connection pool."""
        if self._session is not None:
            await self._session.close()
            self._session = None
