# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime exact-text text-to-speech backend."""

import asyncio
import logging
from contextlib import suppress
from typing import AsyncIterator, Dict, Set

from multi_modal_ai_studio.backends.base import (
    ConfigError,
    ConnectionError,
    TTSBackend,
    TTSChunk,
)
from multi_modal_ai_studio.backends.realtime.client import OpenAIRealtimeClient
from multi_modal_ai_studio.config.schema import TTSConfig

logger = logging.getLogger(__name__)


class OpenAIRealtimeTTSBackend(TTSBackend):
    """Synthesize exact text through an OpenAI-compatible Realtime socket."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        if config.scheme != "openai-realtime":
            raise ConfigError(f"Expected scheme 'openai-realtime', got '{config.scheme}'")
        if config.realtime_transport != "websocket":
            raise ConfigError("Realtime TTS currently requires WebSocket")
        if not (config.realtime_url or config.api_base):
            raise ConfigError("Realtime TTS URL is required")
        self._active_clients: Dict[asyncio.Task, OpenAIRealtimeClient] = {}
        self._cancel_tasks: Dict[asyncio.Task, asyncio.Task] = {}
        self._cancelled_tasks: Set[asyncio.Task] = set()

    def _make_client(self) -> OpenAIRealtimeClient:
        return OpenAIRealtimeClient(
            url=(self.config.realtime_url or self.config.api_base or "").strip(),
            api_key=(self.config.api_key or "").strip(),
            model=self.config.model or "tts-1",
            voice=self.config.voice,
            output_audio_sample_rate=self.config.sample_rate,
            session_type="realtime",
            api_style=self.config.realtime_api_style,
        )

    def cancel_synthesis(self) -> int:
        """Request provider cancellation without cancelling the turn executor."""
        cancelled = 0
        for task, client in list(self._active_clients.items()):
            if task.done() or task in self._cancelled_tasks:
                continue
            self._cancelled_tasks.add(task)
            self._cancel_tasks[task] = asyncio.create_task(client.cancel_response())
            cancelled += 1
        if cancelled:
            logger.info(
                "Requested cancellation of %d active Realtime TTS response(s)",
                cancelled,
            )
        return cancelled

    async def synthesize_stream(self, text: str) -> AsyncIterator[TTSChunk]:
        if not text.strip():
            return
        client = self._make_client()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Realtime TTS synthesis requires an asyncio task")
        self._active_clients[task] = client
        response_started = False
        try:
            await client.connect()
            language = (self.config.language or "en").split("-", 1)[0].lower()
            await client.send_text(text, metadata={"language": language})
            await client.create_response(modalities=["audio"])
            async for event in client.events():
                if event is None:
                    break
                if event.kind == "session_ready":
                    continue
                if event.kind == "audio":
                    response_started = True
                    yield TTSChunk(
                        audio=event.audio or b"",
                        sample_rate=event.sample_rate,
                        is_final=False,
                        metadata={
                            "item_id": event.item_id,
                            "response_id": event.response_id,
                        },
                    )
                    continue
                if event.kind == "audio_done":
                    if task not in self._cancelled_tasks:
                        yield TTSChunk(
                            audio=b"",
                            sample_rate=event.sample_rate,
                            is_final=True,
                            metadata={
                                "item_id": event.item_id,
                                "response_id": event.response_id,
                            },
                        )
                    break
                if event.kind == "response_done":
                    status = (event.raw or {}).get("response", {}).get("status")
                    if status in {"cancelled", "failed"} or task in self._cancelled_tasks:
                        break
                    if not response_started:
                        raise ConnectionError("Realtime TTS response completed without audio")
                if event.kind == "error":
                    raise ConnectionError(event.message or "Realtime TTS error")
        finally:
            cancel_task = self._cancel_tasks.pop(task, None)
            if cancel_task is not None:
                # Barge-in cancels the consumer task immediately.  Make sure the
                # provider sees response.cancel before this socket is closed.
                with suppress(Exception):
                    await asyncio.shield(cancel_task)
            await client.disconnect()
            self._active_clients.pop(task, None)
            self._cancelled_tasks.discard(task)

    async def close(self) -> None:
        self.cancel_synthesis()
        if self._cancel_tasks:
            await asyncio.gather(*self._cancel_tasks.values(), return_exceptions=True)
        await asyncio.gather(
            *(client.disconnect() for client in self._active_clients.values()),
            return_exceptions=True,
        )
        self._active_clients.clear()
        self._cancel_tasks.clear()
        self._cancelled_tasks.clear()
