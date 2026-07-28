# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation regression tests for the blocking Riva TTS stream."""

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.backends.tts.riva import (
    MAX_INTERRUPTIBLE_TTS_CHARS,
    RivaTTSBackend,
)


class _BlockingRivaCall:
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self.started.set()
        self.cancelled.wait(timeout=5)
        raise StopIteration

    def cancel(self):
        self.cancelled.set()
        return True


class _FakeTTSService:
    def __init__(self, call):
        self.call = call

    def synthesize_online(self, *args, **kwargs):
        return self.call


@pytest.mark.asyncio
async def test_cancelling_riva_tts_cancels_blocking_grpc_call():
    call = _BlockingRivaCall()
    backend = RivaTTSBackend.__new__(RivaTTSBackend)
    backend.config = SimpleNamespace(
        voice="Magpie-Multilingual.EN-US.Mia.Neutral",
        sample_rate=22050,
    )
    backend.timeline = None
    backend.tts_service = _FakeTTSService(call)
    backend.logger = logging.getLogger("test.riva.tts")
    backend._active_rpc_calls = {}
    backend._active_rpc_lock = threading.Lock()

    async def consume():
        async for _ in backend.synthesize_stream("Long response"):
            pass

    task = asyncio.create_task(consume())
    started = await asyncio.to_thread(call.started.wait, 1)
    assert started

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call.cancelled.wait(timeout=0.5)


@pytest.mark.asyncio
async def test_riva_backend_can_cancel_active_rpc_before_task_cleanup():
    call = _BlockingRivaCall()
    backend = RivaTTSBackend.__new__(RivaTTSBackend)
    backend.config = SimpleNamespace(
        voice="Magpie-Multilingual.EN-US.Mia.Neutral",
        sample_rate=22050,
    )
    backend.timeline = None
    backend.tts_service = _FakeTTSService(call)
    backend.logger = logging.getLogger("test.riva.tts")
    backend._active_rpc_calls = {}
    backend._active_rpc_lock = threading.Lock()

    async def consume():
        async for _ in backend.synthesize_stream("Long response"):
            pass

    task = asyncio.create_task(consume())
    started = await asyncio.to_thread(call.started.wait, 1)
    assert started

    assert backend.cancel_synthesis() == 1
    await asyncio.wait_for(task, timeout=0.5)
    assert call.cancelled.is_set()


def test_riva_tts_bounds_each_server_request_for_barge_in():
    text = (
        "The history of robotics spans ancient automata and modern autonomous "
        "machines with increasingly capable perception and control systems. "
    ) * 8

    chunks = RivaTTSBackend._split_text_by_sentences(
        text,
        max_chars=MAX_INTERRUPTIBLE_TTS_CHARS,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_INTERRUPTIBLE_TTS_CHARS for chunk in chunks)
    assert " ".join(chunks).split() == text.split()
