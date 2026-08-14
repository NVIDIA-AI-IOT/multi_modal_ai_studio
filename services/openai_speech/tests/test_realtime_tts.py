# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realtime TTS cancellation and event-order tests."""

import asyncio
import json

import pytest
from openai_speech.realtime_tts import RealtimeTTSConnection


class _SendingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(json.loads(text))


class _DelayedTTS:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def load(self):
        return None

    async def synthesize(self, text, voice, language, response_format):
        self.started.set()
        await self.release.wait()
        return b"\x01\x00" * 100, "audio/pcm"


@pytest.mark.asyncio
async def test_response_cancel_stops_audio_delivery_and_is_idempotent():
    websocket = _SendingWebSocket()
    engine = _DelayedTTS()
    connection = RealtimeTTSConnection(websocket, engine, "magpie")
    await connection._create_item(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Do not play this"}],
        }
    )
    await connection._create_response({"output_modalities": ["audio"]})
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    response_id = next(
        message["response"]["id"]
        for message in websocket.messages
        if message["type"] == "response.created"
    )
    await connection._cancel_response(response_id)
    engine.release.set()
    await connection.close()

    event_types = [message["type"] for message in websocket.messages]
    assert "response.output_audio.delta" not in event_types
    done = [message for message in websocket.messages if message["type"] == "response.done"]
    assert len(done) == 1
    assert done[0]["response"]["status"] == "cancelled"
