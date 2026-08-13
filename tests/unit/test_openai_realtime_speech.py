# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for provider-neutral OpenAI Realtime speech adapters."""

import asyncio
import json

import aiohttp
import pytest

from multi_modal_ai_studio.backends.asr.openai_realtime import (
    OpenAIRealtimeASRBackend,
    _resample_pcm16,
)
from multi_modal_ai_studio.backends.realtime.client import (
    OpenAIRealtimeClient,
    RealtimeEvent,
)
from multi_modal_ai_studio.config.schema import ASRConfig, SessionConfig


class _SendingWebSocket:
    closed = False

    def __init__(self):
        self.messages = []

    async def send_str(self, message):
        self.messages.append(json.loads(message))


class _IncomingWebSocket:
    def __init__(self, payloads):
        self._messages = [
            aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(item), None)
            for item in payloads
        ]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


@pytest.mark.asyncio
async def test_ga_transcription_session_uses_nested_audio_schema():
    client = OpenAIRealtimeClient(
        "wss://example.test/v1/realtime",
        "key",
        model="transcribe-model",
        session_type="transcription",
        api_style="openai-ga",
        input_audio_transcription={"model": "transcribe-model", "language": "en"},
        turn_detection={"type": "server_vad", "silence_duration_ms": 700},
    )
    socket = _SendingWebSocket()
    client._ws = socket

    await client._send_session_update()

    session = socket.messages[0]["session"]
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["input"]["transcription"]["model"] == "transcribe-model"
    assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert "input_audio_transcription" not in session


@pytest.mark.asyncio
async def test_beta_transcription_session_supports_speaches_preview_schema():
    client = OpenAIRealtimeClient(
        "ws://localhost:18080/v1/realtime?intent=transcription",
        "",
        model="Systran/faster-whisper-small",
        session_type="transcription",
        api_style="openai-beta",
        input_audio_transcription={
            "model": "Systran/faster-whisper-small",
            "language": "en",
        },
        turn_detection={
            "type": "server_vad",
            "create_response": False,
            "prefix_padding_ms": 500,
        },
    )
    socket = _SendingWebSocket()
    client._ws = socket

    await client._send_session_update()

    session = socket.messages[0]["session"]
    assert session["input_audio_transcription"]["model"].endswith("small")
    assert session["turn_detection"]["create_response"] is False
    assert session["turn_detection"]["prefix_padding_ms"] == 500
    assert "type" not in session
    assert "input_audio_format" not in session
    assert "model=Systran%2Ffaster-whisper-small" in client._connect_url()
    assert "intent=transcription" in client._connect_url()


@pytest.mark.asyncio
async def test_realtime_response_audio_operations_remain_provider_neutral():
    client = OpenAIRealtimeClient("ws://localhost/realtime", "")
    socket = _SendingWebSocket()
    client._ws = socket

    await client.send_text("Say this naturally")
    await client.create_response(modalities=["audio"])
    await client.cancel_response("response-1")

    assert [message["type"] for message in socket.messages] == [
        "conversation.item.create",
        "response.create",
        "response.cancel",
    ]
    assert socket.messages[1]["response"]["output_modalities"] == ["audio"]
    assert socket.messages[2]["response_id"] == "response-1"


@pytest.mark.asyncio
async def test_beta_response_create_uses_preview_modalities_field():
    client = OpenAIRealtimeClient(
        "ws://localhost/realtime",
        "",
        api_style="openai-beta",
    )
    socket = _SendingWebSocket()
    client._ws = socket

    await client.create_response(modalities=["text", "audio"])

    assert socket.messages == [{
        "type": "response.create",
        "response": {"modalities": ["text", "audio"]},
    }]


@pytest.mark.asyncio
async def test_realtime_client_exposes_speech_audio_and_transcription_events():
    audio = "AQI="
    client = OpenAIRealtimeClient("ws://localhost/realtime", "")
    client._ws = _IncomingWebSocket([
        {"type": "input_audio_buffer.speech_started", "item_id": "item-1"},
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-1",
            "delta": "hello ",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-1",
            "transcript": "hello world",
        },
        {"type": "response.output_audio.delta", "item_id": "item-2", "delta": audio},
        {"type": "response.output_audio.done", "item_id": "item-2"},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "item-1"},
    ])

    await client._receive_loop()
    observed = []
    async for event in client.events():
        if event is None:
            break
        observed.append(event)

    assert [event.kind for event in observed] == [
        "speech_started",
        "transcript_delta",
        "transcript_completed",
        "audio",
        "audio_done",
        "speech_stopped",
    ]
    assert observed[3].audio == b"\x01\x02"
    assert all(event.item_id for event in observed)


@pytest.mark.asyncio
async def test_realtime_client_preserves_standalone_whitespace_delta():
    client = OpenAIRealtimeClient("ws://localhost/realtime", "")
    client._ws = _IncomingWebSocket([
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-1",
            "delta": " ",
        },
    ])

    await client._receive_loop()
    event = await client._event_queue.get()

    assert event.kind == "transcript_delta"
    assert event.text == " "


@pytest.mark.asyncio
async def test_speaches_read_only_prefix_notice_is_nonfatal():
    client = OpenAIRealtimeClient(
        "ws://localhost/realtime",
        "",
        api_style="openai-beta",
    )
    client._ws = _IncomingWebSocket([{
        "type": "error",
        "error": {
            "message": "Specifying `session.turn_detection.prefix_padding_ms` is not supported."
        },
    }])

    await client._receive_loop()
    event = await client._event_queue.get()

    assert event.kind == "session_warning"


class _FakeRealtimeClient:
    def __init__(self, events):
        self._events = events

    async def events(self):
        for event in self._events:
            yield event
        yield None


@pytest.mark.asyncio
async def test_realtime_asr_accumulates_deltas_and_preserves_vad_boundaries():
    backend = OpenAIRealtimeASRBackend(ASRConfig(
        scheme="openai-realtime",
        realtime_url="ws://localhost/realtime",
        realtime_session_type="transcription",
        realtime_api_style="openai-beta",
        model="test-model",
        language="en-US",
    ))
    backend._client = _FakeRealtimeClient([
        RealtimeEvent(kind="speech_started", item_id="item-1"),
        RealtimeEvent(kind="transcript_delta", text="hello", item_id="item-1"),
        RealtimeEvent(kind="transcript_delta", text=" ", item_id="item-1"),
        RealtimeEvent(kind="transcript_delta", text="world", item_id="item-1"),
        RealtimeEvent(kind="speech_stopped", item_id="item-1"),
        RealtimeEvent(
            kind="transcript_completed",
            text="hello world",
            item_id="item-1",
        ),
    ])

    results = [result async for result in backend.receive_results()]

    assert [result.metadata.get("control_event") for result in (results[0], results[3])] == [
        "vad_start",
        "vad_end",
    ]
    assert results[1].text == "hello"
    assert results[2].text == "hello world"
    assert results[-1].is_final
    assert results[-1].text == "hello world"
    assert results[-1].end_time >= results[-1].start_time


def test_realtime_asr_config_round_trip_and_resampling():
    config = SessionConfig.from_dict({
        "asr": {
            "scheme": "openai-realtime",
            "realtime_url": "ws://localhost/realtime",
            "realtime_session_type": "transcription",
            "realtime_api_style": "openai-beta",
            "enable_vad": False,
            "interim_results": False,
        }
    })

    assert config.asr.realtime_api_style == "openai-beta"
    assert config.asr.enable_vad is False
    assert config.asr.interim_results is False
    assert len(_resample_pcm16(b"\x00\x00" * 160, 16000, 24000)) == 480


def test_realtime_asr_config_rejects_unknown_wire_format():
    config = ASRConfig(
        scheme="openai-realtime",
        realtime_url="ws://localhost/realtime",
        realtime_api_style="unknown",  # type: ignore[arg-type]
    )

    assert "Unsupported Realtime API wire format" in config.validate()
