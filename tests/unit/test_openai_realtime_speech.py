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
from multi_modal_ai_studio.backends.tts.openai_realtime import (
    OpenAIRealtimeTTSBackend,
)
from multi_modal_ai_studio.config.schema import ASRConfig, SessionConfig, TTSConfig


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

    await client.send_text("Say this naturally", metadata={"language": "en"})
    await client.create_response(modalities=["audio"])
    await client.cancel_response("response-1")

    assert [message["type"] for message in socket.messages] == [
        "conversation.item.create",
        "response.create",
        "response.cancel",
    ]
    assert socket.messages[0]["item"]["metadata"] == {"language": "en"}
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
    client = OpenAIRealtimeClient(
        "ws://localhost/realtime",
        "",
        output_audio_sample_rate=22050,
    )
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
        {
            "type": "response.output_audio.delta",
            "response_id": "response-1",
            "item_id": "item-2",
            "delta": audio,
        },
        {
            "type": "response.output_audio.done",
            "response_id": "response-1",
            "item_id": "item-2",
        },
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
    assert observed[3].response_id == "response-1"
    assert observed[4].response_id == "response-1"
    assert observed[4].sample_rate == 22050
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


class _FakeRealtimeTTSClient:
    def __init__(self, events):
        self._events = events
        self.connected = False
        self.disconnected = False
        self.text = None
        self.metadata = None
        self.modalities = None
        self.cancelled = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def send_text(self, text, *, role="user", metadata=None):
        self.text = text
        self.metadata = metadata

    async def create_response(self, *, modalities=None, instructions=None):
        self.modalities = modalities

    async def cancel_response(self, response_id=None):
        self.cancelled = True

    async def events(self):
        for event in self._events:
            yield event
        yield None


@pytest.mark.asyncio
async def test_realtime_tts_yields_correlated_audio_and_final_marker():
    backend = OpenAIRealtimeTTSBackend(TTSConfig(
        scheme="openai-realtime",
        realtime_url="ws://localhost:8082/v1/realtime",
        realtime_api_style="openai-ga",
        model="nvidia/magpie_tts_multilingual_357m",
        voice="Sofia",
        language="en-US",
        sample_rate=22050,
    ))
    fake = _FakeRealtimeTTSClient([
        RealtimeEvent(kind="session_ready"),
        RealtimeEvent(
            kind="audio",
            audio=b"\x01\x00",
            sample_rate=22050,
            item_id="item-1",
            response_id="resp-1",
        ),
        RealtimeEvent(
            kind="audio_done",
            sample_rate=22050,
            item_id="item-1",
            response_id="resp-1",
        ),
    ])
    backend._make_client = lambda: fake

    chunks = [chunk async for chunk in backend.synthesize_stream("Hello")]

    assert fake.connected and fake.disconnected
    assert fake.text == "Hello"
    assert fake.metadata == {"language": "en"}
    assert fake.modalities == ["audio"]
    assert chunks[0].audio == b"\x01\x00"
    assert chunks[0].metadata == {"item_id": "item-1", "response_id": "resp-1"}
    assert chunks[-1].is_final is True


@pytest.mark.asyncio
async def test_realtime_tts_allows_concurrent_lookahead_requests():
    backend = OpenAIRealtimeTTSBackend(TTSConfig(
        scheme="openai-realtime",
        realtime_url="ws://localhost:8082/v1/realtime",
        model="nvidia/magpie_tts_multilingual_357m",
        sample_rate=22050,
    ))
    clients = [
        _FakeRealtimeTTSClient([
            RealtimeEvent(kind="audio", audio=audio, sample_rate=22050),
            RealtimeEvent(kind="audio_done", sample_rate=22050),
        ])
        for audio in (b"\x01\x00", b"\x02\x00")
    ]
    backend._make_client = lambda: clients.pop(0)

    async def collect(text):
        return [chunk async for chunk in backend.synthesize_stream(text)]

    first, second = await asyncio.gather(collect("First."), collect("Second."))

    assert first[0].audio == b"\x01\x00"
    assert second[0].audio == b"\x02\x00"
    assert backend._active_clients == {}


@pytest.mark.asyncio
async def test_realtime_tts_sends_cancel_before_disconnect():
    calls = []
    release = asyncio.Event()

    class BlockingClient(_FakeRealtimeTTSClient):
        async def cancel_response(self, response_id=None):
            calls.append("cancel")
            self.cancelled = True
            release.set()

        async def disconnect(self):
            calls.append("disconnect")
            self.disconnected = True

        async def events(self):
            await release.wait()
            yield RealtimeEvent(
                kind="response_done",
                raw={"response": {"status": "cancelled"}},
            )

    backend = OpenAIRealtimeTTSBackend(TTSConfig(
        scheme="openai-realtime",
        realtime_url="ws://localhost:8082/v1/realtime",
        model="nvidia/magpie_tts_multilingual_357m",
    ))
    client = BlockingClient([])
    backend._make_client = lambda: client

    synthesis = asyncio.create_task(
        backend.synthesize_stream("Cancel this response.").__anext__()
    )
    while not client.connected:
        await asyncio.sleep(0)

    assert backend.cancel_synthesis() == 1
    synthesis.cancel()
    await asyncio.gather(synthesis, return_exceptions=True)

    assert calls == ["cancel", "disconnect"]


def test_realtime_tts_config_round_trip_and_validation():
    config = SessionConfig.from_dict({
        "tts": {
            "scheme": "openai-realtime",
            "realtime_url": "ws://localhost:8082/v1/realtime",
            "realtime_transport": "websocket",
            "realtime_api_style": "openai-ga",
            "sample_rate": 22050,
        }
    }).tts

    assert config.realtime_url == "ws://localhost:8082/v1/realtime"
    assert config.realtime_api_style == "openai-ga"
    assert config.validate() == []
