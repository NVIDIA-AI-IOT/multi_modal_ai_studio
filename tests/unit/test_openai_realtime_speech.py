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
from multi_modal_ai_studio.backends.realtime.duplex import AcousticTurn, ContinuousPCMInput


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
async def test_voicechat_configures_only_supported_native_fields():
    client = OpenAIRealtimeClient("ws://localhost/realtime", "", api_style="nemo-voicechat",
                                  instructions="Speak briefly.", turn_detection={"type": "server_vad"})
    client._ws = _SendingWebSocket()
    await client._send_session_update()
    config = client._ws.messages[0]["session"]
    assert config == {
        "instructions": "Speak briefly.", "tools": [],
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}},
                  "output": {"format": "pcm16"}},
    }
    with pytest.raises(NotImplementedError):
        await client.cancel_response()
    assert len(client._ws.messages) == 1


@pytest.mark.asyncio
async def test_voicechat_finalizes_multiple_turns_sharing_response_id_and_drains_partial():
    client = OpenAIRealtimeClient("ws://localhost/realtime", "", api_style="nemo-voicechat")
    client._ws = _IncomingWebSocket([
        {"type": "response.output_audio_transcript.done", "transcript": "", "response_id": "stream"},
        {"type": "response.output_audio_transcript.delta", "delta": "Two."},
        {"type": "response.output_audio_transcript.done", "transcript": "Two.", "response_id": "stream"},
        {"type": "response.output_audio_transcript.delta", "delta": "Two."},
        {"type": "response.output_audio_transcript.done", "transcript": "Two.", "response_id": "stream"},
        {"type": "response.output_audio_transcript.delta", "delta": "Four."},
        {"type": "session.end"},
    ])
    await client._receive_loop()
    events = [e async for e in client.events() if e is not None]
    assert [e.text for e in events if e.kind == "output_transcript_completed"] == ["Two.", "Two.", "Four."]
    assert client._session_ended.is_set()


@pytest.mark.asyncio
async def test_native_transcript_done_does_not_duplicate_ga_item_final():
    client = OpenAIRealtimeClient("ws://localhost/realtime", "")
    client._ws = _IncomingWebSocket([
        {"type": "response.output_audio_transcript.done", "transcript": "Two."},
        {"type": "conversation.item.done", "item": {"role": "assistant", "content": [
            {"type": "output_audio", "transcript": "Two."}]}},
    ])
    await client._receive_loop()
    events = [e async for e in client.events() if e is not None]
    assert [e.text for e in events if e.kind == "output_transcript_completed"] == ["Two."]


def test_continuous_input_supplies_silence_after_mute_and_bounds_stale_capture():
    source = ContinuousPCMInput()
    source.feed(b"\x01\x02" * 100)
    assert source.take() == b"\x01\x02" * 100 + bytes(3640)
    source.feed(b"\x01\x02" * 1920)
    source.clear()
    assert source.take() == bytes(3840)
    source.feed(b"\x01\x02" * 20000)
    assert len(source.buffer) == 3840 * 5
    assert source.take() == b"\x01\x02" * 1920


def test_acoustic_turn_waits_for_silence_then_recovers_on_next_utterance():
    turn = AcousticTurn()
    assert turn.observe(bytes(3840), 0) == (False, False)
    assert turn.observe(b"\x00\x80" * 1920, 1) == (True, False)
    assert turn.observe(bytes(3840), 1.2) == (False, False)
    assert turn.observe(bytes(3840), 1.5) == (False, True)
    assert turn.observe(bytes(3840), 2) == (False, False)
    assert turn.observe(b"\x00\x01" * 1920, 2.1) == (True, False)


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


def _asr_with_events(events):
    backend = OpenAIRealtimeASRBackend(ASRConfig(
        scheme="openai-realtime", realtime_url="ws://localhost/realtime",
        realtime_session_type="transcription",
    ))
    backend._client = _FakeRealtimeClient(events)
    timestamps = iter(float(index) for index in range(len(events)))
    backend._timestamp = lambda: next(timestamps)
    return backend


@pytest.mark.asyncio
async def test_realtime_asr_duplicate_boundaries_and_finals_do_not_repeat_turn():
    backend = _asr_with_events([
        RealtimeEvent(kind="speech_started", item_id="a"),
        RealtimeEvent(kind="speech_started", item_id="a"),
        RealtimeEvent(kind="speech_stopped", item_id="a"),
        RealtimeEvent(kind="speech_stopped", item_id="a"),
        RealtimeEvent(kind="transcript_completed", item_id="a", text="Stop"),
        RealtimeEvent(kind="transcript_completed", item_id="a", text="Stop"),
        RealtimeEvent(kind="transcript_delta", item_id="a", text="Stop"),
        RealtimeEvent(kind="speech_stopped", item_id="a"),
        RealtimeEvent(kind="speech_started", item_id="a"),
        RealtimeEvent(kind="error", item_id="a", message="duplicate commit"),
    ])
    results = [result async for result in backend.receive_results()]

    assert len(results) == 4
    assert [r.metadata.get("control_event") for r in results[:2]] == ["vad_start", "vad_end"]
    assert results[2].text == "Stop" and results[2].is_final
    assert (results[2].start_time, results[2].end_time) == (0.0, 2.0)
    assert results[2].metadata["speech_end_observed"] is True
    assert results[3].metadata["error"] == "duplicate commit"


@pytest.mark.asyncio
async def test_realtime_asr_overlapping_items_keep_their_own_boundaries():
    backend = _asr_with_events([
        RealtimeEvent(kind="speech_started", item_id="a"),
        RealtimeEvent(kind="speech_stopped", item_id="a"),
        RealtimeEvent(kind="speech_started", item_id="b"),
        RealtimeEvent(kind="speech_stopped", item_id="b"),
        RealtimeEvent(kind="transcript_completed", item_id="b", text="Second"),
        RealtimeEvent(kind="transcript_completed", item_id="a", text="First"),
    ])
    results = [result async for result in backend.receive_results()]
    finals = [r for r in results if r.is_final]

    assert [(r.text, r.start_time, r.end_time) for r in finals] == [
        ("Second", 2.0, 3.0), ("First", 0.0, 1.0),
    ]


@pytest.mark.asyncio
async def test_realtime_asr_final_before_endpoint_does_not_borrow_previous_end():
    backend = _asr_with_events([
        RealtimeEvent(kind="speech_started", item_id="a"),
        RealtimeEvent(kind="speech_stopped", item_id="a"),
        RealtimeEvent(kind="transcript_completed", item_id="a", text="First"),
        RealtimeEvent(kind="speech_started", item_id="b"),
        RealtimeEvent(kind="transcript_completed", item_id="b", text="Second"),
        RealtimeEvent(kind="speech_stopped", item_id="b"),
        RealtimeEvent(kind="transcript_completed", item_id="unknown", text="Third"),
    ])
    results = [result async for result in backend.receive_results()]
    finals = [r for r in results if r.is_final]

    assert [(r.text, r.start_time, r.end_time) for r in finals] == [
        ("First", 0.0, 1.0), ("Second", 3.0, 4.0), ("Third", None, 6.0),
    ]
    assert [r.metadata["speech_end_observed"] for r in finals] == [True, False, False]
    assert sum(r.metadata.get("control_event") == "vad_end" for r in results) == 1


@pytest.mark.asyncio
async def test_realtime_asr_empty_final_is_terminal_but_anonymous_turns_can_repeat():
    backend = _asr_with_events([
        RealtimeEvent(kind="speech_started", item_id="a"),
        RealtimeEvent(kind="transcript_completed", item_id="a", text=""),
        RealtimeEvent(kind="transcript_delta", item_id="a", text="stale"),
        RealtimeEvent(kind="speech_stopped", item_id="a"),
        RealtimeEvent(kind="speech_started"),
        RealtimeEvent(kind="speech_stopped"),
        RealtimeEvent(kind="transcript_completed", text="Hello"),
        RealtimeEvent(kind="speech_started"),
        RealtimeEvent(kind="speech_stopped"),
        RealtimeEvent(kind="transcript_completed", text="Again"),
    ])
    results = [result async for result in backend.receive_results()]

    assert [(r.text, r.start_time, r.end_time) for r in results if r.is_final] == [
        ("Hello", 4.0, 5.0), ("Again", 7.0, 8.0),
    ]
    assert not any(r.text == "stale" for r in results)


@pytest.mark.asyncio
async def test_realtime_asr_item_history_is_bounded_and_reset_on_reconnect(monkeypatch):
    from multi_modal_ai_studio.backends.asr.openai_realtime import _MAX_TRACKED_ITEMS

    backend = _asr_with_events([
        RealtimeEvent(kind="transcript_completed", item_id=str(index), text="Hello")
        for index in range(_MAX_TRACKED_ITEMS + 1)
    ])
    results = [result async for result in backend.receive_results()]
    assert len(results) == _MAX_TRACKED_ITEMS + 1
    assert len(backend._items) == _MAX_TRACKED_ITEMS

    class ReconnectedClient(_FakeRealtimeClient):
        async def connect(self):
            pass

    client = ReconnectedClient([
        RealtimeEvent(kind="transcript_completed", item_id="1", text="New connection"),
    ])
    backend._client = None
    backend._active_item_id = "stale"
    monkeypatch.setattr(backend, "_make_client", lambda: client)
    monkeypatch.setattr(backend, "_timestamp", lambda: 1.0)
    await backend.start_stream()
    assert not backend._items
    assert backend._active_item_id is None
    results = [result async for result in backend.receive_results()]
    assert len(results) == 1 and results[0].text == "New connection"


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
