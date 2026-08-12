# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for server-side microphone and speaker lifecycle."""

import io
import queue
import threading

from multi_modal_ai_studio.devices import capture, playback


def test_capture_queue_discards_oldest_audio_when_consumer_falls_behind():
    health = capture.CaptureHealth(device="hw:2,0")
    chunks = capture.create_capture_queue(max_chunks=2)

    assert capture.put_capture_item(chunks, b"oldest", health) is False
    assert capture.put_capture_item(chunks, b"middle", health) is False
    assert capture.put_capture_item(chunks, b"newest", health) is True

    assert chunks.qsize() == 2
    assert chunks.get_nowait() == b"middle"
    assert chunks.get_nowait() == b"newest"
    assert health.total_queue_overflows == 1
    assert health.to_dict()["total_queue_overflows"] == 1


def test_capture_control_event_is_not_blocked_by_full_audio_queue():
    chunks = capture.create_capture_queue(max_chunks=1)
    capture.put_capture_item(chunks, b"stale")
    event = capture._make_capture_event("gave_up", device="hw:99,0")

    assert capture.put_capture_item(chunks, event) is True
    queued = chunks.get_nowait()
    assert capture.is_capture_event(queued)
    assert queued["event"] == "gave_up"


def test_unknown_capture_source_does_not_start_thread():
    chunks = queue.Queue()
    stop_event = threading.Event()

    assert (
        capture.start_server_mic_capture(
            "unknown",
            "device",
            chunks,
            stop_event,
        )
        is None
    )


def test_stop_server_mic_capture_terminates_process_and_joins_thread():
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            self.waited = True

    class FakeThread:
        def __init__(self):
            self.joined = False

        def is_alive(self):
            return not self.joined

        def join(self, timeout):
            self.joined = True

    stop_event = threading.Event()
    process = FakeProcess()
    thread = FakeThread()
    holder = [process]

    assert capture.stop_server_mic_capture(stop_event, thread, holder) is True
    assert stop_event.is_set()
    assert process.terminated
    assert process.waited
    assert thread.joined
    assert holder == []


def test_alsa_playback_uses_plughw_and_requested_pcm_format(monkeypatch):
    calls = []

    class FakeProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO()

        def poll(self):
            return None

    fake_process = FakeProcess()

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return fake_process

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(playback.time, "sleep", lambda _seconds: None)

    result = playback.start_server_speaker_playback("hw:2,0", 24000)

    assert result is fake_process
    command, kwargs = calls[0]
    assert command == [
        "aplay",
        "-D",
        "plughw:2,0",
        "-f",
        "S16_LE",
        "-r",
        "24000",
        "-c",
        "1",
        "-t",
        "raw",
    ]
    assert kwargs["stdin"] is playback.subprocess.PIPE
    assert kwargs["stderr"] is playback.subprocess.PIPE


def test_alsa_playback_retries_immediate_process_failure(monkeypatch):
    attempts = []

    class FailedProcess:
        stdin = io.BytesIO()
        stderr = io.BytesIO(b"device busy")

        def poll(self):
            return 1

    def fake_popen(command, **kwargs):
        attempts.append(command)
        return FailedProcess()

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(playback.time, "sleep", lambda _seconds: None)

    assert playback.start_server_speaker_playback("default", 16000) is None
    assert len(attempts) == playback.PLAYBACK_RETRIES
