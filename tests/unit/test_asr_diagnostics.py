# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from multi_modal_ai_studio.backends.asr import diagnostics
from multi_modal_ai_studio.backends.asr.openai_rest import OpenAIRestASRBackend
from multi_modal_ai_studio.config.schema import ASRConfig


def test_capture_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MMAS_ASR_CAPTURE_DIR", raising=False)
    assert diagnostics.capture_request(b"audio", {}) is None
    assert not list(tmp_path.iterdir())


def test_capture_private_bounded_and_unique(tmp_path, monkeypatch):
    root = tmp_path / "private"
    monkeypatch.setenv("MMAS_ASR_CAPTURE_DIR", str(root))
    monkeypatch.setattr(diagnostics, "_MAX_FILES", 2)
    first = diagnostics.capture_request(b"audio", {"language": "ja"})
    second = diagnostics.capture_request(b"audio", {})
    assert first != second
    assert diagnostics.capture_request(b"audio", {}) is None
    assert first.with_suffix(".wav").read_bytes() == b"audio"
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.iterdir())
    diagnostics.finish_capture(first, {"status": "success", "transcript": "日本語"})
    record = json.loads(first.read_text())
    assert record["wav_sha256"] == hashlib.sha256(b"audio").hexdigest()
    assert record["transcript"] == "日本語"
    assert record["status"] == "success"


def test_capture_refuses_public_or_symlink_directory(tmp_path, monkeypatch):
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    monkeypatch.setenv("MMAS_ASR_CAPTURE_DIR", str(root))
    assert diagnostics.capture_request(b"audio", {}) is None
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("MMAS_ASR_CAPTURE_DIR", str(link))
    assert diagnostics.capture_request(b"audio", {}) is None
    assert not list(root.iterdir())


@pytest.mark.parametrize("limit", ["_MAX_BYTES", "_MAX_WAV_BYTES"])
def test_byte_limits(tmp_path, monkeypatch, limit):
    monkeypatch.setenv("MMAS_ASR_CAPTURE_DIR", str(tmp_path / "private"))
    monkeypatch.setattr(diagnostics, limit, 2)
    assert diagnostics.capture_request(b"audio", {}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 500])
async def test_saved_wav_matches_sent_bytes_and_excludes_secrets(tmp_path, monkeypatch, status):
    root = tmp_path / "private"
    monkeypatch.setenv("MMAS_ASR_CAPTURE_DIR", str(root))
    sent = {}

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def text(self):
            return '{"text":"こんにちは"}'

        async def json(self):
            return {"text": "こんにちは"}

    response = Response()
    response.status = status

    def post(_url, *, data, headers):
        sent.update({meta["name"]: value for meta, _, value in data._fields})
        return response

    backend = OpenAIRestASRBackend(
        ASRConfig(
            scheme="openai-rest",
            api_base="http://localhost/v1",
            api_key="secret-key",
            model="Systran/faster-whisper-small",
            language="ja-JP",
        )
    )
    backend._session = SimpleNamespace(post=post)
    backend._results = asyncio.Queue()
    await backend._transcribe(b"\x00\x00" * 160)
    wav = next(root.glob("*.wav"))
    metadata = wav.with_suffix(".json").read_text()
    record = json.loads(metadata)
    assert wav.read_bytes() == sent["file"]
    assert record["language"] == "ja"
    assert record["http_status"] == status
    assert record["status"] == ("success" if status == 200 else "error")
    assert record["request_duration_ms"] >= 0
    assert "secret-key" not in metadata and "localhost" not in metadata
    assert backend._results.get_nowait().is_final
