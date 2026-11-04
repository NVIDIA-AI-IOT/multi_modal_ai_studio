# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import importlib.util
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

_SPEC = importlib.util.spec_from_file_location(
    "replay_rest_asr", Path(__file__).resolve().parents[2] / "scripts/replay_rest_asr.py"
)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)


@pytest.mark.asyncio
@pytest.mark.parametrize("asr_status", [200, 500])
async def test_replay_preserves_input_measures_overlap_and_keeps_errors(tmp_path, asr_status):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    original = stream.getvalue()
    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(original)
    seen = []

    async def asr(request):
        form = await request.post()
        seen.append((form["file"].file.read(), form["language"]))
        await asyncio.sleep(0.02)
        return web.json_response({"text": "recognized"}, status=asr_status)

    async def tts(request):
        assert (await request.json())["response_format"] == "pcm"
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"\x00\x00")
        await asyncio.sleep(0.05)
        await response.write(b"\x00\x00")
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", asr)
    app.router.add_post("/v1/audio/speech", tts)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    args = SimpleNamespace(
        wav=wav_path,
        output=tmp_path / "results.json",
        api_base=f"http://127.0.0.1:{port}/v1",
        model="test",
        language="ja",
        tts_model="tts",
        voice="test",
        tts_text="text",
        repeats=1,
        modes=["asr-only", "tts-overlap", "after-tts"],
        overlap_delay=0,
        pause=0,
        timeout=3,
    )
    try:
        assert await replay.run(args) == (0 if asr_status == 200 else 1)
        assert seen == [(original, "ja")] * 3
        report = json.loads(args.output.read_text())
        assert report["input_sha256"] == hashlib.sha256(original).hexdigest()
        assert len(report["runs"]) == 3
        assert report["runs"][1]["measured_overlap_s"] > 0
        after = report["runs"][2]
        assert after["asr"]["start_s"] >= after["tts"]["end_s"]
        assert all(r["asr"]["http_status"] == asr_status for r in report["runs"])
        assert args.output.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileExistsError):
            await replay.run(args)
    finally:
        await runner.cleanup()
