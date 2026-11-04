# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

from aiohttp import ClientSession, web
import pytest

from multi_modal_ai_studio.webui.server import WebUIServer


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["test-session.json", "older example.json"])
async def test_download_returns_original_file_bytes_and_attachment(tmp_path, filename):
    payload = '{"session_id": "test-session", "name": "日本語", "timeline": []}\n'
    (tmp_path / filename).write_text(payload)
    instance = WebUIServer(session_dir=tmp_path)
    runner = web.AppRunner(instance.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with ClientSession() as client:
            async with client.get(
                f"http://127.0.0.1:{port}/api/sessions/test-session/download"
            ) as response:
                assert response.status == 200
                assert await response.read() == payload.encode()
                assert response.headers["Content-Disposition"].startswith("attachment;")
                assert response.content_type == "application/json"
                assert response.headers["Cache-Control"] == "no-store"
                assert json.loads(payload)["session_id"] == "test-session"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["missing", "../secret", "nested/file", "nested\\file"])
async def test_download_rejects_missing_and_unsafe_session_ids(tmp_path, session_id):
    instance = WebUIServer(session_dir=tmp_path)
    result = await instance.handle_download_session(
        SimpleNamespace(match_info={"session_id": session_id})
    )
    assert result.status == 404


def test_live_stopped_and_recorded_headers_share_download_control():
    root = Path(__file__).resolve().parents[2]
    app = (root / "src/multi_modal_ai_studio/webui/static/app.js").read_text()
    assert "renderSessionFilenameControls(stoppedSessionId, sessionFileName)" in app
    assert "renderSessionFilenameControls(session.session_id, sessionFileName)" in app
    assert 'aria-label="Download session JSON"' in app
    assert "encodeURIComponent(sessionId)" in app
