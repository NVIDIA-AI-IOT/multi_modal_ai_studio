# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
import pytest

from multi_modal_ai_studio.webui import server
from multi_modal_ai_studio.webui.speech_models import (
    ModelDownloadError,
    discover_speaches_download,
    download_speaches_model,
    supports_speaches_download,
)

SCHEMA = {"info": {"title": "Speaches"}, "paths": {"/v1/models/{model_id}": {"post": {}}}}


@asynccontextmanager
async def provider(schema=SCHEMA, status=200):
    calls = []
    app = web.Application()

    async def openapi(request):
        calls.append(("GET", request.path, request.headers.get("Authorization")))
        return web.json_response(schema)

    async def download(request):
        calls.append(("POST", request.match_info["model"], request.headers.get("Authorization")))
        return web.Response(status=status, text="downloaded" if status == 200 else "failure")

    app.router.add_get("/openapi.json", openapi)
    app.router.add_post("/v1/models/{model:.*}", download)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", calls
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {},
        {"info": {"title": "vLLM"}, "paths": SCHEMA["paths"]},
        {"info": {"title": "Speaches"}, "paths": {}},
        {"info": {"title": "Speaches"}, "paths": {"/v1/models/{model_id}": {"get": {}}}},
    ],
)
def test_download_not_assumed_from_openai_compatibility(schema):
    assert not supports_speaches_download(schema)


@pytest.mark.asyncio
async def test_discovery_never_downloads_and_explicit_download_uses_auth():
    async with provider() as (base, calls):
        assert await discover_speaches_download(base, "secret")
        assert [call[0] for call in calls] == ["GET"]
        await download_speaches_model(base, "secret", "Systran/faster-whisper-small")
        assert calls[-1] == ("POST", "Systran/faster-whisper-small", "Bearer secret")


@pytest.mark.asyncio
async def test_generic_provider_receives_no_download_post():
    async with provider({"info": {"title": "Other provider"}, "paths": SCHEMA["paths"]}) as (
        base,
        calls,
    ):
        with pytest.raises(ModelDownloadError, match="does not advertise"):
            await download_speaches_model(base, "", "owner/model")
        assert all(call[0] == "GET" for call in calls)


@pytest.mark.asyncio
async def test_upstream_download_failure_is_not_reported_as_success():
    async with provider(status=403) as (base, _calls):
        with pytest.raises(ModelDownloadError, match="403"):
            await download_speaches_model(base, "", "owner/model")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["../secret", "owner/../secret", "owner/model?x=1", ""])
async def test_download_rejects_invalid_model_before_network(model):
    with pytest.raises(ValueError):
        await download_speaches_model("http://localhost/v1", "", model)


@pytest.mark.asyncio
async def test_http_download_handler_validates_input_and_releases_pending(monkeypatch):
    instance = server.WebUIServer.__new__(server.WebUIServer)

    async def run(body):
        async def payload():
            return body

        return await instance.handle_download_asr_model(SimpleNamespace(json=payload))

    assert (await run([])).status == 400

    async def fail(*args):
        raise ModelDownloadError("upstream unavailable")

    monkeypatch.setattr(server, "download_speaches_model", fail)
    body = {"api_base": "http://localhost/v1", "model": "owner/model"}
    assert (await run(body)).status == 502
    assert not instance._pending_asr_downloads

    async def success(*args):
        pass

    monkeypatch.setattr(server, "download_speaches_model", success)
    assert json.loads((await run(body)).text)["downloaded"] is True
    instance._pending_asr_downloads.add((body["api_base"], body["model"]))
    assert (await run(body)).status == 409


def test_asr_ui_has_explicit_reload_download_and_no_auto_fetch_promise():
    root = Path(__file__).resolve().parents[2]
    app = (root / "src/multi_modal_ai_studio/webui/static/app.js").read_text()
    assert 'id="asr-model-reload"' in app
    assert 'id="asr-model-download"' in app
    assert "async function downloadASRModel()" in app
    assert "window.confirm('Download '" in app
    assert "button.hidden = !(data && data.download_supported" in app
    assert "Undownloaded models are fetched on first use" not in app
    assert "Download before use" in app
    assert "asrModelDiscoverySequence" in app
