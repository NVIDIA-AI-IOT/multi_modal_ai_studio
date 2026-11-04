# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, opt-in model downloads for Speaches (not an OpenAI API)."""

import re
from urllib.parse import quote, urljoin, urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout


class ModelDownloadError(Exception):
    pass


def supports_speaches_download(schema: object) -> bool:
    """Require provider identity and its advertised model-management route."""
    if not isinstance(schema, dict):
        return False
    info = schema.get("info")
    paths = schema.get("paths")
    if not isinstance(info, dict) or str(info.get("title", "")).lower() != "speaches":
        return False
    if not isinstance(paths, dict):
        return False
    route = paths.get("/v1/models/{model_id}", {})
    return isinstance(route, dict) and isinstance(route.get("post"), dict)


def _endpoint(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/v1")
    ):
        raise ValueError("A valid HTTP(S) /v1 API endpoint is required")
    return base


async def discover_speaches_download(api_base: str, api_key: str = "") -> bool:
    """Read capability metadata; never download merely by selecting a model."""
    try:
        base = _endpoint(api_base)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with ClientSession(timeout=ClientTimeout(total=5)) as client:
            async with client.get(
                urljoin(base + "/", "../openapi.json"),
                headers=headers,
            ) as response:
                if response.status != 200:
                    return False
                schema = await response.json()
        return supports_speaches_download(schema)
    except (ClientError, TimeoutError, ValueError):
        return False


async def download_speaches_model(api_base: str, api_key: str, model: str) -> None:
    """Download only after a user action and a successful capability probe."""
    base = _endpoint(api_base)
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", model):
        raise ValueError("Select a valid provider model ID (owner/model)")
    if not await discover_speaches_download(base, api_key):
        raise ModelDownloadError("This provider does not advertise Speaches model downloads")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with ClientSession(timeout=ClientTimeout(total=600)) as client:
        async with client.post(
            f"{base}/models/{quote(model, safe='')}",
            headers=headers,
            allow_redirects=False,
        ) as response:
            if response.status not in {200, 201}:
                detail = (await response.text())[:300]
                raise ModelDownloadError(f"Model download failed ({response.status}): {detail}")
