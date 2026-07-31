# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session preview thumbnail persistence and validation tests."""

import base64
from pathlib import Path

from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core.session import (
    MAX_SESSION_THUMBNAIL_BYTES,
    SESSION_THUMBNAIL_PREFIX,
    Session,
    decode_session_thumbnail,
    validate_session_thumbnail,
)


def _jpeg_data_url(payload: bytes = b"preview") -> str:
    jpeg = b"\xff\xd8" + payload + b"\xff\xd9"
    return SESSION_THUMBNAIL_PREFIX + base64.b64encode(jpeg).decode("ascii")


def test_session_thumbnail_round_trip(tmp_path):
    session = Session(SessionConfig(), session_id="thumbnail-session")
    session.thumbnail = _jpeg_data_url()
    session.tts_playback_segments = [
        {"startTime": 1.0, "endTime": 1.025, "amplitude": 12.0}
    ]
    path = tmp_path / "thumbnail-session.json"

    session.save(path)
    loaded = Session.load(path)

    assert loaded.thumbnail == session.thumbnail
    assert loaded.to_dict()["thumbnail"] == session.thumbnail
    assert loaded.tts_playback_segments == session.tts_playback_segments


def test_thumbnail_validation_rejects_non_jpeg_and_oversized_payload():
    png_disguised_as_jpeg = (
        SESSION_THUMBNAIL_PREFIX
        + base64.b64encode(b"\x89PNG\r\n").decode("ascii")
    )
    oversized = _jpeg_data_url(b"x" * MAX_SESSION_THUMBNAIL_BYTES)

    assert validate_session_thumbnail(png_disguised_as_jpeg) is None
    assert decode_session_thumbnail(oversized) is None


def test_recorded_session_uses_compact_thumbnail_layout():
    repository_root = Path(__file__).resolve().parents[2]
    app = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/app.js"
    ).read_text()
    styles = (
        repository_root
        / "src/multi_modal_ai_studio/webui/static/styles.css"
    ).read_text()

    assert "updateRecordedReviewLayout" in app
    assert "chat-panel--recorded-review-has-thumbnail" in app
    assert ".chat-panel.chat-panel--recorded-review" in styles
    assert "width: 160px" in styles
    assert "height: 90px" in styles
