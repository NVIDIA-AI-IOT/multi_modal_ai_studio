# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded storage of the exact WAV sent to REST transcription.

Raw microphone audio and transcripts are sensitive. Nothing is saved unless
MMAS_ASR_CAPTURE_DIR is explicitly set; credentials are never included.
"""

import hashlib
import json
import logging
import os
import stat
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_MAX_FILES = 32
_MAX_BYTES = 64 * 1024 * 1024
_MAX_WAV_BYTES = 4 * 1024 * 1024


def capture_request(wav: bytes, metadata: dict) -> Path | None:
    """Save a private WAV/JSON pair, or fail open without affecting ASR.

    The cap applies across streams/process restarts in this directory. A
    process-local lock serializes writers; use a separate directory per MMAS
    process. This is a diagnostic aid, not a multi-process storage service.
    """
    directory = os.environ.get("MMAS_ASR_CAPTURE_DIR", "").strip()
    if not directory:
        return None
    try:
        with _LOCK:
            root = Path(directory).expanduser()
            root.mkdir(parents=True, mode=0o700, exist_ok=True)
            info = root.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("capture directory must be a private directory (mode 700)")
            recordings = list(root.glob("asr-*.wav"))
            if (
                len(recordings) >= _MAX_FILES
                or len(wav) > _MAX_WAV_BYTES
                or sum(p.stat().st_size for p in recordings) + len(wav) > _MAX_BYTES
            ):
                logger.warning("ASR diagnostic capture limit reached; ASR continues without saving")
                return None
            path = root / f"asr-{uuid.uuid4().hex}.wav"
            with path.open("xb") as target:
                os.chmod(path, 0o600)
                target.write(wav)
            record = {
                **metadata,
                "capture_id": path.stem,
                "created_at_epoch": time.time(),
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "wav_bytes": len(wav),
                "status": "pending",
            }
            _write_metadata(path.with_suffix(".json"), record, exclusive=True)
            logger.info("[ASR diagnostic] Captured %s", path)
            return path.with_suffix(".json")
    except Exception:
        logger.warning("ASR diagnostic capture failed; ASR continues", exc_info=True)
        return None


def finish_capture(path: Path | None, outcome: dict) -> None:
    """Attach timings/status after the HTTP request, without credentials."""
    if path is None:
        return
    try:
        record = json.loads(path.read_text())
        record.update(outcome)
        _write_metadata(path, record)
    except Exception:
        logger.warning("ASR diagnostic result save failed; ASR continues", exc_info=True)


def _write_metadata(path: Path, record: dict, *, exclusive: bool = False) -> None:
    # Parent is private before files are created. No endpoint, API key, request
    # headers, or complete configuration object is ever passed to this helper.
    with path.open("x" if exclusive else "w") as target:
        os.chmod(path, 0o600)
        json.dump(record, target, ensure_ascii=False, indent=2)
        target.write("\n")
