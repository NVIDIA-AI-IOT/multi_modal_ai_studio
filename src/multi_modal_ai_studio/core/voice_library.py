# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small host-local reference voice library (not model training or a voice service).

Only opaque IDs cross the configuration boundary. Never resolve client paths or
URLs. WAV decoding is deliberately restricted to uncompressed PCM16 mono.
"""
import base64
import io
import json
import os
import re
import tempfile
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

MAX_AUDIO_BYTES = 3 * 1024 * 1024
MAX_VOICES = 100
_ID = re.compile(r"[0-9a-f]{32}\Z")


def validate_wav(audio: bytes) -> tuple[bytes, float, int]:
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("Reference WAV must be at most 3 MiB")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getcomptype() != "NONE":
                raise ValueError("Use an uncompressed 16-bit mono WAV")
            if not 8000 <= rate <= 48000 or not 1 <= frames / rate <= 30:
                raise ValueError("Reference WAV must be 1–30 seconds at 8–48 kHz")
            pcm = wav.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("Reference WAV is truncated")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV; use uncompressed 16-bit mono WAV") from exc
    # Remove ancillary chunks (which can contain unrelated private metadata).
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return output.getvalue(), frames / rate, rate


class VoiceLibrary:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("VAW_VOICE_LIBRARY_DIR") or
                         Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) /
                         "voice-ai-workbench/voices")

    def _path(self, voice_id):
        if not isinstance(voice_id, str) or not _ID.fullmatch(voice_id):
            raise ValueError("Invalid reference voice ID")
        return self.root / (voice_id + ".json")

    def load(self, voice_id):
        path = self._path(voice_id)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as file:
                raw = file.read(MAX_AUDIO_BYTES * 2 + 1)
            if len(raw) > MAX_AUDIO_BYTES * 2:
                raise ValueError("Reference voice file is too large")
            entry = json.loads(raw)
            if (not isinstance(entry, dict) or entry.get("id") != voice_id
                    or entry.get("consent") is not True
                    or not isinstance(entry.get("name"), str)
                    or not isinstance(entry.get("transcript"), str)):
                raise ValueError("Invalid reference voice metadata")
            audio = base64.b64decode(entry["audio_base64"], validate=True)
            audio, duration, rate = validate_wav(audio)
            entry.update(duration=duration, sample_rate=rate)
            return entry, audio
        except FileNotFoundError as exc:
            raise ValueError("Reference voice is unavailable on this server; upload or select it again") from exc
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Reference voice could not be read safely") from exc

    @staticmethod
    def public(entry):
        return {key: entry.get(key) for key in (
            "id", "name", "transcript", "source", "duration", "sample_rate", "created_at"
        )}

    def list(self):
        result = []
        for path in sorted(self.root.glob("*.json")):
            try:
                entry, _ = self.load(path.stem)
                result.append(self.public(entry))
            except ValueError:
                continue
        return result

    def save(self, audio, *, name, transcript="", consent=False, source="upload"):
        if consent is not True:
            raise ValueError("Confirm you have permission to use this voice for synthesis")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("Voice name must contain 1–80 characters")
        if not isinstance(transcript, str) or len(transcript) > 2000:
            raise ValueError("Reference transcript must be at most 2000 characters")
        if source not in ("upload", "voicedesign-preview"):
            raise ValueError("Invalid reference source")
        audio, duration, rate = validate_wav(audio)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if len(list(self.root.glob("*.json"))) >= MAX_VOICES:
            raise ValueError("Voice library is full (100 voices); delete an unused voice first")
        voice_id = uuid.uuid4().hex
        entry = dict(id=voice_id, name=name.strip(), transcript=transcript.strip(),
                     consent=True, source=source, duration=duration, sample_rate=rate,
                     created_at=datetime.now(timezone.utc).isoformat(),
                     audio_base64=base64.b64encode(audio).decode("ascii"))
        # One private, atomic file: no half-written audio/metadata pairs.
        fd, temp = tempfile.mkstemp(prefix=".upload-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(entry, file, ensure_ascii=False)
            os.replace(temp, self._path(voice_id))
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return self.public(entry)

    def delete(self, voice_id):
        # Validate before unlink; refusing symlinks also avoids confusing ownership.
        self.load(voice_id)
        self._path(voice_id).unlink()

    def speech_fields(self, voice_id, mode):
        entry, audio = self.load(voice_id)
        if mode not in ("icl", "x-vector"):
            raise ValueError("Unsupported reference voice mode")
        if mode == "icl" and not entry.get("transcript"):
            raise ValueError("Transcript-guided cloning requires the exact reference transcript")
        fields = {"ref_audio": "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii"),
                  "x_vector_only_mode": mode == "x-vector"}
        if mode == "icl":
            fields["ref_text"] = entry["transcript"]
        return fields
