# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Riva/protobuf dependencies without a GPU or Riva server."""

import tomllib
from pathlib import Path

import pytest
from google.protobuf import any_pb2, json_format, struct_pb2
from packaging.requirements import Requirement
from riva.client.proto import riva_asr_pb2, riva_tts_pb2


def test_distribution_requires_updated_riva_and_protobuf():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    requirements = {r.name: r for r in map(Requirement, project["dependencies"])}
    assert "2.27.0" not in requirements["nvidia-riva-client"].specifier
    assert "2.28.0" in requirements["nvidia-riva-client"].specifier
    assert "6.33.5" not in requirements["protobuf"].specifier
    assert "7.34.1" not in requirements["protobuf"].specifier
    assert "7.36.2" in requirements["protobuf"].specifier


@pytest.mark.parametrize("kind", ["struct", "value", "list", "any"])
def test_protobuf_json_recursion_limit_is_enforced(kind):
    # Small, bounded input exercises the paths from protobuf issues #26432 and
    # #26736 plus the original Any path. This must raise ParseError, not silently
    # accept the input or run into Python's own RecursionError.
    nested = {"leaf": "speech"}
    for _ in range(20):
        nested = {"child": nested}
    if kind == "struct":
        message, payload = struct_pb2.Struct(), nested
    elif kind == "value":
        message, payload = struct_pb2.Value(), nested
    elif kind == "list":
        payload = ["speech"]
        for _ in range(20):
            payload = [payload]
        message = struct_pb2.ListValue()
    else:
        message = any_pb2.Any()
        payload = {"@type": "type.googleapis.com/google.protobuf.Struct", "value": nested}
        for _ in range(20):
            payload = {"@type": "type.googleapis.com/google.protobuf.Any", "value": payload}
    with pytest.raises(json_format.ParseError, match="[Dd]eep|[Nn]est|recursion"):
        json_format.ParseDict(payload, message, max_recursion_depth=5)


def test_riva_asr_generated_messages_roundtrip():
    request = riva_asr_pb2.StreamingRecognizeRequest(audio_content=b"\x00\x01\x02\x03")
    restored = type(request).FromString(request.SerializeToString())
    assert restored.audio_content == request.audio_content
    response = riva_asr_pb2.StreamingRecognizeResponse()
    result = response.results.add(is_final=True)
    result.alternatives.add(transcript="Voice AI Workbench", confidence=0.9)
    restored = type(response).FromString(response.SerializeToString())
    assert restored.results[0].is_final
    assert restored.results[0].alternatives[0].transcript == "Voice AI Workbench"


def test_riva_tts_generated_messages_roundtrip():
    request = riva_tts_pb2.SynthesizeSpeechRequest(
        text="Hello from Voice AI Workbench.", language_code="en-US", sample_rate_hz=22050
    )
    assert type(request).FromString(request.SerializeToString()) == request
    response = riva_tts_pb2.SynthesizeSpeechResponse(audio=b"\x00\x01\x02\x03")
    assert type(response).FromString(response.SerializeToString()).audio == response.audio
