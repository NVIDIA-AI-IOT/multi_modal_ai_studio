# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.config.schema import LLMConfig
from multi_modal_ai_studio.webui.voice_pipeline import _empty_llm_response_message


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", [True, False])
async def test_reasoning_delta_does_not_drop_answer_content(monkeypatch, combined):
    deltas = (
        [{"reasoning_content": "private reasoning", "content": "two"}]
        if combined
        else [
            {"reasoning_content": "private reasoning"},
            {"content": "two"},
        ]
    )
    lines = [("data: " + json.dumps({"choices": [{"delta": delta}]})).encode() for delta in deltas]
    lines += [b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}', b"data: [DONE]"]

    class Content:
        def __aiter__(self):
            self.lines = iter(lines)
            return self

        async def __anext__(self):
            try:
                return next(self.lines)
            except StopIteration:
                raise StopAsyncIteration

    class Response:
        status = 200
        content = Content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("multi_modal_ai_studio.backends.llm.openai.aiohttp.ClientSession", Client)
    llm = OpenAILLMBackend(LLMConfig(api_base="http://localhost/v1", model="test"))
    tokens = [token async for token in llm.generate_stream("What is one plus one?")]
    assert "".join(token.token for token in tokens) == "two"
    assert tokens[-1].metadata["reasoning"] == "private reasoning"
    assert tokens[-1].metadata["full_response"] == "two"
    assert tokens[-1].metadata["finish_reason"] == "stop"


def test_empty_answer_diagnostic_never_contains_reasoning_text():
    message = _empty_llm_response_message("sensitive private analysis")
    assert "reasoning but no answer" in message
    assert "sensitive private analysis" not in message
    assert "no speech was generated" in _empty_llm_response_message("")
