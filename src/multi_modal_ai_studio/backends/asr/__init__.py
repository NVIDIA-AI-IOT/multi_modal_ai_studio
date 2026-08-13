# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASR (Automatic Speech Recognition) backend implementations."""

from multi_modal_ai_studio.backends.asr.openai_rest import OpenAIRestASRBackend
from multi_modal_ai_studio.backends.asr.openai_realtime import OpenAIRealtimeASRBackend
from multi_modal_ai_studio.backends.asr.riva import RivaASRBackend

__all__ = [
    "OpenAIRealtimeASRBackend",
    "OpenAIRestASRBackend",
    "RivaASRBackend",
]
