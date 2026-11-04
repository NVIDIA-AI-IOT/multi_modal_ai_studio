# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Language codes for file transcription and Realtime transcription APIs."""

from typing import Optional


def transcription_language(value: Optional[str]) -> Optional[str]:
    """Map UI locale tags to language codes; omit automatic detection.

    Riva still receives the original locale (for example ``ja-JP``). Only
    OpenAI-compatible transcription adapters should use this conversion.
    """
    value = (value or "").strip().lower().replace("_", "-")
    if not value or value == "auto":
        return None
    return value.split("-", 1)[0]
