# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the speech service with Uvicorn."""

import os

import uvicorn


def main() -> None:
    """Start the configured ASR or TTS service."""
    uvicorn.run(
        "openai_speech.app:app",
        host=os.getenv("SPEECH_HOST", "0.0.0.0"),
        port=int(os.getenv("SPEECH_PORT", "8081")),
        workers=1,
    )


if __name__ == "__main__":
    main()
