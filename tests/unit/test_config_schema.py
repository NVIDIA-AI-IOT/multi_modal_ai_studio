# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config schema (no live backends)."""

import pytest

from multi_modal_ai_studio.config.schema import (
    ASRConfig,
    LLMConfig,
    TTSConfig,
    AppConfig,
)


def test_asr_config_defaults():
    """ASRConfig with minimal required fields."""
    cfg = ASRConfig(
        scheme="riva",
        server="localhost:50051",
        model="conformer",
        language="en-US",
    )
    assert cfg.scheme == "riva"
    assert cfg.server == "localhost:50051"
    assert cfg.model == "conformer"
    assert cfg.language == "en-US"


def test_llm_config_defaults():
    """LLMConfig with minimal required fields."""
    cfg = LLMConfig(
        scheme="openai",
        api_base="http://localhost:11434/v1",
        model="llama3.2:3b",
    )
    assert cfg.scheme == "openai"
    assert cfg.api_base == "http://localhost:11434/v1"
    assert cfg.model == "llama3.2:3b"


def test_tts_config_defaults():
    """TTSConfig with minimal required fields."""
    cfg = TTSConfig(
        scheme="riva",
        server="localhost:50051",
        voice="English-US.Female-1",
        sample_rate=24000,
    )
    assert cfg.scheme == "riva"
    assert cfg.server == "localhost:50051"
    assert cfg.voice == "English-US.Female-1"
    assert cfg.sample_rate == 24000


def test_app_config_minimal():
    """AppConfig can be constructed with defaults."""
    cfg = AppConfig()
    assert hasattr(cfg, "barge_in_enabled")
    assert hasattr(cfg, "timeline_position")
    assert hasattr(cfg, "session_output_dir")
