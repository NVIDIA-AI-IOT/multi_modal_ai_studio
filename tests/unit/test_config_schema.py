# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config schema (no live backends)."""

import pytest

from multi_modal_ai_studio.config.schema import (
    ASRConfig,
    LLMConfig,
    TTSConfig,
    AppConfig,
    DeviceConfig,
    SessionConfig,
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
    assert cfg.barge_in_trigger == "final"
    assert cfg.barge_in_partial_count == 3
    assert hasattr(cfg, "timeline_position")
    assert hasattr(cfg, "session_output_dir")


def test_app_config_validates_partial_barge_in_count():
    assert AppConfig(barge_in_partial_count=1).validate() == []
    assert AppConfig(barge_in_partial_count=20).validate() == []
    assert AppConfig(barge_in_partial_count=0).validate()
    assert AppConfig(barge_in_partial_count=21).validate()


@pytest.mark.parametrize("source", ["alsa", "usb"])
def test_server_audio_device_config_round_trip(source):
    config = SessionConfig(
        devices=DeviceConfig(
            audio_input_source=source,
            audio_input_device="hw:2,0" if source == "alsa" else "2",
            audio_input_device_name="USB microphone",
            audio_output_source=source,
            audio_output_device="hw:3,0" if source == "alsa" else "3",
            audio_output_device_name="USB speaker",
        ),
        app=AppConfig(
            barge_in_enabled=True,
            barge_in_trigger="partial",
            barge_in_partial_count=4,
        ),
    )

    restored = SessionConfig.from_dict(config.to_dict())

    assert restored.devices == config.devices
    assert restored.app == config.app


def test_microphone_selector_overrides_stale_browser_input_fields():
    config = SessionConfig.from_dict(
        {
            "devices": {
                "microphone": "alsa:hw:2,0",
                "microphone_name": "PowerConf (Server USB)",
                "audio_input_source": "browser",
                "audio_input_device": None,
            }
        }
    )

    assert config.devices.audio_input_source == "alsa"
    assert config.devices.audio_input_device == "hw:2,0"
    assert config.devices.audio_input_device_name == "PowerConf (Server USB)"
