#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for backend implementations.

Tests that ASR, LLM, and TTS backends can be instantiated
and configured correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_modal_ai_studio.config.schema import ASRConfig, LLMConfig, TTSConfig
from multi_modal_ai_studio.backends.asr.riva import RivaASRBackend
from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.backends.tts.riva import RivaTTSBackend


def test_asr_backend():
    """Test ASR backend initialization"""
    print("🔍 Testing ASR Backend...")
    
    config = ASRConfig(
        scheme="riva",
        server="localhost:50051",
        model="conformer",
        language="en-US",
        vad_start_threshold=0.5,
        vad_stop_threshold=0.3,
    )
    
    try:
        backend = RivaASRBackend(config)
        print(f"  ✅ ASR backend initialized: {backend.riva_server}")
        return True
    except Exception as e:
        print(f"  ❌ ASR backend error: {e}")
        return False


def test_llm_backend():
    """Test LLM backend initialization"""
    print("\n🔍 Testing LLM Backend...")
    
    config = LLMConfig(
        scheme="openai",
        api_base="http://localhost:11434/v1",
        model="llama3.2:3b",
        temperature=0.7,
    )
    
    try:
        backend = OpenAILLMBackend(config)
        print(f"  ✅ LLM backend initialized: {backend.api_base}")
        return True
    except Exception as e:
        print(f"  ❌ LLM backend error: {e}")
        return False


def test_tts_backend():
    """Test TTS backend initialization"""
    print("\n🔍 Testing TTS Backend...")
    
    config = TTSConfig(
        scheme="riva",
        server="localhost:50051",
        voice="English-US.Female-1",
        sample_rate=24000,
    )
    
    try:
        backend = RivaTTSBackend(config)
        print(f"  ✅ TTS backend initialized: {backend.riva_server}")
        return True
    except Exception as e:
        print(f"  ❌ TTS backend error: {e}")
        return False


def main():
    """Run all backend tests"""
    print("=" * 60)
    print("Backend Initialization Tests")
    print("=" * 60)
    
    results = {
        "ASR": test_asr_backend(),
        "LLM": test_llm_backend(),
        "TTS": test_tts_backend(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for backend, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{backend}: {status}")
    
    if all(results.values()):
        print("\n✅ All backends initialized successfully!")
        print("\nNote: This only tests initialization, not actual connections.")
        print("To test with real services, ensure Riva and Ollama are running.")
        return 0
    else:
        print("\n❌ Some backends failed to initialize.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
