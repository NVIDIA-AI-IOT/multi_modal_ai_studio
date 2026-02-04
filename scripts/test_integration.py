#!/usr/bin/env python3
"""
Integration test script for backends with real services.

Tests actual connections to Riva and Ollama.
Requires:
- Riva server running (ASR + TTS)
- Ollama server running (LLM)
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_modal_ai_studio.config.schema import ASRConfig, LLMConfig, TTSConfig
from multi_modal_ai_studio.backends.asr.riva import RivaASRBackend
from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.backends.tts.riva import RivaTTSBackend


async def test_llm():
    """Test LLM backend with Ollama"""
    print("🔍 Testing LLM Backend (Ollama)...")
    
    config = LLMConfig(
        scheme="openai",
        api_base="http://localhost:11434/v1",
        model="llama3.2:3b",
        temperature=0.7,
        max_tokens=100,
    )
    
    try:
        backend = OpenAILLMBackend(config)
        
        # Try to list models
        print("  → Detecting available models...")
        models = await backend.list_available_models()
        if models:
            print(f"  ✓ Found {len(models)} model(s): {', '.join(models[:3])}")
        else:
            print("  ⚠ No models detected, but will try anyway")
        
        # Test generation
        print("  → Generating response...")
        prompt = "Say 'Hello from Multi-modal AI Studio!' in one sentence."
        
        tokens = []
        token_count = 0
        async for token in backend.generate_stream(prompt):
            if not token.is_final:
                tokens.append(token.token)
                token_count += 1
                # Print first few tokens
                if token_count <= 5:
                    print(f"    Token {token_count}: '{token.token}'")
        
        response = "".join(tokens)
        print(f"\n  ✓ LLM Response ({token_count} tokens):")
        print(f"    \"{response}\"\n")
        
        return True
    
    except Exception as e:
        print(f"  ❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tts():
    """Test TTS backend with Riva"""
    print("🔍 Testing TTS Backend (Riva)...")
    
    config = TTSConfig(
        scheme="riva",
        server="localhost:50051",
        voice="",  # Empty = use default voice
        sample_rate=22050,  # Riva's native rate
    )
    
    try:
        backend = RivaTTSBackend(config)
        
        # List voices
        print("  → Listing available voices...")
        voices = await backend.list_voices()
        if voices:
            print(f"  ✓ Found {len(voices)} voice(s):")
            for voice in voices[:5]:
                print(f"    - {voice}")
        else:
            print("  ⚠ No voices detected")
        
        # Test synthesis
        print("  → Synthesizing speech...")
        text = "Hello from Multi-modal AI Studio!"
        
        chunks = []
        chunk_count = 0
        total_audio_bytes = 0
        
        async for chunk in backend.synthesize_stream(text):
            chunks.append(chunk)
            chunk_count += 1
            total_audio_bytes += len(chunk.audio)
            
            if chunk_count == 1:
                print(f"    First chunk: {len(chunk.audio)} bytes, {chunk.duration_ms:.0f}ms")
            
            if chunk.is_final:
                print(f"    Final chunk received")
        
        print(f"\n  ✓ TTS completed:")
        print(f"    Total chunks: {chunk_count}")
        print(f"    Total audio: {total_audio_bytes:,} bytes ({total_audio_bytes/1024:.1f} KB)")
        duration_sec = total_audio_bytes / (config.sample_rate * 2)  # 16-bit = 2 bytes
        print(f"    Duration: ~{duration_sec:.2f} seconds\n")
        
        return True
    
    except Exception as e:
        print(f"  ❌ TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_asr():
    """Test ASR backend with Riva"""
    print("🔍 Testing ASR Backend (Riva)...")
    
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
        print("  ✓ ASR backend connected to Riva")
        print("  ℹ Note: Full ASR test requires audio input (skipping audio stream test)")
        print("  ℹ ASR will be fully tested in end-to-end pipeline\n")
        return True
    
    except Exception as e:
        print(f"  ❌ ASR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests"""
    print("=" * 70)
    print("Multi-modal AI Studio - Integration Tests")
    print("=" * 70)
    print()
    print("Testing with real services:")
    print("  - Ollama: http://localhost:11434")
    print("  - Riva: localhost:50051")
    print()
    
    results = {}
    
    # Test LLM (most interactive)
    results["LLM"] = await test_llm()
    
    # Test TTS
    results["TTS"] = await test_tts()
    
    # Test ASR (basic connection)
    results["ASR"] = await test_asr()
    
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for backend, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{backend}: {status}")
    
    print()
    
    if all(results.values()):
        print("✅ All integration tests passed!")
        print()
        print("Next steps:")
        print("  1. Implement session management (core/session.py)")
        print("  2. Create end-to-end pipeline test")
        print("  3. Build WebUI (Phase 2)")
        return 0
    else:
        print("❌ Some tests failed.")
        print()
        print("Troubleshooting:")
        print("  - Ensure Riva is running: docker ps | grep riva")
        print("  - Ensure Ollama is running: curl http://localhost:11434/api/tags")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
