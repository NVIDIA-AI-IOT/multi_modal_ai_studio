#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Demo script showing how to emit timeline events with rectangles for ASR/TTS visualization.

This demonstrates the pattern backends should follow to emit duration-based events
that render as rectangles in the timeline visualization.
"""

import asyncio
import json
from multi_modal_ai_studio.core.timeline import Timeline, Lane

async def demo_conversation_timeline():
    """Simulate a conversation with ASR and TTS rectangle events."""
    
    timeline = Timeline()
    timeline.start()
    
    print("🎬 Starting demo conversation timeline...")
    
    # Simulate Turn 1: User speaks
    await asyncio.sleep(0.5)
    
    # VAD detects speech start
    vad_start = timeline.add_event("vad_start", Lane.AUDIO)
    print(f"  🎤 VAD: Speech detected at {vad_start.timestamp:.3f}s")
    
    # User speaks for 2 seconds
    await asyncio.sleep(2.0)
    
    # VAD detects speech end
    vad_end = timeline.add_event("vad_end", Lane.AUDIO)
    print(f"  🔇 VAD: Speech ended at {vad_end.timestamp:.3f}s")
    
    # Add VAD segment as rectangle
    timeline.add_vad_segment(
        start_time=vad_start.timestamp,
        end_time=vad_end.timestamp,
        data={"confidence": 0.95}
    )
    print(f"  📊 VAD Rectangle: {vad_start.timestamp:.3f}s - {vad_end.timestamp:.3f}s")
    
    # ASR processes the audio
    asr_start = timeline.add_event("asr_start", Lane.SPEECH)
    await asyncio.sleep(0.3)  # ASR processing time
    
    asr_final = timeline.add_event(
        "asr_final",
        Lane.SPEECH,
        data={"text": "Hello, how are you?"}
    )
    print(f"  📝 ASR Final: 'Hello, how are you?' at {asr_final.timestamp:.3f}s")
    
    # Add ASR segment as rectangle
    timeline.add_asr_segment(
        start_time=vad_start.timestamp,
        end_time=vad_end.timestamp,
        text="Hello, how are you?",
        data={"confidence": 0.98}
    )
    print(f"  📊 ASR Rectangle: {vad_start.timestamp:.3f}s - {vad_end.timestamp:.3f}s")
    
    # LLM processes
    llm_start = timeline.add_event("llm_start", Lane.LLM)
    await asyncio.sleep(0.5)  # LLM thinking
    
    llm_complete = timeline.add_event(
        "llm_complete",
        Lane.LLM,
        data={"text": "I'm doing great, thank you for asking!"}
    )
    print(f"  🤖 LLM Complete at {llm_complete.timestamp:.3f}s")
    
    # TTS generates audio
    tts_start = timeline.add_event("tts_start", Lane.TTS)
    await asyncio.sleep(0.2)  # TTS latency
    
    tts_first_audio = timeline.add_event("tts_first_audio", Lane.TTS)
    print(f"  🔊 TTS First Audio at {tts_first_audio.timestamp:.3f}s")
    
    # TTS plays for 3 seconds
    await asyncio.sleep(3.0)
    tts_complete = timeline.add_event("tts_complete", Lane.TTS)
    print(f"  ✅ TTS Complete at {tts_complete.timestamp:.3f}s")
    
    # Add TTS segment as rectangle
    timeline.add_tts_segment(
        start_time=tts_first_audio.timestamp,
        end_time=tts_complete.timestamp,
        text="I'm doing great, thank you for asking!",
        data={"voice": "English-US.Female-1"}
    )
    print(f"  📊 TTS Rectangle: {tts_first_audio.timestamp:.3f}s - {tts_complete.timestamp:.3f}s")
    
    # Add some audio amplitude samples for waveform (user audio)
    print("\n  🎵 Adding audio waveform samples...")
    for i in range(10):
        amp = 50 + (i % 5) * 10  # Varying amplitude
        timeline.add_audio_amplitude(
            amplitude=amp,
            source='user',
            data={"sample_index": i}
        )
    
    # Add AI audio waveform samples
    for i in range(15):
        amp = 60 + (i % 7) * 8
        timeline.add_audio_amplitude(
            amplitude=amp,
            source='tts',
            data={"sample_index": i}
        )
    
    # Calculate TTL
    ttl = timeline.calculate_ttl()
    if ttl:
        print(f"\n⚡ Turn-Taking Latency (TTL): {ttl*1000:.0f}ms")
    
    # Export timeline
    timeline_data = timeline.to_dict()
    
    print(f"\n📦 Timeline has {len(timeline_data)} events")
    print(f"   - VAD segments (rectangles): {len([e for e in timeline_data if e.get('event_type') == 'vad_segment'])}")
    print(f"   - ASR segments (rectangles): {len([e for e in timeline_data if e.get('event_type') == 'asr_segment'])}")
    print(f"   - TTS segments (rectangles): {len([e for e in timeline_data if e.get('event_type') == 'tts_segment'])}")
    print(f"   - Audio waveforms: {len([e for e in timeline_data if e.get('event_type') == 'audio_amplitude'])}")
    
    # Save to file
    output_file = "/tmp/demo_timeline.json"
    with open(output_file, 'w') as f:
        json.dump(timeline_data, f, indent=2)
    
    print(f"\n💾 Timeline saved to: {output_file}")
    print("\n✅ Demo complete! You can load this timeline in the WebUI to see rectangles.")
    
    return timeline

if __name__ == "__main__":
    asyncio.run(demo_conversation_timeline())
