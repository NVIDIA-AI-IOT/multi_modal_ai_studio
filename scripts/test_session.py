#!/usr/bin/env python3
"""
Test session management and timeline recording.

Simulates a conversational turn and verifies metrics calculation.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core.session import Session
from multi_modal_ai_studio.core.timeline import Lane, EventType


def simulate_conversation_turn(session: Session):
    """Simulate a single conversation turn with realistic timing."""
    
    print("\n🎬 Simulating Conversation Turn...")
    
    # Start turn
    session.start_turn()
    
    # === User speaks ===
    print("  👤 User starts speaking...")
    session.add_event(EventType.USER_SPEECH_START, Lane.AUDIO)
    time.sleep(0.1)  # User speaks for 1.5 seconds
    
    # === User stops speaking (VAD detects silence) ===
    print("  🔇 User stops speaking (VAD detected)")
    session.add_event(EventType.USER_SPEECH_END, Lane.AUDIO)  # KEY for TTL!
    
    # === ASR processes ===
    print("  🎤 ASR processing...")
    time.sleep(0.05)  # ASR finalization delay
    session.add_event(
        EventType.ASR_FINAL,
        Lane.SPEECH,
        {"text": "Hello, how are you?", "confidence": 0.98}
    )
    session.update_turn_transcript("Hello, how are you?", confidence=0.98)
    
    # === LLM processes ===
    print("  🧠 LLM generating response...")
    session.add_event(EventType.LLM_START, Lane.LLM)
    time.sleep(0.02)  # Prefill time
    
    session.add_event(EventType.LLM_FIRST_TOKEN, Lane.LLM)
    
    # Stream tokens
    response_tokens = ["I'm", " doing", " well", ", thank", " you", "!"]
    for token in response_tokens:
        time.sleep(0.01)  # Token generation time
        session.add_event(EventType.LLM_TOKEN, Lane.LLM, {"token": token})
    
    response = "".join(response_tokens)
    session.add_event(EventType.LLM_COMPLETE, Lane.LLM)
    session.update_turn_response(response)
    
    # === TTS synthesizes ===
    print("  🔊 TTS synthesizing...")
    session.add_event(EventType.TTS_START, Lane.TTS)
    time.sleep(0.03)  # TTS generation delay
    
    session.add_event(
        EventType.TTS_FIRST_AUDIO,
        Lane.TTS,
        {"chunk_size": 2048}
    )  # KEY for TTL!
    
    # More audio chunks
    for i in range(5):
        time.sleep(0.01)
        session.add_event(
            EventType.TTS_AUDIO,
            Lane.TTS,
            {"chunk_size": 2048, "chunk_index": i}
        )
    
    session.add_event(EventType.TTS_COMPLETE, Lane.TTS)
    
    # End turn
    session.end_turn()
    
    print("  ✓ Turn complete")


def test_session_management():
    """Test session creation, recording, and metrics."""
    print("=" * 70)
    print("Session Management Test")
    print("=" * 70)
    
    # Load config from preset
    config = SessionConfig.from_yaml(Path("presets/default.yaml"))
    
    # Create session
    print("\n→ Creating session...")
    session = Session(
        config=config,
        name="Test Session - TTL Measurement"
    )
    session.start()
    print(f"  ✓ Session created: {session.session_id[:8]}...")
    
    # Simulate 3 conversation turns
    for i in range(3):
        print(f"\n--- Turn {i+1} ---")
        simulate_conversation_turn(session)
    
    # Calculate metrics
    print("\n→ Calculating metrics...")
    metrics = session.calculate_metrics()
    
    print("\n" + "=" * 70)
    print("Session Metrics")
    print("=" * 70)
    print()
    print(session.get_metrics_summary())
    print()
    
    # Show detailed metrics
    print("Detailed Metrics:")
    print(f"  Total Turns: {metrics.total_turns}")
    print(f"  Session Duration: {metrics.session_duration:.2f}s")
    print(f"  ")
    print(f"  Turn-Taking Latency (TTL):")
    print(f"    Average: {metrics.avg_ttl * 1000:.0f}ms")
    print(f"    Min: {metrics.min_ttl * 1000:.0f}ms")
    print(f"    Max: {metrics.max_ttl * 1000:.0f}ms")
    print(f"  ")
    print(f"  Component Latencies:")
    print(f"    ASR: {metrics.avg_asr_latency * 1000:.0f}ms")
    print(f"    LLM: {metrics.avg_llm_latency * 1000:.0f}ms")
    print(f"    TTS: {metrics.avg_tts_latency * 1000:.0f}ms")
    
    # Test save/load
    print("\n→ Testing save/load...")
    save_path = Path("sessions") / f"test_session_{session.session_id[:8]}.json"
    session.save(save_path)
    
    # Load it back
    loaded_session = Session.load(save_path)
    print(f"  ✓ Session loaded: {loaded_session.name}")
    print(f"    Turns: {len(loaded_session.turns)}")
    print(f"    Events: {len(loaded_session.timeline.events)}")
    
    # Verify metrics match
    loaded_metrics = loaded_session.calculate_metrics()
    if abs(loaded_metrics.avg_ttl - metrics.avg_ttl) < 0.001:
        print(f"  ✓ Metrics match after save/load")
    else:
        print(f"  ⚠ Metrics mismatch!")
    
    # Clean up test file
    save_path.unlink()
    print(f"  ✓ Test file cleaned up")
    
    print("\n" + "=" * 70)
    print("✅ Session management test complete!")
    print("=" * 70)
    print()
    print("Timeline Event Summary:")
    summary = session.timeline.get_summary()
    print(f"  Total events: {summary['event_count']}")
    print(f"  Events by lane:")
    for lane, count in summary['events_by_lane'].items():
        print(f"    {lane}: {count}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_session_management()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
