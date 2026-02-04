#!/usr/bin/env python3
"""
Generate realistic mock session data for WebUI development.

Creates multiple sessions with different characteristics:
- Low latency (fast, short responses)
- High accuracy (slower, detailed responses)
- Mixed performance (variable latency)
- Long session (many turns for scroll testing)
"""

import sys
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core import Session, Lane, EventType


def simulate_turn(session: Session, user_text: str, ai_response: str, latency_profile: str = "normal"):
    """Simulate a realistic conversation turn with configurable latency."""
    
    # Start turn
    session.start_turn()
    
    # User speech
    session.add_event(EventType.USER_SPEECH_START, Lane.AUDIO)
    time.sleep(random.uniform(0.05, 0.15))  # User speaking duration variance
    
    # ASR processing
    if random.random() > 0.3:  # 70% chance of partial transcript
        session.add_event(EventType.ASR_PARTIAL, Lane.SPEECH, {
            "text": user_text[:len(user_text)//2],
            "is_final": False
        })
        time.sleep(random.uniform(0.01, 0.03))
    
    session.add_event(EventType.USER_SPEECH_END, Lane.AUDIO)  # KEY for TTL!
    time.sleep(random.uniform(0.01, 0.02))
    
    # Final ASR
    confidence = random.uniform(0.92, 0.99)
    session.add_event(EventType.ASR_FINAL, Lane.SPEECH, {
        "text": user_text,
        "confidence": confidence
    })
    session.update_turn_transcript(user_text, confidence)
    
    # Latency profiles
    if latency_profile == "low":
        asr_delay = random.uniform(0.03, 0.05)
        llm_prefill = random.uniform(0.04, 0.06)
        token_delay = 0.002
        tts_delay = random.uniform(0.02, 0.03)
    elif latency_profile == "high":
        asr_delay = random.uniform(0.08, 0.12)
        llm_prefill = random.uniform(0.15, 0.25)
        token_delay = 0.005
        tts_delay = random.uniform(0.05, 0.08)
    else:  # normal
        asr_delay = random.uniform(0.04, 0.08)
        llm_prefill = random.uniform(0.08, 0.15)
        token_delay = 0.003
        tts_delay = random.uniform(0.03, 0.05)
    
    time.sleep(asr_delay)
    
    # LLM generation
    session.add_event(EventType.LLM_START, Lane.LLM)
    time.sleep(llm_prefill)
    session.add_event(EventType.LLM_FIRST_TOKEN, Lane.LLM)
    
    # Simulate streaming tokens
    words = ai_response.split()
    for i, word in enumerate(words):
        session.add_event(EventType.LLM_TOKEN, Lane.LLM, {
            "token": word,
            "index": i
        })
        time.sleep(token_delay)
    
    session.add_event(EventType.LLM_COMPLETE, Lane.LLM)
    session.update_turn_response(ai_response)
    
    # TTS synthesis
    session.add_event(EventType.TTS_START, Lane.TTS)
    time.sleep(tts_delay)
    session.add_event(EventType.TTS_FIRST_AUDIO, Lane.TTS, {"chunk_size": 2048})  # KEY for TTL!
    
    # Simulate audio chunks
    estimated_chunks = len(ai_response) // 20
    for i in range(estimated_chunks):
        session.add_event(EventType.TTS_AUDIO, Lane.TTS, {
            "chunk_index": i,
            "chunk_size": 2048
        })
        time.sleep(random.uniform(0.01, 0.02))
    
    session.add_event(EventType.TTS_COMPLETE, Lane.TTS)
    session.end_turn()


def generate_low_latency_session():
    """Generate a low-latency optimized session."""
    print("📝 Generating low-latency session...")
    config = SessionConfig.from_yaml(Path("presets/low-latency.yaml"))
    session = Session(config, name="Low Latency Test - 2026-02-03")
    session.start()
    
    conversations = [
        ("What's the weather like?", "It's sunny and 72 degrees."),
        ("Set a timer for 5 minutes", "Timer set for 5 minutes."),
        ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
        ("What time is it?", "It's 2:30 PM."),
        ("Thanks", "You're welcome!"),
    ]
    
    for user_text, ai_response in conversations:
        simulate_turn(session, user_text, ai_response, latency_profile="low")
        time.sleep(random.uniform(0.1, 0.3))  # Gap between turns
    
    metrics = session.calculate_metrics()
    print(f"  ✓ 5 turns, Avg TTL: {metrics.avg_ttl*1000:.0f}ms")
    return session


def generate_high_accuracy_session():
    """Generate a high-accuracy session with detailed responses."""
    print("📝 Generating high-accuracy session...")
    config = SessionConfig.from_yaml(Path("presets/high-accuracy.yaml"))
    session = Session(config, name="High Accuracy Test - 2026-02-03")
    session.start()
    
    conversations = [
        ("Explain quantum computing", "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information in ways classical computers cannot. Unlike classical bits that are either 0 or 1, quantum bits or qubits can exist in multiple states simultaneously."),
        ("What are the main challenges?", "The main challenges include maintaining quantum coherence, error correction, scaling to large numbers of qubits, and operating at extremely low temperatures near absolute zero."),
        ("How is it different from classical computing?", "Classical computing uses binary logic gates and sequential processing, while quantum computing uses quantum gates and parallel processing across superposed states, potentially solving certain problems exponentially faster."),
        ("What are practical applications?", "Practical applications include cryptography, drug discovery, financial modeling, climate simulation, optimization problems, and machine learning acceleration."),
        ("When will it be mainstream?", "Widespread quantum computing adoption is likely 10-20 years away, though specialized applications may emerge sooner as the technology matures and becomes more accessible."),
        ("Tell me about quantum entanglement", "Quantum entanglement is a phenomenon where particles become correlated such that the quantum state of one particle cannot be described independently of others, even when separated by large distances."),
        ("How does quantum error correction work?", "Quantum error correction uses redundant encoding of quantum information across multiple physical qubits to protect against errors from decoherence and noise, similar to classical error correction but more complex due to quantum constraints."),
        ("Thank you for the explanation", "You're welcome! Feel free to ask if you have more questions about quantum computing."),
    ]
    
    for user_text, ai_response in conversations:
        simulate_turn(session, user_text, ai_response, latency_profile="high")
        time.sleep(random.uniform(0.2, 0.5))
    
    metrics = session.calculate_metrics()
    print(f"  ✓ 8 turns, Avg TTL: {metrics.avg_ttl*1000:.0f}ms")
    return session


def generate_mixed_performance_session():
    """Generate a session with variable performance."""
    print("📝 Generating mixed performance session...")
    config = SessionConfig.from_yaml(Path("presets/default.yaml"))
    session = Session(config, name="Mixed Performance - 2026-02-03")
    session.start()
    
    conversations = [
        ("Hi there", "Hello! How can I help you today?", "low"),
        ("Tell me about machine learning", "Machine learning is a subset of artificial intelligence focused on building systems that learn from data and improve their performance over time without being explicitly programmed.", "normal"),
        ("What's deep learning?", "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input, enabling impressive results in computer vision, natural language processing, and speech recognition.", "high"),
        ("How do neural networks work?", "Neural networks consist of interconnected nodes organized in layers that process information through weighted connections, learning to recognize patterns by adjusting these weights through backpropagation.", "normal"),
        ("Okay", "Is there anything else you'd like to know?", "low"),
        ("Explain backpropagation", "Backpropagation is an algorithm for training neural networks by computing gradients of the loss function with respect to each weight, then updating weights to minimize error using gradient descent.", "high"),
        ("What about overfitting?", "Overfitting occurs when a model learns training data too well, including noise and outliers, resulting in poor generalization to new data. Techniques like regularization, dropout, and cross-validation help prevent it.", "normal"),
        ("Can you give an example?", "Sure! Imagine a student who memorizes answers to practice problems without understanding concepts. They'll ace practice tests but fail on new questions.", "normal"),
        ("That makes sense", "Great! Understanding the concepts rather than memorizing is key to both learning and building good ML models.", "low"),
        ("What's the difference between supervised and unsupervised learning?", "Supervised learning uses labeled training data to learn input-output mappings, while unsupervised learning finds patterns in unlabeled data without predefined categories or targets.", "normal"),
        ("Tell me more about unsupervised learning", "Unsupervised learning includes clustering algorithms like k-means, dimensionality reduction techniques like PCA, and generative models like autoencoders and GANs.", "high"),
        ("Thanks for explaining", "You're welcome! Machine learning is a fascinating field with lots to explore.", "low"),
    ]
    
    for user_text, ai_response, profile in conversations:
        simulate_turn(session, user_text, ai_response, latency_profile=profile)
        time.sleep(random.uniform(0.1, 0.4))
    
    metrics = session.calculate_metrics()
    print(f"  ✓ 12 turns, Avg TTL: {metrics.avg_ttl*1000:.0f}ms")
    return session


def generate_long_session():
    """Generate a long session for scroll testing."""
    print("📝 Generating long session (30 turns)...")
    config = SessionConfig.from_yaml(Path("presets/default.yaml"))
    session = Session(config, name="Long Conversation - 2026-02-03")
    session.start()
    
    # Generate 30 varied turns
    topics = [
        ("What's your name?", "I'm an AI voice assistant designed to help with various tasks."),
        ("Tell me about yourself", "I can answer questions, help with tasks, and have natural conversations using voice."),
        ("What's the capital of France?", "Paris is the capital of France."),
        ("How far is the moon?", "The moon is approximately 238,855 miles or 384,400 kilometers from Earth."),
        ("What's 15 times 23?", "15 times 23 equals 345."),
        ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
        ("What's the speed of light?", "The speed of light in a vacuum is about 299,792 kilometers per second."),
        ("Tell me a fun fact", "Honey never spoils - archaeologists have found 3,000-year-old honey that's still edible!"),
        ("What's the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("How many continents are there?", "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."),
    ]
    
    # Repeat and vary
    for i in range(30):
        user_text, ai_response = topics[i % len(topics)]
        if i > 10:
            user_text = f"Question {i}: {user_text}"
        profile = ["low", "normal", "high"][i % 3]
        simulate_turn(session, user_text, ai_response, latency_profile=profile)
        time.sleep(random.uniform(0.05, 0.2))
    
    metrics = session.calculate_metrics()
    print(f"  ✓ 30 turns, Avg TTL: {metrics.avg_ttl*1000:.0f}ms")
    return session


def main():
    """Generate all mock sessions."""
    print("=" * 70)
    print("Generating Mock Session Data for WebUI Development")
    print("=" * 70)
    print()
    
    output_dir = Path("mock_sessions")
    output_dir.mkdir(exist_ok=True)
    
    sessions = [
        generate_low_latency_session(),
        generate_high_accuracy_session(),
        generate_mixed_performance_session(),
        generate_long_session(),
    ]
    
    print()
    print("💾 Saving sessions...")
    for session in sessions:
        # Create filename from session name
        filename = session.name.lower().replace(" ", "_").replace("-", "") + ".json"
        filepath = output_dir / filename
        session.save(filepath)
        
        metrics = session.calculate_metrics()
        print(f"  ✓ {filepath.name}")
        print(f"    - Turns: {metrics.total_turns}")
        print(f"    - Events: {len(session.timeline.events)}")
        print(f"    - Avg TTL: {metrics.avg_ttl*1000:.0f}ms")
        print(f"    - Size: {filepath.stat().st_size / 1024:.1f} KB")
    
    print()
    print("=" * 70)
    print("✅ Mock session generation complete!")
    print("=" * 70)
    print()
    print(f"Generated {len(sessions)} session files in {output_dir}/")
    print("Ready for WebUI development!")


if __name__ == "__main__":
    main()
