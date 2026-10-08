# Changelog

All notable changes to Multi-modal AI Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0-rc.1] - 2026-10-09

### Added
- Recommended Jetson quick start using Speaches Faster-Whisper and Kokoro with a text-only Gemma 4 E2B llama.cpp service.
- Jetson Orin and Thor launchers with health checks, persistent model caches, MTP speculative decoding, and reproducible API verification.
- OpenAI-compatible REST ASR and TTS backends with model, language, and voice discovery.
- Provider-neutral OpenAI-compatible Realtime transcription and response-audio backends.
- Native Realtime reference services for NVIDIA Nemotron 3.5 ASR and Magpie TTS.
- NVIDIA open speech model launcher, presets, API documentation, and Jetson qualification notes.
- Browser WebRTC and server USB microphone, speaker, and camera paths.
- Voice preview, speech-speed control, and optional compact OpenAI REST TTS configuration layout.
- Session thumbnails, recorded device and model configuration, and Jetson host information.
- Timeline visualization for VAD, ASR request/streaming/finalization, LLM prefill/generation, TTS, playback, barge-in, cancellation, and discarded audio.
- Manual speech release test plan and automated speech service contract tests.

### Changed
- Separated speech services from the LLM so each stage can be sized, restarted, and replaced independently.
- Made the public quick start independent of Riva, NGC entitlement, Ollama, and hosted APIs.
- Hardened live and recorded timeline timing so TTL and audio waveforms remain associated with the correct turn.
- Increased Jetson GPU telemetry sampling and preserved short utilization peaks in session data.
- Unified browser and server-device session behavior, cancellation, and replay semantics.

### Fixed
- Barge-in now stops browser and server playback, cancels queued TTS work, and prevents cancelled audio from leaking into the next turn.
- REST TTS first-audio events are retained even for very short responses.
- Recorded session replay preserves the live TTL start point and AI waveform timing.
- Bundled and imported sessions remain reviewable when their JSON filename does not match the embedded session ID.
- ASR intervals no longer merge across separate utterances.
- Stale REST connections are retried without requiring a server restart.
- Voice metadata discovery remains optional and no longer breaks providers that do not implement extension endpoints.

### Testing
- Qualified the recommended stack on Jetson Orin Nano, Jetson AGX Orin, and Jetson AGX Thor.
- Added unit coverage for audio devices, VAD, Realtime adapters, cancellation, session replay, system telemetry, voice metadata, speed controls, and previews.
- Added OpenAI-compatible speech service contract tests and idempotent launcher verification commands.

### Known limitations
- The pinned Jetson Speaches image is a release candidate intended for evaluation and demos.
- Speaches Realtime transcription uses VAD-delimited Faster-Whisper requests rather than model-native token-by-token streaming, so partial transcripts may be unavailable.
- `Start speaking before LLM finishes` uses application-level text chunks. Very small chunks can produce prosody resets or playback underruns; keep it disabled for the baseline quality test.
- An 8 GB Orin Nano should stop unrelated GPU workloads and use swap for transient CPU-side allocations during model loading. Swap does not increase GPU memory.
- The bundled local services expose unauthenticated development endpoints. Use them only on a trusted network or behind an authenticated proxy.
- Headless mode remains experimental. Browser WebRTC is the recommended public-preview device path.
- Qwen3-TTS and other provider-specific servers are experimental integrations rather than part of the default quick start.

## [0.1.0] - 2026-02-03

### Project Initialization
- Created Multi-modal AI Studio as replacement for Live RIVA WebUI
- Established project goals: voice-first AI with latency analysis
- Set up development infrastructure and documentation

---

## Version History

- **v0.1.0** (2026-02-03): Project initialization, core infrastructure
- **v0.2.0-rc.1** (2026-10-09): Public preview for interchangeable local speech and LLM services on Jetson
