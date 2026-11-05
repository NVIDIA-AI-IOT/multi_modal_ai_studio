# Changelog

All notable changes to Multi-modal AI Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - Unreleased

This release consolidates the changes developed in rc1 and rc2 below. Publication and the final release date remain pending.

### Highlights
- Run a local voice assistant on Jetson Orin Nano, AGX Orin, or Thor with independent ASR, LLM, and TTS services; the recommended path uses Speaches and text-only Gemma 4 E2B.
- Choose REST or Realtime speech adapters, NVIDIA open-model reference services, or the existing Riva gRPC path.
- Inspect speech, generation, playback, barge-in, and device utilization together in a saved session timeline.
- Install a self-contained UI/preset wheel; launch GPU model services independently using the matching release's repository scripts.

### Release preparation
- Include canonical presets in wheel/sdist builds and expose them in the CLI and Web UI outside a source checkout, with writable local overrides kept out of site-packages.
- Verify clean wheel installation, preset APIs, and static UI assets in CI; attach wheel, sdist, dependency manifests, and checksums to GitHub Releases.
- Publish the same verified wheel/sdist to GitHub and, when explicitly enabled, PyPI for stable releases via approval-gated Trusted Publishing. RC releases do not automatically publish to PyPI; TestPyPI and manual promotion remain explicit operations.
- Record final rc2 wheel qualification on Orin Nano and Thor, including audible interruption recovery and saved-session replay/download checks.

### Fixed
- Discard a cancelled, unheard response's pending TTL measurement without overwriting a later turn's origin or an already completed band.

### Limitations
- See the rc1 known limitations and the [0.2.0 release notes](docs/releases/v0.2.0.md). In particular, backend containers remain independent, headless/USB coverage is limited, and application-level TTS chunking is not model-native streaming.

## [0.2.0-rc.2] - Unreleased

### Fixed
- Normalize UI locale tags such as `ja-JP` to `ja` for REST transcription, without changing Riva locales. Automatic language detection omits the language parameter.
- Pair TTS cancellation completion with its turn even when the server completes cancellation before the browser reports playback stopped.
- Preserve answer content when a streamed LLM delta also contains reasoning. Report reasoning-only or empty answers instead of silently skipping speech.
- Remove the unsupported promise that uninstalled speech models will always download on first use.

### Added
- REST ASR model-list Reload and explicit, confirmed Download for providers advertising the Speaches model-management API.
- Session JSON download next to the filename copy control, including after Stop and during recorded review.
- Disabled-by-default, private, bounded REST ASR WAV diagnostics and an identical-input replay script for ASR-only/TTS-overlap/post-TTS comparisons.
- Regression coverage for language normalization, provider-specific downloads, cancellation event ordering, mixed reasoning/answer deltas, and original JSON downloads.

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
