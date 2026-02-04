# Changelog

All notable changes to Multi-modal AI Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Core project structure with Python packaging (pyproject.toml)
- Configuration schema with dataclasses (ASR, LLM, TTS, Device, App, Session)
- 5 example configuration presets (default, low-latency, high-accuracy, openai-realtime, text-only)
- Abstract backend interfaces (ASRBackend, LLMBackend, TTSBackend)
- Riva ASR backend with streaming recognition and VAD
- OpenAI-compatible LLM backend with streaming generation
- Riva TTS backend with streaming audio synthesis
- Timeline system for recording all pipeline events
- Session management with turn tracking and TTL calculation
- Session save/load to JSON with full timeline preservation
- Comprehensive documentation (SESSION_MANAGEMENT.md, AUDIO_MODES.md, TIMELINE_DESIGN.md)
- Development environment setup script (scripts/setup_dev.sh)
- Integration tests for ASR, LLM, and TTS backends
- Cursor AI rules for consistent development (.cursor/rules/*.mdc)

### Changed
- **[2026-02-03] Removed `timeline_buffer_sec` configuration** (BREAKING)
  - Timeline now always collects ALL events (no buffer limit)
  - Rationale: Data collection should be unlimited; rendering limits belong in UI layer
  - Removed from `AppConfig` and all presets
  - See `docs/TIMELINE_DESIGN.md` for architectural details
- **[2026-02-03] Changed default `timeline_position` from `bottom` to `right`**
  - Matches UI design where timeline is beside session list
  - `text-only` preset still uses `hidden` (no timeline needed)
- Made `pyaudio` an optional dependency (only needed for USB audio devices)

### Fixed
- Riva TTS voice selection now handles missing `list_voices` API gracefully
- Riva TTS correctly uses `en-US` language code instead of invalid `pcm`
- Empty voice string (`""`) now uses Riva's default voice successfully

### Testing
- ✅ Backend initialization test (scripts/test_backends.py)
- ✅ Integration test with live Riva and Ollama (scripts/test_integration.py)
- ✅ Session management test with TTL calculation (scripts/test_session.py)

### Known Issues
- Backends don't emit timeline events yet (timeline integration pending)
- Headless CLI mode not implemented
- WebUI not implemented

## [0.1.0] - 2026-02-03

### Project Initialization
- Created Multi-modal AI Studio as replacement for Live RIVA WebUI
- Established project goals: voice-first AI with latency analysis
- Set up development infrastructure and documentation

---

## Version History

- **v0.1.0** (2026-02-03): Project initialization, core infrastructure
- **Unreleased**: Session management, timeline system, backend implementations
