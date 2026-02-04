"""
Configuration schema for Multi-modal AI Studio.

This module defines the complete configuration structure using dataclasses.
All configuration objects support:
- YAML/JSON serialization
- Validation
- CLI argument generation
- Default values
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
import yaml
import json


@dataclass
class ASRConfig:
    """ASR (Automatic Speech Recognition) configuration.
    
    Attributes:
        scheme: Backend type
        server: Server address (for gRPC backends like Riva)
        api_base: API base URL (for REST backends like OpenAI)
        api_key: API key for authentication
        model: Model identifier
        language: Language code (e.g., en-US)
        vad_start_threshold: Voice Activity Detection start threshold
        vad_stop_threshold: Voice Activity Detection stop threshold
        speech_timeout_ms: Silence timeout in milliseconds
        requires_restart: Whether changing config requires RIVA restart
    """
    scheme: Literal["riva", "openai-rest", "openai-realtime", "azure", "none"] = "riva"
    server: Optional[str] = "localhost:50051"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "conformer"
    language: str = "en-US"
    vad_start_threshold: float = 0.5
    vad_stop_threshold: float = 0.3
    speech_timeout_ms: int = 500
    requires_restart: bool = False
    
    def validate(self) -> List[str]:
        """Validate configuration consistency.
        
        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []
        
        if self.scheme == "riva":
            if not self.server:
                warnings.append("Riva scheme requires server address")
        
        elif self.scheme in ["openai-rest", "openai-realtime"]:
            if not self.api_key:
                warnings.append("OpenAI scheme requires API key")
        
        elif self.scheme == "azure":
            if not self.api_key:
                warnings.append("Azure scheme requires subscription key")
        
        # VAD thresholds validation
        if not (0.0 <= self.vad_start_threshold <= 1.0):
            warnings.append("VAD start threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.vad_stop_threshold <= 1.0):
            warnings.append("VAD stop threshold must be between 0.0 and 1.0")
        
        if self.vad_start_threshold < self.vad_stop_threshold:
            warnings.append("VAD start threshold should be >= stop threshold")
        
        return warnings


@dataclass
class LLMConfig:
    """LLM (Large Language Model) configuration.
    
    Attributes:
        scheme: Backend type (openai-compatible for most)
        api_base: API base URL
        api_key: API key for authentication
        model: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for the conversation
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
    """
    scheme: Literal["openai", "anthropic", "none"] = "openai"
    api_base: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None
    model: str = "llama3.2:3b"
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = "You are a helpful voice assistant."
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def validate(self) -> List[str]:
        """Validate configuration consistency.
        
        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []
        
        if not self.api_base:
            warnings.append("LLM API base URL is required")
        
        if self.scheme == "anthropic" and not self.api_key:
            warnings.append("Anthropic requires API key")
        
        if not (0.0 <= self.temperature <= 2.0):
            warnings.append("Temperature should be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            warnings.append("max_tokens must be positive")
        
        return warnings


@dataclass
class TTSConfig:
    """TTS (Text-to-Speech) configuration.
    
    Attributes:
        scheme: Backend type
        server: Server address (for gRPC backends like Riva)
        api_base: API base URL (for REST backends)
        api_key: API key for authentication
        voice: Voice identifier
        sample_rate: Audio sample rate in Hz
        speed: Speech speed multiplier (0.25-4.0 for OpenAI)
        response_format: Audio format (pcm, mp3, opus, etc.)
    """
    scheme: Literal["riva", "openai-rest", "openai-realtime", "elevenlabs", "none"] = "riva"
    server: Optional[str] = "localhost:50051"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    voice: str = "English-US.Female-1"
    sample_rate: int = 24000
    speed: float = 1.0
    response_format: str = "pcm"
    
    def validate(self) -> List[str]:
        """Validate configuration consistency.
        
        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []
        
        if self.scheme == "riva":
            if not self.server:
                warnings.append("Riva scheme requires server address")
        
        elif self.scheme in ["openai-rest", "openai-realtime"]:
            if not self.api_key:
                warnings.append("OpenAI scheme requires API key")
            if not (0.25 <= self.speed <= 4.0):
                warnings.append("OpenAI speed must be between 0.25 and 4.0")
        
        elif self.scheme == "elevenlabs":
            if not self.api_key:
                warnings.append("ElevenLabs requires API key")
        
        if self.sample_rate not in [8000, 16000, 22050, 24000, 44100, 48000]:
            warnings.append(f"Unusual sample rate: {self.sample_rate} Hz")
        
        return warnings


@dataclass
class DeviceConfig:
    """Device routing configuration.
    
    Attributes:
        video_source: Video input source
        video_device: Device path for USB video (e.g., /dev/video0)
        audio_input_source: Audio input source
        audio_input_device: Device path for USB/ALSA audio
        audio_output_source: Audio output source
        audio_output_device: Device path for USB/ALSA audio
    """
    video_source: Literal["browser", "usb", "none"] = "browser"
    video_device: Optional[str] = None
    audio_input_source: Literal["browser", "usb", "alsa", "none"] = "browser"
    audio_input_device: Optional[str] = None
    audio_output_source: Literal["browser", "usb", "alsa", "none"] = "browser"
    audio_output_device: Optional[str] = None
    
    @property
    def interaction_mode(self) -> str:
        """Determine interaction mode based on device config.
        
        Returns:
            One of: voice_to_voice, voice_to_text, text_to_voice, text_to_text
        """
        mic = self.audio_input_source
        spk = self.audio_output_source
        
        if mic != "none" and spk != "none":
            return "voice_to_voice"
        elif mic != "none" and spk == "none":
            return "voice_to_text"
        elif mic == "none" and spk != "none":
            return "text_to_voice"
        else:
            return "text_to_text"
    
    @property
    def needs_asr(self) -> bool:
        """Check if ASR is needed."""
        return self.audio_input_source != "none"
    
    @property
    def needs_tts(self) -> bool:
        """Check if TTS is needed."""
        return self.audio_output_source != "none"
    
    def get_mode_description(self) -> str:
        """Get human-readable mode description."""
        mode_map = {
            "voice_to_voice": "🎤 Voice Input → 🔊 Voice Output",
            "voice_to_text": "🎤 Voice Input → 📝 Text Output",
            "text_to_voice": "⌨️ Text Input → 🔊 Voice Output",
            "text_to_text": "⌨️ Text Input → 📝 Text Output"
        }
        return mode_map.get(self.interaction_mode, "Unknown")
    
    def validate(self) -> List[str]:
        """Validate device configuration.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        if self.video_source == "usb" and not self.video_device:
            warnings.append("USB video source requires device path")
        
        if self.audio_input_source in ["usb", "alsa"] and not self.audio_input_device:
            warnings.append("USB/ALSA audio input requires device path")
        
        if self.audio_output_source in ["usb", "alsa"] and not self.audio_output_device:
            warnings.append("USB/ALSA audio output requires device path")
        
        return warnings


@dataclass
class AppConfig:
    """Application-level configuration.
    
    Attributes:
        barge_in_enabled: Allow user to interrupt AI speech
        timeline_position: Timeline panel position (right=beside session list, bottom=below config, hidden=no timeline)
        session_auto_save: Automatically save sessions
        session_output_dir: Directory for session storage
        theme: UI theme (dark or light)
        auto_restart_riva: Automatically restart Riva when needed
    
    Note: Timeline always records ALL events (no buffer limit).
    Rendering limits are handled by the UI layer, not data collection.
    """
    barge_in_enabled: bool = True
    timeline_position: Literal["right", "bottom", "hidden"] = "right"
    session_auto_save: bool = True
    session_output_dir: str = "./sessions"
    theme: Literal["dark", "light"] = "dark"
    auto_restart_riva: bool = False
    
    def validate(self) -> List[str]:
        """Validate app configuration.
        
        Returns:
            List of warning messages
        """
        warnings = []
        # No validation needed for current fields
        return warnings


@dataclass
class SessionConfig:
    """Complete session configuration.
    
    This is the top-level configuration object that contains all settings
    for a session. It can be saved/loaded from YAML/JSON and converted to CLI args.
    """
    name: str = "New Session"
    description: str = ""
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    app: AppConfig = field(default_factory=AppConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Path) -> None:
        """Export to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Path) -> None:
        """Export to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionConfig':
        """Create from dictionary."""
        return cls(
            name=data.get('name', 'New Session'),
            description=data.get('description', ''),
            asr=ASRConfig(**data.get('asr', {})),
            llm=LLMConfig(**data.get('llm', {})),
            tts=TTSConfig(**data.get('tts', {})),
            devices=DeviceConfig(**data.get('devices', {})),
            app=AppConfig(**data.get('app', {}))
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'SessionConfig':
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: Path) -> 'SessionConfig':
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate entire configuration.
        
        Returns:
            Dictionary mapping component names to warning lists
        """
        return {
            'asr': self.asr.validate(),
            'llm': self.llm.validate(),
            'tts': self.tts.validate(),
            'devices': self.devices.validate(),
            'app': self.app.validate(),
        }
    
    def get_required_services(self) -> List[str]:
        """Get list of required services based on configuration.
        
        Returns:
            List of service names: ['asr', 'llm', 'tts']
        """
        services = []
        
        if self.devices.needs_asr and self.asr.scheme != "none":
            services.append("asr")
        
        if self.llm.scheme != "none":
            services.append("llm")
        
        if self.devices.needs_tts and self.tts.scheme != "none":
            services.append("tts")
        
        return services
    
    def to_cli_args(self) -> str:
        """Generate CLI command from configuration.
        
        Returns:
            Shell command string
        """
        args = ["multi-modal-ai-studio"]
        
        # ASR args
        if self.asr.scheme != "none":
            args.append(f"--asr-scheme {self.asr.scheme}")
            if self.asr.server:
                args.append(f"--asr-server {self.asr.server}")
            if self.asr.api_key:
                args.append(f"--asr-api-key {self.asr.api_key}")
            args.append(f"--asr-model {self.asr.model}")
            args.append(f"--asr-language {self.asr.language}")
            args.append(f"--asr-vad-start {self.asr.vad_start_threshold}")
            args.append(f"--asr-vad-stop {self.asr.vad_stop_threshold}")
        
        # LLM args
        if self.llm.scheme != "none":
            args.append(f"--llm-scheme {self.llm.scheme}")
            args.append(f"--llm-api-base {self.llm.api_base}")
            if self.llm.api_key:
                args.append(f"--llm-api-key {self.llm.api_key}")
            args.append(f"--llm-model {self.llm.model}")
            args.append(f"--llm-temperature {self.llm.temperature}")
            args.append(f"--llm-max-tokens {self.llm.max_tokens}")
        
        # TTS args
        if self.tts.scheme != "none":
            args.append(f"--tts-scheme {self.tts.scheme}")
            if self.tts.server:
                args.append(f"--tts-server {self.tts.server}")
            if self.tts.api_key:
                args.append(f"--tts-api-key {self.tts.api_key}")
            args.append(f"--tts-voice {self.tts.voice}")
        
        # Device args
        if self.devices.audio_input_source != "browser":
            device_str = self.devices.audio_input_source
            if self.devices.audio_input_device:
                device_str += f":{self.devices.audio_input_device}"
            args.append(f"--audio-input {device_str}")
        
        if self.devices.audio_output_source != "browser":
            device_str = self.devices.audio_output_source
            if self.devices.audio_output_device:
                device_str += f":{self.devices.audio_output_device}"
            args.append(f"--audio-output {device_str}")
        
        # App args
        if self.app.barge_in_enabled:
            args.append("--barge-in")
        
        args.append(f"--timeline-position {self.app.timeline_position}")
        
        return " \\\n  ".join(args)
