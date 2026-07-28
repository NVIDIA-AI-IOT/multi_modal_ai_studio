# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Base classes for ASR, LLM, and TTS backends.

All backend implementations must inherit from these abstract base classes
to ensure consistent interface across different providers (Riva, OpenAI, Azure, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Dict, Any, List
import logging
import unicodedata

logger = logging.getLogger(__name__)

_TTS_SENTENCE_BREAKS = frozenset(
    ".!?。！？｡؟۔።।॥៕៚᙮᜶᜷꓿…"
)
_TTS_CLAUSE_BREAKS = frozenset(
    ",;:，、；：﹐﹑﹔﹕︐︑︔︓､،؛"
)
_TTS_CLOSING_PUNCTUATION = frozenset(
    "\"'”’»›」』】）》〕〉》〗〙〛"
)


def _split_after_punctuation(text: str, boundaries: frozenset) -> List[str]:
    """Return text pieces ending after Unicode punctuation and closing marks."""
    pieces = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\n" or char in boundaries:
            end = index + 1
            while (
                end < len(text)
                and text[end] in _TTS_CLOSING_PUNCTUATION
            ):
                end += 1
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end
            index = end
            continue
        index += 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _unicode_text_clusters(text: str) -> List[str]:
    """Approximate grapheme clusters without a language-specific tokenizer."""
    clusters: List[str] = []
    current = ""
    previous_was_zwj = False
    for char in text:
        codepoint = ord(char)
        is_mark = unicodedata.category(char) in {"Mn", "Mc", "Me"}
        is_variation_selector = (
            0xFE00 <= codepoint <= 0xFE0F
            or 0xE0100 <= codepoint <= 0xE01EF
        )
        is_zwj = codepoint == 0x200D
        if (
            current
            and not is_mark
            and not is_variation_selector
            and not is_zwj
            and not previous_was_zwj
        ):
            clusters.append(current)
            current = char
        else:
            current += char
        previous_was_zwj = is_zwj
    if current:
        clusters.append(current)
    return clusters


def _pack_tts_pieces(
    pieces: List[str],
    max_chars: int,
    separator: str = " ",
) -> List[str]:
    """Pack already-bounded semantic pieces without exceeding max_chars."""
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        joiner = separator if current else ""
        if len(current) + len(joiner) + len(piece) <= max_chars:
            current += joiner + piece
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def split_tts_text(text: str, max_chars: int) -> List[str]:
    """Split multilingual TTS text without requiring a language tokenizer.

    Prefer Unicode sentence endings, then clause punctuation, then whitespace
    boundaries. Text without spaces (for example Japanese or Chinese) falls
    back to approximate grapheme clusters so combining marks and ZWJ sequences
    remain intact whenever possible.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    bounded: List[str] = []
    for sentence in _split_after_punctuation(text, _TTS_SENTENCE_BREAKS):
        if len(sentence) <= max_chars:
            bounded.append(sentence)
            continue
        for clause in _split_after_punctuation(sentence, _TTS_CLAUSE_BREAKS):
            if len(clause) <= max_chars:
                bounded.append(clause)
                continue
            words = clause.split()
            if len(words) > 1:
                word_chunks = _pack_tts_pieces(words, max_chars)
                for word_chunk in word_chunks:
                    if len(word_chunk) <= max_chars:
                        bounded.append(word_chunk)
                    else:
                        bounded.extend(
                            _pack_tts_pieces(
                                _unicode_text_clusters(word_chunk),
                                max_chars,
                                separator="",
                            )
                        )
            else:
                bounded.extend(
                    _pack_tts_pieces(
                        _unicode_text_clusters(clause),
                        max_chars,
                        separator="",
                    )
                )
    # Keep primary sentence boundaries intact. Repacking the final list could
    # join the tail of one sentence to the next and undo the prosody boundary
    # that this splitter is intended to preserve.
    return [piece for piece in bounded if piece]


@dataclass
class ASRResult:
    """Result from ASR backend.
    
    Attributes:
        text: Transcribed text
        is_final: Whether this is a final transcript (vs partial/intermediate)
        confidence: Confidence score (0.0-1.0)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        metadata: Additional backend-specific metadata
    """
    text: str
    is_final: bool
    confidence: float = 1.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMToken:
    """Token from LLM backend.
    
    Attributes:
        token: The generated token/text chunk
        is_final: Whether this is the last token
        metadata: Additional backend-specific metadata (e.g., finish_reason)
    """
    token: str
    is_final: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TTSChunk:
    """Audio chunk from TTS backend.
    
    Attributes:
        audio: Raw audio bytes (PCM format)
        sample_rate: Sample rate in Hz
        is_final: Whether this is the last chunk
        duration_ms: Duration of this chunk in milliseconds
        metadata: Additional backend-specific metadata
    """
    audio: bytes
    sample_rate: int
    is_final: bool = False
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ASRBackend(ABC):
    """Abstract base class for ASR (Automatic Speech Recognition) backends.
    
    Implementations must support streaming audio input and produce
    both partial and final transcription results.
    """
    
    def __init__(self, config):
        """Initialize ASR backend.
        
        Args:
            config: ASRConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def start_stream(self) -> None:
        """Start streaming recognition session.
        
        This should establish connection to the ASR service and
        prepare for receiving audio chunks.
        
        Raises:
            ConnectionError: If unable to connect to ASR service
            ConfigError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio chunk for recognition.
        
        Args:
            audio_chunk: Raw audio bytes (PCM format, 16kHz, 16-bit)
        
        Raises:
            RuntimeError: If stream not started or connection lost
        """
        pass
    
    @abstractmethod
    async def receive_results(self) -> AsyncIterator[ASRResult]:
        """Yield recognition results as they become available.
        
        This should yield both partial results (is_final=False) and
        final results (is_final=True).
        
        Yields:
            ASRResult: Transcription results
        """
        pass
    
    @abstractmethod
    async def stop_stream(self) -> None:
        """Stop streaming recognition session.
        
        This should gracefully close the connection and clean up resources.
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_stream()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_stream()


class LLMBackend(ABC):
    """Abstract base class for LLM (Large Language Model) backends.
    
    Implementations must support streaming text generation with
    conversation history support.
    """
    
    def __init__(self, config):
        """Initialize LLM backend.
        
        Args:
            config: LLMConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[LLMToken]:
        """Generate response tokens in streaming fashion.
        
        Args:
            prompt: User prompt/message
            history: Conversation history in format:
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            system_prompt: Optional system prompt (overrides config if provided)
        
        Yields:
            LLMToken: Generated tokens
        
        Raises:
            ConnectionError: If unable to connect to LLM service
            ConfigError: If configuration is invalid
        """
        pass
    
    async def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate complete response (non-streaming convenience method).
        
        Args:
            prompt: User prompt/message
            history: Conversation history
            system_prompt: Optional system prompt
        
        Returns:
            Complete generated response
        """
        response_tokens = []
        async for token in self.generate_stream(prompt, history, system_prompt):
            response_tokens.append(token.token)
        return "".join(response_tokens)


class TTSBackend(ABC):
    """Abstract base class for TTS (Text-to-Speech) backends.
    
    Implementations must support streaming audio generation from text.
    """
    
    def __init__(self, config):
        """Initialize TTS backend.
        
        Args:
            config: TTSConfig instance
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[TTSChunk]:
        """Generate audio chunks from text in streaming fashion.
        
        Args:
            text: Text to synthesize
        
        Yields:
            TTSChunk: Audio chunks
        
        Raises:
            ConnectionError: If unable to connect to TTS service
            ConfigError: If configuration is invalid
        """
        pass
    
    async def synthesize(self, text: str) -> bytes:
        """Generate complete audio (non-streaming convenience method).
        
        Args:
            text: Text to synthesize
        
        Returns:
            Complete audio as bytes (PCM format)
        """
        audio_chunks = []
        async for chunk in self.synthesize_stream(text):
            audio_chunks.append(chunk.audio)
        return b"".join(audio_chunks)


class BackendError(Exception):
    """Base exception for backend errors."""
    pass


class ConnectionError(BackendError):
    """Raised when unable to connect to backend service."""
    pass


class ConfigError(BackendError):
    """Raised when backend configuration is invalid."""
    pass


class StreamError(BackendError):
    """Raised when stream operation fails."""
    pass
