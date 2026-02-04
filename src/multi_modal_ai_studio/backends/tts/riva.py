"""
NVIDIA Riva TTS Backend

Implements TTS using NVIDIA Riva's speech synthesis service.
Adapted from live-riva-webui with streaming support and timeline events.
"""

import asyncio
import logging
import re
from typing import AsyncIterator, Optional

import riva.client
import riva.client.proto.riva_tts_pb2 as rtts

from multi_modal_ai_studio.backends.base import (
    TTSBackend,
    TTSChunk,
    ConnectionError,
    ConfigError,
)
from multi_modal_ai_studio.config.schema import TTSConfig

logger = logging.getLogger(__name__)

# Riva TTS has a max input length of 2000 characters
# Use 1800 as safe limit to account for variations
MAX_TTS_CHARS = 1800


class RivaTTSBackend(TTSBackend):
    """NVIDIA Riva TTS backend with streaming support.
    
    Features:
    - Streaming audio generation
    - Automatic text chunking (respects 2000 char limit)
    - Voice selection
    - Multiple sample rates
    """
    
    def __init__(self, config: TTSConfig):
        """Initialize Riva TTS backend.
        
        Args:
            config: TTSConfig instance with Riva settings
        
        Raises:
            ConfigError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        if config.scheme != "riva":
            raise ConfigError(f"Expected scheme 'riva', got '{config.scheme}'")
        
        if not config.server:
            raise ConfigError("Riva server address is required")
        
        # Parse server address
        self.riva_server = config.server
        self.use_ssl = config.server.startswith("https://") or ":443" in config.server
        self.ssl_cert = None  # TODO: Add SSL cert support in config
        
        # Initialize Riva client
        try:
            self.auth = riva.client.Auth(self.ssl_cert, self.use_ssl, self.riva_server)
            self.tts_service = riva.client.SpeechSynthesisService(self.auth)
            self.logger.info(f"Initialized Riva TTS: {config.voice} @ {self.riva_server}")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Riva TTS: {e}")
    
    async def list_voices(self) -> list:
        """List available TTS voices from Riva.
        
        Returns:
            List of voice names
        """
        try:
            loop = asyncio.get_running_loop()
            
            def _list_voices():
                # Use the service's list_voices method if available
                # Note: Some Riva versions may not support ListVoices
                try:
                    voices = self.tts_service.list_voices()
                    return voices
                except AttributeError:
                    self.logger.warning("list_voices not available in this Riva version")
                    return []
            
            voices = await loop.run_in_executor(None, _list_voices)
            if voices:
                self.logger.info(f"Available voices: {voices}")
            return voices
        
        except Exception as e:
            self.logger.warning(f"Could not list voices: {e}")
            return []
    
    @staticmethod
    def _split_text_by_sentences(text: str, max_chars: int = MAX_TTS_CHARS) -> list:
        """Split text into chunks by sentences, respecting max character limit.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        # Split by sentence boundaries (., !, ?, ;, :, newlines)
        sentences = re.split(r'(?<=[.!?;:\n])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If single sentence is too long, force split by words
            if len(sentence) > max_chars:
                logger.warning(
                    f"Single sentence exceeds max length ({len(sentence)} > {max_chars}), "
                    "splitting by words"
                )
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += (" " if current_chunk else "") + word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            # Normal case: accumulate sentences
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
        return chunks
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[TTSChunk]:
        """Generate audio chunks from text in streaming fashion.
        
        Args:
            text: Text to synthesize
        
        Yields:
            TTSChunk: Audio chunks
        
        Raises:
            ConnectionError: If unable to connect to TTS service
        """
        if not text.strip():
            self.logger.warning("Empty text for TTS, skipping")
            return
        
        self.logger.debug(f"TTS synthesizing: {text[:50]}... ({len(text)} chars)")
        
        # Split text into chunks if needed
        text_chunks = self._split_text_by_sentences(text)
        
        try:
            loop = asyncio.get_running_loop()
            total_chunks = len(text_chunks)
            
            for chunk_idx, text_chunk in enumerate(text_chunks):
                is_last_chunk = (chunk_idx == total_chunks - 1)
                
                if total_chunks > 1:
                    self.logger.debug(f"TTS chunk {chunk_idx + 1}/{total_chunks}: {len(text_chunk)} chars")
                
                # Synthesize this chunk (blocking call, run in executor)
                def _synthesize():
                    # Determine language code from voice name or use default
                    language_code = "en-US"  # Default
                    if self.config.voice:
                        # Extract language from voice name (e.g., "English-US.Female-1" -> "en-US")
                        if "English-US" in self.config.voice:
                            language_code = "en-US"
                        elif "English-GB" in self.config.voice:
                            language_code = "en-GB"
                        # Add more as needed
                    
                    return list(self.tts_service.synthesize_online(
                        text_chunk,
                        voice_name=self.config.voice or "",
                        language_code=language_code,
                        encoding=riva.client.AudioEncoding.LINEAR_PCM,
                        sample_rate_hz=self.config.sample_rate,
                    ))
                
                responses = await loop.run_in_executor(None, _synthesize)
                
                # Yield audio chunks
                for audio_idx, response in enumerate(responses):
                    if response.audio:
                        is_final = is_last_chunk and (audio_idx == len(responses) - 1)
                        
                        # Calculate approximate duration
                        audio_bytes = len(response.audio)
                        bytes_per_second = self.config.sample_rate * 2  # 16-bit = 2 bytes per sample
                        duration_ms = (audio_bytes / bytes_per_second) * 1000 if bytes_per_second > 0 else 0
                        
                        yield TTSChunk(
                            audio=response.audio,
                            sample_rate=self.config.sample_rate,
                            is_final=is_final,
                            duration_ms=duration_ms,
                            metadata={
                                "voice": self.config.voice,
                                "chunk_index": chunk_idx,
                                "audio_chunk_index": audio_idx,
                                "total_text_chunks": total_chunks,
                            }
                        )
        
        except Exception as e:
            self.logger.error(f"TTS synthesis error: {e}", exc_info=True)
            raise ConnectionError(f"Failed to synthesize speech: {e}")
