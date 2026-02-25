"""
NVIDIA Riva TTS Backend

Implements TTS using NVIDIA Riva's speech synthesis service.
Adapted from live-riva-webui with streaming support and timeline events.
Streaming: yield audio chunks as Riva produces them (no buffering full sentence).
"""

import asyncio
import logging
import queue
import re
import threading
from typing import AsyncIterator, List, Optional

import riva.client
import riva.client.proto.riva_tts_pb2 as rtts

from multi_modal_ai_studio.backends.base import (
    TTSBackend,
    TTSChunk,
    ConnectionError,
    ConfigError,
)
from multi_modal_ai_studio.config.schema import TTSConfig
from multi_modal_ai_studio.core.timeline import Timeline, Lane

logger = logging.getLogger(__name__)


def list_riva_tts_voices_sync(
    server: str,
    use_ssl: bool = False,
    ssl_cert: Optional[str] = None,
    language_code: Optional[str] = "en-US",
):
    """List TTS voices and model info from Riva (sync, for use from thread/executor).

    Uses GetRivaSynthesisConfig and parses subvoices (same as Live RIVA WebUI).
    Optionally filters by language (e.g. en-US -> EN-US).

    Returns:
        dict with:
            voices: list of dicts with name, id, language, speaker, emotion, model.
            model_name: str, the active/first TTS model name from Riva (e.g. Magpie-Multilingual).
            model_names: list of str, all TTS model names returned by GetRivaSynthesisConfig.
    On error returns {"voices": [], "model_name": None, "model_names": []}.
    """
    try:
        auth = riva.client.Auth(ssl_cert, use_ssl, server)
        tts_service = riva.client.SpeechSynthesisService(auth)
        request = rtts.RivaSynthesisConfigRequest()
        response = tts_service.stub.GetRivaSynthesisConfig(
            request,
            metadata=auth.get_auth_metadata(),
        )
        # Collect all model names from response (Riva can expose multiple models)
        # Proto field may be model_name or name depending on Riva version
        model_names = []
        if response.model_config:
            for cfg in response.model_config:
                name = getattr(cfg, "model_name", None) or getattr(cfg, "name", None)
                if name:
                    model_names.append(str(name).strip())

        subvoices_str = None
        model_name = "Magpie-Multilingual"
        if response.model_config:
            first = response.model_config[0]
            name = getattr(first, "model_name", None) or getattr(first, "name", None)
            if name:
                model_name = str(name).strip()
            for key, value in first.parameters.items():
                if key == "subvoices":
                    subvoices_str = value
                    break

        if not subvoices_str:
            logger.debug("No subvoices in Riva TTS config")
            return {"voices": [], "model_name": model_name if model_names else None, "model_names": model_names}

        voices = []
        for item in subvoices_str.split(","):
            if ":" not in item:
                continue
            voice_name, voice_id = item.split(":", 1)
            parts = voice_name.split(".")
            full_voice_name = f"{model_name}.{voice_name}"
            lang = parts[0] if len(parts) > 0 else "unknown"
            voices.append({
                "name": full_voice_name,
                "id": voice_id,
                "language": lang,
                "speaker": parts[1] if len(parts) > 1 else voice_name,
                "emotion": parts[2] if len(parts) > 2 else None,
                "model": model_name,
            })

        if language_code:
            # Filter by language (e.g. en-US -> EN-US)
            lang_match = language_code.upper().replace("-", "-")
            voices = [v for v in voices if v["language"] == lang_match]

        # Ensure model_name/model_names are always set when we have voices (fallback if proto didn't expose model_name)
        if voices and not model_names and model_name:
            model_names = [model_name]
        if voices and not model_names and not model_name:
            model_name = voices[0].get("model") or "RIVA-TTS"
            model_names = [model_name] if model_name else []

        logger.info(
            "Discovered %d TTS voice(s) from Riva (language=%s, model=%s)",
            len(voices), language_code or "all", model_name,
        )
        return {"voices": voices, "model_name": model_name, "model_names": model_names}
    except Exception as e:
        logger.warning("Failed to list Riva TTS voices: %s", e)
        return {"voices": [], "model_name": None, "model_names": []}


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

    # Pre-buffer initial TTS audio before yielding (bytes).
    # Riva's synthesize_online often produces tiny initial chunks (10-20ms).
    # Playing those immediately causes stuttering/broken first words because
    # the browser's AudioContext runs out of audio before the next chunk
    # arrives.  Accumulating ~400ms first ensures the browser has enough
    # buffered audio while subsequent chunks stream in.
    _TTS_PREBUFFER_BYTES = 17640  # ~400ms at 22050 Hz, 16-bit mono

    # Minimum audio size for any yielded chunk AFTER the initial pre-buffer.
    # Prevents the browser from receiving tiny 10-20ms fragments that cause
    # playback gaps when there's any jitter in Riva's response timing.
    _MIN_YIELD_BYTES = 4410  # ~100ms at 22050 Hz, 16-bit mono

    def __init__(self, config: TTSConfig, timeline: Optional[Timeline] = None):
        """Initialize Riva TTS backend.

        Args:
            config: TTSConfig instance with Riva settings
            timeline: Optional Timeline instance for event recording

        Raises:
            ConfigError: If configuration is invalid
        """
        super().__init__(config)

        # Timeline for recording events (rectangle rendering, metrics)
        self.timeline = timeline

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

        # Emit TTS start event for timeline
        tts_start_time = None
        tts_first_audio_time = None
        if self.timeline:
            tts_start_event = self.timeline.add_event("tts_start", Lane.TTS, data={"text": text})
            tts_start_time = tts_start_event.timestamp

        # Split text into chunks if needed
        text_chunks = self._split_text_by_sentences(text)

        try:
            loop = asyncio.get_running_loop()
            total_chunks = len(text_chunks)

            for chunk_idx, text_chunk in enumerate(text_chunks):
                is_last_chunk = (chunk_idx == total_chunks - 1)

                if total_chunks > 1:
                    self.logger.debug(f"TTS chunk {chunk_idx + 1}/{total_chunks}: {len(text_chunk)} chars")

                # Stream this text chunk: run Riva's generator in a thread, yield chunks as they arrive (Live RIVA WebUI style)
                language_code = "en-US"
                if self.config.voice:
                    if "English-US" in self.config.voice:
                        language_code = "en-US"
                    elif "English-GB" in self.config.voice:
                        language_code = "en-GB"

                chunk_queue = queue.Queue()
                sentinel = object()

                def producer():
                    try:
                        for response in self.tts_service.synthesize_online(
                            text_chunk,
                            voice_name=self.config.voice or "",
                            language_code=language_code,
                            encoding=riva.client.AudioEncoding.LINEAR_PCM,
                            sample_rate_hz=self.config.sample_rate,
                        ):
                            if response.audio:
                                chunk_queue.put(response)
                    except Exception as e:
                        chunk_queue.put(e)
                    chunk_queue.put(sentinel)

                thread = threading.Thread(target=producer)
                thread.start()
                audio_idx = 0
                audio_buf = bytearray()
                # First text chunk uses a large initial buffer; subsequent
                # chunks use the smaller minimum-yield threshold.
                initial_target = self._TTS_PREBUFFER_BYTES if chunk_idx == 0 else self._MIN_YIELD_BYTES
                initial_flushed = False
                bytes_per_second = self.config.sample_rate * 2

                def _make_chunk(data: bytes) -> TTSChunk:
                    nonlocal audio_idx
                    audio_idx += 1
                    dur = (len(data) / bytes_per_second) * 1000 if bytes_per_second > 0 else 0
                    return TTSChunk(
                        audio=data,
                        sample_rate=self.config.sample_rate,
                        is_final=is_last_chunk,
                        duration_ms=dur,
                        metadata={
                            "voice": self.config.voice,
                            "chunk_index": chunk_idx,
                            "audio_chunk_index": audio_idx,
                            "total_text_chunks": total_chunks,
                        },
                    )

                try:
                    while True:
                        response = await loop.run_in_executor(None, chunk_queue.get)
                        if response is sentinel:
                            if audio_buf:
                                if self.timeline and tts_first_audio_time is None:
                                    first_audio_event = self.timeline.add_event("tts_first_audio", Lane.TTS)
                                    tts_first_audio_time = first_audio_event.timestamp
                                yield _make_chunk(bytes(audio_buf))
                            break
                        if isinstance(response, Exception):
                            raise response
                        if not response.audio:
                            continue

                        audio_buf.extend(response.audio)

                        # Initial accumulation phase (prebuffer or min-yield)
                        if not initial_flushed:
                            if len(audio_buf) >= initial_target:
                                initial_flushed = True
                                if self.timeline and tts_first_audio_time is None:
                                    first_audio_event = self.timeline.add_event("tts_first_audio", Lane.TTS)
                                    tts_first_audio_time = first_audio_event.timestamp
                                    self.logger.debug(f"TTS first audio at {tts_first_audio_time:.3f}s")
                                yield _make_chunk(bytes(audio_buf))
                                audio_buf.clear()
                            continue

                        # Post-initial: yield when buffer reaches minimum size
                        min_yield = self._MIN_YIELD_BYTES
                        while len(audio_buf) >= min_yield:
                            out = bytes(audio_buf[:min_yield])
                            del audio_buf[:min_yield]
                            if self.timeline and tts_first_audio_time is None:
                                first_audio_event = self.timeline.add_event("tts_first_audio", Lane.TTS)
                                tts_first_audio_time = first_audio_event.timestamp
                            yield _make_chunk(out)
                finally:
                    thread.join(timeout=1.0)

            # Emit TTS complete event and rectangle for timeline
            if self.timeline and tts_first_audio_time is not None:
                tts_complete_event = self.timeline.add_event("tts_complete", Lane.TTS)

                # Add TTS segment rectangle
                self.timeline.add_tts_segment(
                    start_time=tts_first_audio_time,
                    end_time=tts_complete_event.timestamp,
                    text=text,
                    data={"voice": self.config.voice}
                )

                self.logger.debug(
                    f"TTS rectangle: {tts_first_audio_time:.3f}s - {tts_complete_event.timestamp:.3f}s"
                )

        except Exception as e:
            self.logger.error(f"TTS synthesis error: {e}", exc_info=True)
            raise ConnectionError(f"Failed to synthesize speech: {e}")
