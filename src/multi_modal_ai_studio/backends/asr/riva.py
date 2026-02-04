"""
NVIDIA Riva ASR Backend

Implements ASR using NVIDIA Riva's gRPC streaming API.
Adapted from live-riva-webui with timeline event support.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

import riva.client
import riva.client.proto.riva_asr_pb2 as rasr

from multi_modal_ai_studio.backends.base import (
    ASRBackend,
    ASRResult,
    ConnectionError,
    ConfigError,
)
from multi_modal_ai_studio.config.schema import ASRConfig

logger = logging.getLogger(__name__)


class RivaASRBackend(ASRBackend):
    """NVIDIA Riva ASR backend with streaming support.
    
    Features:
    - Streaming recognition with partial and final results
    - Runtime VAD tuning (start/stop thresholds, timeouts)
    - Automatic punctuation
    - Confidence scores
    """
    
    def __init__(self, config: ASRConfig):
        """Initialize Riva ASR backend.
        
        Args:
            config: ASRConfig instance with Riva settings
        
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
            self.asr_service = riva.client.ASRService(self.auth)
            self.logger.info(f"Initialized Riva ASR: {self.riva_server}")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Riva ASR: {e}")
        
        # Streaming state
        self.audio_queue: Optional[asyncio.Queue] = None
        self.stream_task: Optional[asyncio.Task] = None
        self._results_queue: Optional[asyncio.Queue] = None
    
    def _create_streaming_config(
        self,
        sample_rate: int = 16000
    ) -> riva.client.StreamingRecognitionConfig:
        """Create Riva streaming recognition config with VAD tuning.
        
        Args:
            sample_rate: Audio sample rate in Hz
        
        Returns:
            StreamingRecognitionConfig with VAD parameters
        """
        # Create EndpointingConfig for runtime VAD tuning
        endpointing_config = rasr.EndpointingConfig(
            start_history=int(self.config.speech_timeout_ms / 2),  # Pre-speech padding
            start_threshold=self.config.vad_start_threshold,        # Start detection sensitivity
            stop_history=self.config.speech_timeout_ms,             # Silence window for end detection
            stop_threshold=self.config.vad_stop_threshold,          # End detection sensitivity
        )
        
        self.logger.info(
            f"VAD config: start_threshold={self.config.vad_start_threshold}, "
            f"stop_threshold={self.config.vad_stop_threshold}, "
            f"timeout={self.config.speech_timeout_ms}ms"
        )
        
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.config.language,
                model=self.config.model if self.config.model != "conformer" else None,
                max_alternatives=1,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                verbatim_transcripts=False,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
                endpointing_config=endpointing_config,  # Runtime VAD tuning!
            ),
            interim_results=True,  # Send partial results as they're recognized
        )
        return config
    
    async def start_stream(self) -> None:
        """Start streaming recognition session.
        
        Raises:
            ConnectionError: If unable to connect to Riva
        """
        if self.audio_queue is not None:
            raise RuntimeError("Stream already started")
        
        self.audio_queue = asyncio.Queue()
        self._results_queue = asyncio.Queue()
        
        # Start background task to stream audio to Riva
        self.stream_task = asyncio.create_task(self._stream_to_riva())
        
        self.logger.info("Riva ASR stream started")
    
    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio chunk for recognition.
        
        Args:
            audio_chunk: Raw PCM audio bytes (16kHz, 16-bit, mono)
        
        Raises:
            RuntimeError: If stream not started
        """
        if self.audio_queue is None:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        await self.audio_queue.put(audio_chunk)
    
    async def receive_results(self) -> AsyncIterator[ASRResult]:
        """Yield recognition results as they become available.
        
        Yields:
            ASRResult: Transcription results (both partial and final)
        """
        if self._results_queue is None:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        while True:
            result = await self._results_queue.get()
            
            # None signals end of stream
            if result is None:
                break
            
            yield result
    
    async def stop_stream(self) -> None:
        """Stop streaming recognition session."""
        if self.audio_queue is None:
            return
        
        # Signal end of stream
        await self.audio_queue.put(None)
        
        # Wait for stream task to finish
        if self.stream_task:
            try:
                await asyncio.wait_for(self.stream_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Stream task timeout, cancelling")
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass
        
        self.audio_queue = None
        self._results_queue = None
        self.stream_task = None
        
        self.logger.info("Riva ASR stream stopped")
    
    async def _stream_to_riva(self) -> None:
        """Background task to stream audio to Riva and receive results."""
        
        config = self._create_streaming_config()
        
        def audio_chunk_generator():
            """Generator for Riva streaming (runs in thread pool)"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while True:
                try:
                    # Get audio chunk from queue (blocking)
                    chunk = asyncio.run_coroutine_threadsafe(
                        self.audio_queue.get(), asyncio.get_running_loop()
                    ).result(timeout=10.0)
                    
                    if chunk is None:
                        # End of stream
                        break
                    
                    yield chunk
                
                except Exception as e:
                    self.logger.error(f"Error in audio generator: {e}")
                    break
        
        try:
            # Call Riva streaming API (blocking call, runs in thread pool)
            loop = asyncio.get_running_loop()
            responses = await loop.run_in_executor(
                None,
                lambda: self.asr_service.streaming_response_generator(
                    audio_chunks=audio_chunk_generator(),
                    streaming_config=config,
                )
            )
            
            # Process responses
            for response in responses:
                if not response.results:
                    continue
                
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    alternative = result.alternatives[0]
                    text = alternative.transcript.strip()
                    
                    if not text:
                        continue
                    
                    # Create ASRResult
                    asr_result = ASRResult(
                        text=text,
                        is_final=result.is_final,
                        confidence=alternative.confidence if alternative.confidence > 0 else 1.0,
                        metadata={
                            "stability": alternative.stability if hasattr(alternative, "stability") else 1.0,
                            "language_code": self.config.language,
                        }
                    )
                    
                    # Send to results queue
                    await self._results_queue.put(asr_result)
                    
                    if result.is_final:
                        self.logger.info(f"Final transcript: {text} (confidence: {asr_result.confidence:.2f})")
                    else:
                        self.logger.debug(f"Partial transcript: {text}")
        
        except Exception as e:
            self.logger.error(f"Riva streaming error: {e}", exc_info=True)
        
        finally:
            # Signal end of results
            if self._results_queue:
                await self._results_queue.put(None)
