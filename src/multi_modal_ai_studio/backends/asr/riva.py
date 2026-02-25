"""
NVIDIA Riva ASR Backend

Implements ASR using NVIDIA Riva's gRPC streaming API.
Adapted from live-riva-webui with timeline event support.
"""

import asyncio
import logging
import queue
import time
from typing import AsyncIterator, List, Optional

import riva.client
import riva.client.proto.riva_asr_pb2 as rasr

from multi_modal_ai_studio.backends.base import (
    ASRBackend,
    ASRResult,
    ConnectionError,
    ConfigError,
)
from multi_modal_ai_studio.config.schema import ASRConfig
from multi_modal_ai_studio.core.timeline import Timeline, Lane

logger = logging.getLogger(__name__)

# Default ASR model when config does not set one (Silero VAD; same as Live RIVA WebUI).
DEFAULT_ASR_MODEL = "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"


def list_riva_asr_models_sync(server: str, use_ssl: bool = False, ssl_cert: Optional[str] = None) -> List[str]:
    """Query Riva server for available ASR model names (sync, for use from thread/executor).
    Returns empty list on error or if the RPC is not supported."""
    try:
        auth = riva.client.Auth(ssl_cert, use_ssl, server)
        asr_service = riva.client.ASRService(auth)
        stub = getattr(asr_service, "stub", None)
        if stub is None:
            return []
        req = rasr.RivaSpeechRecognitionConfigRequest()
        resp = stub.GetRivaSpeechRecognitionConfig(req, metadata=auth.get_auth_metadata())
        # Response has repeated model_config, each with model_name
        configs = getattr(resp, "model_config", None) or []
        names = [str(getattr(c, "model_name", "") or "").strip() for c in configs]
        return [n for n in names if n]
    except Exception as e:
        logger.debug("Failed to list Riva ASR models: %s", e)
        return []


class RivaASRBackend(ASRBackend):
    """NVIDIA Riva ASR backend with streaming support.

    Features:
    - Streaming recognition with partial and final results
    - Runtime VAD tuning (start/stop thresholds, timeouts)
    - Automatic punctuation
    - Confidence scores
    """

    def __init__(self, config: ASRConfig, timeline: Optional[Timeline] = None):
        """Initialize Riva ASR backend.

        Args:
            config: ASRConfig instance with Riva settings
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
            self.asr_service = riva.client.ASRService(self.auth)
            self.logger.info(f"Initialized Riva ASR: {self.riva_server}")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Riva ASR: {e}")

        # Streaming state (sync queue so thread-pool generator can block on get)
        self._sync_audio_queue: Optional[queue.Queue] = None
        self.stream_task: Optional[asyncio.Task] = None
        self._results_queue: Optional[asyncio.Queue] = None

        # VAD/ASR segment tracking for rectangle rendering
        self._speech_start_time: Optional[float] = None
        self._vad_start_time: Optional[float] = None
        # Dedupe repeated asr_final (Riva sometimes sends same final twice)
        self._last_final_text: Optional[str] = None

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
        # Create EndpointingConfig for runtime VAD tuning (Live RIVA WebUI-style)
        endpointing_config = rasr.EndpointingConfig(
            start_history=self.config.speech_pad_ms,                 # Speech pad (ms)
            start_threshold=self.config.vad_start_threshold,          # Start detection sensitivity
            stop_history=self.config.speech_timeout_ms,               # Silence duration (ms) before end
            stop_threshold=self.config.vad_stop_threshold,            # End detection sensitivity
        )

        self.logger.info(
            f"VAD config: speech_pad={self.config.speech_pad_ms}ms, "
            f"silence_duration={self.config.speech_timeout_ms}ms, "
            f"threshold={self.config.vad_start_threshold}"
        )

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.config.language,
                model=self._riva_model_for_config(),
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

    def _riva_model_for_config(self) -> Optional[str]:
        """Return model name for Riva API. Defaults to Silero VAD model to match Live RIVA WebUI."""
        m = (self.config.model or "").strip()
        if not m or m.lower() in ("conformer", "default", "parakeet 1.1b", "parakeet"):
            return DEFAULT_ASR_MODEL
        return m

    async def start_stream(self) -> None:
        """Start streaming recognition session.

        Raises:
            ConnectionError: If unable to connect to Riva
        """
        if self._sync_audio_queue is not None:
            raise RuntimeError("Stream already started")

        self._sync_audio_queue = queue.Queue()
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
        if self._sync_audio_queue is None:
            raise RuntimeError("Stream not started. Call start_stream() first.")

        self._sync_audio_queue.put(audio_chunk)

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
        if self._sync_audio_queue is None:
            return

        # Signal end of stream (thread-safe; unblocks generator in executor)
        self._sync_audio_queue.put(None)

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

        self._sync_audio_queue = None
        self._results_queue = None
        self.stream_task = None

        self.logger.info("Riva ASR stream stopped")

    # Pre-buffer size for the FIRST audio chunk only (bytes).
    # Browser sends tiny PCM chunks (~4096 B = 128ms).  Accumulating ~300ms
    # before the first yield gives Riva enough initial context to produce a
    # more stable first partial.  After the first yield, chunks stream
    # immediately with no added latency.
    _PREBUFFER_BYTES = 9600  # ~300ms at 16kHz 16-bit mono

    async def _stream_to_riva(self) -> None:
        """Background task to stream audio to Riva and receive results.

        First audio chunk is pre-buffered for stability; subsequent chunks
        stream immediately.  Results are pushed directly into the async
        _results_queue via loop.call_soon_threadsafe (no intermediate polling).
        """
        config = self._create_streaming_config()
        loop = asyncio.get_running_loop()
        rq = self._results_queue
        _last_partial_text: list = [""]  # mutable container for dedup across calls

        def audio_chunk_generator():
            """Pre-buffer first chunk, then yield immediately."""
            q = self._sync_audio_queue
            if not q:
                return

            # Phase 1: accumulate initial audio for a stable first partial
            prebuf = bytearray()
            while len(prebuf) < self._PREBUFFER_BYTES:
                try:
                    chunk = q.get(timeout=10.0)
                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error("Error in audio generator: %s", e)
                    if prebuf:
                        yield bytes(prebuf)
                    return
                if chunk is None:
                    if prebuf:
                        yield bytes(prebuf)
                    return
                prebuf.extend(chunk)
            if prebuf:
                yield bytes(prebuf)

            # Phase 2: stream remaining chunks immediately (no buffering)
            while True:
                try:
                    chunk = q.get(timeout=10.0)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error("Error in audio generator: %s", e)
                    break

        def _enqueue(item):
            """Thread-safe push into the async results queue."""
            loop.call_soon_threadsafe(rq.put_nowait, item)

        def process_response(response):
            """Extract ASRResults from one Riva response, dedup identical partials."""
            out = []
            if not response.results:
                return out
            if not hasattr(process_response, "_logged_first"):
                process_response._logged_first = True
                self.logger.info("Riva ASR first response: %d result(s)", len(response.results))
            for result in response.results:
                if not result.alternatives:
                    continue
                alternative = result.alternatives[0]
                text = alternative.transcript.strip()
                if not text:
                    continue

                is_final = result.is_final

                # Deduplicate consecutive identical partials
                if not is_final and text == _last_partial_text[0]:
                    continue
                if not is_final:
                    _last_partial_text[0] = text
                else:
                    _last_partial_text[0] = ""

                if self.timeline and self._speech_start_time is None:
                    event = self.timeline.add_event("vad_start", Lane.AUDIO)
                    self._speech_start_time = event.timestamp
                    self._vad_start_time = event.timestamp
                    self._last_final_text = None
                    self.logger.debug("Speech started at %.3fs", self._speech_start_time)
                asr_result = ASRResult(
                    text=text,
                    is_final=is_final,
                    confidence=alternative.confidence if alternative.confidence > 0 else 1.0,
                    metadata={
                        "stability": getattr(alternative, "stability", 1.0),
                        "language_code": self.config.language,
                    },
                )
                if is_final:
                    finals_emitted = getattr(process_response, "_finals_emitted", 0) + 1
                    process_response._finals_emitted = finals_emitted
                    self.logger.info(
                        "Final transcript #%d: %s (confidence=%.2f)", finals_emitted, text, asr_result.confidence
                    )
                    if self.timeline and self._speech_start_time is not None:
                        if text == self._last_final_text:
                            self.logger.debug("Skipping duplicate asr_final in timeline: %r", text[:50])
                            asr_result.metadata["event_timestamp"] = time.time() - (self.timeline.start_time or 0)
                            self._speech_start_time = None
                            self._vad_start_time = None
                        else:
                            self._last_final_text = text
                            vad_end_event = self.timeline.add_event("vad_end", Lane.AUDIO)
                            asr_final_event = self.timeline.add_event(
                                "asr_final", Lane.SPEECH,
                                data={"text": text, "confidence": asr_result.confidence},
                            )
                            speech_end_time = asr_final_event.timestamp
                            self.timeline.add_vad_segment(
                                self._vad_start_time, vad_end_event.timestamp, data={"confidence": 0.95}
                            )
                            self.timeline.add_asr_segment(
                                self._speech_start_time, speech_end_time, text=text,
                                data={"confidence": asr_result.confidence},
                            )
                            self._speech_start_time = None
                            self._vad_start_time = None
                            asr_result.metadata["event_timestamp"] = asr_final_event.timestamp
                else:
                    if not getattr(process_response, "_logged_partial", False):
                        process_response._logged_partial = True
                        self.logger.info("Riva ASR first partial: %r", text[:80])
                    self.logger.debug("Partial transcript: %s", text)
                out.append(asr_result)
            return out

        def run_riva_in_executor():
            """Run Riva stream in thread. Push results directly to async queue.
            Auto-reconnects on stream failure up to max_retries times.
            """
            max_retries = 5
            retry_delay = 1.0
            retry_count = 0

            while retry_count < max_retries:
                try:
                    if retry_count > 0:
                        self.logger.info("Riva ASR reconnecting (attempt %d/%d)...", retry_count + 1, max_retries)
                        import time as _time
                        _time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 10.0)

                    responses = self.asr_service.streaming_response_generator(
                        audio_chunks=audio_chunk_generator(),
                        streaming_config=config,
                    )
                    for response in responses:
                        for asr_result in process_response(response):
                            _enqueue(asr_result)
                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.error("Riva streaming error (attempt %d/%d): %s", retry_count, max_retries, e)
                    if retry_count >= max_retries:
                        self.logger.error("Riva ASR max retries reached, giving up")
                        break
                    q = self._sync_audio_queue
                    if q:
                        try:
                            while not q.empty():
                                item = q.get_nowait()
                                if item is None:
                                    return
                        except Exception:
                            pass

        try:
            await loop.run_in_executor(None, run_riva_in_executor)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error("Riva stream task error: %s", e, exc_info=True)
        finally:
            if self._results_queue:
                await self._results_queue.put(None)
