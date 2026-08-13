# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy model engines for Nemotron ASR and Magpie TTS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
import threading
import tempfile
from typing import AsyncIterator, Dict, Protocol, Tuple
import wave

import numpy as np

from openai_speech.config import Settings

logger = logging.getLogger(__name__)

MAGPIE_SAMPLE_RATE = 22050
MAGPIE_VOICES: Dict[str, int] = {
    "aria": 0,
    "jason": 1,
    "john": 2,
    "leo": 3,
    "sofia": 4,
    # Friendly OpenAI-style aliases keep existing clients usable.
    "alloy": 0,
    "echo": 1,
    "fable": 2,
    "onyx": 3,
    "nova": 4,
    "shimmer": 4,
}


class ASREngine(Protocol):
    """Interface consumed by the REST and Realtime layers."""

    async def load(self) -> None:
        """Load model weights."""

    async def transcribe(self, audio: bytes, filename: str, language: str) -> str:
        """Transcribe uploaded audio."""

    async def create_stream(self, language: str) -> "ASRRealtimeStream":
        """Create one cache-aware streaming utterance decoder."""


class ASRRealtimeStream(Protocol):
    """One native streaming ASR utterance."""

    async def send_audio(self, pcm16: bytes) -> None:
        """Append 16 kHz mono PCM16 audio."""

    async def finish(self) -> None:
        """Signal end of utterance and pad the final model chunk."""

    async def events(self) -> AsyncIterator[Tuple[str, str]]:
        """Yield ``delta``, ``completed``, or ``error`` events."""

    async def close(self) -> None:
        """Release the decoder thread."""


class TTSEngine(Protocol):
    """Interface consumed by the REST layer."""

    async def load(self) -> None:
        """Load model weights."""

    async def synthesize(
        self, text: str, voice: str, language: str, response_format: str
    ) -> Tuple[bytes, str]:
        """Synthesize speech and return bytes plus MIME type."""


@dataclass
class NemotronASREngine:
    """Transformers inference for Nemotron 3.5 streaming ASR."""

    settings: Settings

    def __post_init__(self) -> None:
        self._processor = None
        self._model = None
        self._load_lock = asyncio.Lock()
        # Both REST inference and native streaming execute in worker threads.
        # A threading lock serializes access to the mutable Transformers RNNT
        # generation state without blocking the asyncio event loop.
        self._inference_lock = threading.Lock()

    async def load(self) -> None:
        """Load the processor and RNNT model once."""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from transformers import AutoModelForRNNT, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "ASR dependencies are missing; install mmas-openai-speech[asr]"
            ) from exc

        model_kwargs = {}
        revision_kwargs = {}
        if self.settings.device == "auto":
            model_kwargs["device_map"] = "auto"
        if self.settings.dtype != "auto":
            model_kwargs["dtype"] = self.settings.dtype
        if self.settings.model_revision:
            revision_kwargs["revision"] = self.settings.model_revision
        logger.info("Loading ASR model %s", self.settings.model_id)
        self._processor = AutoProcessor.from_pretrained(self.settings.model_id, **revision_kwargs)
        self._model = AutoModelForRNNT.from_pretrained(
            self.settings.model_id, **revision_kwargs, **model_kwargs
        )
        if self.settings.device != "auto":
            self._model.to(self.settings.device)
        self._model.eval()
        try:
            self._processor.set_num_lookahead_tokens(
                self.settings.realtime_lookahead_tokens
            )
        except ValueError as exc:
            raise RuntimeError(
                "Unsupported SPEECH_REALTIME_LOOKAHEAD_TOKENS value "
                f"{self.settings.realtime_lookahead_tokens}: {exc}"
            ) from exc

    async def transcribe(self, audio: bytes, filename: str, language: str) -> str:
        """Run one file transcription without blocking the event loop."""
        await self.load()
        return await asyncio.to_thread(
            self._transcribe_locked_sync,
            audio,
            filename,
            language,
        )

    def _transcribe_locked_sync(self, audio: bytes, filename: str, language: str) -> str:
        with self._inference_lock:
            return self._transcribe_sync(audio, filename, language)

    async def create_stream(self, language: str) -> "NemotronRealtimeStream":
        """Create a native cache-aware streaming decoder."""
        await self.load()
        return NemotronRealtimeStream(self, language or "auto")

    def _transcribe_sync(self, audio: bytes, filename: str, language: str) -> str:
        import torch
        from transformers.audio_utils import load_audio

        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as uploaded:
            uploaded.write(audio)
            uploaded.flush()
            sample_rate = self._processor.feature_extractor.sampling_rate
            waveform = load_audio(uploaded.name, sampling_rate=sample_rate)

        inputs = self._processor(
            waveform,
            sampling_rate=sample_rate,
            language=language or "auto",
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        with torch.inference_mode():
            output = self._model.generate(**inputs, return_dict_in_generate=True)
        decoded = self._processor.decode(output.sequences, skip_special_tokens=True)
        # Transformers 5.13 returned a string for this model, while 5.14 may
        # return a one-item list for batched RNNT sequences.
        if isinstance(decoded, list):
            decoded = decoded[0] if decoded else ""
        return str(decoded).strip()


class _StreamingPCMBuffer:
    """Thread-safe PCM store consumed by the Transformers feature generator."""

    def __init__(self) -> None:
        self._pcm = bytearray()
        self._finished = False
        self._condition = threading.Condition()

    def append(self, pcm16: bytes) -> None:
        with self._condition:
            self._pcm.extend(pcm16[: len(pcm16) - (len(pcm16) % 2)])
            self._condition.notify_all()

    def finish(self) -> None:
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def samples(self, start: int, end: int, *, processed_end: int) -> np.ndarray | None:
        """Wait for a full slice, padding only the final partially-new chunk."""
        with self._condition:
            while len(self._pcm) // 2 < end and not self._finished:
                self._condition.wait()
            available = len(self._pcm) // 2
            if self._finished and available <= processed_end:
                return None
            stop = min(end, available)
            raw = bytes(self._pcm[start * 2 : stop * 2])
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) < end - start:
            samples = np.pad(samples, (0, end - start - len(samples)))
        return samples


class _RNNTDeltaStreamer:
    """Decode append-only RNNT tokens and publish transcript deltas."""

    def __init__(self, processor, callback) -> None:
        self.processor = processor
        self.callback = callback
        self.token_ids: list[int] = []
        self.text = ""

    def put(self, value) -> None:
        self.token_ids.extend(int(token) for token in value.reshape(-1).tolist())
        decoded = self.processor.decode(
            self.token_ids,
            skip_special_tokens=True,
        )
        if isinstance(decoded, list):
            decoded = decoded[0] if decoded else ""
        decoded = str(decoded)
        if decoded.startswith(self.text) and len(decoded) > len(self.text):
            delta = decoded[len(self.text) :]
            self.text = decoded
            self.callback("delta", delta)

    def end(self) -> None:
        return None


class NemotronRealtimeStream:
    """Live Transformers generator for one Nemotron ASR utterance."""

    def __init__(self, engine: NemotronASREngine, language: str) -> None:
        self.engine = engine
        self.language = language
        self._buffer = _StreamingPCMBuffer()
        self._loop = asyncio.get_running_loop()
        self._events: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="nemotron-realtime-asr",
            daemon=True,
        )
        self._thread.start()

    async def send_audio(self, pcm16: bytes) -> None:
        if not self._closed:
            self._buffer.append(pcm16)

    async def finish(self) -> None:
        self._buffer.finish()

    async def events(self) -> AsyncIterator[Tuple[str, str]]:
        while True:
            event = await self._events.get()
            yield event
            if event[0] in {"completed", "error"}:
                break

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.finish()
        await asyncio.to_thread(self._thread.join, 5.0)

    def _publish(self, kind: str, text: str) -> None:
        self._loop.call_soon_threadsafe(self._events.put_nowait, (kind, text))

    def _feature_generator(self, first_inputs):
        processor = self.engine._processor
        yield first_inputs.input_features[
            :, : processor.num_mel_frames_first_audio_chunk, :
        ]
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        processed_end = processor.num_samples_first_audio_chunk

        while True:
            start = mel_frame_idx * hop_length - n_fft // 2
            end = start + processor.num_samples_per_audio_chunk
            samples = self._buffer.samples(
                start,
                end,
                processed_end=processed_end,
            )
            if samples is None:
                break
            inputs = processor(
                samples,
                sampling_rate=processor.feature_extractor.sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=self.language,
                return_tensors="pt",
            ).to(self.engine._model.device, dtype=self.engine._model.dtype)
            yield inputs.input_features
            processed_end = end
            mel_frame_idx += processor.num_mel_frames_per_audio_chunk

    def _run(self) -> None:
        try:
            processor = self.engine._processor
            first_count = processor.num_samples_first_audio_chunk
            first_audio = self._buffer.samples(
                0,
                first_count,
                processed_end=0,
            )
            if first_audio is None:
                self._publish("completed", "")
                return
            first_inputs = processor(
                first_audio,
                sampling_rate=processor.feature_extractor.sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=True,
                language=self.language,
                return_tensors="pt",
            ).to(self.engine._model.device, dtype=self.engine._model.dtype)
            streamer = _RNNTDeltaStreamer(processor, self._publish)
            generate_kwargs = dict(first_inputs)
            generate_kwargs["input_features"] = self._feature_generator(first_inputs)
            generate_kwargs["streamer"] = streamer
            with self.engine._inference_lock:
                import torch

                with torch.inference_mode():
                    output = self.engine._model.generate(**generate_kwargs)
            decoded = self.engine._processor.decode(
                output.sequences,
                skip_special_tokens=True,
            )
            if isinstance(decoded, list):
                decoded = decoded[0] if decoded else ""
            self._publish("completed", str(decoded).strip())
        except Exception as exc:
            logger.exception("Native Realtime ASR failed")
            self._publish("error", str(exc))


@dataclass
class MagpieTTSEngine:
    """NeMo Speech inference for the public Magpie multilingual checkpoint."""

    settings: Settings

    def __post_init__(self) -> None:
        self._model = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load Magpie once and move it to the selected accelerator."""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            import torch
            from nemo.collections.tts.models import MagpieTTSModel
        except ImportError as exc:
            raise RuntimeError(
                "TTS dependencies are missing; install mmas-openai-speech[tts]"
            ) from exc

        logger.info("Loading TTS model %s", self.settings.model_id)
        if self.settings.model_revision:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id=self.settings.model_id,
                filename="magpie_tts_multilingual_357m.nemo",
                revision=self.settings.model_revision,
            )
            self._model = MagpieTTSModel.restore_from(checkpoint)
        else:
            self._model = MagpieTTSModel.from_pretrained(self.settings.model_id)
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()

    async def synthesize(
        self, text: str, voice: str, language: str, response_format: str
    ) -> Tuple[bytes, str]:
        """Run one utterance synthesis without blocking the event loop."""
        await self.load()
        async with self._inference_lock:
            return await asyncio.to_thread(
                self._synthesize_sync, text, voice, language, response_format
            )

    def _synthesize_sync(
        self, text: str, voice: str, language: str, response_format: str
    ) -> Tuple[bytes, str]:
        speaker_index = MAGPIE_VOICES.get(voice.lower())
        if speaker_index is None:
            supported = ", ".join(sorted({name.title() for name in MAGPIE_VOICES}))
            raise ValueError(f"Unsupported voice '{voice}'. Supported voices: {supported}")
        language = (language or "en").split("-")[0].lower()
        audio, audio_len = self._model.do_tts(
            text,
            language=language,
            apply_TN=True,
            speaker_index=speaker_index,
        )
        length = int(audio_len.reshape(-1)[0].item())
        samples = audio.reshape(-1)[:length].detach().float().cpu().numpy()
        pcm = _float_audio_to_pcm16(samples)
        if response_format == "pcm":
            return pcm, "audio/pcm"
        if response_format == "wav":
            return _pcm16_to_wav(pcm, MAGPIE_SAMPLE_RATE), "audio/wav"
        raise ValueError("Magpie local service currently supports response_format 'pcm' or 'wav'")


def _float_audio_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert normalized floating-point audio into little-endian PCM16."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap mono PCM16 bytes in a WAV container."""
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()
