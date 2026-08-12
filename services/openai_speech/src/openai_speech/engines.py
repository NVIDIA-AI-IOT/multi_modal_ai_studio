# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy model engines for Nemotron ASR and Magpie TTS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
import tempfile
from typing import Dict, Protocol, Tuple
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
    """Interface consumed by the REST layer."""

    async def load(self) -> None:
        """Load model weights."""

    async def transcribe(self, audio: bytes, filename: str, language: str) -> str:
        """Transcribe uploaded audio."""


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
        self._inference_lock = asyncio.Lock()

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

    async def transcribe(self, audio: bytes, filename: str, language: str) -> str:
        """Run one file transcription without blocking the event loop."""
        await self.load()
        async with self._inference_lock:
            return await asyncio.to_thread(self._transcribe_sync, audio, filename, language)

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
