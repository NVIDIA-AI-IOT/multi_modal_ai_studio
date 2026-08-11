#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Strict OpenAI-compatible CustomVoice adapter for faster-qwen3-tts."""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import queue
import threading
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

LOGGER = logging.getLogger("mmas.faster_qwen3_tts")
SAMPLE_RATE = 24000


class SpeechRequest(BaseModel):
    """Subset of the OpenAI speech request plus an optional language hint."""

    model: str = "tts-1"
    input: str
    voice: str = "Ono_Anna"
    response_format: str = "pcm"
    speed: float = 1.0
    language: str | None = None


class StreamingRuntime(Protocol):
    """Small seam that keeps API tests independent of CUDA and model weights."""

    model_id: str
    speaker: str
    backend: str
    quant: str
    chunk_size: int

    def stream(self, text: str, language: str):
        """Yield ``(float32 audio, sample_rate, timing)`` tuples."""


def _pcm16(audio: np.ndarray) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    return np.clip(samples * 32768.0, -32768, 32767).astype("<i2").tobytes()


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap complete PCM in a standard seekable WAV container."""

    result = io.BytesIO()
    with wave.open(result, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return result.getvalue()


def _language_for(text: str, requested: str | None) -> str:
    if requested:
        normalized = requested.lower()
        if normalized in {"ja", "ja-jp", "japanese"}:
            return "Japanese"
        if normalized in {"en", "en-us", "en-gb", "english"}:
            return "English"
        raise HTTPException(status_code=400, detail="language must be Japanese or English")
    if any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in text):
        return "Japanese"
    return "English"


def _voice_matches(requested: str, configured: str) -> bool:
    def normalize(value: str) -> str:
        return value.lower().replace("-", "_").replace(" ", "_")

    return normalize(requested) == normalize(configured)


async def _audio_chunks(
    runtime: StreamingRuntime,
    text: str,
    language: str,
) -> AsyncIterator[bytes]:
    output: queue.Queue[object] = queue.Queue(maxsize=4)
    done = object()
    cancelled = threading.Event()

    def enqueue(item: object) -> bool:
        while not cancelled.is_set():
            try:
                output.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for chunk, _sample_rate, _timing in runtime.stream(text, language):
                if cancelled.is_set() or not enqueue(_pcm16(chunk)):
                    return
        except Exception as exc:  # noqa: BLE001 - propagate backend failures
            enqueue(exc)
        finally:
            enqueue(done)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            item = await asyncio.to_thread(output.get)
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        cancelled.set()


def create_app(runtime: StreamingRuntime) -> FastAPI:
    """Create an application around one already-loaded, single-stream runtime."""

    app = FastAPI(title="MMAS faster-qwen3-tts CustomVoice adapter")
    inference_lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": True,
            "model": runtime.model_id,
            "speaker": runtime.speaker,
            "backend": runtime.backend,
            "quant": runtime.quant,
            "chunk_size": runtime.chunk_size,
        }

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": runtime.model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest):
        if not request.input.strip():
            raise HTTPException(status_code=400, detail="input must not be empty")
        if request.model not in {runtime.model_id, "tts-1", "tts-1-hd"}:
            raise HTTPException(status_code=404, detail=f"unknown model {request.model!r}")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")
        if request.speed != 1.0:
            raise HTTPException(
                status_code=400,
                detail="speed is not implemented by faster-qwen3-tts; use 1.0",
            )
        if not _voice_matches(request.voice, runtime.speaker):
            raise HTTPException(
                status_code=400,
                detail=f"voice must be the configured CustomVoice speaker {runtime.speaker!r}",
            )

        language = _language_for(request.input, request.language)

        async def serialized_stream() -> AsyncIterator[bytes]:
            async with inference_lock:
                async for chunk in _audio_chunks(runtime, request.input, language):
                    yield chunk

        if request.response_format == "pcm":
            return StreamingResponse(serialized_stream(), media_type="audio/L16")

        async with inference_lock:
            chunks = [
                chunk
                async for chunk in _audio_chunks(runtime, request.input, language)
            ]
        return Response(
            _wav_bytes(b"".join(chunks), SAMPLE_RATE),
            media_type="audio/wav",
        )

    return app


@dataclass
class FasterRuntime:
    """Runtime wrapper for Torch/CUDA-graph or the experimental GGML adapter."""

    model_id: str
    speaker: str
    backend: str
    quant: str
    chunk_size: int
    model: object

    def stream(self, text: str, language: str):
        return self.model.generate_custom_voice_streaming(
            text=text,
            speaker=self.speaker,
            language=language,
            chunk_size=self.chunk_size,
            max_new_tokens=512,
        )

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        speaker: str,
        backend: str,
        quant: str,
        chunk_size: int,
        max_seq_len: int,
        model_revision: str | None,
        ggml_cache_dir: str,
        qwentts_library_path: str | None,
    ) -> FasterRuntime:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        resolved_model = model_id
        if model_revision and "/" in model_id:
            from huggingface_hub import snapshot_download

            resolved_model = snapshot_download(model_id, revision=model_revision)

        if backend == "cuda-graph":
            model = FasterQwen3TTS.from_pretrained(
                resolved_model,
                device="cuda",
                dtype=torch.bfloat16,
                attn_implementation="eager",
                max_seq_len=max_seq_len,
            )
        elif backend == "ggml":
            model = FasterQwen3TTS.from_pretrained(
                model_id,
                backend="ggml",
                quant=quant,
                cache_dir=ggml_cache_dir,
                qwentts_library_path=qwentts_library_path or None,
            )
        else:
            raise ValueError(f"unsupported backend: {backend}")

        supported = model.model.get_supported_speakers() if backend == "cuda-graph" else model.get_supported_speakers()
        if speaker.lower() not in {value.lower() for value in supported or []}:
            raise ValueError(f"speaker {speaker!r} unavailable; supported={supported}")
        return cls(model_id, speaker, backend, quant, chunk_size, model)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"))
    parser.add_argument("--model-revision", default=os.getenv("QWEN_TTS_MODEL_REVISION"))
    parser.add_argument("--speaker", default=os.getenv("QWEN_TTS_SPEAKER", "ono_anna"))
    parser.add_argument(
        "--backend",
        choices=("cuda-graph", "ggml"),
        default=os.getenv("QWEN_TTS_BACKEND", "cuda-graph"),
    )
    parser.add_argument("--quant", default=os.getenv("QWEN_TTS_QUANT", "BF16"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("QWEN_TTS_CHUNK_SIZE", "2")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.getenv("QWEN_TTS_MAX_SEQ_LEN", "2048")))
    parser.add_argument(
        "--ggml-cache-dir",
        default=os.getenv("QWEN_TTS_GGML_CACHE_DIR", "/models/qwentts"),
    )
    parser.add_argument(
        "--qwentts-library-path",
        default=os.getenv("QWEN_TTS_QWENTTS_LIBRARY_PATH"),
    )
    parser.add_argument("--host", default=os.getenv("QWEN_TTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QWEN_TTS_PORT", "18082")))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _arguments()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    runtime = FasterRuntime.load(
        model_id=args.model,
        speaker=args.speaker,
        backend=args.backend,
        quant=args.quant,
        chunk_size=args.chunk_size,
        max_seq_len=args.max_seq_len,
        model_revision=args.model_revision,
        ggml_cache_dir=args.ggml_cache_dir,
        qwentts_library_path=args.qwentts_library_path,
    )
    import uvicorn

    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
