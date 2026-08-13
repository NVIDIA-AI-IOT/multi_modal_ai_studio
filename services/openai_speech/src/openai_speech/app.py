# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI application implementing OpenAI-compatible speech endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from openai_speech.config import Settings
from openai_speech.engines import (
    ASREngine,
    MagpieTTSEngine,
    NemotronASREngine,
    TTSEngine,
)
from openai_speech.realtime import RealtimeTranscriptionConnection


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech generation request."""

    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "Sofia"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: Optional[str] = None
    # Extension understood by local multilingual services.
    language: str = "en"


def create_app(
    settings: Optional[Settings] = None,
    *,
    asr_engine: Optional[ASREngine] = None,
    tts_engine: Optional[TTSEngine] = None,
) -> FastAPI:
    """Create an ASR-mode or TTS-mode application."""
    settings = settings or Settings.from_env()
    if settings.mode == "asr":
        asr_engine = asr_engine or NemotronASREngine(settings)
    else:
        tts_engine = tts_engine or MagpieTTSEngine(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.eager_load:
            engine = asr_engine if settings.mode == "asr" else tts_engine
            await engine.load()
        yield

    service = FastAPI(
        title=f"MMAS OpenAI-compatible {settings.mode.upper()}",
        version="0.2.0",
        lifespan=lifespan,
    )

    @service.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "service": settings.mode,
            "model": settings.model_id,
            "model_revision": settings.model_revision,
        }

    @service.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nvidia",
                    "revision": settings.model_revision,
                }
            ],
        }

    @service.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()],
        language: Annotated[str, Form()] = "auto",
        response_format: Annotated[str, Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
    ):
        if settings.mode != "asr":
            raise HTTPException(status_code=404, detail="ASR endpoint is disabled")
        if model not in {settings.model_id, "whisper-1"}:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model}'")
        if response_format not in {"json", "text"}:
            raise HTTPException(
                status_code=400,
                detail="response_format must be 'json' or 'text'",
            )
        if temperature != 0:
            raise HTTPException(
                status_code=400,
                detail="This deterministic RNNT backend only supports temperature=0",
            )
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        try:
            text = await asr_engine.transcribe(payload, file.filename or "audio.wav", language)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"ASR model inference failed: {exc}",
            ) from exc
        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    @service.websocket("/v1/realtime")
    async def realtime_transcription(websocket: WebSocket, model: Optional[str] = None):
        if settings.mode != "asr":
            await websocket.accept()
            await websocket.close(code=1008, reason="ASR endpoint is disabled")
            return
        requested_model = model or settings.model_id
        if requested_model != settings.model_id:
            await websocket.accept()
            await websocket.close(code=1008, reason=f"Unknown model '{requested_model}'")
            return
        connection = RealtimeTranscriptionConnection(
            websocket,
            asr_engine,
            settings.model_id,
        )
        await connection.run()

    @service.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if settings.mode != "tts":
            raise HTTPException(status_code=404, detail="TTS endpoint is disabled")
        if request.model not in {settings.model_id, "tts-1", "tts-1-hd"}:
            raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'")
        if request.speed != 1.0:
            raise HTTPException(
                status_code=400,
                detail="Magpie local service currently supports speed=1.0 only",
            )
        try:
            audio, media_type = await tts_engine.synthesize(
                request.input,
                request.voice,
                request.language,
                request.response_format,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"TTS model inference failed: {exc}",
            ) from exc
        return Response(
            content=audio,
            media_type=media_type,
            headers={
                "X-Model": settings.model_id,
                "X-Request-Timestamp": str(int(time.time())),
            },
        )

    return service


app = create_app()
