#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Thin OpenAI Speech adapter for the standalone TensorRT Edge-LLM TTS CLI."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Optional, Tuple
import wave

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SPEAKERS = {
    "ryan",
    "serena",
    "aiden",
    "vivian",
    "dylan",
    "eric",
    "uncle_fu",
    "ono_anna",
    "sohee",
}


class SpeechRequest(BaseModel):
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "ono_anna"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: Optional[str] = None


@dataclass(frozen=True)
class AdapterSettings:
    workspace: Path
    runtime_image: str = "mmas/tensorrt-edgellm-dev:v0.9.1-cu130"
    sequence_length: int = 2048
    chunk_frames: int = 0

    @classmethod
    def from_env(cls) -> "AdapterSettings":
        workspace = Path(
            os.getenv(
                "EDGE_LLM_WORKSPACE",
                str(Path.home() / "tensorrt-edgellm-workspace"),
            )
        ).resolve()
        return cls(
            workspace=workspace,
            runtime_image=os.getenv(
                "EDGE_LLM_RUNTIME_IMAGE",
                "mmas/tensorrt-edgellm-dev:v0.9.1-cu130",
            ),
            sequence_length=int(os.getenv("EDGE_LLM_SEQUENCE_LENGTH", "2048")),
            chunk_frames=int(os.getenv("EDGE_LLM_CHUNK_FRAMES", "0")),
        )


class EdgeLLMEngine:
    """Serialize subprocess access to the upstream batch=1 standalone runtime."""

    def __init__(self, settings: AdapterSettings):
        self.settings = settings
        self._lock = asyncio.Lock()

    def validate(self) -> None:
        if self.settings.sequence_length not in {1024, 2048, 4096}:
            raise RuntimeError("EDGE_LLM_SEQUENCE_LENGTH must be 1024, 2048, or 4096")
        if self.settings.chunk_frames < 0:
            raise RuntimeError("EDGE_LLM_CHUNK_FRAMES must be zero or positive")
        expected = [
            self.settings.workspace / "build-v0.9.1/examples/omni/qwen3_tts_inference",
            self.settings.workspace
            / f"engines-fp16/mxsl{self.settings.sequence_length}/talker/llm.engine",
            self.settings.workspace
            / f"engines-fp16/mxsl{self.settings.sequence_length}/code_predictor/llm.engine",
            self.settings.workspace
            / f"engines-fp16/mxsl{self.settings.sequence_length}/code2wav/code2wav.engine",
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError("TensorRT Edge-LLM artifacts are missing: " + ", ".join(missing))

    async def synthesize(self, text: str, voice: str, response_format: str) -> Tuple[bytes, str]:
        async with self._lock:
            return await asyncio.to_thread(
                self._synthesize_sync,
                text,
                voice,
                response_format,
            )

    def _synthesize_sync(self, text: str, voice: str, response_format: str) -> Tuple[bytes, str]:
        workspace = self.settings.workspace
        temporary_root = workspace / "server-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="request-", dir=temporary_root) as tmp:
            request_dir = Path(tmp)
            relative = request_dir.relative_to(workspace)
            container_dir = Path("/workspace") / relative
            input_path = request_dir / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "batch_size": 1,
                        "speaker": voice,
                        "max_audio_length": 512,
                        "requests": [
                            {
                                "messages": [
                                    {
                                        "role": "assistant",
                                        "content": text,
                                    }
                                ]
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            engine_base = (
                f"/workspace/engines-fp16/mxsl{self.settings.sequence_length}"
            )
            command = [
                "docker",
                "run",
                "--rm",
                "--runtime",
                "nvidia",
                "--network",
                "none",
                "--ipc",
                "host",
                "--ulimit",
                "memlock=-1",
                "--ulimit",
                "stack=67108864",
                "-v",
                f"{workspace}:/workspace",
                self.settings.runtime_image,
                "/workspace/build-v0.9.1/examples/omni/qwen3_tts_inference",
                f"--talkerEngineDir={engine_base}/talker",
                f"--code2wavEngineDir={engine_base}/code2wav",
                f"--tokenizerDir={engine_base}/talker",
                f"--inputFile={container_dir}/input.json",
                f"--outputFile={container_dir}/output.json",
                f"--outputAudioDir={container_dir}/audio",
                "--batchSize=1",
            ]
            if self.settings.chunk_frames:
                command.extend(
                    [
                        "--streaming",
                        f"--chunkFrames={self.settings.chunk_frames}",
                    ]
                )
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    "qwen3_tts_inference failed: " + completed.stdout[-2000:]
                )
            wav_path = request_dir / "audio/audio_req0.wav"
            if not wav_path.is_file():
                raise RuntimeError("qwen3_tts_inference did not produce audio_req0.wav")
            wav_bytes = wav_path.read_bytes()
            if response_format == "wav":
                return wav_bytes, "audio/wav"
            if response_format == "pcm":
                with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
                    if (
                        wav_file.getnchannels(),
                        wav_file.getsampwidth(),
                        wav_file.getframerate(),
                    ) != (1, 2, 24000):
                        raise RuntimeError("unexpected Edge-LLM WAV format")
                    return wav_file.readframes(wav_file.getnframes()), "audio/pcm"
            raise ValueError("response_format must be 'pcm' or 'wav'")


def create_app(
    settings: Optional[AdapterSettings] = None,
    *,
    engine: Optional[EdgeLLMEngine] = None,
) -> FastAPI:
    settings = settings or AdapterSettings.from_env()
    engine = engine or EdgeLLMEngine(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.validate()
        yield

    app = FastAPI(
        title="MMAS Qwen3-TTS TensorRT Edge-LLM adapter",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "precision": "fp16",
            "batch_size": 1,
            "sequence_length": settings.sequence_length,
            "chunk_frames": settings.chunk_frames,
        }

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "qwen"}],
        }

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if request.model not in {MODEL_ID, "tts-1", "tts-1-hd"}:
            raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'")
        voice = request.voice.strip().lower()
        if voice not in SPEAKERS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported voice '{request.voice}'",
            )
        if request.speed != 1.0:
            raise HTTPException(
                status_code=400,
                detail="TensorRT Edge-LLM Qwen3-TTS supports speed=1.0 only",
            )
        try:
            audio, media_type = await engine.synthesize(
                request.input,
                voice,
                request.response_format,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            audio,
            media_type=media_type,
            headers={
                "X-Model": MODEL_ID,
                "X-Request-Timestamp": str(int(time.time())),
                "X-Edge-LLM-Sequence-Length": str(settings.sequence_length),
            },
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
