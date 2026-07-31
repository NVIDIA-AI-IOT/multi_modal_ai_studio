# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1
FROM ${BASE_IMAGE}

ARG FASTER_QWEN3_TTS_COMMIT=a70afc0f81f7f5f8801c3227968f1102f43f211c

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/andimarafioti/faster-qwen3-tts.git \
        /opt/faster-qwen3-tts \
    && git -C /opt/faster-qwen3-tts checkout --detach "${FASTER_QWEN3_TTS_COMMIT}" \
    && test "$(git -C /opt/faster-qwen3-tts rev-parse HEAD)" = \
        "${FASTER_QWEN3_TTS_COMMIT}" \
    && python3 -m pip install --no-cache-dir \
        "transformers==4.57.3" \
        "huggingface-hub==0.36.2" \
        "accelerate==1.12.0" \
        "fastapi==0.116.1" \
        "onnxruntime==1.28.0" \
        "soundfile==0.13.1" \
        "sox==1.5.0" \
        "uvicorn==0.35.0" \
    && python3 -m pip install --no-cache-dir --no-deps \
        "qwen-tts==0.1.1" \
        "/opt/faster-qwen3-tts" \
    && python3 -m pip install --no-cache-dir \
        "https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/resolve/main/whl/cu130/qwentts_cpp_python-0.3.1%2Bcu130-py3-none-manylinux_2_39_aarch64.whl"

COPY openai_server.py /opt/mmas/faster_qwen3_tts_openai_server.py
COPY torchaudio_stub /opt/mmas/torchaudio_stub
RUN chmod -R a+rX /opt/mmas

ENV PYTHONPATH=/opt/mmas/torchaudio_stub \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub \
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    QWEN_TTS_MODEL_REVISION=85e237c12c027371202489a0ec509ded67b5e4b5 \
    QWEN_TTS_SPEAKER=ono_anna \
    QWEN_TTS_BACKEND=cuda-graph \
    QWEN_TTS_QUANT=BF16 \
    QWEN_TTS_CHUNK_SIZE=2 \
    QWEN_TTS_PORT=18082

EXPOSE 18082

ENTRYPOINT ["python3", "/opt/mmas/faster_qwen3_tts_openai_server.py"]
