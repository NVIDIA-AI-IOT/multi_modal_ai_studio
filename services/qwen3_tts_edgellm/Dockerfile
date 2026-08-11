# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Build-only/runtime environment for TensorRT Edge-LLM on Jetson Thor.

ARG BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm@sha256:08ec0a116a09e50b74e26eca91f52bfe2c95977f0eb4cb3e079cfb4b3c3932c6
FROM ${BASE_IMAGE}

ARG LIBC6_DEV_VERSION=2.39-0ubuntu8.8
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "libc6-dev=${LIBC6_DEV_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/tensorrt/include /opt/tensorrt/lib \
    && cp -a /usr/include/Nv*.h /opt/tensorrt/include/ \
    && for library in \
        /usr/lib/aarch64-linux-gnu/libnvinfer.so* \
        /usr/lib/aarch64-linux-gnu/libnvonnxparser.so*; do \
        ln -s "${library}" "/opt/tensorrt/lib/$(basename "${library}")"; \
    done
