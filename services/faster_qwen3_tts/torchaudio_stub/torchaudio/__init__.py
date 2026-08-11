# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Import shim for Qwen3-TTS 12 Hz on NVIDIA PyTorch CUDA 13 containers.

Qwen imports its unused 25 Hz tokenizer eagerly, which imports
``torchaudio.compliance.kaldi``. The 12 Hz CustomVoice path qualified here does
not call it. A wheel built for CUDA 13.0 cannot be loaded into the NVIDIA
PyTorch CUDA 13.3 container, so this shim keeps the unused import explicit and
fails loudly if a 25 Hz path tries to use it.
"""
