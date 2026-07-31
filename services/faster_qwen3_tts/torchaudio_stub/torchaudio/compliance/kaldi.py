# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unavailable 25 Hz Kaldi features; the qualified 12 Hz path never calls this."""


def fbank(*_args, **_kwargs):
    raise RuntimeError(
        "torchaudio Kaldi features are unavailable in the 12 Hz-only Jetson image"
    )
