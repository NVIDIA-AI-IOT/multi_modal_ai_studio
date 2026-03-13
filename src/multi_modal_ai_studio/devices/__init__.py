# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device detection and routing for audio/video inputs."""

from multi_modal_ai_studio.devices.local import (
    list_local_cameras,
    list_local_audio_inputs,
    list_local_audio_outputs,
)

__all__ = [
    "list_local_cameras",
    "list_local_audio_inputs",
    "list_local_audio_outputs",
]
