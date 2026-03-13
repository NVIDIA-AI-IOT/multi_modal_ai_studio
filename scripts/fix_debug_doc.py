#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off fix for USB_MIC_AUDIO_DEBUG.md step 4 paragraph."""
path = "docs/cursor/USB_MIC_AUDIO_DEBUG.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# File uses Unicode: — (U+2014), ' (U+2019), " " (U+201C U+201D), and possibly non-breaking space before "ms"
old = (
    "   Only one process can capture from the same ALSA device at a time. If **mic-preview** was showing the green bar (Server USB) and you click START, the client closes the preview WebSocket and immediately opens the voice WebSocket\u2014the server may not have released the device yet, so the voice pipeline\u2019s arecord fails with \u201cDevice or resource busy\u201d. With the single-connection flow there is no handoff. (Previously: The client now waited **450\u00a0ms** after closing the preview when Server USB mic is selected, then opens the voice WebSocket, so the server has time to stop the preview capture and release the device. If you still see \u201cDevice or resource busy\u201d, another app may be using the device, or the delay may need to be increased.\n"
)
new = "   Only one process can capture from the same ALSA device at a time. With the single-connection flow, Server USB uses one `/ws/voice` for preview and live, so there is no handoff. If you still see \"Device or resource busy\", another app may be using the device.\n"

if old in content:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Old string not found")
