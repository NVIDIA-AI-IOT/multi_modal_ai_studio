---
name: device-management
description: >
  Enumerate and configure server-side USB cameras, microphones, and speakers
  for Multi-modal AI Studio. Stream camera preview via WebRTC or MJPEG.
  Use when deploying on Jetson or headless servers with USB peripherals.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
---

# Device Management

Manage server-attached USB cameras, microphones, and speakers.

## Overview

When running on a Jetson or headless server, MMAS can use USB devices attached directly to the server (instead of browser media). The server enumerates V4L2 cameras and ALSA audio devices and exposes them via REST APIs.

## Instructions

### List USB cameras

```bash
curl -s http://localhost:8092/api/devices/cameras | python3 -m json.tool
```

Response:
```json
[
  {"id": "/dev/video0", "label": "USB 3.0 Camera (Server USB)"},
  {"id": "/dev/video2", "label": "HD Webcam (Server USB)"}
]
```

### List microphones

```bash
curl -s http://localhost:8092/api/devices/audio-inputs | python3 -m json.tool
```

Response:
```json
[
  {"id": "alsa:hw:1,0", "label": "Blue Yeti Nano (Server USB)"},
  {"id": "alsa:hw:2,0", "label": "NVIDIA Jetson Thor AGX APE (Server USB)"}
]
```

### List speakers

```bash
curl -s http://localhost:8092/api/devices/audio-outputs | python3 -m json.tool
```

### Camera preview (MJPEG stream)

```bash
# Stream in browser or with curl
curl -s "http://localhost:8092/api/camera/stream?device=/dev/video0" --output -
```

The MJPEG stream is `multipart/x-mixed-replace` with JPEG frames. It also pushes frames into the internal `FrameBroker` for VLM consumption.

### Camera preview (WebRTC)

Connect via WebSocket at `/ws/camera-webrtc?device=/dev/video0` for low-latency video preview with WebRTC signaling (offer/answer/ICE).

### Microphone preview

Connect via WebSocket at `/ws/mic-preview` to stream server mic audio levels. Send a config message first, then receive `user_amplitude` JSON events.

### Configure devices in a session

```json
{
  "devices": {
    "video_source": "usb",
    "video_device": "/dev/video0",
    "audio_input_source": "alsa",
    "audio_input_device": "hw:1,0",
    "audio_output_source": "alsa",
    "audio_output_device": "hw:0,0"
  }
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `video_source` | `browser`, `usb`, `local`, `none` | Where camera frames come from |
| `audio_input_source` | `browser`, `usb`, `alsa` | Where microphone audio comes from |
| `audio_output_source` | `browser`, `alsa` | Where TTS audio plays |

## Input Schema

Camera list endpoint:
- No parameters required

Audio endpoints:
- No parameters required

Camera stream:
- `device` (string, optional): V4L2 device path (e.g., `/dev/video0`)

## Output Schema

Device list:
```json
[
  {"id": "device_path_or_id", "label": "Human-readable name (Server USB)"}
]
```

## Guidelines

- Cameras create multiple `/dev/video*` nodes — MMAS filters out metadata nodes automatically using V4L2 capability checks
- USB mics may only support 48kHz natively — MMAS uses `plughw:` ALSA devices for automatic rate conversion to 16kHz
- Only one process can open a V4L2 camera at a time — if WebRTC is active, MJPEG will fail on the same device
- Connect USB devices before starting the server for reliable enumeration
- The `FrameBroker` stores frames in a ring buffer (max 100 frames, 10s max age) for the VLM pipeline
