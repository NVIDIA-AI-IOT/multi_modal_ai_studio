# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
WebRTC server-side camera stream: capture /dev/video0 (or device) on the server
and send it to the browser as a WebRTC video track so the client can use <video> natively.

Requires: aiortc, av, opencv-python-headless (optional extra: pip install -e ".[webrtc-camera]").

Encoding: aiortc/av use software encoding (VP8/H.264) on the server. If CPU load is too high,
consider feeding a hardware-encoded stream (e.g. NVENC, PyNvVideoCodec) into the
WebRTC pipeline instead; see docs/AUDIO_MODES.md "Encoding load and hardware acceleration".

Frame Broker Integration:
Frames captured here are also stored in FrameBroker so VLM can access them
without opening the camera again. This solves the camera lock issue.
See: https://github.com/NVIDIA-AI-IOT/live-vlm-webui for inspiration.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

# Frame broker for VLM access to server camera frames
_frame_broker = None

def _get_frame_broker():
    """Lazy import frame broker to avoid circular imports."""
    global _frame_broker
    if _frame_broker is None:
        try:
            from multi_modal_ai_studio.backends.vision.frame_broker import get_frame_broker
            _frame_broker = get_frame_broker()
        except ImportError:
            logger.debug("FrameBroker not available")
            _frame_broker = False  # Mark as unavailable
    return _frame_broker if _frame_broker else None

# Lazy imports so the app runs without aiortc/av/cv2
_cv2 = None
_aiortc = None
_av = None


def _ensure_imports():
    global _cv2, _aiortc, _av
    if _cv2 is not None:
        return True
    try:
        import cv2
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.rtcrtpsender import RTCRtpSender
        from av import VideoFrame
        _cv2 = cv2
        _aiortc = type("aiortc", (), {"RTCPeerConnection": RTCPeerConnection, "RTCSessionDescription": RTCSessionDescription, "RTCRtpSender": RTCRtpSender})()
        _aiortc.RTCPeerConnection = RTCPeerConnection
        _aiortc.RTCSessionDescription = RTCSessionDescription
        _av = VideoFrame
        return True
    except ImportError as e:
        logger.debug("WebRTC camera not available: %s", e)
        return False


class CameraTrack:
    """Video track that reads from a V4L2 device and yields VideoFrames for WebRTC."""

    kind = "video"

    def __init__(self, device: str):
        self._device = device
        self._cap = None
        self._started = False

    def _open(self):
        if self._cap is not None:
            return
        import cv2
        from av import VideoFrame
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open camera %s" % self._device)
        self._started = True

    async def recv(self):
        from av import VideoFrame
        if self._cap is None:
            self._open()
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, lambda: self._cap.read())
        if not ret or frame is None:
            return None
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._started = False


# Parse ICE candidate string "candidate:foundation component protocol priority ip port typ type"
_CANDIDATE_RE = re.compile(
    r"candidate:(\S+) (\d+) (\S+) (\d+) (\S+) (\d+) typ (\w+)"
)


def _parse_ice_candidate(candidate_str: str):
    """Parse browser candidate string into dict for RTCIceCandidate."""
    m = _CANDIDATE_RE.match((candidate_str or "").strip())
    if not m:
        return None
    foundation, component, protocol, priority, ip, port, _, typ = m.groups()
    return {
        "component": int(component),
        "foundation": foundation,
        "ip": ip,
        "port": int(port),
        "priority": int(priority),
        "protocol": protocol,
        "type": typ,
    }


# Subclass aiortc's VideoStreamTrack so next_timestamp() exists (it's not on MediaStreamTrack)
def _make_track_class():
    from aiortc.mediastreams import MediaStreamError, VideoStreamTrack
    from av import VideoFrame

    class _CameraStreamTrack(VideoStreamTrack):
        kind = "video"

        def __init__(self, device: str):
            super().__init__()
            self._device = device
            self._cap = None

        def _open(self):
            if self._cap is not None:
                return
            import cv2
            self._cap = cv2.VideoCapture(self._device)
            if not self._cap.isOpened():
                # MediaStreamError ends the track cleanly; avoid % in message so aiortc log format doesn't break
                raise MediaStreamError("Could not open camera " + str(self._device))

        async def recv(self):
            if self._cap is None:
                try:
                    self._open()
                except MediaStreamError:
                    raise
                except Exception as e:
                    raise MediaStreamError("Camera open failed: " + str(e))
            pts, time_base = await self.next_timestamp()
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(None, lambda: self._cap.read())
            if not ret or frame is None:
                # aiortc encoder expects a frame, not None; ending the track is the clean response
                raise MediaStreamError("Camera read failed")
            
            # Store frame in FrameBroker for VLM access (every frame for smooth capture)
            broker = _get_frame_broker()
            if broker:
                try:
                    # Store asynchronously to not block WebRTC
                    frame_copy = frame.copy()  # Copy to avoid race with next recv()
                    await loop.run_in_executor(None, lambda: broker.store_frame(frame_copy, jpeg_quality=70))
                except Exception as e:
                    logger.debug("FrameBroker store failed: %s", e)
            
            vf = VideoFrame.from_ndarray(frame, format="bgr24")
            vf.pts = pts
            vf.time_base = time_base
            return vf

        def stop(self):
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            super().stop()

    return _CameraStreamTrack


async def handle_camera_webrtc_ws(request: web.Request) -> web.WebSocketResponse:
    """
    WebSocket for WebRTC signaling: client sends offer + ICE candidates,
    server sends answer + ICE candidates. Server captures from device and sends video track.
    Query param: device=/dev/video0 (required).
    """
    device = (request.query.get("device") or "").strip()
    if not device or not device.startswith("/dev/"):
        return web.Response(text="Missing or invalid query device= (e.g. /dev/video0)", status=400)

    logger.info("[Camera WebRTC] Client connecting for device=%s", device)

    if not _ensure_imports():
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps({"type": "error", "error": "WebRTC camera requires aiortc and av. pip install aiortc av opencv-python-headless"}))
        await ws.close()
        return ws

    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc = None
    track = None
    CameraStreamTrack = _make_track_class()

    def cleanup():
        nonlocal pc, track
        if track is not None:
            try:
                track.stop()
                logger.info("[Camera WebRTC] Track stopped for device=%s", device)
            except Exception:
                pass
            track = None
        if pc is not None:
            asyncio.create_task(pc.close())
            logger.info("[Camera WebRTC] PeerConnection closing for device=%s", device)
        pc = None

    try:
        pc = RTCPeerConnection()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc is None:
                return
            logger.info("[Camera WebRTC] Connection state=%s for device=%s", pc.connectionState, device)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                cleanup()

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            try:
                if candidate:
                    await ws.send_str(json.dumps({
                        "type": "ice",
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    }))
                else:
                    await ws.send_str(json.dumps({"type": "ice", "candidate": None}))
            except Exception as e:
                logger.debug("Send ICE to client: %s", e)

        offer_received = False
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            msg_type = data.get("type")
            if msg_type == "offer" and not offer_received:
                offer_received = True
                sdp = data.get("sdp")
                typ = data.get("type")
                if not sdp or typ != "offer":
                    await ws.send_str(json.dumps({"type": "error", "error": "Invalid offer"}))
                    break
                try:
                    track = CameraStreamTrack(device)
                    pc.addTrack(track)
                    logger.info("[Camera WebRTC] Track added for device=%s", device)
                except Exception as e:
                    logger.warning("Camera track failed for %s: %s", device, e)
                    await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
                    break
                try:
                    offer = RTCSessionDescription(sdp=sdp, type=typ)
                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_str(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                        "answerType": pc.localDescription.type,  # "answer" - avoid duplicate key with "type"
                    }))
                except Exception as e:
                    logger.exception("WebRTC offer/answer failed")
                    await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
            elif msg_type == "ice" and pc is not None:
                cand = data.get("candidate")
                sdp_mid = data.get("sdpMid")
                sdp_mline = data.get("sdpMLineIndex")
                try:
                    if cand is None or cand == "":
                        await pc.addIceCandidate(None)
                    else:
                        parsed = _parse_ice_candidate(cand)
                        if parsed:
                            ice = RTCIceCandidate(
                                component=parsed["component"],
                                foundation=parsed["foundation"],
                                ip=parsed["ip"],
                                port=parsed["port"],
                                priority=parsed["priority"],
                                protocol=parsed["protocol"],
                                type=parsed["type"],
                                sdpMid=sdp_mid,
                                sdpMLineIndex=sdp_mline,
                            )
                            await pc.addIceCandidate(ice)
                except Exception as e:
                    logger.debug("Add ICE candidate: %s", e)
    except Exception as e:
        logger.exception("Camera WebRTC WebSocket: %s", e)
    finally:
        cleanup()
        logger.info("[Camera WebRTC] WebSocket closed for device=%s", device)

    return ws
