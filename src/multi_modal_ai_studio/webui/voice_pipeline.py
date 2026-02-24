"""
Voice WebSocket pipeline: ASR (Riva or OpenAI Realtime) -> LLM (Ollama) -> TTS (Riva).

When ASR is OpenAI Realtime API (WebSocket, full voice), uses a single Realtime WebSocket
instead of Riva ASR + LLM + Riva TTS.
Handles WebSocket messages: config (first JSON), then binary PCM audio.
Sends back: timeline events (JSON) and TTS audio (base64).
On stop/disconnect: saves session to session_dir.
"""

import asyncio
import base64
import json
import logging
import math
import os
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from aiohttp import web

from multi_modal_ai_studio.devices.capture import start_server_mic_capture

try:
    from multi_modal_ai_studio.devices.playback import (
        start_server_speaker_playback,
        stop_server_speaker_playback,
    )
except ImportError:
    # playback.py may be missing on some branches (e.g. upstream main); stub so app still starts.
    def start_server_speaker_playback(device: str, sample_rate: int, proc_holder: Optional[list] = None):
        return None

    def stop_server_speaker_playback(proc) -> None:
        pass

from multi_modal_ai_studio.config.schema import (
    SessionConfig,
    ASRConfig,
    LLMConfig,
    TTSConfig,
)
from multi_modal_ai_studio import __version__
from multi_modal_ai_studio.core.session import Session
from multi_modal_ai_studio.core.timeline import Lane
from multi_modal_ai_studio.webui import system_stats as system_stats_module
from multi_modal_ai_studio.backends.base import ASRResult
from multi_modal_ai_studio.backends.asr.riva import RivaASRBackend
from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.backends.tts.riva import RivaTTSBackend
from multi_modal_ai_studio.backends.realtime import (
    DISABLE_TURN_DETECTION,
    REALTIME_SAMPLE_RATE,
    OpenAIRealtimeClient,
    RealtimeEvent,
)

logger = logging.getLogger(__name__)


class TTSChunkBuffer:
    """Word-count based buffer that chunks LLM tokens for streamed TTS.

    Uses a smaller word threshold for the first chunk (fast time-to-first-audio)
    and a larger threshold for subsequent chunks (better prosody).  Flushes
    eagerly at natural break characters when enough words have accumulated.
    """

    FIRST_CHUNK_WORDS = 6
    MAX_CHUNK_WORDS = 18
    TTS_BREAKS = frozenset(".!?,;:\n\u2014-")

    def __init__(self) -> None:
        self._buf = ""
        self._first_sent = False

    def add(self, token: str) -> Optional[str]:
        """Add a token. Returns a chunk when one is ready to speak."""
        self._buf += token
        words = len(self._buf.split())
        limit = self.FIRST_CHUNK_WORDS if not self._first_sent else self.MAX_CHUNK_WORDS
        hit_break = any(c in token for c in self.TTS_BREAKS) and words >= 2
        if hit_break or words >= limit:
            chunk = self._buf.strip()
            self._buf = ""
            self._first_sent = True
            return chunk or None
        return None

    def flush(self) -> Optional[str]:
        """Return whatever remains in the buffer (call when LLM stream ends)."""
        remainder = self._buf.strip()
        self._buf = ""
        return remainder or None


# Only one mic-preview capture at a time (ALSA device is exclusive).
_mic_preview_lock = threading.Lock()

# Frontend sends asr.backend, asr.riva_server; schema expects asr.scheme, asr.server. Same for tts.
def _normalize_frontend_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize frontend config keys to SessionConfig.from_dict shape."""
    data = dict(payload)
    if "asr" in data:
        asr = dict(data["asr"])
        if "backend" in asr and "scheme" not in asr:
            asr["scheme"] = asr.pop("backend", "riva")
        if "riva_server" in asr and "server" not in asr:
            asr["server"] = asr.pop("riva_server")
        # Realtime: pass through realtime_url; derive from api_base when missing (e.g. preset)
        if asr.get("scheme") == "openai-realtime":
            if not (asr.get("realtime_url") or "").strip():
                base = (asr.get("api_base") or "").strip()
                if base:
                    asr["realtime_url"] = base
            if not asr.get("realtime_transport"):
                asr["realtime_transport"] = "websocket"
            if not asr.get("realtime_session_type"):
                asr["realtime_session_type"] = "full"
        data["asr"] = asr
    if "tts" in data:
        tts = dict(data["tts"])
        if "backend" in tts and "scheme" not in tts:
            tts["scheme"] = tts.pop("backend", "riva")
        if "riva_server" in tts and "server" not in tts:
            tts["server"] = tts.pop("riva_server")
        if data.get("tts_model_name") and "riva_model_name" not in tts:
            tts["riva_model_name"] = data["tts_model_name"]
        data["tts"] = tts
    if "llm" in data:
        llm = dict(data["llm"])
        if "ollama_url" in llm and "api_base" not in llm:
            base = (llm.get("ollama_url") or "").rstrip("/")
            llm["api_base"] = f"{base}/v1" if base else "http://localhost:11434/v1"
        data["llm"] = llm
    return data


def _pcm_rms_to_amplitude(pcm_bytes: bytes) -> float:
    """Compute RMS of 16-bit LE PCM and scale to 0-100 for timeline."""
    if len(pcm_bytes) < 2:
        return 0.0
    try:
        n = len(pcm_bytes) // 2
        samples = struct.unpack(f"{n}h", pcm_bytes)
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / n) if n else 0
        # Normalize by int16 range; scale to 0-100, cap at 100
        return min(100.0, (rms / 32768.0) * 100.0)
    except Exception:
        return 0.0


def _pcm_rms_slices(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    window_s: float = 0.025,
) -> List[float]:
    """Compute RMS per fixed-time window (e.g. 25ms) so timeline gets one amplitude per bar.
    Returns list of 0-100 amplitudes, one per window. Matches client tStep=0.025 (~40 Hz)."""
    if len(pcm_bytes) < 2:
        return []
    try:
        samples = struct.unpack(f"{len(pcm_bytes) // 2}h", pcm_bytes)
    except Exception:
        return []
    samples_per_window = max(1, int(sample_rate * window_s))
    result: List[float] = []
    for i in range(0, len(samples), samples_per_window):
        window = samples[i : i + samples_per_window]
        if not window:
            break
        sum_sq = sum(s * s for s in window)
        rms = math.sqrt(sum_sq / len(window))
        amp = min(100.0, (rms / 32768.0) * 100.0)
        result.append(amp)
    return result


def _resample_pcm_to_24k(pcm_bytes: bytes, from_rate: int) -> bytes:
    """Resample 16-bit mono PCM to 24 kHz for Realtime API."""
    if from_rate == REALTIME_SAMPLE_RATE:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    n_old = len(samples)
    n_new = int(round(n_old * REALTIME_SAMPLE_RATE / from_rate))
    x_old = np.arange(n_old)
    x_new = np.linspace(0, n_old - 1, n_new)
    resampled = np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.int16)
    return resampled.tobytes()


# Browser sends 16 kHz PCM (TARGET_SAMPLE_RATE in app.js).
INPUT_SAMPLE_RATE_FOR_REALTIME = 16000


async def _run_realtime_loop(
    ws: web.WebSocketResponse,
    session: Session,
    config: SessionConfig,
    session_dir: Path,
    use_server_mic: bool,
    use_server_speaker: bool,
    capture_queue: Optional[queue.Queue],
    stop_capture: Optional[threading.Event],
    capture_thread: Optional[threading.Thread],
) -> Optional[str]:
    """
    Run OpenAI Realtime WebSocket loop: PCM in -> Realtime client -> TTS audio + events out.
    Same device handling as classic (browser or server mic, browser or server speaker).
    """
    asr_config = config.asr
    llm_config = config.llm
    tts_config = config.tts
    url = (asr_config.realtime_url or asr_config.api_base or "").strip()
    api_key = (asr_config.api_key or "").strip()
    model = (asr_config.model or "gpt-realtime").strip()
    instructions = (llm_config.system_prompt or "You are a helpful assistant.").strip()
    voice = getattr(tts_config, "voice", "alloy") or "alloy"
    # OpenAI Realtime voices are short names (alloy, echo, ...); Riva uses long names
    if len(voice) > 20:
        voice = "alloy"

    # VAD / turn_detection from ASR config (server_vad or null)
    turn_detection = DISABLE_TURN_DETECTION
    if getattr(asr_config, "enable_vad", True):
        turn_detection = {
            "type": "server_vad",
            "threshold": getattr(asr_config, "vad_start_threshold", 0.5),
            "prefix_padding_ms": getattr(asr_config, "speech_pad_ms", 300),
            "silence_duration_ms": getattr(asr_config, "speech_timeout_ms", 500),
        }

    # Enable input transcription so we get asr_partial/asr_final (ASR lane); client sends audio.input.transcription.
    input_transcription = {"model": "gpt-4o-transcribe"}
    client = OpenAIRealtimeClient(
        url=url,
        api_key=api_key,
        model=model,
        instructions=instructions,
        voice=voice,
        turn_detection=turn_detection,
        input_audio_transcription=input_transcription,
    )
    pipeline_live = asyncio.Event()
    pipeline_live.set() if not use_server_mic else pipeline_live.clear()
    stopped = asyncio.Event()
    session_ready = asyncio.Event()  # Set when Realtime session.updated received; must wait before send_audio
    _user_amplitude_sent = False
    preview_start_time = time.time()
    amplitude_interval = 0.05
    # Optional: record 24 kHz PCM sent to Realtime (export REALTIME_DEBUG_RECORD_PCM=1; optional REALTIME_DEBUG_PCM_DIR)
    _debug_record_pcm = os.environ.get("REALTIME_DEBUG_RECORD_PCM", "").strip().lower() in ("1", "true", "yes")
    _debug_pcm_file_holder: list = []  # [file_handle] shared by receive_loop and server_capture_consumer
    if _debug_record_pcm:
        _debug_pcm_dir = Path(os.environ.get("REALTIME_DEBUG_PCM_DIR", "").strip()) or session_dir
        _debug_pcm_dir = _debug_pcm_dir.resolve()
        logger.info("Realtime debug: PCM recording enabled; will write to %s when first audio is sent", _debug_pcm_dir)
    # Optional: record 24 kHz PCM received from Realtime (response/AI audio) (export REALTIME_DEBUG_RECORD_RESPONSE=1)
    _debug_record_response = os.environ.get("REALTIME_DEBUG_RECORD_RESPONSE", "").strip().lower() in ("1", "true", "yes")
    _debug_response_file_holder: list = []
    if _debug_record_response:
        _debug_response_dir = Path(os.environ.get("REALTIME_DEBUG_PCM_DIR", "").strip()) or session_dir
        _debug_response_dir = _debug_response_dir.resolve()
        logger.info("Realtime debug: response audio recording enabled; will write to %s when first response is received", _debug_response_dir)

    async def send_event(event_dict: Dict[str, Any]) -> None:
        try:
            await ws.send_str(json.dumps({"type": "event", "event": event_dict}))
        except Exception as e:
            logger.warning("Realtime send event failed: %s", e)

    try:
        await client.connect()
    except Exception as e:
        logger.exception("Realtime connect failed: %s", e)
        return None

    def pcm_for_realtime(pcm_bytes: bytes) -> bytes:
        return _resample_pcm_to_24k(pcm_bytes, INPUT_SAMPLE_RATE_FOR_REALTIME)

    def _write_debug_pcm(pcm_24: bytes) -> None:
        if not _debug_record_pcm or not pcm_24:
            return
        try:
            if not _debug_pcm_file_holder:
                out_dir = Path(os.environ.get("REALTIME_DEBUG_PCM_DIR", "").strip()) or session_dir
                out_dir = out_dir.resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                # Same base name as session JSON so PCM is easy to associate: {session_id}.json / {session_id}_realtime_sent_24k.pcm
                path = out_dir / f"{session.session_id}_realtime_sent_24k.pcm"
                _debug_pcm_file_holder.append(open(path, "wb"))
                logger.info("Realtime debug: recording 24 kHz PCM to %s", path)
            _debug_pcm_file_holder[0].write(pcm_24)
        except Exception as e:
            logger.debug("Realtime debug PCM write failed: %s", e)

    def _write_debug_response_audio(pcm_24: bytes) -> None:
        if not _debug_record_response or not pcm_24:
            return
        try:
            if not _debug_response_file_holder:
                out_dir = Path(os.environ.get("REALTIME_DEBUG_PCM_DIR", "").strip()) or session_dir
                out_dir = out_dir.resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{session.session_id}_realtime_response_24k.pcm"
                _debug_response_file_holder.append(open(path, "wb"))
                logger.info("Realtime debug: recording response 24 kHz PCM to %s", path)
            _debug_response_file_holder[0].write(pcm_24)
        except Exception as e:
            logger.debug("Realtime debug response PCM write failed: %s", e)

    server_speaker_proc = None
    tts_start_sent = False
    tts_first_audio_sent = False
    last_output_transcript = ""
    # Cursor for TTS amplitude timestamps: anchored to tts_start so saved-session purple spans tts_start..tts_complete
    tts_amplitude_cursor: Optional[float] = None

    async def realtime_event_consumer() -> None:
        nonlocal server_speaker_proc, tts_start_sent, tts_first_audio_sent, last_output_transcript, tts_amplitude_cursor
        try:
            async for ev in client.events():
                if ev is None or stopped.is_set():
                    break
                if ev.kind == "error":
                    logger.warning("Realtime error: %s", ev.message)
                    ts = (time.time() - session.timeline.start_time) if session.timeline.start_time is not None else 0
                    session.timeline.add_event("error", Lane.SYSTEM, data={"message": ev.message})
                    await send_event({"event_type": "error", "lane": "system", "data": {"message": ev.message}, "timestamp": ts})
                    continue
                if ev.kind == "session_ready":
                    session_ready.set()
                    ts = (time.time() - session.timeline.start_time) if session.timeline.start_time is not None else 0
                    session.timeline.add_event("realtime_session_ready", Lane.SYSTEM, data={})
                    await send_event({"event_type": "realtime_session_ready", "lane": "system", "data": {}, "timestamp": ts})
                    continue
                if ev.kind == "transcript_delta":
                    if ev.text and session.timeline.start_time is not None:
                        ts = time.time() - session.timeline.start_time
                        session.timeline.add_event("asr_partial", Lane.SPEECH, data={"text": ev.text, "confidence": 1.0})
                        await send_event({"event_type": "asr_partial", "lane": "speech", "data": {"text": ev.text, "confidence": 1.0}, "timestamp": ts})
                        logger.info("[asr] asr_partial @ %.2fs: %r", ts, (ev.text[:60] + "..." if len(ev.text) > 60 else ev.text))
                if ev.kind == "transcript_completed":
                    if ev.text and session.timeline.start_time is not None:
                        ts = time.time() - session.timeline.start_time
                        session.timeline.add_event("asr_final", Lane.SPEECH, data={"text": ev.text, "confidence": 1.0})
                        await send_event({"event_type": "asr_final", "lane": "speech", "data": {"text": ev.text, "confidence": 1.0}, "timestamp": ts})
                        logger.info("[asr] asr_final @ %.2fs: %r", ts, (ev.text[:80] + "..." if len(ev.text) > 80 else ev.text))
                if ev.kind == "output_transcript_delta":
                    if ev.text and session.timeline.start_time is not None:
                        ts = time.time() - session.timeline.start_time
                        session.timeline.add_event("realtime_output_partial", Lane.TTS, data={"text": ev.text})
                        await send_event({"event_type": "realtime_output_partial", "lane": "tts", "data": {"text": ev.text}, "timestamp": ts})
                        logger.info("[tts] realtime_output_partial @ %.2fs: %r", ts, (ev.text[:100] + "..." if len(ev.text) > 100 else ev.text))
                if ev.kind == "output_transcript_completed":
                    if ev.text and session.timeline.start_time is not None:
                        ts = time.time() - session.timeline.start_time
                        last_output_transcript = ev.text
                        session.timeline.add_event("realtime_output_final", Lane.TTS, data={"text": ev.text})
                        await send_event({"event_type": "realtime_output_final", "lane": "tts", "data": {"text": ev.text}, "timestamp": ts})
                        logger.info("[tts] realtime_output_final @ %.2fs: %r", ts, ev.text)
                if ev.kind == "audio" and ev.audio:
                    _write_debug_response_audio(ev.audio)
                    amplitude_segments: List[Dict[str, Any]] = []
                    if session.timeline.start_time is not None:
                        ts = time.time() - session.timeline.start_time
                        if not tts_start_sent:
                            tts_start_sent = True
                            session.timeline.add_event("tts_start", Lane.TTS)
                            await send_event({"event_type": "tts_start", "lane": "tts", "data": {}, "timestamp": ts})
                            await ws.send_str(json.dumps({"type": "tts_start"}))
                            logger.info("[tts] tts_start @ %.2fs", ts)
                            # Anchor TTS amplitude to tts_start so saved-session purple spans tts_start..tts_complete
                            tts_amplitude_cursor = ts
                        amp = _pcm_rms_to_amplitude(ev.audio)
                        if amp > 0 and not tts_first_audio_sent:
                            tts_first_audio_sent = True
                            session.timeline.add_event("tts_first_audio", Lane.TTS)
                            await send_event({"event_type": "tts_first_audio", "lane": "tts", "data": {}, "timestamp": ts})
                            logger.info("[tts] tts_first_audio @ %.2fs", ts)
                        # 25ms RMS slices; timeline uses cursor (tts_start-anchored) so replay purple aligns with TTS blocks
                        _window_s = 0.025  # same as Classic _amplitude_window_s (25ms)
                        amps = _pcm_rms_slices(
                            ev.audio,
                            sample_rate=ev.sample_rate,
                            window_s=_window_s,
                        )
                        if tts_amplitude_cursor is not None:
                            for i, a in enumerate(amps):
                                t_start = tts_amplitude_cursor
                                t_end = t_start + _window_s
                                session.timeline.add_audio_amplitude(
                                    amplitude=a, source="tts", timestamp=t_start
                                )
                                amplitude_segments.append({
                                    "startTime": round(t_start, 3),
                                    "endTime": round(t_end, 3),
                                    "amplitude": round(a, 2),
                                })
                                tts_amplitude_cursor = t_end
                    if use_server_speaker and config.devices.audio_output_device:
                        if server_speaker_proc is None:
                            server_speaker_proc = start_server_speaker_playback(
                                config.devices.audio_output_device,
                                ev.sample_rate,
                            )
                        if server_speaker_proc is not None and server_speaker_proc.stdin and not server_speaker_proc.stdin.closed:
                            try:
                                server_speaker_proc.stdin.write(ev.audio)
                                server_speaker_proc.stdin.flush()
                            except (BrokenPipeError, OSError):
                                server_speaker_proc = None
                    b64 = base64.b64encode(ev.audio).decode("ascii")
                    payload = {
                        "type": "tts_audio",
                        "data": b64,
                        "sample_rate": ev.sample_rate,
                        "is_final": False,
                    }
                    if amplitude_segments:
                        payload["amplitude_segments"] = amplitude_segments
                    await ws.send_str(json.dumps(payload))
                if ev.kind == "response_done":
                    if session.timeline.start_time is not None:
                        ts = time.time() - session.timeline.start_time
                        session.timeline.add_event("tts_complete", Lane.TTS, data={"text": last_output_transcript})
                        await send_event({"event_type": "tts_complete", "lane": "tts", "data": {"text": last_output_transcript}, "timestamp": ts})
                        logger.info("[tts] tts_complete @ %.2fs: %r", ts, last_output_transcript)
                    # Reset so next response sends tts_start/tts_first_audio again (TTS lane rectangles for 2nd+ turns)
                    tts_start_sent = False
                    tts_first_audio_sent = False
        except asyncio.CancelledError:
            pass
        finally:
            if server_speaker_proc is not None:
                stop_server_speaker_playback(server_speaker_proc)

    async def receive_loop() -> None:
        nonlocal _user_amplitude_sent
        last_amplitude_time = 0.0
        user_amp_buf: list = []  # last 3 raw amplitudes for smoothing green waveform (16 kHz PCM)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        obj = json.loads(msg.data)
                        if obj.get("type") == "start_session":
                            if use_server_mic and not pipeline_live.is_set():
                                session.start()
                                await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})
                                pipeline_live.set()
                                logger.info("Realtime: start_session received; capture now feeds Realtime")
                        if obj.get("type") == "stop":
                            for key, attr in [("system_stats", "system_stats"), ("tts_playback_segments", "tts_playback_segments"), ("audio_amplitude_history", "audio_amplitude_history"), ("ttl_bands", "ttl_bands")]:
                                val = obj.get(key)
                                if isinstance(val, list) and hasattr(session, attr):
                                    setattr(session, attr, val)
                            if getattr(session, "ttl_bands", None):
                                session.apply_ttl_bands()
                            session.app_version = __version__
                            stopped.set()
                            return
                    except json.JSONDecodeError:
                        pass
                    continue
                if msg.type == web.WSMsgType.BINARY and not use_server_mic:
                    await session_ready.wait()
                    pcm_24 = pcm_for_realtime(msg.data)
                    _write_debug_pcm(pcm_24)
                    await client.send_audio(pcm_24)
                    if session.timeline.start_time is not None:
                        now = time.time() - session.timeline.start_time
                        if now - last_amplitude_time >= amplitude_interval:
                            amp = _pcm_rms_to_amplitude(msg.data)
                            session.timeline.add_audio_amplitude(amplitude=amp, source="user")
                            last_amplitude_time = now
                            user_amp_buf.append(amp)
                            if len(user_amp_buf) > 3:
                                user_amp_buf.pop(0)
                            smoothed = sum(user_amp_buf) / len(user_amp_buf)
                            try:
                                await ws.send_str(json.dumps({"type": "user_amplitude", "timestamp": round(now, 3), "amplitude": round(smoothed, 2)}))
                                _user_amplitude_sent = True
                            except Exception:
                                pass
                if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    stopped.set()
                    return
        except Exception as e:
            logger.debug("Realtime receive_loop ended: %s", e)
        finally:
            stopped.set()

    async def server_capture_consumer() -> None:
        nonlocal _user_amplitude_sent
        if capture_queue is None:
            return
        loop = asyncio.get_event_loop()
        last_amplitude_time = 0.0
        user_amp_buf: list = []
        while not stopped.is_set():
            try:
                chunk = await loop.run_in_executor(None, capture_queue.get)
            except Exception:
                break
            if chunk is None:
                break
            if pipeline_live.is_set():
                await session_ready.wait()
                pcm_24 = pcm_for_realtime(chunk)
                _write_debug_pcm(pcm_24)
                await client.send_audio(pcm_24)
                if session.timeline.start_time is not None:
                    now = time.time() - session.timeline.start_time
                    if now - last_amplitude_time >= amplitude_interval:
                        amp = _pcm_rms_to_amplitude(chunk)
                        session.timeline.add_audio_amplitude(amplitude=amp, source="user")
                        last_amplitude_time = now
                        user_amp_buf.append(amp)
                        if len(user_amp_buf) > 3:
                            user_amp_buf.pop(0)
                        smoothed = sum(user_amp_buf) / len(user_amp_buf)
                        try:
                            await ws.send_str(json.dumps({"type": "user_amplitude", "timestamp": round(now, 3), "amplitude": round(smoothed, 2)}))
                            _user_amplitude_sent = True
                        except Exception:
                            pass
            else:
                now = time.time() - preview_start_time
                if now - last_amplitude_time >= amplitude_interval:
                    amp = _pcm_rms_to_amplitude(chunk)
                    last_amplitude_time = time.time() - preview_start_time
                    try:
                        await ws.send_str(json.dumps({"type": "user_amplitude", "timestamp": round(now, 3), "amplitude": round(amp, 2)}))
                        _user_amplitude_sent = True
                    except Exception:
                        pass

    if not use_server_mic:
        session.start()
        await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})

    event_task = asyncio.create_task(realtime_event_consumer())
    recv_task = asyncio.create_task(receive_loop())
    server_capture_task: Optional[asyncio.Task] = None
    if use_server_mic and capture_queue is not None:
        server_capture_task = asyncio.create_task(server_capture_consumer())

    if use_server_mic:
        await asyncio.wait(
            [asyncio.create_task(stopped.wait()), asyncio.create_task(pipeline_live.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # If pipeline_live was set, receive_loop already called session.start() and send_event on start_session

    await stopped.wait()

    if stop_capture is not None:
        stop_capture.set()
    if server_capture_task is not None:
        server_capture_task.cancel()
    recv_task.cancel()
    event_task.cancel()
    try:
        await client.disconnect()
    except Exception as e:
        logger.debug("Realtime disconnect: %s", e)
    try:
        await recv_task
    except asyncio.CancelledError:
        pass
    if server_capture_task is not None:
        try:
            await server_capture_task
        except asyncio.CancelledError:
            pass
    try:
        await event_task
    except asyncio.CancelledError:
        pass

    if _debug_pcm_file_holder:
        try:
            _debug_pcm_file_holder[0].close()
        except Exception:
            pass
        _debug_pcm_file_holder.clear()
    if _debug_response_file_holder:
        try:
            _debug_response_file_holder[0].close()
        except Exception:
            pass
        _debug_response_file_holder.clear()

    if session.timeline.start_time is None:
        return None

    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{session.session_id}.json"
    session.save(path)
    logger.info("Realtime session saved: %s", path)
    return session.session_id


async def _run_voice_pipeline(
    ws: web.WebSocketResponse,
    config: SessionConfig,
    session_dir: Path,
) -> Optional[str]:
    """
    Run ASR -> LLM -> TTS pipeline. Receive PCM from ws, send events + TTS audio.
    Returns session_id on success, None on error or stop.
    When use_server_mic: capture starts immediately and we stream user_amplitude at 50 Hz (preview).
    Client sends type "start_session" to begin the full pipeline (same capture, now feeding ASR + timeline).
    """
    logger.info("Voice pipeline starting")
    session = Session(config=config)
    use_server_mic = config.devices.audio_input_source in ("alsa", "usb") and bool(
        config.devices.audio_input_device
    )
    use_server_speaker = config.devices.audio_output_source in ("alsa", "usb") and bool(
        config.devices.audio_output_device
    )
    if use_server_speaker:
        logger.info(
            "Server speaker enabled: device=%s (TTS will play to ALSA)",
            config.devices.audio_output_device,
        )
    # Session started on start_session (both mics). Browser mic: no session.start() here so preview PCM is amplitude-only until START.
    asr_config = config.asr
    llm_config = config.llm
    tts_config = config.tts

    # Device capture for server mic (used by both Realtime and classic paths)
    capture_queue: Optional[queue.Queue] = queue.Queue() if use_server_mic else None
    stop_capture: Optional[threading.Event] = threading.Event() if use_server_mic else None
    capture_thread: Optional[threading.Thread] = None
    if use_server_mic and capture_queue is not None and stop_capture is not None:
        capture_thread = start_server_mic_capture(
            config.devices.audio_input_source,
            config.devices.audio_input_device,
            capture_queue,
            stop_capture,
        )
        if not capture_thread:
            logger.warning(
                "Server mic capture could not start (source=%s device=%s); no voice input from server device",
                getattr(config.devices, "audio_input_source", None),
                getattr(config.devices, "audio_input_device", None),
            )
            use_server_mic = False
        else:
            logger.info("Server mic capture thread started; waiting for first PCM chunk and user_amplitude")

    # Realtime full voice (WebSocket): one client, no Riva ASR/LLM/TTS
    if (
        asr_config.scheme == "openai-realtime"
        and asr_config.realtime_session_type == "full"
        and asr_config.realtime_transport == "websocket"
    ):
        url = (asr_config.realtime_url or asr_config.api_base or "").strip()
        if not url:
            await ws.send_str(json.dumps({"type": "error", "error": "OpenAI Realtime requires realtime_url or api_base"}))
            return None
        if not (asr_config.model or "").strip():
            await ws.send_str(json.dumps({"type": "error", "error": "OpenAI Realtime requires model"}))
            return None
        if "localhost" not in url.split("//")[-1].split("/")[0].lower() and not (asr_config.api_key or "").strip():
            await ws.send_str(json.dumps({"type": "error", "error": "OpenAI Realtime (non-localhost) requires API key"}))
            return None
        try:
            return await _run_realtime_loop(
                ws=ws,
                session=session,
                config=config,
                session_dir=session_dir,
                use_server_mic=use_server_mic,
                use_server_speaker=use_server_speaker,
                capture_queue=capture_queue,
                stop_capture=stop_capture,
                capture_thread=capture_thread,
            )
        except Exception as e:
            logger.exception("Realtime loop error: %s", e)
            try:
                await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
            except Exception:
                pass
            return None

    # Classic: Riva ASR + LLM + Riva TTS
    if asr_config.scheme != "riva" or tts_config.scheme != "riva":
        await ws.send_str(json.dumps({"type": "error", "error": "This pipeline requires ASR and TTS scheme 'riva'"}))
        return None
    if not asr_config.server or not tts_config.server:
        await ws.send_str(json.dumps({"type": "error", "error": "Riva server address required for ASR and TTS"}))
        return None
    if not llm_config.api_base:
        await ws.send_str(json.dumps({"type": "error", "error": "LLM api_base required"}))
        return None

    try:
        asr = RivaASRBackend(config=asr_config, timeline=session.timeline)
        llm = OpenAILLMBackend(config=llm_config)
        tts = RivaTTSBackend(config=tts_config, timeline=session.timeline)
    except Exception as e:
        logger.exception("Failed to create backends")
        await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
        return None

    # ASR starts only after client sends start_session (both mics). Avoids Riva timeout during preview and keeps logic identical.
    asr_stream_started = False
    conversation_history = []
    stopped = asyncio.Event()
    finals_queue: asyncio.Queue = asyncio.Queue()
    pipeline_live = asyncio.Event()  # set when client sends start_session; then we start ASR and feed pipeline
    pipeline_live.clear()

    logger.info(
        "Voice pipeline devices: audio_input_source=%s audio_input_device=%s use_server_mic=%s",
        getattr(config.devices, "audio_input_source", None),
        getattr(config.devices, "audio_input_device", None),
        use_server_mic,
    )
    capture_queue: Optional[queue.Queue] = queue.Queue() if use_server_mic else None
    stop_capture: Optional[threading.Event] = threading.Event() if use_server_mic else None
    capture_thread: Optional[threading.Thread] = None
    if use_server_mic and capture_queue is not None and stop_capture is not None:
        capture_thread = start_server_mic_capture(
            config.devices.audio_input_source,
            config.devices.audio_input_device,
            capture_queue,
            stop_capture,
        )
        if not capture_thread:
            logger.warning(
                "Server mic capture could not start (source=%s device=%s); no voice input from server device",
                getattr(config.devices, "audio_input_source", None),
                getattr(config.devices, "audio_input_device", None),
            )
            use_server_mic = False
        else:
            logger.info("Server mic capture thread started; waiting for first PCM chunk and user_amplitude")

    # VLM: Multi-frame capture for vision models
    # Frames are captured continuously in browser ring buffer (browser camera)
    # or in server-side FrameBroker (USB camera), then selected on ASR final
    vision_configured = getattr(llm_config, "enable_vision", False)  # User checked "Enable Vision (VLM)"
    vision_enabled = vision_configured
    
    # Disable vision if camera is set to "none"
    if vision_enabled and config.devices.video_source == "none":
        vision_enabled = False
        logger.info("[VLM] Vision disabled: camera set to 'none'")
    
    vision_frames_count = getattr(llm_config, "vision_frames", 4)
    vision_quality = getattr(llm_config, "vision_quality", 0.7)
    vision_max_width = getattr(llm_config, "vision_max_width", 640)
    vision_buffer_fps = getattr(llm_config, "vision_buffer_fps", 3.0)
    
    # Determine if using server-side camera (USB)
    use_server_camera = (
        config.devices.video_source == "usb" 
        and bool(config.devices.video_device)
    )
    
    # Multi-frame response handling (for browser camera)
    browser_frames_event = asyncio.Event()
    browser_frames_data: dict = {"frames": [], "t_start": 0.0, "t_end": 0.0}
    
    # Track speech timing for frame selection
    speech_start_time: Optional[float] = None  # Set on first asr_partial
    
    # Frame broker for server camera
    _frame_broker = None
    if vision_enabled and use_server_camera:
        try:
            from multi_modal_ai_studio.backends.vision.frame_broker import get_frame_broker
            _frame_broker = get_frame_broker()
            logger.info("[VLM] Using FrameBroker for server camera frames")
        except ImportError:
            logger.warning("[VLM] FrameBroker not available, server camera VLM disabled")
    
    if vision_enabled:
        logger.info(
            "VLM vision enabled: n_frames=%d, quality=%.1f, max_width=%d, buffer_fps=%.1f, server_camera=%s",
            vision_frames_count, vision_quality, vision_max_width, vision_buffer_fps, use_server_camera
        )
    
    async def start_vlm_capture() -> None:
        """Tell browser to start capturing frames into ring buffer.
        
        For server camera (USB), WebRTC already stores frames in FrameBroker,
        so we only need to notify browser for browser camera.
        """
        if not vision_enabled:
            return
        if use_server_camera:
            # Server camera uses FrameBroker, no browser action needed
            logger.debug("[VLM] Server camera uses FrameBroker, no browser capture needed")
            return
        try:
            await ws.send_str(json.dumps({
                "type": "vlm_start_capture",
                "fps": vision_buffer_fps,
                "quality": vision_quality,
                "max_width": vision_max_width,
            }))
            logger.debug("[VLM] Started browser frame capture")
        except Exception as e:
            logger.warning("[VLM] Failed to start capture: %s", e)
    
    async def request_vlm_frames(t_start: float, t_end: float, n_frames: int, timeout: float = 3.0) -> list:
        """Request n_frames evenly spaced between t_start and t_end.
        
        For browser camera: requests from browser's JavaScript ring buffer.
        For server camera: reads from FrameBroker (populated by WebRTC track).
        
        Returns list of frame data URLs, or empty list if unavailable.
        """
        if not vision_enabled:
            return []
        
        # Server camera: read from FrameBroker
        if use_server_camera and _frame_broker is not None:
            try:
                loop = asyncio.get_event_loop()
                frames = await loop.run_in_executor(
                    None, 
                    lambda: _frame_broker.get_frames(t_start, t_end, n_frames, vision_max_width)
                )
                logger.info("[VLM] Retrieved %d frames from FrameBroker (t=%.2f to %.2f)", 
                           len(frames), t_start, t_end)
                return frames
            except Exception as e:
                logger.warning("[VLM] FrameBroker get_frames failed: %s", e)
                return []
        
        # Browser camera: request from browser's JavaScript ring buffer
        browser_frames_event.clear()
        browser_frames_data["frames"] = []
        
        try:
            await ws.send_str(json.dumps({
                "type": "vlm_get_frames",
                "t_start": t_start,
                "t_end": t_end,
                "n_frames": n_frames,
            }))
            logger.info("[VLM] Requested %d frames from browser (t=%.2f to %.2f)", n_frames, t_start, t_end)
        except Exception as e:
            logger.warning("[VLM] Failed to request frames: %s", e)
            return []
        
        try:
            await asyncio.wait_for(browser_frames_event.wait(), timeout=timeout)
            frames = browser_frames_data.get("frames", [])
            logger.info("[VLM] Received %d frames from browser", len(frames))
            return frames
        except asyncio.TimeoutError:
            logger.warning("[VLM] Timeout waiting for browser frames")
            return []
    
    async def stop_vlm_capture() -> None:
        """Tell browser to stop capturing frames."""
        if not vision_enabled:
            return
        if use_server_camera:
            # Server camera uses FrameBroker, no browser action needed
            logger.debug("[VLM] Server camera uses FrameBroker, no browser stop needed")
            return
        try:
            await ws.send_str(json.dumps({"type": "vlm_stop_capture"}))
            logger.debug("[VLM] Stopped browser frame capture")
        except Exception as e:
            logger.warning("[VLM] Failed to stop capture: %s", e)

    async def send_event(event_dict: Dict[str, Any]) -> None:
        try:
            await ws.send_str(json.dumps({"type": "event", "event": event_dict}))
        except Exception as e:
            logger.warning("Send event failed: %s", e)

    connection_start_time = time.time()
    _amplitude_window_s = 0.025  # 25ms window so timeline bars get one value per 25ms (matches client tStep=0.025)

    async def _feed_pcm_preview_only(
        pcm_bytes: bytes,
        last_amplitude_time: float,
        amplitude_interval: float,
    ) -> Tuple[float, bool, float, float]:
        """Preview (before session start): amplitude only. No ASR, no timeline. Sends user_amplitude at throttle."""
        now = time.time() - connection_start_time
        amplitudes = _pcm_rms_slices(pcm_bytes, sample_rate=16000, window_s=_amplitude_window_s)
        did_send = False
        amp = 0.0
        for i, a in enumerate(amplitudes):
            t = now - (len(amplitudes) - 1 - i) * _amplitude_window_s - _amplitude_window_s / 2
            if t < 0:
                continue
            if t - last_amplitude_time >= amplitude_interval:
                # Clamp so timestamps are strictly increasing (avoids non-monotonic when now jitters)
                t = max(t, last_amplitude_time + amplitude_interval)
                last_amplitude_time = t
                amp = a
                did_send = True
                try:
                    await ws.send_str(
                        json.dumps({"type": "user_amplitude", "timestamp": round(t, 3), "amplitude": round(amp, 2)})
                    )
                except Exception as e:
                    logger.warning("Send user_amplitude (preview) failed: %s", e)
        return (last_amplitude_time, did_send, amp, now)

    async def _feed_pcm_to_pipeline(
        pcm_bytes: bytes,
        last_amplitude_time: float,
        amplitude_interval: float,
    ) -> Tuple[float, bool, float, float]:
        """Full feed: ASR + timeline + user_amplitude. Must only be called after start_session."""
        if session.timeline.start_time is None:
            return (last_amplitude_time, False, 0.0, 0.0)
        now = time.time() - session.timeline.start_time
        await asr.send_audio(pcm_bytes)
        amplitudes = _pcm_rms_slices(pcm_bytes, sample_rate=16000, window_s=_amplitude_window_s)
        did_send = False
        amp = 0.0
        for i, a in enumerate(amplitudes):
            t = now - (len(amplitudes) - 1 - i) * _amplitude_window_s - _amplitude_window_s / 2
            if t < 0:
                continue
            if t - last_amplitude_time >= amplitude_interval:
                # Clamp so timestamps are strictly increasing (avoids non-monotonic when now jitters)
                t = max(t, last_amplitude_time + amplitude_interval)
                last_amplitude_time = t
                amp = a
                did_send = True
                session.timeline.add_audio_amplitude(amplitude=amp, source="user", timestamp=t)
                try:
                    await ws.send_str(
                        json.dumps({"type": "user_amplitude", "timestamp": round(t, 3), "amplitude": round(amp, 2)})
                    )
                except Exception as e:
                    logger.warning("Send user_amplitude failed: %s", e)
        return (last_amplitude_time, did_send, amp, now)

    async def receive_loop() -> None:
        """Read from WebSocket: config (first), then binary PCM or stop."""
        last_amplitude_time = 0.0
        last_amplitude_log_time = 0.0  # throttle debug log to ~1s (see INVESTIGATE_USER_AMPLITUDE_ARTIFACT.md)
        amplitude_interval = 0.025  # one per 25ms slice; smoother user waveform
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        obj = json.loads(msg.data)
                        if obj.get("type") == "start_session":
                            # Optional: client sends current config so saved session has correct devices (e.g. speaker changed after preview)
                            if "config" in obj:
                                try:
                                    payload = _normalize_frontend_config(obj["config"])
                                    existing = session.config.to_dict()
                                    if "devices" in payload:
                                        existing["devices"] = {**existing.get("devices", {}), **payload["devices"]}
                                    if "device_labels" in payload:
                                        existing["device_labels"] = {**(existing.get("device_labels") or {}), **payload["device_labels"]}
                                    if "device_types" in payload:
                                        existing["device_types"] = {**(existing.get("device_types") or {}), **payload["device_types"]}
                                    session.config = SessionConfig.from_dict(existing)
                                    # Log if speaker was updated to Server USB/ALSA so TTS will play there
                                    dc = session.config.devices
                                    if dc.audio_output_source in ("alsa", "usb") and dc.audio_output_device:
                                        logger.info(
                                            "Server speaker from start_session config: device=%s (TTS will play to ALSA)",
                                            dc.audio_output_device,
                                        )
                                except Exception as e:
                                    logger.warning("Could not merge start_session config: %s", e)
                            # Both mics: start session and set pipeline_live so ASR/timeline run from here on
                            if session.timeline.start_time is None:
                                session.start()
                                await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})
                                # VLM: Start browser frame capture when session starts
                                await start_vlm_capture()
                                pipeline_live.set()
                                logger.info("Voice pipeline: start_session received; pipeline live (ASR + timeline)")
                        if obj.get("type") == "stop":
                            stats = obj.get("system_stats")
                            if isinstance(stats, list) and (not getattr(session, "system_stats", None) or len(session.system_stats) == 0):
                                session.system_stats = stats
                            tts_segments = obj.get("tts_playback_segments")
                            if isinstance(tts_segments, list):
                                session.tts_playback_segments = tts_segments
                            amp_history = obj.get("audio_amplitude_history")
                            if isinstance(amp_history, list):
                                session.audio_amplitude_history = amp_history
                            ttl_bands = obj.get("ttl_bands")
                            if isinstance(ttl_bands, list):
                                session.ttl_bands = ttl_bands
                                session.apply_ttl_bands()  # single source of truth: metrics from bands (first audio_amplitude tts)
                            session.app_version = __version__
                            stopped.set()
                            return
                        # VLM: handle multi-frame response from browser
                        if obj.get("type") == "vlm_frames":
                            frames = obj.get("frames", [])
                            t_start = obj.get("t_start", 0.0)
                            t_end = obj.get("t_end", 0.0)
                            # Extract just the data URLs from frames
                            frame_urls = []
                            for f in frames:
                                if isinstance(f, dict) and f.get("data") and f["data"].startswith("data:"):
                                    frame_urls.append(f["data"])
                                elif isinstance(f, str) and f.startswith("data:"):
                                    frame_urls.append(f)
                            browser_frames_data["frames"] = frame_urls
                            browser_frames_data["t_start"] = t_start
                            browser_frames_data["t_end"] = t_end
                            browser_frames_event.set()
                            if frame_urls:
                                logger.info("[VLM] Received %d frames from browser (t=%.2f to %.2f)", len(frame_urls), t_start, t_end)
                            else:
                                logger.warning("[VLM] Browser frames response empty (no camera?)")
                    except json.JSONDecodeError:
                        pass
                    continue
                if msg.type == web.WSMsgType.BINARY:
                    # When using Server USB mic, audio comes from server capture task; ignore browser PCM.
                    if not use_server_mic:
                        if session.timeline.start_time is None:
                            last_amplitude_time, did_send, amp, now = await _feed_pcm_preview_only(
                                msg.data, last_amplitude_time, amplitude_interval
                            )
                        else:
                            last_amplitude_time, did_send, amp, now = await _feed_pcm_to_pipeline(
                                msg.data, last_amplitude_time, amplitude_interval
                            )
                        if did_send and amp >= 20.0:
                            try:
                                n = len(msg.data) // 2
                                raw_rms = math.sqrt(sum(s * s for s in struct.unpack(f"{n}h", msg.data)) / n) if n else 0.0
                                logger.warning(
                                    "[user_amplitude_high] session_t=%.2fs amp_0_100=%.2f raw_rms=%.1f chunk_len=%d",
                                    now, amp, raw_rms, len(msg.data),
                                )
                            except Exception:
                                pass
                        if did_send and now - last_amplitude_log_time >= 1.0:
                            last_amplitude_log_time = now
                            logger.info("[user_amplitude] session_t=%.1fs chunk_len=%d amp_0_100=%.2f", now, len(msg.data), amp)
                if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    stopped.set()
                    return
        except Exception as e:
            logger.debug("receive_loop ended: %s", e)
        finally:
            stopped.set()

    async def asr_consumer() -> None:
        """Independent ASR task: forward every partial/final to client immediately; enqueue finals for turn_executor.
        Enables barge-in (turn_executor can be cancelled when new final arrives) and avoids phantom partial at tts_complete.
        On stream end, if we had a partial but no final (e.g. user stopped before VAD), enqueue a synthetic final so one turn runs."""
        last_asr_final_text: Optional[str] = None
        last_asr_final_ts: Optional[float] = None
        last_partial_text: Optional[str] = None
        last_partial_ts: Optional[float] = None
        asr_received_count = 0
        try:
            async for result in asr.receive_results():
                if stopped.is_set():
                    break
                if result is None:
                    break

                asr_received_count += 1
                if asr_received_count == 1:
                    logger.info("[asr] First result received from Riva: is_final=%s text=%r", getattr(result, "is_final", True), (result.text or "").strip()[:80])

                is_final = getattr(result, "is_final", True)
                text = (result.text or "").strip()
                if not text:
                    continue

                now_ts = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                ts = (getattr(result, "metadata", {}) or {}).get("event_timestamp")
                if ts is None:
                    ts = now_ts
                ev_type = "asr_partial" if not is_final else "asr_final"

                if is_final and last_asr_final_text is not None and text == last_asr_final_text:
                    if last_asr_final_ts is not None and abs(ts - last_asr_final_ts) < 2.0:
                        logger.debug("[asr] Skipping duplicate asr_final: %r", text[:50])
                        continue
                if not is_final and last_asr_final_ts is not None:
                    if ts <= last_asr_final_ts:
                        logger.debug("[asr] Skipping stale asr_partial (ts=%.2f <= last_final=%.2f): %r", ts, last_asr_final_ts, text[:50])
                        continue
                    # Only treat as same-utterance phantom if within ~2.5s of last final.
                    # Later partials are a new utterance (e.g. user said "Me a joke" then "Me again").
                    same_utterance_window = 2.5
                    if (ts - last_asr_final_ts) > same_utterance_window:
                        pass  # New utterance; do not skip
                    elif last_asr_final_text and (
                        text == last_asr_final_text
                        or last_asr_final_text.startswith(text)
                        or text.startswith(last_asr_final_text)
                    ):
                        logger.debug("[asr] Skipping phantom asr_partial (same utterance as last final): %r", text[:50])
                        continue

                if not is_final:
                    # VLM: Track speech start time for frame synchronization.
                    # Detect "ghost partial" gaps — if the previous partial was
                    # > SPEECH_GAP_THRESH seconds ago, the earlier partial was
                    # likely triggered by background noise.  Reset so the frame
                    # window starts from the *real* speech, not the ghost.
                    nonlocal speech_start_time
                    if vision_enabled:
                        SPEECH_GAP_THRESH = 3.0  # seconds – generous enough for natural pauses
                        if speech_start_time is None:
                            speech_start_time = ts
                            logger.debug("[VLM] Speech started at t=%.2f", ts)
                        elif last_partial_ts is not None and (ts - last_partial_ts) > SPEECH_GAP_THRESH:
                            logger.info(
                                "[VLM] Partial gap %.1fs detected (ghost partial?) — "
                                "resetting speech_start from %.2f to %.2f",
                                ts - last_partial_ts, speech_start_time, ts,
                            )
                            speech_start_time = ts
                    last_partial_text = text
                    last_partial_ts = ts
                    session.timeline.add_event(ev_type, Lane.SPEECH, data={"text": text, "confidence": getattr(result, "confidence", 1.0)})
                    # Fire-and-forget so we don't block on slow client; keeps asr_consumer able to receive 2nd turn
                    asyncio.create_task(send_event({
                        "event_type": ev_type,
                        "lane": "speech",
                        "data": {"text": text, "confidence": getattr(result, "confidence", 1.0)},
                        "timestamp": ts,
                    }))
                else:
                    await send_event({
                        "event_type": ev_type,
                        "lane": "speech",
                        "data": {"text": text, "confidence": getattr(result, "confidence", 1.0)},
                        "timestamp": ts,
                    })
                    last_asr_final_text = text
                    last_asr_final_ts = ts
                    finals_count = getattr(asr_consumer, "_finals_count", 0) + 1
                    asr_consumer._finals_count = finals_count
                    logger.info("[asr] asr_final #%d enqueued for LLM/TTS: %r", finals_count, text[:80])
                    finals_queue.put_nowait(result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("asr_consumer error: %s", e)
        finally:
            logger.info("[asr] Stream ended; received %d ASR result(s) total", asr_received_count)
            # Only create a synthetic final when the stream had no final at all (e.g. user stopped before
            # VAD sent a final). Do NOT create one when we already had a final and the last partial is
            # different (e.g. Riva sent early final "How about computer?" then partials "joke") — that
            # would create a phantom extra turn; the partial is the tail of the same utterance.
            if last_partial_text and last_asr_final_text is None:
                try:
                    now_ts = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                    syn_ts = last_partial_ts if last_partial_ts is not None else now_ts
                    synthetic = ASRResult(
                        text=last_partial_text,
                        is_final=True,
                        confidence=1.0,
                        metadata={"event_timestamp": syn_ts},
                    )
                    logger.info("[asr] Stream ended with only partial; enqueueing synthetic final for LLM/TTS: %r", last_partial_text[:80])
                    finals_queue.put_nowait(synthetic)
                    # Add to timeline so replay/saved session has this final (use partial's time, not stream-end)
                    session.timeline.add_event("asr_final", Lane.SPEECH, data={"text": last_partial_text, "confidence": 1.0})
                    # Send asr_final to client so UI shows final_transcript (otherwise only partials were sent)
                    await send_event({
                        "event_type": "asr_final",
                        "lane": "speech",
                        "data": {"text": last_partial_text, "confidence": 1.0},
                        "timestamp": syn_ts,
                    })
                except asyncio.QueueFull:
                    pass
                except Exception as e:
                    logger.warning("Failed to send synthetic asr_final event: %s", e)
            try:
                finals_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def turn_executor() -> None:
        """Process ASR finals one at a time: LLM -> TTS. Waits on finals_queue (fed by asr_consumer).
        Future: can be cancelled when new final arrives for barge-in."""
        nonlocal speech_start_time
        turn_index = 0
        try:
            while not stopped.is_set():
                result = await finals_queue.get()
                if result is None:
                    break
                text = (result.text or "").strip()
                if not text:
                    continue

                turn_index += 1
                logger.info("[timing] turn #%d start (asr_final received): %r", turn_index, text[:80])
                session.start_turn(user_transcript=text)
                session.update_turn_transcript(text, confidence=getattr(result, "confidence", 1.0))
                session.timeline.add_event("user_speech_end", Lane.SYSTEM)

                ts_llm_start = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                logger.info("[timing] llm_start @ %.2fs", ts_llm_start)
                await send_event({
                    "event_type": "llm_start",
                    "lane": "llm",
                    "data": {},
                    "timestamp": ts_llm_start,
                })
                session.timeline.add_event("llm_start", Lane.LLM)
                
                # VLM: Request frames (time-synchronized with speech)
                image_data_urls: list = []
                speech_duration_secs: float = 0.0
                _model_lower = (llm_config.model or "").lower()
                is_cosmos = "cosmos" in _model_lower and "reason" in _model_lower
                if vision_enabled:
                    # Calculate speech time window
                    t_end = ts_llm_start  # End time is now (ASR final arrived)
                    # Use speech_start_time if tracked, otherwise use a default window
                    t_start = speech_start_time if speech_start_time is not None else max(0, t_end - 3.0)
                    # ASR produces its first partial ~0.5s after the user actually
                    # starts speaking.  Pull t_start back to capture those early
                    # frames so the video has full context.
                    ASR_LATENCY_LOOKBACK = 0.5  # seconds
                    t_start = max(0, t_start - ASR_LATENCY_LOOKBACK)
                    speech_duration_secs = t_end - t_start
                    
                    # Guard: cap the speech window to MAX_SPEECH_WINDOW_SECS.
                    # Ghost asr_partials (background noise triggering a partial) can
                    # set speech_start_time far too early, inflating the window to
                    # 15-30s when the real speech was only 2-3s.  Capping prevents
                    # pulling in dozens of irrelevant frames.
                    MAX_SPEECH_WINDOW_SECS = 10.0
                    if speech_duration_secs > MAX_SPEECH_WINDOW_SECS:
                        logger.warning(
                            "[VLM] Speech window %.1fs exceeds cap %.1fs — likely ghost partial. "
                            "Clamping t_start from %.2f to %.2f",
                            speech_duration_secs, MAX_SPEECH_WINDOW_SECS,
                            t_start, t_end - MAX_SPEECH_WINDOW_SECS,
                        )
                        t_start = t_end - MAX_SPEECH_WINDOW_SECS
                        speech_duration_secs = MAX_SPEECH_WINDOW_SECS
                    
                    # Cosmos models: request ALL available frames for video encoding.
                    # Video preserves temporal info (motion, actions, state changes).
                    # FrameBroker stores at ~10fps → 3s speech ≈ 30 frames, 5s ≈ 50.
                    # Non-Cosmos: keep few frames (each image ≈ 1000 tokens).
                    if is_cosmos:
                        n_frames_request = 100  # large cap; FrameBroker returns what's available
                    else:
                        n_frames_request = vision_frames_count  # default 4
                    
                    source = "FrameBroker" if use_server_camera else "browser"
                    logger.info(
                        "[VLM] Requesting %d frames from %s (speech: %.2fs–%.2fs, dur=%.2fs, cosmos=%s)",
                        n_frames_request, source, t_start, t_end, speech_duration_secs, is_cosmos,
                    )
                    
                    ts_frame_request = time.time()
                    image_data_urls = await request_vlm_frames(
                        t_start=t_start,
                        t_end=t_end,
                        n_frames=n_frames_request,
                        timeout=3.0,
                    )
                    
                    if image_data_urls:
                        frame_latency_ms = (time.time() - ts_frame_request) * 1000
                        logger.info("[VLM] Received %d frames, latency=%.0fms", len(image_data_urls), frame_latency_ms)
                        session.timeline.add_event("vlm_frames_captured", Lane.LLM, data={
                            "n_frames": len(image_data_urls),
                            "t_start": round(t_start, 2),
                            "t_end": round(t_end, 2),
                            "speech_duration": round(speech_duration_secs, 2),
                            "latency_ms": round(frame_latency_ms),
                            "source": source,
                            "cosmos_video": is_cosmos,
                        })
                    else:
                        logger.warning("[VLM] Vision enabled but frame capture failed")
                    
                    # Reset speech start time for next turn
                    speech_start_time = None
                
                full_response = ""
                llm_first_token_sent = False

                effective_system_prompt = llm_config.system_prompt
                if vision_enabled and image_data_urls:
                    effective_system_prompt = getattr(llm_config, "vision_system_prompt", llm_config.system_prompt)
                elif vision_enabled and not image_data_urls:
                    logger.warning("[VLM] Vision enabled but no frames captured")

                # ── Interleaved vs sequential TTS ──
                use_stream_tts = getattr(tts_config, "stream_tts", True)

                # ── Shared TTS state ──
                tts_first_sent = False
                ts_tts_first = 0.0
                last_tts_amplitude_time = 0.0
                tts_amplitude_interval = 0.05
                server_speaker_proc = None
                tts_consumer_error: Optional[Exception] = None

                async def _send_tts_audio(chunk):
                    nonlocal tts_first_sent, ts_tts_first, last_tts_amplitude_time, server_speaker_proc
                    if not tts_first_sent:
                        ts_tts_first = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                        ref_label = "llm_first_token" if use_stream_tts else "llm_complete"
                        ref_ts = ts_first if use_stream_tts else ts_llm_complete
                        logger.info("[timing] tts_first_audio @ %.2fs (%.2fs after %s)", ts_tts_first, ts_tts_first - ref_ts, ref_label)
                        session.timeline.add_event("tts_first_audio", Lane.TTS)
                        await send_event({"event_type": "tts_first_audio", "lane": "tts", "data": {}, "timestamp": ts_tts_first})
                        tts_first_sent = True
                    _use_speaker = session.config.devices.audio_output_source in ("alsa", "usb") and bool(
                        session.config.devices.audio_output_device
                    )
                    _out_device = session.config.devices.audio_output_device
                    if _use_speaker and chunk.audio:
                        if server_speaker_proc is None:
                            server_speaker_proc = start_server_speaker_playback(_out_device, chunk.sample_rate)
                            if server_speaker_proc is None:
                                logger.warning(
                                    "Server speaker playback could not start for %s; check aplay and device",
                                    _out_device,
                                )
                        if server_speaker_proc is not None and server_speaker_proc.stdin and not server_speaker_proc.stdin.closed:
                            try:
                                server_speaker_proc.stdin.write(chunk.audio)
                                server_speaker_proc.stdin.flush()
                            except (BrokenPipeError, OSError) as e:
                                logger.debug("Server speaker write failed: %s", e)
                                server_speaker_proc = None
                    if session.timeline.start_time is not None and chunk.audio:
                        now = time.time() - session.timeline.start_time
                        if now - last_tts_amplitude_time >= tts_amplitude_interval:
                            amp = _pcm_rms_to_amplitude(chunk.audio)
                            session.timeline.add_audio_amplitude(amplitude=amp, source="tts")
                            last_tts_amplitude_time = now
                    b64 = base64.b64encode(chunk.audio).decode("ascii")
                    await ws.send_str(json.dumps({
                        "type": "tts_audio",
                        "data": b64,
                        "sample_rate": chunk.sample_rate,
                        "is_final": chunk.is_final,
                    }))

                async def _tts_consumer(tts_q: asyncio.Queue) -> None:
                    """Background task: pull text chunks from queue, synthesize, send audio."""
                    nonlocal tts_consumer_error
                    try:
                        chunk_idx = 0
                        while True:
                            text_chunk = await tts_q.get()
                            if text_chunk is None:
                                break
                            chunk_idx += 1
                            logger.info("[stream_tts] TTS chunk #%d (%d words, %d chars)", chunk_idx, len(text_chunk.split()), len(text_chunk))
                            async for audio_chunk in tts.synthesize_stream(text_chunk):
                                if stopped.is_set():
                                    return
                                await _send_tts_audio(audio_chunk)
                    except Exception as e:
                        logger.exception("[stream_tts] TTS consumer error: %s", e)
                        tts_consumer_error = e

                # ── LLM generation + TTS ──
                ts_first = ts_llm_start
                ts_llm_complete = ts_llm_start
                tts_started = False
                chunk_buf = TTSChunkBuffer() if use_stream_tts else None
                tts_q: Optional[asyncio.Queue] = None
                tts_task: Optional[asyncio.Task] = None

                try:
                    async for token in llm.generate_stream(
                        prompt=text,
                        history=None,
                        system_prompt=effective_system_prompt,
                        image_data_urls=image_data_urls if image_data_urls else None,
                        speech_duration=speech_duration_secs if speech_duration_secs > 0 else None,
                    ):
                        if stopped.is_set():
                            break
                        if token.token:
                            full_response += token.token
                            if not llm_first_token_sent:
                                llm_first_token_sent = True
                                ts_first = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                                logger.info("[timing] llm_first_token @ %.2fs (prefill took %.2fs)", ts_first, ts_first - ts_llm_start)
                                session.timeline.add_event("llm_first_token", Lane.LLM)
                                await send_event({"event_type": "llm_first_token", "lane": "llm", "data": {}, "timestamp": ts_first})

                            if use_stream_tts:
                                ready = chunk_buf.add(token.token)
                                if ready:
                                    if not tts_started:
                                        tts_started = True
                                        tts_q = asyncio.Queue()
                                        tts_task = asyncio.create_task(_tts_consumer(tts_q))
                                        session.timeline.add_event("tts_start", Lane.TTS)
                                        await send_event({"event_type": "tts_start", "lane": "tts", "data": {"stream_tts": True}, "timestamp": (time.time() - session.timeline.start_time) if session.timeline.start_time else 0})
                                        await ws.send_str(json.dumps({"type": "tts_start"}))
                                    await tts_q.put(ready)

                    session.timeline.add_event("llm_complete", Lane.LLM, data={"text": full_response})
                except Exception as e:
                    logger.exception("LLM error: %s", e)
                    full_response = full_response or "Sorry, I had an error."
                    session.timeline.add_event("llm_complete", Lane.LLM, data={"text": full_response, "error": str(e)})

                ts_llm_complete = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                logger.info("[timing] llm_complete @ %.2fs (llm took %.2fs)", ts_llm_complete, ts_llm_complete - ts_llm_start)
                await send_event({"event_type": "llm_complete", "lane": "llm", "data": {"text": full_response}, "timestamp": ts_llm_complete})

                session.update_turn_response(full_response)
                await send_event({"event_type": "chat", "user": text, "assistant": full_response})

                if not full_response.strip():
                    session.end_turn()
                    continue

                try:
                    if use_stream_tts:
                        remainder = chunk_buf.flush()
                        if remainder:
                            if not tts_started:
                                tts_started = True
                                tts_q = asyncio.Queue()
                                tts_task = asyncio.create_task(_tts_consumer(tts_q))
                                session.timeline.add_event("tts_start", Lane.TTS)
                                await send_event({"event_type": "tts_start", "lane": "tts", "data": {"stream_tts": True}, "timestamp": (time.time() - session.timeline.start_time) if session.timeline.start_time else 0})
                                await ws.send_str(json.dumps({"type": "tts_start"}))
                            await tts_q.put(remainder)
                        if tts_q is not None:
                            await tts_q.put(None)
                        if tts_task is not None:
                            await tts_task
                        if tts_consumer_error:
                            raise tts_consumer_error
                    else:
                        ts_tts_start = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                        session.timeline.add_event("tts_start", Lane.TTS)
                        await send_event({
                            "event_type": "tts_start",
                            "lane": "tts",
                            "data": {},
                            "timestamp": ts_tts_start,
                        })
                        await ws.send_str(json.dumps({"type": "tts_start"}))

                        tts_first_sent = False
                        tts_amplitude_next_t = 0.0
                        server_speaker_proc = None
                        async for chunk in tts.synthesize_stream(full_response):
                            if stopped.is_set():
                                break
                            if not tts_first_sent:
                                ts_tts_first = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                                tts_amplitude_next_t = ts_tts_first
                                logger.info("[timing] tts_first_audio @ %.2fs (tts first chunk after %.2fs from llm_complete)", ts_tts_first, ts_tts_first - ts_llm_complete)
                                session.timeline.add_event("tts_first_audio", Lane.TTS)
                                await send_event({
                                    "event_type": "tts_first_audio",
                                    "lane": "tts",
                                    "data": {},
                                    "timestamp": ts_tts_first,
                                })
                                tts_first_sent = True
                            _use_speaker = session.config.devices.audio_output_source in ("alsa", "usb") and bool(
                                session.config.devices.audio_output_device
                            )
                            _out_device = session.config.devices.audio_output_device
                            if _use_speaker and chunk.audio:
                                if server_speaker_proc is None:
                                    server_speaker_proc = start_server_speaker_playback(
                                        _out_device,
                                        chunk.sample_rate,
                                    )
                                    if server_speaker_proc is None:
                                        logger.warning(
                                            "Server speaker playback could not start for %s; check aplay and device (e.g. same device as mic may be busy)",
                                            _out_device,
                                        )
                                if server_speaker_proc is not None and server_speaker_proc.stdin and not server_speaker_proc.stdin.closed:
                                    try:
                                        server_speaker_proc.stdin.write(chunk.audio)
                                        server_speaker_proc.stdin.flush()
                                    except (BrokenPipeError, OSError) as e:
                                        logger.debug("Server speaker write failed (aplay may have exited): %s", e)
                                        server_speaker_proc = None
                            amplitude_segments: List[Dict[str, Any]] = []
                            if session.timeline.start_time is not None and chunk.audio:
                                amps = _pcm_rms_slices(
                                    chunk.audio,
                                    sample_rate=chunk.sample_rate,
                                    window_s=_amplitude_window_s,
                                )
                                ts_send = time.time() - session.timeline.start_time
                                for i, a in enumerate(amps):
                                    t_start = ts_send - (len(amps) - i) * _amplitude_window_s
                                    t_end = t_start + _amplitude_window_s
                                    session.timeline.add_audio_amplitude(
                                        amplitude=a, source="tts", timestamp=t_start
                                    )
                                    amplitude_segments.append({
                                        "startTime": round(t_start, 3),
                                        "endTime": round(t_end, 3),
                                        "amplitude": round(a, 2),
                                    })
                                tts_amplitude_next_t = ts_send
                            b64 = base64.b64encode(chunk.audio).decode("ascii")
                            payload = {
                                "type": "tts_audio",
                                "data": b64,
                                "sample_rate": chunk.sample_rate,
                                "is_final": chunk.is_final,
                            }
                            if amplitude_segments:
                                payload["amplitude_segments"] = amplitude_segments
                            await ws.send_str(json.dumps(payload))
                    ts_tts_complete = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                    logger.info("[timing] tts_complete @ %.2fs (tts took %.2fs)", ts_tts_complete, ts_tts_complete - ts_tts_first if tts_first_sent else 0)
                    session.timeline.add_event("tts_complete", Lane.TTS)
                    await send_event({"event_type": "tts_complete", "lane": "tts", "data": {}, "timestamp": ts_tts_complete})
                except Exception as e:
                    logger.exception("TTS error: %s", e)
                finally:
                    if tts_task is not None and not tts_task.done():
                        tts_task.cancel()
                        try:
                            await tts_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    if server_speaker_proc is not None:
                        stop_server_speaker_playback(server_speaker_proc)

                session.end_turn()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("turn_executor error: %s", e)

    if not use_server_mic:
        await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})
        # VLM: Start browser frame capture when session starts
        await start_vlm_capture()
    _user_amplitude_sent = False

    async def server_capture_consumer() -> None:
        """Read PCM from server mic capture queue. Preview: _feed_pcm_preview_only. Live: _feed_pcm_to_pipeline."""
        nonlocal _user_amplitude_sent
        if capture_queue is None:
            return
        loop = asyncio.get_event_loop()
        last_amplitude_time = 0.0
        amplitude_interval = 0.025
        first_get = True
        while not stopped.is_set():
            try:
                chunk = await loop.run_in_executor(None, capture_queue.get)
            except Exception as e:
                logger.warning("Server capture consumer get failed: %s", e)
                break
            if chunk is None:
                if first_get:
                    logger.warning(
                        "Server mic capture sent None on first get (capture failed to produce any PCM); check arecord and device"
                    )
                else:
                    logger.info("Server mic capture ended (None received)")
                break
            first_get = False
            if pipeline_live.is_set():
                if last_amplitude_time > 1.0:
                    last_amplitude_time = 0.0
                last_amplitude_time, did_send, amp, _ = await _feed_pcm_to_pipeline(
                    chunk, last_amplitude_time, amplitude_interval
                )
                if did_send and not _user_amplitude_sent:
                    _user_amplitude_sent = True
                    logger.info("First user_amplitude sent to client (live); amp=%.2f", amp)
            else:
                last_amplitude_time, did_send, amp, _ = await _feed_pcm_preview_only(
                    chunk, last_amplitude_time, amplitude_interval
                )
                if did_send and not _user_amplitude_sent:
                    _user_amplitude_sent = True
                    logger.info("First user_amplitude sent to client (preview); amp=%.2f", amp)

    server_capture_task: Optional[asyncio.Task] = None
    if use_server_mic and capture_thread is not None and capture_queue is not None:
        server_capture_task = asyncio.create_task(server_capture_consumer())

    recv_task = asyncio.create_task(receive_loop())
    asr_task: Optional[asyncio.Task] = None
    turn_task: Optional[asyncio.Task] = None
    system_stats_task: Optional[asyncio.Task] = None

    async def _system_stats_loop() -> None:
        """Send CPU/GPU at 10 Hz over WebSocket and append to session.system_stats for save."""
        loop = asyncio.get_event_loop()
        interval = 0.1
        while not stopped.is_set():
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            if stopped.is_set():
                break
            if session.timeline.start_time is None:
                continue
            try:
                stats = await loop.run_in_executor(None, system_stats_module.gather_system_stats)
            except Exception as e:
                logger.debug("System stats gather failed: %s", e)
                continue
            t = time.time() - session.timeline.start_time
            cpu = stats.get("cpu_percent")
            gpu = stats.get("gpu_percent")
            session.system_stats.append({"t": t, "cpu": cpu, "gpu": gpu})
            try:
                await ws.send_str(
                    json.dumps({"type": "system_stats", "timestamp": t, "cpu_percent": cpu, "gpu_percent": gpu})
                )
            except Exception as e:
                logger.debug("Send system_stats failed: %s", e)

    # Wait for client to send start_session (both mics); then start ASR stream and turn executor
    await asyncio.wait(
        [asyncio.create_task(stopped.wait()), asyncio.create_task(pipeline_live.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not stopped.is_set():
        session.system_stats = []
        await asr.start_stream()
        asr_task = asyncio.create_task(asr_consumer())
        turn_task = asyncio.create_task(turn_executor())
        system_stats_task = asyncio.create_task(_system_stats_loop())
        await stopped.wait()
    if stop_capture is not None:
        stop_capture.set()
    if server_capture_task is not None:
        server_capture_task.cancel()
    if system_stats_task is not None:
        system_stats_task.cancel()
        try:
            await system_stats_task
        except asyncio.CancelledError:
            pass
    await asr.stop_stream()
    recv_task.cancel()
    if asr_task is not None:
        asr_task.cancel()
    # Do not cancel turn_task: asr_consumer's finally enqueues a synthetic final when stream
    # ends with only partials (user stopped before Riva VAD sent a final). Let turn_executor
    # drain finals_queue so it processes that synthetic final and runs LLM/TTS before we save.
    try:
        await recv_task
    except asyncio.CancelledError:
        pass
    if server_capture_task is not None:
        try:
            await server_capture_task
        except asyncio.CancelledError:
            pass
    if asr_task is not None:
        try:
            await asr_task
        except asyncio.CancelledError:
            pass
    if turn_task is not None:
        try:
            await asyncio.wait_for(turn_task, timeout=120.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning("turn_executor did not finish within 120s")
            turn_task.cancel()
            try:
                await turn_task
            except asyncio.CancelledError:
                pass

    # VLM: Stop browser frame capture
    await stop_vlm_capture()

    if session.timeline.start_time is None:
        return None  # preview-only (Server USB) and client closed before start_session

    metrics = session.calculate_metrics()

    # Optional: use LLM to generate a short title for the conversation
    if session.turns:
        async def _generate_title() -> None:
            transcript_parts = []
            for t in session.turns[:5]:  # first 5 turns to avoid huge prompt
                transcript_parts.append(f"User: {t.user_transcript or ''}")
                transcript_parts.append(f"Assistant: {(t.ai_response or '')[:200]}")
            transcript = "\n".join(transcript_parts).strip()[:800]
            prompt = f"""Based on this conversation, suggest a very short title (3-8 words). Reply with only the title, no quotes or extra punctuation.

{transcript}

Title:"""
            title_text = ""
            async for token in llm.generate_stream(
                prompt=prompt,
                history=None,
                system_prompt="You reply with only a short phrase. No explanation.",
            ):
                if token.token:
                    title_text += token.token
            title_text = title_text.strip().split("\n")[0].strip()[:50]
            if title_text:
                session.name = title_text
                logger.info("Session title: %s", session.name)

        try:
            await asyncio.wait_for(_generate_title(), timeout=15.0)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Could not generate session title: %s", e)

    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{session.session_id}.json"
    session.save(path)
    logger.info("Session saved: %s", path)
    return session.session_id


async def handle_voice_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for /ws/voice. First message must be { type: 'config', config: {...} }."""
    logger.info("Voice WebSocket connection requested")
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("Voice WebSocket prepared, waiting for config")

    if not request.app.get("session_dir"):
        await ws.send_str(json.dumps({"type": "error", "error": "Server missing session_dir"}))
        await ws.close()
        return ws

    config_payload = None
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=30.0)
        if msg.type != web.WSMsgType.TEXT:
            await ws.send_str(json.dumps({"type": "error", "error": "First message must be JSON config"}))
            await ws.close()
            return ws
        obj = json.loads(msg.data)
        if obj.get("type") != "config" or "config" not in obj:
            await ws.send_str(json.dumps({"type": "error", "error": "First message must be { type: 'config', config: {...} }"}))
            await ws.close()
            return ws
        config_payload = _normalize_frontend_config(obj["config"])
        session_config = SessionConfig.from_dict(config_payload)
        logger.info("Voice config received: asr=%s llm=%s tts=%s", session_config.asr.scheme, session_config.llm.api_base, session_config.tts.scheme)
    except asyncio.TimeoutError:
        await ws.send_str(json.dumps({"type": "error", "error": "Timeout waiting for config"}))
        await ws.close()
        return ws
    except Exception as e:
        logger.exception("Config parse error: %s", e)
        await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
        await ws.close()
        return ws

    # Use current effective session dir (e.g. mock_sessions); server keeps override in sync
    server = request.app.get("_server")
    if server is not None and hasattr(server, "_get_effective_session_dir"):
        session_dir = server._get_effective_session_dir().resolve()
    else:
        session_dir = Path(request.app.get("session_dir")).resolve()

    session_id = None
    try:
        session_id = await _run_voice_pipeline(ws, session_config, session_dir)
    except Exception as e:
        logger.exception("Voice pipeline error: %s", e)
        try:
            await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass
    finally:
        if session_id:
            try:
                await ws.send_str(json.dumps({"type": "session_saved", "session_id": session_id}))
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass

    return ws


async def handle_mic_preview_ws(request: web.Request) -> web.WebSocketResponse:
    """
    WebSocket for mic level preview only (no ASR/LLM/TTS).
    First message must be { type: 'config', config: { devices: { microphone: 'alsa:hw:2,0' } } }.
    Streams user_amplitude so the client can show the green waveform before the user clicks START.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("Mic preview WebSocket connection requested")

    config_payload = None
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=15.0)
        if msg.type != web.WSMsgType.TEXT:
            await ws.send_str(json.dumps({"type": "error", "error": "First message must be JSON config"}))
            await ws.close()
            return ws
        obj = json.loads(msg.data)
        if obj.get("type") != "config" or "config" not in obj:
            await ws.send_str(json.dumps({"type": "error", "error": "First message must be { type: 'config', config: {...} }"}))
            await ws.close()
            return ws
        config_payload = _normalize_frontend_config(obj["config"])
        session_config = SessionConfig.from_dict(config_payload)
    except asyncio.TimeoutError:
        await ws.send_str(json.dumps({"type": "error", "error": "Timeout waiting for config"}))
        await ws.close()
        return ws
    except Exception as e:
        logger.exception("Mic preview config parse error: %s", e)
        await ws.send_str(json.dumps({"type": "error", "error": str(e)}))
        await ws.close()
        return ws

    use_server_mic = session_config.devices.audio_input_source in ("alsa", "usb") and bool(
        session_config.devices.audio_input_device
    )
    if not use_server_mic:
        await ws.send_str(json.dumps({"type": "error", "error": "Mic preview requires Server USB/ALSA microphone"}))
        await ws.close()
        return ws

    if not _mic_preview_lock.acquire(timeout=3):
        await ws.send_str(
            json.dumps({"type": "error", "error": "Microphone preview in use; try again in a moment"})
        )
        await ws.close()
        return ws

    lock_released = False
    try:
        capture_queue = queue.Queue()
        stop_capture = threading.Event()
        proc_holder: list = []  # ALSA: capture thread appends the arecord process so we can terminate it to release the device quickly
        capture_thread = start_server_mic_capture(
            session_config.devices.audio_input_source,
            session_config.devices.audio_input_device,
            capture_queue,
            stop_capture,
            proc_holder if session_config.devices.audio_input_source == "alsa" else None,
        )
        if not capture_thread:
            await ws.send_str(json.dumps({"type": "error", "error": "Failed to start server mic capture"}))
            await ws.close()
            return ws

        stopped = asyncio.Event()
        loop = asyncio.get_event_loop()
        amplitude_interval = 0.025  # 40 Hz for preview waveform
        last_amplitude_time = 0.0
        _first_sent = False

        async def capture_consumer() -> None:
            nonlocal last_amplitude_time, _first_sent
            while not stopped.is_set():
                try:
                    chunk = await loop.run_in_executor(None, capture_queue.get)
                except asyncio.CancelledError:
                    break
                except Exception:
                    break
                if chunk is None:
                    break
                now = time.time()
                if now - last_amplitude_time >= amplitude_interval:
                    last_amplitude_time = now
                    amp = _pcm_rms_to_amplitude(chunk)
                    try:
                        await ws.send_str(
                            json.dumps({"type": "user_amplitude", "timestamp": round(now, 3), "amplitude": round(amp, 2)})
                        )
                        if not _first_sent:
                            _first_sent = True
                            logger.info("Mic preview: first user_amplitude sent; amp=%.2f", amp)
                    except Exception as e:
                        logger.warning("Mic preview send user_amplitude failed: %s", e)
                        break

        async def receive_loop() -> None:
            try:
                async for msg in ws:
                    if msg.type == web.WSMsgType.TEXT:
                        try:
                            o = json.loads(msg.data)
                            if o.get("type") == "stop":
                                stopped.set()
                                return
                        except json.JSONDecodeError:
                            pass
                    if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                        stopped.set()
                        return
            except Exception:
                stopped.set()

        recv_task = asyncio.create_task(receive_loop())
        cons_task = asyncio.create_task(capture_consumer())
        await stopped.wait()
        stop_capture.set()
        # Terminate arecord immediately so the capture thread exits and releases the ALSA device (otherwise voice pipeline gets "Device or resource busy")
        if proc_holder and len(proc_holder) > 0:
            try:
                proc_holder[0].terminate()
            except Exception as e:
                logger.debug("Mic preview: terminate arecord: %s", e)
            await asyncio.sleep(0.15)  # give OS time to release the device before we release the lock
        _mic_preview_lock.release()  # release early so next preview or voice pipeline can acquire the device
        lock_released = True
        cons_task.cancel()
        try:
            await cons_task
        except asyncio.CancelledError:
            pass
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Mic preview WebSocket closed")
    finally:
        if not lock_released:
            try:
                _mic_preview_lock.release()
            except RuntimeError:
                pass
    return ws
