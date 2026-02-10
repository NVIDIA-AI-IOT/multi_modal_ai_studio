"""
Voice WebSocket pipeline: ASR (Riva) -> LLM (Ollama) -> TTS (Riva).

Handles WebSocket messages: config (first JSON), then binary PCM audio.
Sends back: timeline events (JSON) and TTS audio (base64).
On stop/disconnect: saves session to session_dir.
"""

import asyncio
import base64
import json
import logging
import math
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from multi_modal_ai_studio.devices.capture import start_server_mic_capture

from multi_modal_ai_studio.config.schema import (
    SessionConfig,
    ASRConfig,
    LLMConfig,
    TTSConfig,
)
from multi_modal_ai_studio import __version__
from multi_modal_ai_studio.core.session import Session
from multi_modal_ai_studio.core.timeline import Lane
from multi_modal_ai_studio.backends.base import ASRResult
from multi_modal_ai_studio.backends.asr.riva import RivaASRBackend
from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.backends.tts.riva import RivaTTSBackend

logger = logging.getLogger(__name__)

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
    if not use_server_mic:
        session.start()

    asr_config = config.asr
    llm_config = config.llm
    tts_config = config.tts

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

    await asr.start_stream()
    conversation_history = []
    stopped = asyncio.Event()
    finals_queue: asyncio.Queue = asyncio.Queue()
    pipeline_live = asyncio.Event()  # when set, Server USB capture feeds ASR + timeline (after client sent start_session)
    if use_server_mic:
        pipeline_live.clear()
    else:
        pipeline_live.set()

    # When user selected a Server USB microphone, capture on server; preview streams amplitude until start_session.
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
                config.devices.audio_input_source,
                config.devices.audio_input_device,
            )
            use_server_mic = False
        else:
            logger.info("Server mic capture thread started; waiting for first PCM chunk and user_amplitude")

    async def send_event(event_dict: Dict[str, Any]) -> None:
        try:
            await ws.send_str(json.dumps({"type": "event", "event": event_dict}))
        except Exception as e:
            logger.warning("Send event failed: %s", e)

    async def receive_loop() -> None:
        """Read from WebSocket: config (first), then binary PCM or stop."""
        nonlocal conversation_history
        last_amplitude_time = 0.0
        last_amplitude_log_time = 0.0  # throttle debug log to ~1s (see INVESTIGATE_USER_AMPLITUDE_ARTIFACT.md)
        amplitude_interval = 0.05  # record amplitude at most every 50ms
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        obj = json.loads(msg.data)
                        if obj.get("type") == "start_session":
                            # Server USB: transition from preview to full pipeline (same capture, now feed ASR + timeline)
                            if use_server_mic and not pipeline_live.is_set():
                                session.start()
                                await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})
                                pipeline_live.set()
                                logger.info("Voice pipeline: start_session received; capture now feeds ASR + timeline")
                        if obj.get("type") == "stop":
                            stats = obj.get("system_stats")
                            if isinstance(stats, list):
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
                    except json.JSONDecodeError:
                        pass
                    continue
                if msg.type == web.WSMsgType.BINARY:
                    # When using Server USB mic, audio comes from server capture task; ignore browser PCM.
                    if not use_server_mic:
                        await asr.send_audio(msg.data)
                        # Record audio amplitude for timeline (throttled)
                        if session.timeline.start_time is not None:
                            now = time.time() - session.timeline.start_time
                            if now - last_amplitude_time >= amplitude_interval:
                                amp = _pcm_rms_to_amplitude(msg.data)
                                session.timeline.add_audio_amplitude(amplitude=amp, source="user")
                                last_amplitude_time = now
                                # Log every high user amplitude so we can trace false green on replay (live was correct; replay shows bars = they're in saved timeline)
                                if amp >= 20.0:
                                    try:
                                        n = len(msg.data) // 2
                                        raw_rms = math.sqrt(sum(s * s for s in struct.unpack(f"{n}h", msg.data)) / n) if n else 0.0
                                        logger.warning(
                                            "[user_amplitude_high] session_t=%.2fs amp_0_100=%.2f raw_rms=%.1f chunk_len=%d (no ASR nearby => false green on replay)",
                                            now, amp, raw_rms, len(msg.data),
                                        )
                                    except Exception:
                                        pass
                                # Debug: log user mic amplitude ~every 1s (INVESTIGATE_USER_AMPLITUDE_ARTIFACT.md)
                                if now - last_amplitude_log_time >= 1.0:
                                    try:
                                        n = len(msg.data) // 2
                                        if n:
                                            samples = struct.unpack(f"{n}h", msg.data)
                                            sum_sq = sum(s * s for s in samples)
                                            raw_rms = math.sqrt(sum_sq / n)
                                        else:
                                            raw_rms = 0.0
                                        logger.info(
                                            "[user_amplitude] session_t=%.1fs chunk_len=%d raw_rms=%.1f amp_0_100=%.2f",
                                            now, len(msg.data), raw_rms, amp,
                                        )
                                        last_amplitude_log_time = now
                                    except Exception:
                                        pass
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
        nonlocal conversation_history
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
                full_response = ""
                llm_first_token_sent = False
                try:
                    async for token in llm.generate_stream(
                        prompt=text,
                        history=conversation_history,
                        system_prompt=llm_config.system_prompt,
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
                                await send_event({
                                    "event_type": "llm_first_token",
                                    "lane": "llm",
                                    "data": {},
                                    "timestamp": ts_first,
                                })
                    session.timeline.add_event("llm_complete", Lane.LLM, data={"text": full_response})
                except Exception as e:
                    logger.exception("LLM error: %s", e)
                    full_response = "Sorry, I had an error."
                    session.timeline.add_event("llm_complete", Lane.LLM, data={"text": full_response, "error": str(e)})

                ts_llm_complete = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                logger.info("[timing] llm_complete @ %.2fs (llm took %.2fs)", ts_llm_complete, ts_llm_complete - ts_llm_start)
                await send_event({
                    "event_type": "llm_complete",
                    "lane": "llm",
                    "data": {"text": full_response},
                    "timestamp": ts_llm_complete,
                })

                session.update_turn_response(full_response)
                conversation_history.append({"role": "user", "content": text})
                conversation_history.append({"role": "assistant", "content": full_response})

                await send_event({
                    "event_type": "chat",
                    "user": text,
                    "assistant": full_response,
                })

                if not full_response.strip():
                    session.end_turn()
                    continue

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
                last_tts_amplitude_time = 0.0
                tts_amplitude_interval = 0.05
                try:
                    async for chunk in tts.synthesize_stream(full_response):
                        if stopped.is_set():
                            break
                        if not tts_first_sent:
                            ts_tts_first = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                            logger.info("[timing] tts_first_audio @ %.2fs (tts first chunk after %.2fs from llm_complete)", ts_tts_first, ts_tts_first - ts_llm_complete)
                            session.timeline.add_event("tts_first_audio", Lane.TTS)
                            await send_event({
                                "event_type": "tts_first_audio",
                                "lane": "tts",
                                "data": {},
                                "timestamp": ts_tts_first,
                            })
                            tts_first_sent = True
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
                    ts_tts_complete = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                    logger.info("[timing] tts_complete @ %.2fs (tts stream took %.2fs)", ts_tts_complete, ts_tts_complete - ts_tts_first if tts_first_sent else 0)
                    session.timeline.add_event("tts_complete", Lane.TTS)
                    await send_event({
                        "event_type": "tts_complete",
                        "lane": "tts",
                        "data": {},
                        "timestamp": ts_tts_complete,
                    })
                except Exception as e:
                    logger.exception("TTS error: %s", e)

                session.end_turn()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("turn_executor error: %s", e)

    if not use_server_mic:
        await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})

    _user_amplitude_sent = False
    preview_start_time = time.time()

    async def server_capture_consumer() -> None:
        """Read PCM from server mic capture queue. Preview: stream user_amplitude at 50 Hz. Live: also feed ASR + timeline."""
        nonlocal _user_amplitude_sent
        if capture_queue is None:
            return
        loop = asyncio.get_event_loop()
        last_amplitude_time = 0.0
        amplitude_interval = 0.05  # 50 Hz for both preview and live
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
                await asr.send_audio(chunk)
                if session.timeline.start_time is not None:
                    now = time.time() - session.timeline.start_time
                    if now - last_amplitude_time >= amplitude_interval:
                        amp = _pcm_rms_to_amplitude(chunk)
                        session.timeline.add_audio_amplitude(amplitude=amp, source="user")
                        last_amplitude_time = now
                        try:
                            await ws.send_str(
                                json.dumps({"type": "user_amplitude", "timestamp": round(now, 3), "amplitude": round(amp, 2)})
                            )
                            if not _user_amplitude_sent:
                                _user_amplitude_sent = True
                                logger.info("First user_amplitude sent to client (live); amp=%.2f", amp)
                        except Exception as e:
                            logger.warning("Send user_amplitude failed: %s", e)
            else:
                # Preview: stream amplitude only (50 Hz), timestamp relative to connection
                now = time.time() - preview_start_time
                if now - last_amplitude_time >= amplitude_interval:
                    amp = _pcm_rms_to_amplitude(chunk)
                    last_amplitude_time = time.time() - preview_start_time
                    try:
                        await ws.send_str(
                            json.dumps({"type": "user_amplitude", "timestamp": round(now, 3), "amplitude": round(amp, 2)})
                        )
                        if not _user_amplitude_sent:
                            _user_amplitude_sent = True
                            logger.info("First user_amplitude sent to client (preview); amp=%.2f", amp)
                    except Exception as e:
                        logger.warning("Send user_amplitude failed: %s", e)

    server_capture_task: Optional[asyncio.Task] = None
    if use_server_mic and capture_thread is not None and capture_queue is not None:
        server_capture_task = asyncio.create_task(server_capture_consumer())

    recv_task = asyncio.create_task(receive_loop())
    asr_task: Optional[asyncio.Task] = None
    turn_task: Optional[asyncio.Task] = None

    if use_server_mic:
        # Wait for client to send start_session; then start ASR + turn executor (same capture keeps streaming)
        done, _ = await asyncio.wait(
            [asyncio.create_task(stopped.wait()), asyncio.create_task(pipeline_live.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stopped.is_set():
            pass  # goto cleanup
        else:
            asr_task = asyncio.create_task(asr_consumer())
            turn_task = asyncio.create_task(turn_executor())
            await stopped.wait()
    else:
        asr_task = asyncio.create_task(asr_consumer())
        turn_task = asyncio.create_task(turn_executor())
        await stopped.wait()
    if stop_capture is not None:
        stop_capture.set()
    if server_capture_task is not None:
        server_capture_task.cancel()
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

    session_dir = request.app.get("session_dir")
    if not session_dir:
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
