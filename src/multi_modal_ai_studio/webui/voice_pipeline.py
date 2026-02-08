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
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from multi_modal_ai_studio.config.schema import (
    SessionConfig,
    ASRConfig,
    LLMConfig,
    TTSConfig,
)
from multi_modal_ai_studio.core.session import Session
from multi_modal_ai_studio.core.timeline import Lane
from multi_modal_ai_studio.backends.asr.riva import RivaASRBackend
from multi_modal_ai_studio.backends.llm.openai import OpenAILLMBackend
from multi_modal_ai_studio.backends.tts.riva import RivaTTSBackend

logger = logging.getLogger(__name__)

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
    """
    logger.info("Voice pipeline starting")
    session = Session(config=config)
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
                            stopped.set()
                            return
                    except json.JSONDecodeError:
                        pass
                    continue
                if msg.type == web.WSMsgType.BINARY:
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

    async def results_loop() -> None:
        """Consume ASR results; on final -> LLM -> TTS; emit events and TTS audio."""
        nonlocal conversation_history
        last_asr_final_text: Optional[str] = None
        last_asr_final_ts: Optional[float] = None
        try:
            async for result in asr.receive_results():
                if stopped.is_set():
                    break
                if result is None:
                    break

                is_final = getattr(result, "is_final", True)
                text = (result.text or "").strip()
                if not text:
                    continue

                # Use backend event time for asr_final so the timeline dot isn't tied to pipeline processing (e.g. same moment as tts_complete)
                now_ts = (time.time() - session.timeline.start_time) if session.timeline.start_time else 0
                ts = (getattr(result, "metadata", {}) or {}).get("event_timestamp")
                if ts is None:
                    ts = now_ts
                ev_type = "asr_partial" if not is_final else "asr_final"

                # Skip duplicate asr_final (same text as last final, within 2s) to avoid repeated final_transcript
                if is_final and last_asr_final_text is not None and text == last_asr_final_text:
                    if last_asr_final_ts is not None and abs(ts - last_asr_final_ts) < 2.0:
                        logger.debug("[asr] Skipping duplicate asr_final: %r", text[:50])
                        continue
                # Skip phantom partials: (1) partial at or before last final timestamp; (2) late partial from same utterance (arrives after LLM/TTS, gets ts=now so appears at tts_complete)
                if not is_final and last_asr_final_ts is not None:
                    if ts <= last_asr_final_ts:
                        logger.debug("[asr] Skipping stale asr_partial (ts=%.2f <= last_final=%.2f): %r", ts, last_asr_final_ts, text[:50])
                        continue
                    # Same-utterance late partial: text matches or overlaps with last final (Riva sent partial after we already had final)
                    if last_asr_final_text and (
                        text == last_asr_final_text
                        or (text in last_asr_final_text or last_asr_final_text.startswith(text) or text.startswith(last_asr_final_text))
                    ):
                        logger.debug("[asr] Skipping phantom asr_partial (same utterance as last final): %r", text[:50])
                        continue

                # Riva backend already adds asr_final to the timeline with correct timestamp; only add partials here
                if not is_final:
                    session.timeline.add_event(ev_type, Lane.SPEECH, data={"text": text, "confidence": getattr(result, "confidence", 1.0)})
                await send_event({
                    "event_type": ev_type,
                    "lane": "speech",
                    "data": {"text": text, "confidence": getattr(result, "confidence", 1.0)},
                    "timestamp": ts,
                })

                if not is_final:
                    continue

                last_asr_final_text = text
                last_asr_final_ts = ts

                logger.info("[timing] asr_final received (turn start)")
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

                # Notify client that TTS is starting so it can resume AudioContext and schedule first chunk immediately
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
                        # Record TTS amplitude for AUDIO lane (purple waveform in replay)
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
            logger.exception("results_loop error: %s", e)

    await send_event({"event_type": "session_start", "lane": "system", "data": {}, "timestamp": 0})

    recv_task = asyncio.create_task(receive_loop())
    results_task = asyncio.create_task(results_loop())

    await stopped.wait()
    await asr.stop_stream()
    recv_task.cancel()
    results_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass
    try:
        await results_task
    except asyncio.CancelledError:
        pass

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
