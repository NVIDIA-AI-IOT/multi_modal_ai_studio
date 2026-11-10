# SPDX-License-Identifier: Apache-2.0
"""Read models for full-duplex recordings, independent of cascade Turns.

Input/output associations are temporal candidates, not semantic ground truth.
Never substitute a later answer for an earlier reply lacking audio alignment.
"""
import math
from statistics import fmean


def is_full_duplex(config):
    asr = (config or {}).get("asr") or {}
    return (
        (asr.get("scheme") or asr.get("backend")) == "openai-realtime"
        and asr.get("realtime_session_type", "full") == "full"
        and asr.get("realtime_transport", "websocket") == "websocket"
    )


def _time(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def derive_full_duplex(session):
    """Derive display data without changing the original recording."""
    if not is_full_duplex(session.get("config")):
        return None
    raw = session.get("timeline") or []
    if isinstance(raw, dict):
        raw = raw.get("events", [])
    events = sorted(
        (e for e in raw if isinstance(e, dict) and _time(e.get("timestamp")) is not None),
        key=lambda e: e["timestamp"],
    )
    users, assistants, captures, onsets, fallback = [], {}, {}, {}, {}
    region_onsets, region_links = {}, []
    has_transcripts = any(e.get("event_type") == "realtime_output_final" for e in events)
    for index, event in enumerate(events):
        kind, t, data = event.get("event_type"), event["timestamp"], event.get("data") or {}
        text = str(data.get("text") or "").strip()
        if kind == "asr_final" and text:
            users.append(dict(id=f"user-{len(users)+1}", role="user", text=text,
                              timestamp=t, final_timestamp=t, finalized=True))
        elif kind in ("realtime_output_partial", "realtime_output_final") and text:
            key = data.get("utterance_id")
            # Legacy providers without utterance IDs are reconstructed from finals.
            if key is None and kind == "realtime_output_partial":
                continue
            key = str(key) if key is not None else f"legacy-{index}"
            item = assistants.setdefault(key, dict(id=f"assistant-{key}", role="assistant",
                                                   timestamp=t, finalized=False))
            item["text"] = text
            if kind == "realtime_output_final":
                item.update(finalized=True, final_timestamp=t)
        elif kind == "tts_complete" and text and not has_transcripts:
            fallback[f"audio-{index}"] = dict(id=f"assistant-audio-{index}", role="assistant",
                                                   text=text, timestamp=t, finalized=True)
        elif kind in ("duplex_input_start", "duplex_input_end"):
            capture = captures.setdefault(str(data.get("input_id")), {})
            capture["start" if kind == "duplex_input_start" else "end"] = t
        elif kind == "duplex_audio_link":
            region_links.append((str(data.get("utterance_id")), str(data.get("audio_region_id"))))
        elif kind == "duplex_audio_playback":
            # Muted, suspended, unknown or server-delivery timestamps are not playout evidence.
            if data.get("source") == "browser_schedule" and data.get("audible") is True:
                if data.get("utterance_id") is not None:
                    key = str(data["utterance_id"])
                    onsets[key] = min(t, onsets.get(key, math.inf))
                if data.get("audio_region_id") is not None:
                    key = str(data["audio_region_id"])
                    region_onsets[key] = min(t, region_onsets.get(key, math.inf))

    if not assistants:
        assistants = fallback
    previous_final = -math.inf
    for user in users:
        # A region crossing a final transcript can contain the next question.
        # Assign completed regions by their end, without consuming that future
        # endpoint merely because the region started before this transcript.
        intervals = [c for c in captures.values()
                     if "start" in c and previous_final < c.get("end", math.inf) <= user["final_timestamp"]]
        if not intervals:
            intervals = [dict(start=c["start"]) for c in captures.values()
                         if previous_final < c.get("start", math.inf) <= user["final_timestamp"]
                         and c.get("end", math.inf) > user["final_timestamp"]]
        if intervals:
            user["input_start"] = min(c["start"] for c in intervals)
            ends = [c["end"] for c in intervals if "end" in c]
            user["input_end"] = max(ends) if len(ends) == len(intervals) else None
            user["input_boundary_ambiguous"] = len(intervals) > 1
            if ends:
                user["input_end_range"] = [min(ends), max(ends)]
            # Short energy bursts are not automatically competing speech ends.
            # Use a clearly dominant, completed region as an explicit heuristic;
            # retain later bursts as an endpoint range rather than moving the
            # endpoint forward and making the latency artificially shorter.
            complete = [c for c in intervals if "end" in c]
            ranked = sorted(complete, key=lambda c: c["end"]-c["start"], reverse=True)
            if len(ranked) > 1 and len(complete) == len(intervals):
                main, others = ranked[0], ranked[1:]
                duration = main["end"] - main["start"]
                other_duration = max(c["end"]-c["start"] for c in others)
                if (main["end"] <= user["final_timestamp"] and duration >= 0.4
                        and duration >= 3 * other_duration):
                    user["input_end"] = main["end"]
                    user["input_boundary_ambiguous"] = False
                    user["endpoint_method"] = "dominant_energy_region"
                    user["input_end_range"] = [main["end"], max(ends)]
                    user["secondary_energy_regions"] = len(others)
            user["timestamp"] = user["input_start"]
            user["association_start"] = max(user["input_start"], previous_final)
        previous_final = user["final_timestamp"]
    for uid, rid in region_links:
        if (rid in region_onsets
                and len({r for u, r in region_links if u == uid}) == 1
                and len({u for u, r in region_links if r == rid}) == 1):
            onsets[uid] = min(onsets.get(uid, math.inf), region_onsets[rid])
    for key, item in assistants.items():
        item["audio_onset"] = onsets.get(key)

    replies, gaps, reactions = [], [], []
    for index, user in enumerate(users):
        reply = dict(input_id=user["id"], reply_id=None, fd_ttl_ms=None,
                     reaction_ms=None, association="temporal_candidate",
                     status="input_audio_timing_missing")
        if "input_start" in user:
            boundary = users[index+1].get("association_start", users[index+1]["timestamp"]) if index+1 < len(users) else math.inf
            candidate_time = lambda a: min(a["timestamp"], a["audio_onset"]) if a["audio_onset"] is not None else a["timestamp"]
            candidates = sorted((a for a in assistants.values()
                                 if user.get("association_start", user["input_start"]) <= candidate_time(a) < boundary),
                                key=lambda a: a["timestamp"])
            reply["status"] = "no_reply_candidate"
            if candidates:
                first = candidates[0]  # Includes short acknowledgements such as "Sure".
                reply.update(reply_id=first["id"], reply_text=first["text"], status="audio_alignment_missing")
                if first["audio_onset"] is not None:
                    if user.get("input_boundary_ambiguous"):
                        reply.update(status="input_boundary_ambiguous", input_end_range=user.get("input_end_range"))
                        replies.append(reply)
                        continue
                    reaction = (first["audio_onset"] - user["input_start"]) * 1000
                    reply.update(reaction_ms=reaction, status="input_end_missing")
                    reactions.append(reaction)
                    if user.get("input_end") is not None:
                        gap = (first["audio_onset"] - user["input_end"]) * 1000
                        reply.update(fd_ttl_ms=gap, status="estimated", input_end=user["input_end"], audio_onset=first["audio_onset"])
                        reply["endpoint_method"] = user.get("endpoint_method", "single_energy_region")
                        reply["input_end_range"] = user.get("input_end_range")
                        if user.get("endpoint_method") == "dominant_energy_region":
                            reply["fd_ttl_range_ms"] = [
                                (first["audio_onset"]-end)*1000
                                for end in reversed(user["input_end_range"])]
                        gaps.append(gap)  # Negative gaps represent overlapping speech.
        replies.append(reply)
    stops = [e["timestamp"] for e in events if e.get("event_type") == "session_stop"]
    duration = max(stops) if stops else max((e["timestamp"] for e in events), default=0)
    return dict(
        schema_version=1,
        messages=sorted(users + list(assistants.values()), key=lambda m: m["timestamp"]),
        replies=replies,
        metrics=dict(user_utterances=len(users), assistant_utterances=len(assistants),
                     duration_s=duration, duration_basis="session_stop" if stops else "last_observed_event",
                     avg_fd_ttl_ms=fmean(gaps) if gaps else None,
                     avg_reaction_ms=fmean(reactions) if reactions else None,
                     matched_replies=len(gaps), unmatched_replies=len(users)-len(gaps),
                     ambiguous_inputs=sum(bool(u.get("input_boundary_ambiguous")) for u in users),
                     dominant_endpoint_estimates=sum(r.get("endpoint_method") == "dominant_energy_region" for r in replies),
                     timing_basis="input_energy_to_browser_scheduled_audio",
                     association="temporal_candidate"),
    )
