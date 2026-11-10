# SPDX-License-Identifier: Apache-2.0
"""Browser acoustic gaps, independent of model turns and transcripts."""
import math
from statistics import fmean


def browser_timing_event(event, elapsed):
    """Validate and bound the small client observation payload before saving."""
    if not isinstance(event, dict):
        return None
    kind, timestamp, data = event.get("event_type"), event.get("timestamp"), event.get("data")
    if kind not in {"browser_timing_started", "browser_capture_ready", "browser_input_start", "browser_input_end",
                    "browser_output_start", "browser_output_end"} or not isinstance(data, dict):
        return None
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp) or not 0 <= timestamp <= elapsed+5:
        return None
    if data.get("clock") != "browser_monotonic":
        return None
    clean = {"clock": "browser_monotonic", "source": "browser_capture" if "input" in kind or "capture" in kind else "browser_schedule"}
    if kind == "browser_timing_started":
        clean.update(source="browser_audio", detector_version=1)
    elif kind == "browser_capture_ready":
        if data.get("method") not in {"audio_worklet", "script_processor_fallback"}:
            return None
        clean["method"] = data["method"]
    elif kind != "browser_timing_started":
        uid = data.get("region_id")
        if not isinstance(uid, int) or isinstance(uid, bool) or not 0 < uid <= 100000:
            return None
        clean.update(region_id=uid, resolution_ms=10)
        if data.get("reason") in {"silence", "energy", "muted", "stopped", "capture_disconnected", "cancelled"}:
            clean["reason"] = data["reason"]
    return {"event_type": kind, "timestamp": float(timestamp), "data": clean}


def derive_browser_timing(events):
    """Same acoustic-sequence rule as speech_timing_helpers.gaps."""
    if not any(e.get("event_type") == "browser_timing_started" and (e.get("data") or {}).get("clock") == "browser_monotonic" for e in events):
        return None
    inputs, outputs = {}, {}
    for event in events:
        kind, t, data = event.get("event_type", ""), event.get("timestamp"), event.get("data") or {}
        if data.get("clock") != "browser_monotonic" or not isinstance(t, (int,float)) or not math.isfinite(t):
            continue
        parts = kind.split("_")
        if len(parts) != 3 or parts[0] != "browser" or parts[1] not in {"input","output"} or parts[2] not in {"start","end"}:
            continue
        group = inputs if parts[1] == "input" else outputs
        item = group.setdefault(data.get("region_id"), {})
        item[parts[2]] = t
        if parts[2] == "end":
            item["reason"] = data.get("reason")
    ins = sorted(((uid,i) for uid,i in inputs.items() if "start" in i), key=lambda pair:pair[1]["start"])
    outs = sorted((o for o in outputs.values() if "start" in o), key=lambda o:o["start"])
    replies = []
    for index,(uid,i) in enumerate(ins):
        reply = dict(input_id=f"browser-{uid}", fd_ttl_ms=None, reaction_ms=None, input_end=i.get("end"),
                     audio_onset=None, association="acoustic_sequence", endpoint_method="browser_energy", status="input_end_missing")
        if "end" in i:
            if i.get("reason") not in {None,"silence"}:
                reply["status"] = "input_interrupted"
            else:
                boundary = ins[index+1][1]["start"] if index+1 < len(ins) else math.inf
                output = next((o for o in outs if i["start"] <= o["start"] < boundary), None)
                if output:
                    reply.update(status="estimated", audio_onset=output["start"],
                                 fd_ttl_ms=(output["start"]-i["end"])*1000,
                                 reaction_ms=(output["start"]-i["start"])*1000)
                else:
                    reply["status"] = "continued_input" if boundary < math.inf else (
                        "overlapping_audio" if any(o["start"] < i["start"] and o.get("end",math.inf)>i["end"] for o in outs) else "no_audio_onset")
        replies.append(reply)
    gaps = [r["fd_ttl_ms"] for r in replies if r["fd_ttl_ms"] is not None]
    reactions = [r["reaction_ms"] for r in replies if r["reaction_ms"] is not None]
    return dict(schema_version=1, replies=replies, metrics=dict(avg_fd_ttl_ms=fmean(gaps) if gaps else None,
                avg_reaction_ms=fmean(reactions) if reactions else None, matched_replies=len(gaps),
                unmatched_replies=len(ins)-len(gaps), measurement_inputs=len(ins), ambiguous_inputs=0,
                dominant_endpoint_estimates=0, timing_basis="browser_capture_to_scheduled_audio",
                association="acoustic_sequence"))
