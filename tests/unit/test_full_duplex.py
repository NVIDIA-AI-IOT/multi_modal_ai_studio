"""Full-duplex read models preserve speech without inventing cascade turns."""
from copy import deepcopy

import pytest

from multi_modal_ai_studio.core.full_duplex import derive_full_duplex
from multi_modal_ai_studio.webui.server import WebUIServer


def event(kind, t, **data):
    return {"event_type": kind, "timestamp": t, "data": data}


def recording(events):
    return {"config": {"asr": {"scheme": "openai-realtime", "realtime_api_style": "nemo-voicechat"}},
            "timeline": events, "turns": [], "metrics": {},
            "ttl_bands": [{"start": 10, "end": 99, "ttlMs": 89000}]}


def test_legacy_recording_restores_users_and_keeps_greetings_and_repeated_answers():
    rows = [event("realtime_output_final", 1, text="Hello", utterance_id=1)]
    for i in range(4):
        rows += [event("asr_final", 3+i*4, text=f"Question {i}"),
                 event("tts_complete", 4+i*4, text="Two."),
                 event("realtime_output_final", 5+i*4, text="Two.", utterance_id=i+2)]
    rows += [event("realtime_output_final", 18, text="Sure", utterance_id=6)]
    original = recording(rows)
    before = deepcopy(original)
    result = derive_full_duplex(original)
    assert result["metrics"]["user_utterances"] == 4
    assert result["metrics"]["assistant_utterances"] == 6
    assert sum(m["role"] == "user" for m in result["messages"]) == 4
    assert result["metrics"]["avg_fd_ttl_ms"] is None
    assert result["metrics"]["matched_replies"] == 0
    assert original == before  # Read-time adaptation never edits original JSON.


def acknowledgement_trace():
    return [
        event("duplex_input_start", 10, input_id=1),
        event("realtime_output_partial", 10.5, text="Sure", utterance_id=5),
        event("realtime_output_final", 10.6, text="Sure.", utterance_id=5),
        event("duplex_input_end", 11, input_id=1),
        event("asr_final", 11.5, text="Stop. What is one plus one?"),
        event("realtime_output_final", 12, text="Two.", utterance_id=6),
        event("duplex_audio_playback", 12.3, utterance_id=6, source="browser_schedule", audible=True),
    ]


def test_unaligned_sure_is_not_replaced_with_later_substantive_answer():
    result = derive_full_duplex(recording(acknowledgement_trace()))
    reply = result["replies"][0]
    assert reply["reply_id"] == "assistant-5"
    assert reply["reply_text"] == "Sure."
    assert reply["status"] == "audio_alignment_missing"
    assert reply["fd_ttl_ms"] is None


def test_short_acknowledgement_counts_and_negative_gap_preserves_overlap():
    rows = acknowledgement_trace()+[
        event("duplex_audio_playback", 10.7, utterance_id=5, source="browser_schedule", audible=True),
    ]
    result = derive_full_duplex(recording(rows))
    assert result["replies"][0]["fd_ttl_ms"] == pytest.approx(-300)
    assert result["replies"][0]["reaction_ms"] == pytest.approx(700)
    assert result["metrics"]["matched_replies"] == 1
    assert result["replies"][0]["association"] == "temporal_candidate"


@pytest.mark.parametrize("source,audible", [("server_pcm_delivery", True), ("browser_schedule", False)])
def test_non_played_audio_is_not_counted_as_audible_response(source, audible):
    rows = acknowledgement_trace()+[
        event("duplex_audio_playback", 10.7, utterance_id=5, source=source, audible=audible),
    ]
    assert derive_full_duplex(recording(rows))["replies"][0]["fd_ttl_ms"] is None


def test_partial_snapshots_count_once_and_stop_is_independent_of_drain():
    rows = [event("realtime_output_partial", 1, text="One", utterance_id=1),
            event("realtime_output_partial", 2, text="One plus one", utterance_id=1),
            event("session_stop", 3),
            event("realtime_output_final", 4, text="One plus one is two.", utterance_id=1)]
    result = derive_full_duplex(recording(rows))
    assert result["metrics"]["assistant_utterances"] == 1
    assert result["messages"][0]["text"] == "One plus one is two."
    assert result["metrics"]["duration_s"] == 3


def test_session_list_keeps_counts_but_not_transcript_or_secret():
    data = recording([event("asr_final", 1, text="private speech")])
    data["config"]["asr"]["api_key"] = "private-key"
    summary = WebUIServer._session_list_summary(data)
    assert summary["full_duplex"]["metrics"]["user_utterances"] == 1
    assert "messages" not in summary["full_duplex"]
    assert "private speech" not in str(summary)
    assert "private-key" not in str(summary)


def test_cascade_recordings_keep_their_existing_metrics_path():
    assert derive_full_duplex({"config": {"asr": {"scheme": "riva"}}}) is None


def test_short_tail_retains_dominant_endpoint_and_latency_range():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 2, input_id=1),
            event("duplex_input_start", 2.6, input_id=2),
            event("duplex_input_end", 2.7, input_id=2),
            event("asr_final", 3, text="Question"),
            event("realtime_output_final", 3.2, text="Sure", utterance_id=1),
            event("duplex_audio_playback", 3.3, utterance_id=1, source="browser_schedule", audible=True)]
    result = derive_full_duplex(recording(rows))
    reply = result["replies"][0]
    assert reply["endpoint_method"] == "dominant_energy_region"
    assert reply["input_end"] == 2
    assert reply["fd_ttl_ms"] == pytest.approx(1300)
    assert reply["fd_ttl_range_ms"] == pytest.approx([600, 1300])
    assert result["metrics"]["dominant_endpoint_estimates"] == 1


def test_comparable_regions_remain_ambiguous():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 2, input_id=1),
            event("duplex_input_start", 2.6, input_id=2),
            event("duplex_input_end", 3.2, input_id=2),
            event("asr_final", 3.5, text="Question"),
            event("realtime_output_final", 3.6, text="Sure", utterance_id=1),
            event("duplex_audio_playback", 3.7, utterance_id=1, source="browser_schedule", audible=True)]
    result = derive_full_duplex(recording(rows))
    assert result["replies"][0]["status"] == "input_boundary_ambiguous"
    assert result["replies"][0]["input_end_range"] == [2, 3.2]
    assert result["replies"][0]["fd_ttl_ms"] is None
    assert result["metrics"]["avg_fd_ttl_ms"] is None
    assert result["metrics"]["avg_reaction_ms"] is None
    assert result["metrics"]["ambiguous_inputs"] == 1


def test_short_regions_before_dominant_speech_do_not_suppress_endpoint():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 1.1, input_id=1),
            event("duplex_input_start", 2, input_id=2),
            event("duplex_input_end", 3.8, input_id=2),
            event("asr_final", 4.2, text="Question"),
            event("realtime_output_final", 4.4, text="Answer", utterance_id=1),
            event("duplex_audio_playback", 4.8, utterance_id=1, source="browser_schedule", audible=True)]
    reply = derive_full_duplex(recording(rows))["replies"][0]
    assert reply["input_end"] == 3.8
    assert reply["fd_ttl_ms"] == pytest.approx(1000)
    assert reply["fd_ttl_range_ms"] == pytest.approx([1000, 1000])


def test_region_extending_past_transcript_is_not_selected_as_dominant():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 1.1, input_id=1),
            event("duplex_input_start", 2, input_id=2),
            event("duplex_input_end", 3.8, input_id=2),
            event("asr_final", 2.5, text="Stop"),
            event("realtime_output_final", 2.6, text="Sure", utterance_id=1),
            event("duplex_audio_playback", 2.8, utterance_id=1, source="browser_schedule", audible=True)]
    reply = derive_full_duplex(recording(rows))["replies"][0]
    assert reply["endpoint_method"] == "single_energy_region"
    assert reply["input_end"] == 1.1  # Never borrow the future endpoint at 3.8.


def test_relative_dominance_handles_a_260ms_secondary_region():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 2.3, input_id=1),
            event("duplex_input_start", 2.6, input_id=2),
            event("duplex_input_end", 2.86, input_id=2),
            event("asr_final", 3, text="Question"),
            event("realtime_output_final", 3.1, text="Answer", utterance_id=1),
            event("duplex_audio_playback", 3.4, utterance_id=1, source="browser_schedule", audible=True)]
    reply = derive_full_duplex(recording(rows))["replies"][0]
    assert reply["input_end"] == 2.3
    assert reply["fd_ttl_ms"] == pytest.approx(1100)
    assert reply["fd_ttl_range_ms"] == pytest.approx([540, 1100])


def test_region_crossing_stop_transcript_remains_available_for_next_question():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 1.2, input_id=1),
            event("duplex_input_start", 1.8, input_id=2),
            event("asr_final", 2, text="Stop"),
            event("realtime_output_partial", 1.99, text="Sure", utterance_id=1),
            event("realtime_output_final", 2.1, text="Sure.", utterance_id=1),
            event("duplex_input_end", 3, input_id=2),
            event("asr_final", 3.5, text="Next question"),
            event("realtime_output_final", 3.6, text="Answer", utterance_id=2),
            event("duplex_audio_playback", 3.9, utterance_id=2, source="browser_schedule", audible=True)]
    result = derive_full_duplex(recording(rows))
    assert result["replies"][0]["reply_text"] == "Sure."
    assert result["replies"][0]["status"] == "audio_alignment_missing"
    assert result["replies"][1]["input_end"] == 3
    assert result["replies"][1]["fd_ttl_ms"] == pytest.approx(900)


def test_audio_first_link_uses_original_browser_onset_despite_late_text():
    rows = [event("duplex_input_start", 1, input_id=1),
            event("duplex_input_end", 1.5, input_id=1),
            event("asr_final", 2, text="Question"),
            event("duplex_audio_playback", 1.8, audio_region_id=7, source="browser_schedule", audible=True),
            event("duplex_input_start", 2.1, input_id=2),
            event("duplex_input_end", 2.5, input_id=2),
            event("asr_final", 3, text="Next question"),
            event("realtime_output_final", 3.2, text="Sure", utterance_id=1),
            event("duplex_audio_link", 3.2, audio_region_id=7, utterance_id=1)]
    reply = derive_full_duplex(recording(rows))["replies"][0]
    assert reply["reply_text"] == "Sure"
    assert reply["audio_onset"] == 1.8
    assert reply["fd_ttl_ms"] == pytest.approx(300)


def test_conflicting_audio_region_links_do_not_invent_onsets():
    rows = acknowledgement_trace()+[
        event("duplex_audio_playback", 10.7, audio_region_id=1, source="browser_schedule", audible=True),
        event("duplex_audio_link", 11, audio_region_id=1, utterance_id=5),
        event("duplex_audio_link", 11.1, audio_region_id=1, utterance_id=6),
    ]
    assert derive_full_duplex(recording(rows))["replies"][0]["fd_ttl_ms"] is None
