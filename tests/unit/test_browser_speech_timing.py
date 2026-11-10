"""Shared acoustic fixtures prove TTL does not depend on ASR text or model IDs."""
import json
from pathlib import Path

import pytest

from multi_modal_ai_studio.config.schema import SessionConfig
from multi_modal_ai_studio.core.full_duplex import derive_full_duplex
from multi_modal_ai_studio.core.session import Session
from multi_modal_ai_studio.core.speech_timing import browser_timing_event, derive_browser_timing
from multi_modal_ai_studio.core.timeline import Lane

CASES = json.loads((Path(__file__).parents[1]/'fixtures/browser_acoustic_gaps.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_shared_acoustic_gaps(case):
    result = derive_browser_timing(case['events'])
    assert [r['status'] for r in result['replies']] == case['statuses']
    for reply,expected in zip(result['replies'],case['ttl_ms']):
        assert reply['fd_ttl_ms'] == (pytest.approx(expected) if expected is not None else None)


def test_full_duplex_without_transcripts_still_measures_audio():
    result = derive_full_duplex({'config':{'asr':{'scheme':'openai-realtime'}},'timeline':CASES[0]['events']})
    assert result['metrics']['user_utterances'] == 0
    assert result['metrics']['matched_replies'] == 1
    assert result['metrics']['avg_fd_ttl_ms'] == 500
    assert result['metrics']['association'] == 'acoustic_sequence'
    changed = derive_full_duplex({'config':{'asr':{'scheme':'openai-realtime'}},'timeline':CASES[0]['events']+[
        {'event_type':'asr_final','timestamp':10,'data':{'text':'arbitrary input'}},
        {'event_type':'realtime_output_final','timestamp':20,'data':{'text':'unrelated subtitle','utterance_id':99}},
    ]})
    assert changed['replies'] == result['replies']
    assert changed['metrics']['avg_fd_ttl_ms'] == result['metrics']['avg_fd_ttl_ms']


def test_cascade_serialization_uses_same_gap_and_does_not_invent_missing_samples():
    for case,expected in [(CASES[0],.5),(CASES[5],None)]:
        session = Session(SessionConfig())
        for e in case['events']:
            session.timeline.add_event(e['event_type'],Lane.AUDIO,timestamp=e['timestamp'],data=e['data'])
        saved = session.to_dict()
        assert saved['metrics']['avg_ttl'] == expected
        assert saved['browser_timing']['metrics']['matched_replies'] == (1 if expected else 0)


@pytest.mark.parametrize('timestamp',[float('nan'),float('inf'),-1,True,100])
def test_client_observation_rejects_invalid_or_future_time(timestamp):
    event = {'event_type':'browser_input_start','timestamp':timestamp,'data':{'region_id':1,'clock':'browser_monotonic'}}
    assert browser_timing_event(event,10) is None


def test_client_observation_strips_unrelated_payload_and_requires_region_id():
    event = {'event_type':'browser_input_end','timestamp':1,'data':{'region_id':1,'clock':'browser_monotonic','secret':'not persisted','reason':'silence'}}
    assert 'secret' not in browser_timing_event(event,10)['data']
    event['data']['region_id'] = False
    assert browser_timing_event(event,10) is None


def test_capture_method_is_recorded_with_a_bounded_schema():
    e = {'event_type':'browser_capture_ready','timestamp':1,'data':{'clock':'browser_monotonic','method':'audio_worklet'}}
    assert browser_timing_event(e,10)['data']['method'] == 'audio_worklet'
    e['data']['method'] = 'unrecognized'
    assert browser_timing_event(e,10) is None
