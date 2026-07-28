// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
    applySpeechTimingEvent,
    buildTtsSegmentsFromTimeline,
    hasDenseTtsAmplitudeTimeline,
    truncateAudioSegmentsAt,
} = require('../../src/multi_modal_ai_studio/webui/static/timeline_helpers.js');

test('coarse TTS replay starts at first audio and stays within each turn', () => {
    const segments = buildTtsSegmentsFromTimeline([
        { event_type: 'tts_complete', timestamp: 12 },
        { event_type: 'tts_start', timestamp: 10 },
        { event_type: 'tts_first_audio', timestamp: 2 },
        { event_type: 'tts_complete', timestamp: 5 },
        { event_type: 'tts_start', timestamp: 1 },
        { event_type: 'tts_first_audio', timestamp: 10.5 },
    ]);

    assert.deepEqual(segments, [
        { startTime: 2, endTime: 5, amplitude: 50 },
        { startTime: 10.5, endTime: 12, amplitude: 50 },
    ]);
});

test('legacy TTS replay falls back to synthesis time without first-audio event', () => {
    const segments = buildTtsSegmentsFromTimeline([
        { event_type: 'tts_start', timestamp: 3 },
        { event_type: 'tts_complete', timestamp: 4 },
    ]);

    assert.deepEqual(segments, [
        { startTime: 3, endTime: 4, amplitude: 50 },
    ]);
});

test('dense TTS amplitude requires adjacent samples at 100 ms or less', () => {
    const sample = (timestamp, source = 'tts') => ({
        event_type: 'audio_amplitude',
        source,
        timestamp,
    });

    assert.equal(hasDenseTtsAmplitudeTimeline([sample(1), sample(1.025)]), true);
    assert.equal(hasDenseTtsAmplitudeTimeline([sample(1), sample(1.2)]), false);
    assert.equal(hasDenseTtsAmplitudeTimeline([sample(1, 'user'), sample(1.025, 'user')]), false);
});

test('VAD start and end drive the REST ASR turn and TTL state', () => {
    const state = {
        liveTtlBandStartTime: null,
        voiceSilenceCandidate: 1.5,
        voiceSilenceConsecutiveCount: 3,
        voiceTurnActive: false,
    };

    assert.equal(
        applySpeechTimingEvent(state, { event_type: 'vad_start', timestamp: 1 }),
        true,
    );
    assert.equal(state.voiceTurnActive, true);
    assert.equal(state.voiceSilenceCandidate, null);
    assert.equal(state.voiceSilenceConsecutiveCount, 0);

    assert.equal(
        applySpeechTimingEvent(state, { event_type: 'vad_end', timestamp: 2.75 }),
        true,
    );
    assert.equal(state.liveTtlBandStartTime, 2.75);
});

test('final-only REST ASR gets a TTL fallback without a partial transcript', () => {
    const state = {
        lastAsrPartialTime: null,
        liveTtlBandStartTime: null,
        voiceTurnActive: false,
    };

    assert.equal(
        applySpeechTimingEvent(state, { event_type: 'asr_final', timestamp: 7 }),
        false,
    );
    assert.equal(state.liveTtlBandStartTime, 6.8);
    assert.equal(state.voiceTurnActive, true);
});

test('ASR final preserves the last partial as the TTL start when available', () => {
    const state = {
        lastAsrPartialTime: 4.25,
        liveTtlBandStartTime: null,
        voiceTurnActive: false,
    };

    applySpeechTimingEvent(state, { event_type: 'asr_final', timestamp: 5 });

    assert.equal(state.liveTtlBandStartTime, 4.25);
});

test('barge-in truncates scheduled AI audio at the actual stop time', () => {
    const segments = truncateAudioSegmentsAt([
        { startTime: 4, endTime: 5, amplitude: 10 },
        { startTime: 5, endTime: 7, amplitude: 20 },
        { startTime: 7, endTime: 8, amplitude: 30 },
    ], 6.25);

    assert.deepEqual(segments, [
        { startTime: 4, endTime: 5, amplitude: 10 },
        { startTime: 5, endTime: 6.25, amplitude: 20 },
    ]);
});
