// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
    applySpeechTimingEvent,
    buildBargeInWindows,
    buildIntervalPeakMarkers,
    buildPeakPreservingPoints,
    buildTtsSegmentsFromTimeline,
    closeTtlBandAt,
    dedupeTimelineEventsByTimestamp,
    getAsrLegendSpec,
    hasDenseTtsAmplitudeTimeline,
    matchAsrRequestIntervalsToFinals,
    pairTimelineEvents,
    rebuildTtsPlaybackSegments,
    selectFirstPlaybackTimes,
    resolveTtsFirstAudioTimes,
    splitAudioSegmentsAt,
    syncLiveSessionClock,
    splitPointsAtTimeGaps,
    truncateAudioSegmentsAt,
} = require('../../src/multi_modal_ai_studio/webui/static/timeline_helpers.js');

test('ASR legend describes REST request timing', () => {
    const spec = getAsrLegendSpec({
        asr: {
            scheme: 'openai-rest',
            api_base: 'http://localhost:8081/v1',
            // Stale fields from a previous UI selection must not override the
            // explicit backend.
            riva_server: 'localhost:50051',
        },
    });

    assert.equal(spec.mode, 'rest');
    assert.equal(spec.activeLabel, 'ASR Request');
    assert.equal(spec.endpointLabel, 'Endpoint Wait');
});

test('ASR legend describes persistent streaming transports', () => {
    const realtime = getAsrLegendSpec({
        asr: { scheme: 'openai-realtime' },
    });
    const riva = getAsrLegendSpec({
        asr: { scheme: 'riva', riva_server: 'localhost:50051' },
    });

    assert.equal(realtime.activeLabel, 'Streaming ASR');
    assert.equal(realtime.endpointLabel, 'Finalization');
    assert.equal(riva.activeLabel, 'Streaming ASR');
    assert.equal(riva.endpointLabel, 'Endpoint / Finalize');
});

test('ASR request starts and ends pair without crossing the next request', () => {
    const intervals = pairTimelineEvents([
        { event_type: 'asr_inference_end', timestamp: 5.3 },
        { event_type: 'asr_inference_start', timestamp: 2.0 },
        { event_type: 'asr_inference_start', timestamp: 5.0 },
        { event_type: 'asr_inference_end', timestamp: 2.2 },
    ], 'asr_inference_start', 'asr_inference_end');

    assert.deepEqual(intervals.map(interval => [interval.start, interval.end]), [
        [2.0, 2.2],
        [5.0, 5.3],
    ]);
});

test('turn-scoped first audio survives a short barge-in playback gap', () => {
    const playbackTimes = selectFirstPlaybackTimes(
        2,
        [12.645, 22.784],
        [12.653, 22.788],
        [
            [
                { startTime: 12.653, endTime: 22.098 },
                // Only 0.69 s separates these replies, so legacy gap-based
                // waveform grouping merges them.
                { startTime: 22.788, endTime: 23.138 },
            ],
        ]
    );
    assert.deepEqual(playbackTimes, [12.645, 22.784]);
});

test('ASR request matching excludes an empty request before the next speech turn', () => {
    const matched = matchAsrRequestIntervalsToFinals([
        { start: 4.263, end: 4.629 },
        { start: 8.245, end: 8.523 },
        { start: 10.639, end: 11.101 },
    ], [
        { timestamp: 4.630 },
        { timestamp: 11.102 },
    ]);

    assert.deepEqual(matched.map(interval => [interval.start, interval.end]), [
        [4.263, 4.629],
        [10.639, 11.101],
    ]);
});

test('ASR GPU marker selects only the strongest sample inside each request', () => {
    const markers = buildIntervalPeakMarkers([
        { t: 1.9, gpu: 99 },
        { t: 2.02, gpu: 41 },
        { t: 2.08, gpu: 96 },
        { t: 2.2, gpu: 88 },
        { t: 5.1, gpu: 3 },
    ], [
        { start: 2.0, end: 2.1 },
        { start: 5.0, end: 5.2 },
    ], 5);

    assert.equal(markers.length, 1);
    assert.equal(markers[0].t, 2.08);
    assert.equal(markers[0].value, 96);
});

test('GPU plotting retains the peak when samples share a screen pixel', () => {
    const points = buildPeakPreservingPoints(
        [
            { t: 1.00, gpu: 0 },
            { t: 1.01, gpu: 73 },
            { t: 1.02, gpu: 12 },
            { t: 1.20, gpu: 4 },
        ],
        sample => sample.t * 10,
        sample => sample.gpu,
    );

    assert.deepEqual(points.map(point => point.value), [73, 4]);
});

test('utilization plotting splits rather than interpolating missing telemetry', () => {
    const runs = splitPointsAtTimeGaps([
        { t: 6.30, value: 56 },
        { t: 6.35, value: 56 },
        { t: 8.28, value: 0 },
        { t: 8.33, value: 0 },
    ], 0.2);

    assert.deepEqual(runs.map(run => run.map(point => point.t)), [
        [6.30, 6.35],
        [8.28, 8.33],
    ]);
});

test('GPU fallback samples remain continuous despite 250 ms cadence jitter', () => {
    const runs = splitPointsAtTimeGaps([
        { t: 4.9826, value: 87.2 },
        { t: 5.2391, value: 84.1 },
        { t: 5.4972, value: 99.7 },
        { t: 5.7535, value: 99.6 },
        // A missed 250 ms fallback sample must still create a gap.
        { t: 6.2704, value: 99.4 },
    ], 0.4);

    assert.deepEqual(runs.map(run => run.map(point => point.t)), [
        [4.9826, 5.2391, 5.4972, 5.7535],
        [6.2704],
    ]);
});

test('TTFA prefers the browser-observed playback start over server first bytes', () => {
    const times = resolveTtsFirstAudioTimes(
        [
            { timestamp: 5.020 },
            { timestamp: 11.492 },
        ],
        [
            { timestamp: 6.196 },
            { timestamp: 12.714 },
        ],
        [
            { startTime: 5.974, endTime: 5.999 },
            { startTime: 12.493, endTime: 12.518 },
        ],
    );

    assert.deepEqual(times, [5.974, 12.493]);
});

test('ASR boundary aliases at the same timestamp count as one turn boundary', () => {
    const boundaries = dedupeTimelineEventsByTimestamp([
        { event_type: 'vad_end', timestamp: 2.5 },
        { event_type: 'user_speech_end', timestamp: 2.5 },
        { event_type: 'vad_end', timestamp: 8.0 },
        { event_type: 'user_speech_end', timestamp: 8.002 },
    ]);

    assert.deepEqual(boundaries.map(event => event.timestamp), [2.5, 8.0]);
});

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

test('legacy generated TTS envelopes are rebuilt as continuous browser playout', () => {
    const segments = rebuildTtsPlaybackSegments([
        { event_type: 'tts_start', timestamp: 10 },
        { event_type: 'tts_first_audio', timestamp: 12 },
        { event_type: 'audio_amplitude', source: 'tts', timestamp: 11.6, amplitude: 4 },
        { event_type: 'audio_amplitude', source: 'tts', timestamp: 11.625, amplitude: 8 },
        // The next generated chunk overlaps in server time, but follows the
        // first chunk in WebAudio playout order.
        { event_type: 'audio_amplitude', source: 'tts', timestamp: 11.5, amplitude: 12 },
        { event_type: 'tts_playback_stopped', timestamp: 12.29 },
        { event_type: 'tts_start', timestamp: 20 },
        { event_type: 'tts_first_audio', timestamp: 21 },
        { event_type: 'audio_amplitude', source: 'tts', timestamp: 20.7, amplitude: 16 },
    ], [
        { start: 9, end: 12.25 },
        { start: 19, end: 21.5 },
    ]);

    assert.deepEqual(segments.map(segment => segment.amplitude), [4, 8, 16]);
    assert.ok(Math.abs(segments[0].startTime - 12.25) < 1e-9);
    assert.ok(Math.abs(segments[1].startTime - 12.275) < 1e-9);
    assert.ok(Math.abs(segments[1].endTime - 12.29) < 1e-9);
    assert.ok(Math.abs(segments[2].startTime - 21.5) < 1e-9);
});

test('live session clock replaces pre-connect click time with server timeline origin', () => {
    // START was clicked at wall-clock 1000.0, but session_start(timestamp=0)
    // arrived two seconds later. The live origin must become 1002.0 so a
    // server amplitude at t=12 renders at wall-clock 1014, not 1012.
    assert.equal(syncLiveSessionClock(1002, 0), 1002);
    assert.equal(syncLiveSessionClock(1002.125, 0.125), 1002);
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

test('barge-in preserves discarded scheduled audio as a separate diagnostic region', () => {
    const result = splitAudioSegmentsAt([
        { startTime: 4, endTime: 5, amplitude: 10 },
        { startTime: 5, endTime: 7, amplitude: 20 },
        { startTime: 7, endTime: 8, amplitude: 30 },
    ], 6.25);

    assert.deepEqual(result.played, [
        { startTime: 4, endTime: 5, amplitude: 10 },
        { startTime: 5, endTime: 6.25, amplitude: 20 },
    ]);
    assert.deepEqual(result.discarded, [
        { startTime: 6.25, endTime: 7, amplitude: 20 },
        { startTime: 7, endTime: 8, amplitude: 30 },
    ]);
    assert.equal(result.discardedEnd, 8);
});

test('barge-in windows stop at cancellation and never borrow the next turn', () => {
    const windows = buildBargeInWindows([
        { event_type: 'tts_start', timestamp: 2 },
        {
            event_type: 'tts_playback_stopped',
            timestamp: 4,
            data: { reason: 'barge_in', discarded_audio_end: 5.5 },
        },
        { event_type: 'tts_cancelled', timestamp: 4.8 },
        { event_type: 'tts_start', timestamp: 8 },
        {
            event_type: 'tts_playback_stopped',
            timestamp: 9,
            data: { reason: 'barge_in', discarded_audio_end: 11 },
        },
        { event_type: 'tts_start', timestamp: 10 },
        { event_type: 'tts_cancelled', timestamp: 10.5 },
    ]);

    assert.equal(windows.length, 2);
    assert.equal(windows[0].cancelEnd, 4.8);
    assert.equal(windows[0].discardedEnd, 5.5);
    assert.equal(windows[1].cancelEnd, null);
    assert.equal(windows[1].discardedEnd, 10);
});

test('old TTS audio cannot close the new user turn TTL band', () => {
    const state = {
        liveTtlBands: [],
        liveTtlBandStartTime: 15.5,
        voiceTurnActive: true,
        ttsEligibleForCurrentTtl: false,
    };

    assert.equal(closeTtlBandAt(state, 16.2), false);
    assert.equal(state.liveTtlBandStartTime, 15.5);
    assert.deepEqual(state.liveTtlBands, []);
});

test('TTL band rejects negative duration and accepts current response audio', () => {
    const state = {
        liveTtlBands: [],
        liveTtlBandStartTime: 15.5,
        voiceTurnActive: true,
        ttsEligibleForCurrentTtl: true,
        lastAsrPartialTime: 15.2,
        firstTtsPlayTimeThisResponse: 12.3,
        earliestTtsPlayTimeAboveThreshold: 12.4,
    };

    assert.equal(closeTtlBandAt(state, 12.3), false);
    assert.equal(state.liveTtlBandStartTime, 15.5);

    state.firstTtsPlayTimeThisResponse = 17.1;
    state.earliestTtsPlayTimeAboveThreshold = 17.2;
    assert.equal(closeTtlBandAt(state, 17.1), true);
    assert.deepEqual(state.liveTtlBands, [
        { start: 15.5, end: 17.1, ttlMs: 1600 },
    ]);
    assert.equal(state.liveTtlBandStartTime, null);
});
