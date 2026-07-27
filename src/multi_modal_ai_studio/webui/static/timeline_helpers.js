// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(function (root, factory) {
    const helpers = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = helpers;
    }
    root.MMASTimelineHelpers = helpers;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    // Build coarse TTS playback segments for sessions recorded before dense
    // 25 ms TTS RMS samples were persisted. Playback starts at first audio,
    // not at synthesis start.
    function buildTtsSegmentsFromTimeline(timeline) {
        if (!timeline || !timeline.length) return [];
        const starts = timeline.filter(function (e) { return e.event_type === 'tts_start'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        const firstAudios = timeline.filter(function (e) { return e.event_type === 'tts_first_audio'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        const completes = timeline.filter(function (e) { return e.event_type === 'tts_complete'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        if (!starts.length) return [];
        return starts.map(function (s, i) {
            const synthesisStart = s.timestamp != null ? Number(s.timestamp) : 0;
            const nextSynthesisStart = starts[i + 1] && starts[i + 1].timestamp != null ? Number(starts[i + 1].timestamp) : Infinity;
            const firstAudio = firstAudios.find(function (e) {
                const t = e.timestamp != null ? Number(e.timestamp) : NaN;
                return !isNaN(t) && t >= synthesisStart && t < nextSynthesisStart;
            });
            const endEvent = completes.find(function (e) {
                const t = e.timestamp != null ? Number(e.timestamp) : NaN;
                return !isNaN(t) && t >= synthesisStart && t < nextSynthesisStart;
            });
            const startTime = firstAudio && firstAudio.timestamp != null ? Number(firstAudio.timestamp) : synthesisStart;
            const endTime = endEvent && endEvent.timestamp != null ? Math.max(startTime + 0.025, Number(endEvent.timestamp)) : startTime + 0.1;
            return { startTime: startTime, endTime: endTime, amplitude: 50 };
        });
    }

    // A legacy OpenAI-REST recording stored one RMS value per multi-second
    // HTTP chunk. Treat that as sparse; otherwise interpolation draws only
    // isolated purple pixels.
    function hasDenseTtsAmplitudeTimeline(timeline) {
        if (!timeline || !timeline.length) return false;
        const samples = timeline.filter(function (e) {
            return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai');
        }).map(function (e) {
            return e.timestamp != null ? Number(e.timestamp) : NaN;
        }).filter(function (t) {
            return !isNaN(t);
        }).sort(function (a, b) {
            return a - b;
        });
        for (let i = 1; i < samples.length; i++) {
            if (samples[i] - samples[i - 1] <= 0.1) return true;
        }
        return false;
    }

    // Apply the speech timing state shared by the REST and Realtime paths.
    // Returns true when the caller should not process the event further.
    function applySpeechTimingEvent(state, evt) {
        if (!state || !evt) return false;
        if (evt.event_type === 'vad_start' || evt.event_type === 'user_speech_start') {
            state.voiceTurnActive = true;
            state.voiceSilenceCandidate = null;
            state.voiceSilenceConsecutiveCount = 0;
            return true;
        }
        if (evt.event_type === 'vad_end' || evt.event_type === 'user_speech_end') {
            const vadEndTime = evt.timestamp != null ? Number(evt.timestamp) : NaN;
            if (!isNaN(vadEndTime) && state.liveTtlBandStartTime == null) {
                state.liveTtlBandStartTime = vadEndTime;
                state.voiceTurnActive = true;
            }
            return true;
        }
        if (evt.event_type === 'asr_final' && state.liveTtlBandStartTime == null) {
            const finalTime = evt.timestamp != null ? Number(evt.timestamp) : NaN;
            if (!isNaN(finalTime)) {
                state.liveTtlBandStartTime = state.lastAsrPartialTime != null
                    ? state.lastAsrPartialTime
                    : finalTime - 0.2;
                state.voiceTurnActive = true;
            }
        }
        return false;
    }

    return {
        applySpeechTimingEvent: applySpeechTimingEvent,
        buildTtsSegmentsFromTimeline: buildTtsSegmentsFromTimeline,
        hasDenseTtsAmplitudeTimeline: hasDenseTtsAmplitudeTimeline,
    };
}));
