// SPDX-License-Identifier: Apache-2.0
(function (root) {
    'use strict';
    function reset(state) {
        state.nativeUtteranceOwners = Object.create(null);
        state.nativePendingUtterances = Object.create(null);
        state.nativeAwaitingUserFinal = false;
    }
    function observeUserSpeech(state) {
        if (!state.nativeAwaitingUserFinal) state.nativePendingUtterances = Object.create(null);
        state.nativeAwaitingUserFinal = true;
    }
    function attach(state, turn, data) {
        if (!turn.nativeUtterances) turn.nativeUtterances = Object.create(null);
        const id = String(data.utterance_id);
        turn.nativeUtterances[id] = String(data.text).trim();
        turn.assistant = Object.values(turn.nativeUtterances).filter(Boolean).join(' ');
        state.nativeUtteranceOwners[id] = turn;
    }
    function applyOutput(state, data) {
        const owner = state.nativeUtteranceOwners[String(data.utterance_id)];
        if (owner) {
            attach(state, owner, data);
        } else if (state.nativeAwaitingUserFinal) {
            state.nativePendingUtterances[String(data.utterance_id)] = data;
            return;
        } else {
            if (!state.liveChatTurns.length) state.liveChatTurns.push({ user: '', assistant: '' });
            attach(state, state.liveChatTurns[state.liveChatTurns.length - 1], data);
        }
    }
    function completeUser(state) {
        state.nativeAwaitingUserFinal = false;
        const turn = state.liveChatTurns[state.liveChatTurns.length - 1];
        Object.values(state.nativePendingUtterances).forEach(data => attach(state, turn, data));
        state.nativePendingUtterances = Object.create(null);
    }
    function transcriptMarkers(timeline) {
        const byId = new Map();
        timeline.forEach((event, index) => {
            const kind = event.event_type;
            if (kind !== 'realtime_output_partial' && kind !== 'realtime_output_final') return;
            const data = event.data || {};
            if (!data.text || !Number.isFinite(event.timestamp)) return;
            if (data.utterance_id == null && kind === 'realtime_output_partial') return;
            const id = data.utterance_id == null ? 'legacy-' + index : String(data.utterance_id);
            const previous = byId.get(id);
            if (!previous || !previous.finalized || kind === 'realtime_output_final') {
                byId.set(id, {id, timestamp:event.timestamp, text:String(data.text), finalized:kind === 'realtime_output_final'});
            }
        });
        if (!byId.size) timeline.forEach((event, index) => {
            if (event.event_type === 'tts_complete' && event.data && event.data.text) {
                byId.set('audio-' + index, {id:'audio-' + index, timestamp:event.timestamp, text:String(event.data.text), finalized:true});
            }
        });
        return Array.from(byId.values()).sort((a, b) => a.timestamp-b.timestamp);
    }
    function fitLabel(text, width, measure) {
        if (width < 24) return '';
        if (measure(text) <= width) return text;
        let count = Math.min(text.length, 60);
        while (count && measure(text.slice(0, count) + '…') > width) count--;
        return count ? text.slice(0, count) + '…' : '';
    }
    function layoutTranscriptLabels(markers, left, right, measure) {
        let limit = right;
        const placed = [];
        for (let i = markers.length - 1; i >= 0; i--) {
            const marker = markers[i];
            if (marker.x < left - 0.5 || marker.x > right + 0.5) continue;
            const x = Math.max(left, Math.min(right, marker.x));
            const room = Math.min(260, limit - x - 14);
            const onLeft = room < 24;
            const label = fitLabel(marker.text.trim(), onLeft ? Math.min(260, x-left-14) : room, measure);
            const textX = onLeft ? x - 9 - measure(label) : x + 9;
            placed.unshift({...marker, x, label, textX});
            limit = onLeft ? textX - 6 : x - 6;
        }
        return placed;
    }
    const api = { reset, observeUserSpeech, applyOutput, completeUser, transcriptMarkers, fitLabel, layoutTranscriptLabels };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    root.MMASDuplexHelpers = api;
})(typeof window !== 'undefined' ? window : globalThis);
