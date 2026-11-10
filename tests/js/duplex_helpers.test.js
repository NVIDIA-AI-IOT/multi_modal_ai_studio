'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const duplex = require('../../src/multi_modal_ai_studio/webui/static/duplex_helpers.js');

test('greeting, answer and follow-up survive the next user turn and late final events', () => {
    const state = { liveChatTurns: [] };
    duplex.reset(state);
    duplex.applyOutput(state, { utterance_id: 1, text: 'Hello.' });
    duplex.observeUserSpeech(state);
    duplex.applyOutput(state, { utterance_id: 2, text: 'One' });
    state.liveChatTurns.push({ user: 'One plus one?', assistant: '' });
    duplex.completeUser(state);
    duplex.applyOutput(state, { utterance_id: 2, text: 'One plus one equals two.' });
    duplex.applyOutput(state, { utterance_id: 3, text: 'What is two plus two?' });
    duplex.observeUserSpeech(state);
    duplex.applyOutput(state, { utterance_id: 2, text: 'One plus one equals two.' });
    duplex.applyOutput(state, { utterance_id: 4, text: 'Four.' });
    state.liveChatTurns.push({ user: 'Two plus two?', assistant: '' });
    duplex.completeUser(state);
    duplex.applyOutput(state, { utterance_id: 3, text: 'What is two plus two?' });
    assert.deepEqual(state.liveChatTurns.map(t => [t.user, t.assistant]), [
        ['', 'Hello.'],
        ['One plus one?', 'One plus one equals two. What is two plus two?'],
        ['Two plus two?', 'Four.'],
    ]);
});

test('one label per utterance prefers final, without duplicating audio completion', () => {
    const rows = [
        {event_type:'realtime_output_partial', timestamp:1, data:{utterance_id:1, text:'One'}},
        {event_type:'realtime_output_partial', timestamp:2, data:{utterance_id:1, text:'One plus one'}},
        {event_type:'tts_complete', timestamp:3, data:{text:'Two.'}},
        {event_type:'realtime_output_final', timestamp:4, data:{utterance_id:1, text:'Two.'}},
        {event_type:'realtime_output_final', timestamp:5, data:{utterance_id:2, text:'Two.'}},
    ];
    const labels = duplex.transcriptMarkers(rows);
    assert.deepEqual(labels.map(x => x.text), ['Two.', 'Two.']);
    assert.ok(labels.every(x => x.finalized));
    assert.equal(duplex.fitLabel('a long label', 40, text => text.length * 8), 'a lo…');
    assert.equal(duplex.fitLabel('a long label', 12, text => text.length * 8), '');
});

test('final label at the viewport edge is placed left without label overlap', () => {
    const measure = text => text.length * 6;
    const placed = duplex.layoutTranscriptLabels([
        {x:350, text:'Earlier answer'}, {x:500, text:'One plus one equals two.'},
    ], 100, 500, measure);
    assert.equal(placed[1].label, 'One plus one equals two.');
    assert.ok(placed[1].textX < placed[1].x);
    assert.ok(placed[0].textX + measure(placed[0].label) < placed[1].textX);
    assert.equal(duplex.layoutTranscriptLabels([{x:500.00001,text:'Final'}],100,500,measure).length,1);
});
