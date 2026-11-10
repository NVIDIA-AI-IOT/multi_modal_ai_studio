'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {EnergyObserver,gaps} = require('../../src/multi_modal_ai_studio/webui/static/speech_timing_helpers.js');
const cases = JSON.parse(fs.readFileSync(path.join(__dirname,'../fixtures/browser_acoustic_gaps.json'),'utf8'));
for (const fixture of cases) test(fixture.name,()=> {
    const replies = gaps(fixture.events);
    assert.deepEqual(replies.map(r=>r.status),fixture.statuses);
    replies.forEach((r,i)=> fixture.ttl_ms[i] == null ? assert.equal(r.fd_ttl_ms,null) : assert.ok(Math.abs(r.fd_ttl_ms-fixture.ttl_ms[i])<1e-6));
});
function observe(segments,chunks,options={}) {
    const samples = Float32Array.from(segments.flatMap(([seconds,value])=>Array(Math.round(seconds*16000)).fill(value)));
    const observer = new EnergyObserver({sampleRate:16000,source:'input',...options});
    const events = [];
    let offset=0,index=0;
    while(offset<samples.length) {
        const size=chunks[index++%chunks.length];
        events.push(...observer.process(samples.slice(offset,offset+size),offset/16000));
        offset+=size;
    }
    return events;
}
test('audio-frame timing is independent of 128ms versus 20ms message delivery',()=> {
    const audio=[[.2,0],[.1,.05],[.05,0],[.1,.05],[.3,0]];
    const slow=observe(audio,[2048]),fast=observe(audio,[320]);
    assert.equal(slow.length,2);assert.equal(fast.length,2);
    slow.forEach((e,i)=>assert.ok(Math.abs(e.timestamp-fast[i].timestamp)<1e-6));
    assert.ok(Math.abs(fast[0].timestamp-.2)<1e-6);
    assert.ok(Math.abs(fast[1].timestamp-.45)<1e-6);
});
test('brief noise is rejected but a short speech region is retained without transcript gating',()=> {
    const events=observe([[.04,.05],[.2,0],[.08,.05],[.3,0]],[128]);
    assert.equal(events.length,2);
    assert.ok(Math.abs(events[0].timestamp-.24)<1e-6);
    assert.ok(Math.abs(events[1].timestamp-.32)<1e-6);
});
test('leading and all-silent output chunks do not create an onset',()=> {
    const events=observe([[.2,0],[.04,.05],[.2,0]],[320],{source:'output',startRms:.004,stopRms:.004,minimumMs:20});
    assert.equal(events[0].event_type,'browser_output_start');
    assert.ok(Math.abs(events[0].timestamp-.2)<1e-6);
    assert.equal(observe([[1,0]],[320],{source:'output',minimumMs:20}).length,0);
});
test('output onset excludes silence within the first active 10ms window',()=> {
    const events=observe([[.20375,0],[.05,.05],[.2,0]],[320],{source:'output',startRms:.004,stopRms:.004,minimumMs:20});
    assert.ok(Math.abs(events[0].timestamp-.20375)<1/16000);
});
