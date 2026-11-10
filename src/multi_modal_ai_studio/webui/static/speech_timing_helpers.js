// SPDX-License-Identifier: Apache-2.0
(function (root) {
    'use strict';
    // Observation only: these thresholds never gate model input or turn-taking.
    class EnergyObserver {
        constructor({sampleRate, source, startRms=0.0125, stopRms=0.0075, minimumMs=60, silenceMs=150}) {
            Object.assign(this, {sampleRate, source, startRms, stopRms, minimumMs, silenceMs});
            this.windowSize = Math.round(sampleRate * .01);
            this.sequence = 0;
            this.clear();
        }
        clear() {
            this.pending = []; this.pendingStart = null; this.candidate = null;
            this.active = null; this.lastActive = null; this.lastTime = null;
        }
        event(kind, time, reason='silence') {
            return {event_type:'browser_' + this.source + '_' + kind, timestamp:time,
                lane:this.source === 'input' ? 'speech' : 'tts',
                data:{region_id:this.active, source:this.source === 'input' ? 'browser_capture' : 'browser_schedule',
                    clock:'browser_monotonic', resolution_ms:10, reason}};
        }
        advance(time) {
            if (this.active != null && time-this.lastActive >= this.silenceMs/1000-1e-7) {
                const event = this.event('end', this.lastActive);
                this.active = null; this.candidate = null;
                return [event];
            }
            return [];
        }
        abort(time, reason) {
            const events = this.active == null ? [] : [this.event('end', time, reason)];
            this.clear();
            return events;
        }
        process(samples, startTime) {
            const events = [];
            if (this.lastTime != null && startTime-this.lastTime > .03) {
                events.push(...this.advance(startTime));
                this.pending = []; this.pendingStart = null; this.candidate = null;
            }
            for (let i=0; i<samples.length; i++) {
                if (!this.pending.length) this.pendingStart = startTime+i/this.sampleRate;
                this.pending.push(samples[i]);
                if (this.pending.length < this.windowSize) continue;
                const t = this.pendingStart, end = t+this.windowSize/this.sampleRate;
                const rms = Math.sqrt(this.pending.reduce((sum, s) => sum+s*s, 0)/this.windowSize);
                const voiced = rms >= (this.active == null ? this.startRms : this.stopRms);
                if (voiced) {
                    const first = this.source === 'output' ? this.pending.findIndex(s=>Math.abs(s)>=this.startRms) : 0;
                    if (this.active == null) {
                        if (this.candidate == null) this.candidate = t+Math.max(0,first)/this.sampleRate;
                        if (end-this.candidate >= this.minimumMs/1000-1e-7) {
                            this.active = ++this.sequence;
                            events.push(this.event('start', this.candidate, 'energy'));
                        }
                    }
                    this.lastActive = end;
                } else {
                    if (this.active == null) this.candidate = null;
                    events.push(...this.advance(end));
                }
                this.pending = []; this.pendingStart = null;
            }
            this.lastTime = startTime+samples.length/this.sampleRate;
            return events;
        }
    }
    function gaps(events) {
        const inputs = new Map(), outputs = new Map();
        events.forEach(event => {
            if (!event || !Number.isFinite(event.timestamp) || !event.data || event.data.clock !== 'browser_monotonic') return;
            const match = /^browser_(input|output)_(start|end)$/.exec(event.event_type);
            if (!match) return;
            const map = match[1] === 'input' ? inputs : outputs;
            const id = event.data.region_id;
            if (!map.has(id)) map.set(id,{id});
            map.get(id)[match[2]] = event.timestamp;
            if (match[2] === 'end') map.get(id).reason = event.data.reason;
        });
        const ins = [...inputs.values()].filter(i=>Number.isFinite(i.start)).sort((a,b)=>a.start-b.start);
        const outs = [...outputs.values()].filter(i=>Number.isFinite(i.start)).sort((a,b)=>a.start-b.start);
        return ins.map((input,index) => {
            const reply = {input_id:'browser-'+input.id, fd_ttl_ms:null, reaction_ms:null,
                input_end:input.end, audio_onset:null, association:'acoustic_sequence', endpoint_method:'browser_energy',status:'input_end_missing'};
            if (input.end == null) return reply;
            if (input.reason && input.reason !== 'silence') return {...reply,status:'input_interrupted'};
            const next = index+1 < ins.length ? ins[index+1].start : Infinity;
            const output = outs.find(o=>o.start>=input.start && o.start<next);
            if (!output) return {...reply,status:next<Infinity ? 'continued_input' :
                (outs.some(o=>o.start<input.start && (o.end==null || o.end>input.end)) ? 'overlapping_audio' : 'no_audio_onset')};
            return {...reply,status:'estimated',audio_onset:output.start,
                fd_ttl_ms:(output.start-input.end)*1000,reaction_ms:(output.start-input.start)*1000};
        });
    }
    const api = {EnergyObserver, gaps};
    root.MMASSpeechTiming = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
