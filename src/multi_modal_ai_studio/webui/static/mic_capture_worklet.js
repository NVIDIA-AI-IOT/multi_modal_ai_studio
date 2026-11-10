// SPDX-License-Identifier: Apache-2.0
// Keep capture off the main thread; frame indices survive delayed delivery.
class VawMicCapture extends AudioWorkletProcessor {
    constructor() {
        super();
        this.pcm = new Int16Array(Math.round(sampleRate*.02));
        this.offset = 0; this.startFrame = 0;
    }
    process(inputs) {
        const input = inputs[0] && inputs[0][0];
        if (!input) return true;
        for (let i=0; i<input.length; i++) {
            if (!this.offset) this.startFrame = currentFrame+i;
            const value = Math.max(-1,Math.min(1,input[i]));
            this.pcm[this.offset++] = value < 0 ? value*32768 : value*32767;
            if (this.offset === this.pcm.length) {
                this.port.postMessage({pcm:this.pcm.buffer,startFrame:this.startFrame},[this.pcm.buffer]);
                this.pcm = new Int16Array(Math.round(sampleRate*.02)); this.offset = 0;
            }
        }
        return true;
    }
}
registerProcessor('vaw-mic-capture',VawMicCapture);
