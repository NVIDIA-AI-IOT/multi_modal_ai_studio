// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const helpers = require('../../src/multi_modal_ai_studio/webui/static/config_helpers.js');
const source = fs.readFileSync(path.join(__dirname, '../../src/multi_modal_ai_studio/webui/static/app.js'), 'utf8');

// Run the actual UI functions without starting the browser's device/session loop.
function harness(profile = 'qwen3-customvoice', metadata = {}) {
    const elements = {};
    for (const id of ['voice-select', 'language-select', 'model-select', 'model-hint', 'metadata-hint']) {
        elements['tts-rest-' + id] = {innerHTML: '', textContent: '', isConnected: true};
    }
    elements['config-tab-content'] = {innerHTML: '', querySelector: () => null};
    const logs = [];
    const context = vm.createContext({
        currentConfig: {tts: {
            backend: 'openai-rest', scheme: 'openai-rest', api_base: 'http://localhost:18083/v1',
            model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', voice: 'ryan', language: 'en',
            sample_rate: 24000, speed: 1, request_profile: profile, instructions: 'Speak warmly.',
        }},
        state: {activeConfigTab: 'tts', isLiveSession: false, sessionState: 'setup'},
        window: {MMASConfigHelpers: helpers},
        document: {getElementById: id => elements[id]},
        fetch: async () => ({ok: true, json: async () => metadata}),
        escapeHtml: text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;'),
        renderReadonlySessionConfigNote: () => '',
        updateConfigTabStates: () => {}, refreshPipelineDisplay: () => {},
        renderEditableConfigForm: () => 'rerendered', setTimeout: () => {},
        getApiBase: () => '', stopTTSVoicePreview: () => {},
        console: {log: value => logs.push(value)},
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../../src/multi_modal_ai_studio/webui/static/voice_cloning.js'), 'utf8'), context);
    for (const name of ['renderTTSConfig', 'restTTSFallbackVoices', 'restTTSLanguageLabel', 'fetchOpenAITTSMetadata', 'updateConfig']) {
        const start = source.search(new RegExp('(?:async )?function ' + name + '\\('));
        assert.notEqual(start, -1);
        vm.runInContext(source.slice(start, source.indexOf('\n}', start) + 2), context);
    }
    return {context, elements, logs};
}

test('voice instructions are escaped, optional for CustomVoice, and read-only on review', () => {
    const {context} = harness();
    context.currentConfig.tts.instructions = '</textarea><script>alert(1)</script>';
    let html = context.renderTTSConfig(context.currentConfig.tts);
    assert.ok(html.includes('&lt;/textarea&gt;&lt;script&gt;'));
    assert.ok(!html.includes('<script>alert(1)</script>'));
    assert.match(html, /<textarea\s+id="tts-rest-instructions"/);
    html = context.renderTTSConfig(context.currentConfig.tts, true);
    assert.match(html, /<textarea disabled\s+id="tts-rest-instructions"/);
    assert.match(html, /<select disabled id="tts-rest-request-profile"/);
    assert.ok(!html.includes('id="tts-rest-preview-btn"'));
});

test('VoiceDesign requires instructions and disables named speakers', () => {
    const {context} = harness('qwen3-voicedesign');
    const html = context.renderTTSConfig(context.currentConfig.tts);
    assert.match(html, /<textarea\s+required id="tts-rest-instructions"/);
    assert.match(html, /<select disabled id="tts-rest-voice-select"/);
    assert.ok(html.includes('Designed from instructions'));
});

test('only explicit Qwen profiles lock the PCM sample rate and numeric speed', () => {
    for (const profile of ['openai', 'qwen3-customvoice', 'qwen3-voicedesign']) {
        const {context} = harness(profile);
        const html = context.renderTTSConfig(context.currentConfig.tts);
        assert.equal(/<select disabled id="tts-rest-sample-rate"/.test(html), profile !== 'openai');
        assert.equal(/<input type="range" disabled id="tts-rest-speed"/.test(html), profile !== 'openai');
    }
});

test('Qwen-specific voice tools are grouped separately from language and audio settings', () => {
    for (const [profile, title] of [
        ['qwen3-customvoice', 'CustomVoice'],
        ['qwen3-voicedesign', 'VoiceDesign'],
        ['qwen3-base', 'Voice Clone'],
    ]) {
        const {context} = harness(profile);
        for (const readonly of [false, true]) {
            const html = context.renderTTSConfig(context.currentConfig.tts, readonly);
            const start = html.indexOf('<section class="qwen-voice-panel');
            const end = html.indexOf('</section>', start);
            assert.ok(start > html.indexOf('id="tts-rest-language-select"'));
            assert.ok(end < html.indexOf('tts-audio-settings-grid'));
            const group = html.slice(start, end);
            assert.ok(group.includes('aria-labelledby="tts-qwen-voice-heading"'));
            assert.ok(group.includes('· ' + title));
            assert.ok(group.includes('id="tts-rest-voice-select"'));
            assert.equal(group.includes('id="tts-rest-instructions"'), profile !== 'qwen3-base');
            if (profile === 'qwen3-base') {
                assert.ok(group.includes(readonly ? 'Reference voice' : 'reference-voice-mode'));
            }
        }
    }
    const {context} = harness('openai');
    assert.ok(!context.renderTTSConfig(context.currentConfig.tts).includes('qwen-voice-panel'));
});

test('REST audio output settings contain sample rate, speed and quality without a duplicate quality control', () => {
    const {context} = harness('openai');
    context.currentConfig.tts.quality = 'medium';
    for (const readonly of [false, true]) {
        const html = context.renderTTSConfig(context.currentConfig.tts, readonly);
        const start = html.indexOf('tts-audio-settings-grid');
        const group = html.slice(start, html.indexOf('</section>', start));
        for (const id of ['tts-rest-sample-rate', 'tts-rest-speed', 'tts-rest-quality']) {
            assert.ok(group.includes('id="' + id + '"'));
            assert.ok(group.includes('for="' + id + '"'));
        }
        assert.equal((html.match(/class="form-group tts-quality-setting"/g) || []).length, 1);
        assert.match(group, /<option value="medium" selected>Medium/);
        assert.equal(/<select disabled id="tts-rest-quality"/.test(group), readonly);
    }
});

test('CustomVoice fallback exposes all nine speakers and ten languages plus auto', async () => {
    const {context, elements} = harness();
    await context.fetchOpenAITTSMetadata();
    assert.equal((elements['tts-rest-voice-select'].innerHTML.match(/<option /g) || []).length, 9);
    assert.equal((elements['tts-rest-language-select'].innerHTML.match(/<option /g) || []).length, 11);
    assert.match(elements['tts-rest-language-select'].innerHTML, /value="ja"/);
    assert.equal(context.currentConfig.tts.voice, 'ryan');
    assert.match(elements['tts-rest-metadata-hint'].textContent, /fallback/);
});

test('native speaker accent metadata does not filter Qwen multilingual speakers', async () => {
    const {context, elements} = harness('qwen3-customvoice', {
        voices: [{id: 'ryan', language: 'en'}, {id: 'ono_anna', language: 'ja'}],
        languages: ['English', 'Japanese'],
    });
    context.currentConfig.tts.language = 'Japanese';
    await context.fetchOpenAITTSMetadata();
    assert.match(elements['tts-rest-language-select'].innerHTML, /value="ja" selected/);
    assert.match(elements['tts-rest-voice-select'].innerHTML, /value="ryan" selected/);
    assert.match(elements['tts-rest-voice-select'].innerHTML, /value="ono_anna"/);
    assert.equal(context.currentConfig.tts.language, 'Japanese');
});

test('VoiceDesign discovery does not select a stale configured voice', async () => {
    const {context, elements} = harness('qwen3-voicedesign');
    await context.fetchOpenAITTSMetadata();
    assert.equal(elements['tts-rest-voice-select'].disabled, true);
    assert.equal(elements['tts-rest-voice-select'].innerHTML, '<option value="">Designed from instructions</option>');
});

test('late discovery cannot rewrite a changed profile or detached review form', async () => {
    for (const change of ['profile', 'detached']) {
        const {context, elements} = harness();
        let resolve;
        context.fetch = () => new Promise(done => { resolve = done; });
        const pending = context.fetchOpenAITTSMetadata();
        if (change === 'profile') context.currentConfig.tts.request_profile = 'qwen3-voicedesign';
        else elements['tts-rest-voice-select'].isConnected = false;
        resolve({ok: true, json: async () => ({voices: [{id: 'other'}]})});
        await pending;
        assert.equal(elements['tts-rest-voice-select'].innerHTML, '');
        assert.equal(context.currentConfig.tts.voice, 'ryan');
    }
});

test('typing instructions does not replace the focused form or log the instruction', () => {
    const {context, elements, logs} = harness();
    context.updateConfig('tts', 'instructions', 'Quiet and thoughtful.');
    assert.equal(context.currentConfig.tts.instructions, 'Quiet and thoughtful.');
    assert.equal(elements['config-tab-content'].innerHTML, '');
    assert.equal(logs.length, 0);
});

test('profile selection normalizes audio settings without redirecting endpoints/models', () => {
    const {context, elements} = harness('openai');
    context.currentConfig.tts.sample_rate = 22050;
    context.currentConfig.tts.speed = 1.25;
    const {api_base, model} = context.currentConfig.tts;
    context.updateConfig('tts', 'request_profile', 'qwen3-voicedesign');
    assert.equal(context.currentConfig.tts.sample_rate, 24000);
    assert.equal(context.currentConfig.tts.speed, 1);
    assert.equal(context.currentConfig.tts.api_base, api_base);
    assert.equal(context.currentConfig.tts.model, model);
    assert.equal(elements['config-tab-content'].innerHTML, 'rerendered');
});

test('recorded instructions/profile survive review merge; legacy sessions keep defaults', () => {
    const defaults = {backend: 'riva', request_profile: 'openai', instructions: ''};
    const recorded = {scheme: 'openai-rest', request_profile: 'qwen3-voicedesign', instructions: 'A calm robot.'};
    const merged = helpers.mergeRecordedSpeechConfig(defaults, recorded);
    assert.equal(merged.request_profile, 'qwen3-voicedesign');
    assert.equal(merged.instructions, 'A calm robot.');
    const legacy = helpers.mergeRecordedSpeechConfig(defaults, {scheme: 'openai-rest', voice: 'af_heart'});
    assert.equal(legacy.request_profile, 'openai');
    assert.equal(legacy.instructions, '');
});

test('Base exposes reference upload and mode, not VoiceDesign instructions', () => {
    const {context} = harness('qwen3-base');
    const html = context.renderTTSConfig(context.currentConfig.tts);
    assert.match(html, /reference-voice-file/);
    assert.match(html, /reference-voice-consent/);
    assert.match(html, /reference-voice-mode/);
    assert.match(html, /<input type="text" id="tts-preview-text"/);
    assert.match(html, /<input type="text" id="reference-voice-name"/);
    assert.ok(!html.includes('id="tts-rest-instructions"'));
    const review = context.renderTTSConfig(context.currentConfig.tts, true);
    assert.ok(!review.includes('reference-voice-file'));
    assert.ok(!review.includes('saveReferenceVoice'));
    assert.ok(!review.includes('tts-preview-text'));
});

test('Base lists server references and never sends named speaker choices', async () => {
    const {context, elements} = harness('qwen3-base', {voices: [{id: 'a'.repeat(32), name: 'My robot', duration: 5, transcript: 'Hello'}]});
    await context.loadReferenceVoices();
    assert.match(elements['tts-rest-voice-select'].innerHTML, /My robot/);
    assert.ok(!elements['tts-rest-voice-select'].innerHTML.includes('ryan'));
    context.selectReferenceVoice('a'.repeat(32));
    assert.equal(context.currentConfig.tts.reference_voice_id, 'a'.repeat(32));
    assert.equal(context.currentConfig.tts.voice, 'My robot');
});

test('late reference list cannot overwrite a detached or recorded form', async () => {
    for (const change of ['detached', 'recorded', 'config']) {
        const {context, elements} = harness('qwen3-base');
        let resolve;
        context.fetch = () => new Promise(done => { resolve = done; });
        const pending = context.loadReferenceVoices();
        if (change === 'detached') elements['tts-rest-voice-select'].isConnected = false;
        if (change === 'recorded') context.state.sessionState = 'review';
        if (change === 'config') context.currentConfig.tts = {...context.currentConfig.tts};
        resolve({ok: true, json: async () => ({voices: [{id: 'a'.repeat(32), name: 'Unwanted', duration: 1}]})});
        await pending;
        assert.equal(elements['tts-rest-voice-select'].innerHTML, '');
    }
});

test('VoiceDesign preview save is invalidated by changed voice settings', () => {
    const {context} = harness('qwen3-voicedesign');
    const config = context.currentConfig.tts;
    context.rememberDesignedVoicePreview(config, {}, encodeURIComponent('Hello robot.'));
    let html = context.renderReferenceVoiceControls(config, false);
    assert.ok(!/id="save-designed-voice"\s+disabled/.test(html));
    config.instructions = 'A different voice.';
    html = context.renderReferenceVoiceControls(config, false);
    assert.match(html, /id="save-designed-voice"\s+disabled/);
    assert.equal(context.renderReferenceVoiceControls(config, true), '');
});

test('switching to Base clears unsupported instructions and retains reference metadata', () => {
    const {context} = harness();
    context.currentConfig.tts.reference_voice_id = 'b'.repeat(32);
    context.updateConfig('tts', 'request_profile', 'qwen3-base');
    assert.equal(context.currentConfig.tts.instructions, '');
    const merged = helpers.mergeRecordedSpeechConfig({}, context.currentConfig.tts);
    assert.equal(merged.reference_voice_id, 'b'.repeat(32));
    assert.equal(merged.request_profile, 'qwen3-base');
});
