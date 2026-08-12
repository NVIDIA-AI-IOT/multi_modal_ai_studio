// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
    buildResetConfig,
    getTtsModelName,
    matchesOpenAiAsrDiscovery,
    matchesOpenAiTtsDiscovery,
    matchesRivaDiscovery,
    mergeRecordedSpeechConfig,
    normalizeLanguageTag,
    normalizeSpeechBackendConfig,
    resolveTtsVoiceSelection,
    ttsVoicesForLanguage,
} = require('../../src/multi_modal_ai_studio/webui/static/config_helpers.js');

test('reset defaults retain the server preset without mutating either input', () => {
    const defaults = {
        asr: {backend: 'riva', server: 'localhost:50051', language: 'en-US'},
        tts: {backend: 'riva', voice: ''},
        devices: {microphone: 'browser'},
    };
    const preset = {
        name: 'NVIDIA Open Models Speech',
        asr: {
            backend: 'openai-rest',
            api_base: 'http://localhost:8081/v1',
            model: 'nvidia/nemotron-3.5-asr-streaming-0.6b',
        },
        tts: {
            backend: 'openai-rest',
            api_base: 'http://localhost:8082/v1',
            model: 'nvidia/magpie_tts_multilingual_357m',
        },
    };

    const result = buildResetConfig(defaults, preset);

    assert.equal(result.asr.backend, 'openai-rest');
    assert.equal(result.asr.server, 'localhost:50051');
    assert.equal(result.asr.model, 'nvidia/nemotron-3.5-asr-streaming-0.6b');
    assert.equal(result.tts.backend, 'openai-rest');
    assert.equal(result.devices.microphone, 'browser');
    assert.equal(defaults.asr.backend, 'riva');
    assert.equal(preset.asr.server, undefined);
});

test('stale Riva ASR discovery cannot overwrite an OpenAI REST preset', () => {
    const config = {
        backend: 'openai-rest',
        scheme: 'openai-rest',
        api_base: 'http://localhost:8081/v1',
        model: 'nvidia/nemotron-3.5-asr-streaming-0.6b',
    };

    assert.equal(matchesRivaDiscovery(config, 'localhost:50051'), false);
    assert.equal(config.model, 'nvidia/nemotron-3.5-asr-streaming-0.6b');
});

test('Riva discovery applies only to the current server and language', () => {
    const config = {
        backend: 'riva',
        scheme: 'riva',
        riva_server: 'localhost:50051',
        language: 'en-US',
    };

    assert.equal(matchesRivaDiscovery(config, 'localhost:50051'), true);
    assert.equal(matchesRivaDiscovery(config, 'other:50051'), false);
    assert.equal(matchesRivaDiscovery(config, 'localhost:50051', 'en-US'), true);
    assert.equal(matchesRivaDiscovery(config, 'localhost:50051', 'ja-JP'), false);
});

test('conflicting backend and scheme fail closed', () => {
    const config = {
        backend: 'riva',
        scheme: 'openai-rest',
        riva_server: 'localhost:50051',
    };

    assert.equal(matchesRivaDiscovery(config, 'localhost:50051'), false);
});

test('OpenAI REST TTS discovery is scoped to endpoint and model', () => {
    const config = {
        backend: 'openai-rest',
        scheme: 'openai-rest',
        api_base: 'http://localhost:18080/v1',
        model: 'kokoro-fp16',
    };
    assert.equal(matchesOpenAiTtsDiscovery(config, 'http://localhost:18080/v1', 'kokoro-fp16'), true);
    assert.equal(matchesOpenAiTtsDiscovery(config, 'http://other:18080/v1', 'kokoro-fp16'), false);
    assert.equal(matchesOpenAiTtsDiscovery(config, 'http://localhost:18080/v1', 'other-model'), false);
    assert.equal(matchesOpenAiTtsDiscovery({...config, backend: 'riva', scheme: 'riva'}, 'http://localhost:18080/v1', 'kokoro-fp16'), false);
});

test('OpenAI REST ASR discovery is scoped to the current endpoint', () => {
    const config = {
        backend: 'openai-rest',
        scheme: 'openai-rest',
        api_base: 'http://localhost:18080/v1',
    };
    assert.equal(matchesOpenAiAsrDiscovery(config, 'http://localhost:18080/v1'), true);
    assert.equal(matchesOpenAiAsrDiscovery(config, 'http://other:18080/v1'), false);
    assert.equal(matchesOpenAiAsrDiscovery({...config, backend: 'riva', scheme: 'riva'}, 'http://localhost:18080/v1'), false);
});

test('saved speech config scheme selects the correct review backend', () => {
    const recorded = {
        scheme: 'openai-rest',
        api_base: 'http://127.0.0.1:18080/v1',
    };
    const normalized = normalizeSpeechBackendConfig(recorded);

    assert.equal(normalized.backend, 'openai-rest');
    assert.equal(normalized.scheme, 'openai-rest');
    assert.equal(recorded.backend, undefined);
});

test('recorded REST scheme overrides the Riva UI default during review merge', () => {
    const merged = mergeRecordedSpeechConfig(
        {backend: 'riva', scheme: 'riva', server: 'localhost:50051'},
        {
            scheme: 'openai-rest',
            api_base: 'http://127.0.0.1:18080/v1',
            model: 'Systran/faster-whisper-tiny.en',
        },
    );

    assert.equal(merged.backend, 'openai-rest');
    assert.equal(merged.scheme, 'openai-rest');
    assert.equal(merged.api_base, 'http://127.0.0.1:18080/v1');
});

test('speech backend normalization retains explicit editable backend', () => {
    const normalized = normalizeSpeechBackendConfig({
        backend: 'openai-realtime',
        scheme: 'openai-rest',
    });
    assert.equal(normalized.backend, 'openai-realtime');
});

test('REST TTS voices are filtered with normalized language tags', () => {
    const voices = [
        {id: 'af_heart', language: 'en-us'},
        {id: 'bf_emma', language: 'en-GB'},
        {id: 'jf_alpha', language: 'ja'},
        {id: 'untagged'},
    ];
    assert.equal(normalizeLanguageTag('en_us'), 'en-US');
    assert.deepEqual(
        ttsVoicesForLanguage(voices, 'en-US').map(v => v.id),
        ['af_heart', 'untagged'],
    );
    assert.deepEqual(
        ttsVoicesForLanguage(voices, 'en').map(v => v.id),
        ['af_heart', 'bf_emma', 'untagged'],
    );
    assert.deepEqual(
        ttsVoicesForLanguage(voices, 'ja').map(v => v.id),
        ['jf_alpha', 'untagged'],
    );
});

test('REST TTS language change replaces a voice from the previous language', () => {
    const voices = [
        {id: 'af_heart', language: 'en-US'},
        {id: 'jf_alpha', language: 'ja'},
        {id: 'jf_nezumi', language: 'ja'},
    ];
    const result = resolveTtsVoiceSelection(voices, 'ja', 'af_heart', true);
    assert.deepEqual(result.voices.map(v => v.id), ['jf_alpha', 'jf_nezumi']);
    assert.equal(result.selectedVoice, 'jf_alpha');
});

test('REST TTS keeps a configured value only when provider metadata is unavailable', () => {
    const result = resolveTtsVoiceSelection([], 'ja', 'custom-voice', false);
    assert.deepEqual(result.voices.map(v => v.id), ['custom-voice']);
    assert.equal(result.selectedVoice, 'custom-voice');
});

test('REST TTS pipeline metadata uses model instead of voice', () => {
    const config = {
        tts_model_name: 'Sofia',
        tts: {
            scheme: 'openai-rest',
            model: 'nvidia/magpie_tts_multilingual_357m',
            voice: 'Sofia',
            riva_model_name: 'Sofia',
        },
    };

    assert.equal(
        getTtsModelName(config),
        'nvidia/magpie_tts_multilingual_357m',
    );
});

test('Riva TTS pipeline metadata retains discovered model name', () => {
    const config = {
        tts_model_name: 'magpie_tts_ensemble-Magpie-Multilingual',
        tts: {
            scheme: 'riva',
            riva_model_name: 'magpie_tts_ensemble-Magpie-Multilingual',
            voice: 'Magpie-Multilingual.EN-US.Sofia',
        },
    };

    assert.equal(
        getTtsModelName(config),
        'magpie_tts_ensemble-Magpie-Multilingual',
    );
});
