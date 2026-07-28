// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
    matchesRivaDiscovery,
} = require('../../src/multi_modal_ai_studio/webui/static/config_helpers.js');

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
