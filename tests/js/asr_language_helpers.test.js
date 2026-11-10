'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const languages = require('../../src/multi_modal_ai_studio/webui/static/asr_language_helpers.js');
const native = {backend:'openai-realtime', realtime_api_style:'nemo-voicechat',
    model:'nemotron-voicechat', realtime_url:'ws://localhost:18088/v1/realtime'};

test('known 11B profile is explicit and provider metadata takes priority', () => {
    assert.deepEqual(languages.languageInfo(native), {languages:['en'], source:'model_profile'});
    assert.deepEqual(languages.languageInfo(native, ['en','ja']), {languages:['en','ja'], source:'provider'});
    assert.equal(languages.languageInfo({...native,model:'unknown'}).source, 'unknown');
    assert.equal(languages.languageInfo({...native,realtime_api_style:'openai-ga'}).source, 'unknown');
    assert.deepEqual(languages.languageInfo({...native,model:'unknown'}, ['multilingual', null, 12]).languages, []);
});

test('English locale aliases preserve configuration while unsupported languages stay invalid', () => {
    assert.deepEqual(languages.choices(['en','ja'], 'en-US'), {supported:true,options:[
        {code:'en',value:'en-US',selected:true}, {code:'ja',value:'ja',selected:false},
    ]});
    assert.equal(languages.choices(['en'], 'ja-JP').supported, false);
    assert.equal(languages.choices(['en-GB'], 'en-US').supported, false);
    assert.equal(languages.choices(['ja-JP'], 'ja_JP').supported, true);
});

test('discovery keys prevent stale languages from leaking across model, endpoint or credentials', () => {
    assert.equal(languages.apiBase(native), 'http://localhost:18088/v1');
    assert.equal(languages.apiBase({...native,realtime_url:'wss://example.org/v1/realtime?model=x'}), 'https://example.org/v1');
    assert.equal(languages.apiBase({...native,realtime_url:'invalid'}), '');
    for (const change of [{model:'next'}, {api_key:'different'}, {realtime_url:'ws://other/v1/realtime'}, {realtime_api_style:'openai-ga'}]) {
        assert.notEqual(languages.identity(native), languages.identity({...native,...change}));
    }
});
