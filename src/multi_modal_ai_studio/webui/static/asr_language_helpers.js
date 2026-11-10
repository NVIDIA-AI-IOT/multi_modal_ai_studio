// SPDX-License-Identifier: Apache-2.0
(function (root) {
    'use strict';
    function normalize(value) {
        return String(value || '').trim().replace(/_/g, '-').toLowerCase();
    }
    function apiBase(config) {
        if ((config.backend || config.scheme) !== 'openai-realtime') {
            return String(config.api_base || config.openai_url || '').replace(/\/$/, '');
        }
        try {
            const url = new URL(config.realtime_url || '');
            if (!['ws:', 'wss:', 'http:', 'https:'].includes(url.protocol)) return '';
            url.protocol = url.protocol === 'wss:' || url.protocol === 'https:' ? 'https:' : 'http:';
            url.pathname = url.pathname.replace(/\/realtime\/?$/, '').replace(/\/$/, '');
            url.search = ''; url.hash = '';
            return url.toString().replace(/\/$/, '');
        } catch (_) { return ''; }
    }
    function identity(config) {
        return JSON.stringify([config.backend || config.scheme, apiBase(config), config.model || '',
            config.realtime_api_style || '', config.api_key || '']);
    }
    function languageInfo(config, reported) {
        const languages = Array.isArray(reported) ? Array.from(new Set(reported.filter(code =>
            typeof code === 'string' && /^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$/i.test(code.trim())
        ).map(code => code.trim().replace(/_/g, '-')))) : [];
        if (languages.length) return {languages, source:'provider'};
        // Explicit profile for the public 11B preset/alias, not all S2S servers.
        const model = String(config.model || '').toLowerCase();
        if ((config.backend || config.scheme) === 'openai-realtime'
                && config.realtime_api_style === 'nemo-voicechat'
                && ['nemotron-voicechat', 'nvidia/nvidia-nemotronlabs-voicechat-11b', 'nvidia-nemotronlabs-voicechat-11b'].includes(model)) {
            return {languages:['en'], source:'model_profile'};
        }
        return {languages:[], source:'unknown'};
    }
    function choices(languages, configured) {
        const current = normalize(configured);
        let selected = languages.find(code => normalize(code) === current);
        if (!selected) selected = languages.find(code => !normalize(code).includes('-') && current.startsWith(normalize(code)+'-'));
        return {supported:!!selected, options:languages.map(code => ({code,
            value:code === selected ? configured : code, selected:code === selected}))};
    }
    const api = {apiBase, identity, languageInfo, choices};
    root.MMASAsrLanguages = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
