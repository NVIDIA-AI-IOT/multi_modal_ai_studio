// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(function (root, factory) {
    const helpers = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = helpers;
    }
    root.MMASConfigHelpers = helpers;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    /**
     * Return true only while a Riva discovery response still belongs to the
     * active config. Initial rendering can start discovery with built-in Riva
     * defaults before an asynchronous server preset switches to OpenAI REST.
     */
    function matchesRivaDiscovery(config, requestedServer, requestedLanguage) {
        if (!config || !requestedServer) return false;
        if (config.backend && config.backend !== 'riva') return false;
        if (config.scheme && config.scheme !== 'riva') return false;
        if (config.backend !== 'riva' && config.scheme !== 'riva') return false;

        const currentServer = config.riva_server || config.server || 'localhost:50051';
        if (currentServer !== requestedServer) return false;

        if (requestedLanguage != null) {
            const currentLanguage = config.language || 'en-US';
            if (currentLanguage !== requestedLanguage) return false;
        }
        return true;
    }

    function normalizeLanguageTag(value) {
        const parts = String(value || '').trim().replace(/_/g, '-').split('-').filter(Boolean);
        if (!parts.length) return '';
        return [parts[0].toLowerCase()].concat(parts.slice(1).map(function (part) {
            return (part.length === 2 || part.length === 3) ? part.toUpperCase() : part;
        })).join('-');
    }

    /**
     * Build the configuration selected by "Reset default". A server preset
     * supplied with --preset is part of that default and must remain layered
     * over the built-in fallback values.
     */
    function buildResetConfig(defaults, serverPreset) {
        function clone(value) {
            if (Array.isArray(value)) return value.map(clone);
            if (value && typeof value === 'object') {
                const result = {};
                Object.keys(value).forEach(function (key) {
                    result[key] = clone(value[key]);
                });
                return result;
            }
            return value;
        }

        function merge(target, source) {
            if (!source || typeof source !== 'object') return target;
            Object.keys(source).forEach(function (key) {
                const value = source[key];
                if (value && typeof value === 'object' && !Array.isArray(value)) {
                    const existing = target[key];
                    target[key] = merge(
                        existing && typeof existing === 'object' && !Array.isArray(existing)
                            ? existing
                            : {},
                        value
                    );
                } else {
                    target[key] = clone(value);
                }
            });
            return target;
        }

        return merge(clone(defaults || {}), serverPreset || {});
    }

    /**
     * Saved session schemas use `scheme`, while the editable UI historically
     * used `backend`. Backfill either alias without mutating recorded data so
     * read-only session review selects the backend that actually ran.
     */
    function normalizeSpeechBackendConfig(config) {
        const normalized = Object.assign({}, config || {});
        const backend = normalized.backend || normalized.scheme || 'riva';
        normalized.backend = backend;
        if (!normalized.scheme) normalized.scheme = backend;
        return normalized;
    }

    /**
     * Describe which endpointing controls are meaningful for an ASR backend.
     * REST transcription needs MMAS's local energy endpointing, while a
     * Realtime provider owns server_vad and may allow it to be disabled.
     */
    function getAsrVadControlProfile(config) {
        const normalized = normalizeSpeechBackendConfig(config);
        if (normalized.backend === 'openai-rest') {
            return {
                backend: 'openai-rest',
                title: 'Local endpointing (MMAS Energy VAD)',
                description: 'MMAS buffers microphone audio, detects the utterance boundary locally, then sends one WAV file to /v1/audio/transcriptions.',
                enabled: true,
                canDisable: false,
                showStopThreshold: true,
            };
        }
        if (normalized.backend === 'openai-realtime') {
            return {
                backend: 'openai-realtime',
                title: 'Server VAD (Realtime API)',
                description: 'These values are sent in the Realtime session turn_detection configuration. Provider support and accepted ranges may vary.',
                enabled: normalized.enable_vad !== false,
                canDisable: true,
                showStopThreshold: false,
            };
        }
        return null;
    }

    /** Merge UI defaults only after the recorded speech backend is normalized. */
    function mergeRecordedSpeechConfig(defaults, recorded) {
        return Object.assign(
            {},
            defaults || {},
            normalizeSpeechBackendConfig(recorded)
        );
    }

    /** Reject an asynchronous REST discovery response after endpoint/model changes. */
    function matchesOpenAiTtsDiscovery(config, requestedApiBase, requestedModel) {
        if (!config || !requestedApiBase) return false;
        const isRest = config.backend === 'openai-rest' || config.scheme === 'openai-rest';
        if (!isRest) return false;
        const currentApiBase = String(config.api_base || config.openai_url || '').replace(/\/$/, '');
        const currentModel = String(config.model || '');
        return currentApiBase === String(requestedApiBase).replace(/\/$/, '')
            && currentModel === String(requestedModel || '');
    }

    /** Reject an asynchronous REST ASR model response after endpoint/backend changes. */
    function matchesOpenAiAsrDiscovery(config, requestedApiBase) {
        if (!config || !requestedApiBase) return false;
        const isRest = config.backend === 'openai-rest' || config.scheme === 'openai-rest';
        if (!isRest) return false;
        const currentApiBase = String(config.api_base || config.openai_url || '').replace(/\/$/, '');
        return currentApiBase === String(requestedApiBase).replace(/\/$/, '');
    }

    /** Filter provider voice records by a selected language without dropping untagged voices. */
    function ttsVoicesForLanguage(voices, selectedLanguage) {
        const selected = normalizeLanguageTag(selectedLanguage);
        if (!Array.isArray(voices) || !selected) return Array.isArray(voices) ? voices.slice() : [];
        const selectedBase = selected.split('-', 1)[0];
        return voices.filter(function (voice) {
            const language = normalizeLanguageTag(voice && voice.language);
            if (!language) return true;
            if (selected.includes('-')) return language === selected;
            return language.split('-', 1)[0] === selectedBase;
        });
    }

    /**
     * Resolve the REST TTS voice list and selected value for a language.
     * A provider-qualified list is authoritative: a configured voice from a
     * different language must not leak into the new language's options.
     */
    function resolveTtsVoiceSelection(voices, selectedLanguage, configuredVoice, providerQualified) {
        const configured = String(configuredVoice || '');
        const options = ttsVoicesForLanguage(voices, selectedLanguage);
        const hasConfigured = options.some(function (voice) {
            return String((voice && (voice.id || voice.name)) || voice) === configured;
        });

        if (!providerQualified && configured && !hasConfigured) {
            options.unshift({id: configured, name: configured, configuredOnly: true});
        }
        if (!providerQualified && !options.length) {
            options.push({id: configured, name: configured || 'Default', configuredOnly: true});
        }

        const selectedVoice = hasConfigured
            ? configured
            : String((options[0] && (options[0].id || options[0].name)) || options[0] || '');
        return {voices: options, selectedVoice: selectedVoice};
    }

    /**
     * Return the model identifier used for pipeline/session metadata.
     *
     * Voice and model are separate concepts. Older clients selected
     * riva_model_name -> voice -> model for every backend, which caused a REST
     * Magpie session to be recorded as model "Sofia". Prefer the REST model,
     * while retaining Riva's discovered model-name behavior.
     */
    function getTtsModelName(config) {
        if (!config) return null;
        const tts = config.tts || config;
        const topLevelName = config.tts ? config.tts_model_name : null;
        const isRiva = tts.backend === 'riva' || tts.scheme === 'riva';
        if (isRiva) {
            return topLevelName
                || tts.riva_model_name
                || tts.model
                || tts.voice
                || null;
        }
        return tts.model
            || topLevelName
            || tts.voice
            || null;
    }

    return {
        buildResetConfig,
        getTtsModelName,
        getAsrVadControlProfile,
        matchesRivaDiscovery,
        matchesOpenAiAsrDiscovery,
        matchesOpenAiTtsDiscovery,
        mergeRecordedSpeechConfig,
        normalizeLanguageTag,
        normalizeSpeechBackendConfig,
        resolveTtsVoiceSelection,
        ttsVoicesForLanguage,
    };
}));
