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

    return {
        matchesRivaDiscovery,
    };
}));
