// Multi-modal AI Studio - WebUI App
// Handles session loading, display, and timeline visualization

// ===== State Management =====
const state = {
    sessions: [],
    selectedSession: null,
    activeConfigTab: 'asr',
    timelineZoom: 1.0,
    timelineOffset: 0,
    timelineDuration: 0, // Total timeline duration in seconds
    isLiveSession: false,
    sessionState: 'setup', // 'setup', 'live', 'stopped'
    /** MediaStream from getUserMedia for camera/mic preview (setup mode); stop on STOP or config change */
    previewStream: null,
    /** Mic waveform: last 2000ms amplitude ring buffer (one value per ~16ms at 60fps) */
    micAmplitudeBuffer: [],
    micWaveformAnimId: null,
    micAudioContext: null,
    micAnalyser: null,

    /** Voice pipeline WebSocket (live session) */
    voiceWs: null,
    /** Live session: timeline events streamed from backend */
    liveTimelineEvents: [],
    /** Live session: chat turns [{ user, assistant }] for display */
    liveChatTurns: [],
    /** Last N WebSocket messages (JSON strings) for voice debug panel */
    voiceMessageLog: [],
    /** AudioContext for TTS playback (created when needed) */
    ttsAudioContext: null,
    /** Next start time for TTS chunk scheduling (context time) */
    ttsNextStartTime: 0,
    /** MediaStream used for voice pipeline mic (may be same as previewStream) */
    voiceMicStream: null,
    /** Selected browser microphone deviceId (from enumerateDevices); null = use browser default */
    selectedBrowserMicId: null,
    /** Selected browser camera deviceId; null = use browser default */
    selectedBrowserCameraId: null,
    /** Selected browser speaker/audiooutput deviceId; null = use browser default */
    selectedBrowserSpeakerId: null,
    /** ScriptProcessor or worklet node for PCM capture; disconnect on stop */
    voicePcmNode: null,
    /** Live session: wall-clock time when session_start was received (for amplitude timestamps) */
    liveSessionStartTime: 0,
    /** requestAnimationFrame id for live timeline scroll ticker; cancel when leaving live */
    liveTimelineRafId: null,
    /** Live session: (timestamp_sec, amplitude 0–100) for AUDIO lane waveform; max ~15 sec at ~20 Hz */
    liveAudioAmplitudeHistory: [],
    /** Live session: TTS (AI) segments for purple waveform on AUDIO lane: { startTime, endTime, amplitude } */
    liveTtsAmplitudeHistory: [],
    /** Live session: CPU/GPU samples for bottom timeline lane: { t, cpu, gpu } (t = session-relative sec) */
    liveSystemStats: [],
    /** Live session: interval id for system stats polling; cleared on disconnect */
    liveSystemStatsPollIntervalId: null,
    /** Live session: have we set initial 15s zoom once */
    liveTimelineInitialZoomSet: false,

    // UI state (for future persistence)
    ui: {
        configPanelCollapsed: false,
        timelinePanelCollapsed: false,
        sessionListVisible: true, // For mobile
    }
};

// ===== UI Settings (Global, persisted in localStorage) =====
const uiSettings = {
    combineSpeechLanes: false,
    showSessionThumbnails: true,
    autoScrollChat: true,
    showTimestamps: false,
    showDebugInfo: false,
    /** UI-only gain for user (mic) waveform on timeline: 1, 2, or 4 */
    userAudioGain: 2,
    /** UI-only gain for AI (TTS) waveform on timeline: 1, 2, or 4 */
    aiAudioGain: 2
};

// Load UI settings from localStorage
function loadUISettings() {
    const saved = localStorage.getItem('uiSettings');
    if (saved) {
        try {
            Object.assign(uiSettings, JSON.parse(saved));
        } catch (e) {
            console.error('Failed to load UI settings:', e);
        }
    }
}

// Save UI settings to localStorage
function saveUISettings() {
    localStorage.setItem('uiSettings', JSON.stringify(uiSettings));
    console.log('UI settings saved:', uiSettings);
}

// ===== Session Loading =====
async function loadSessions() {
    console.log('loadSessions() called');
    try {
        // In development, load from mock_sessions
        console.log('Fetching from /api/sessions...');
        const response = await fetch('/api/sessions');
        console.log('Response received:', response.status);
        const sessions = await response.json();
        console.log('Sessions loaded:', sessions.length);
        state.sessions = sessions;
        renderSessionList();
    } catch (error) {
        console.error('Failed to load sessions:', error);
        // For now, show error in UI
        document.getElementById('session-items').innerHTML = `
            <div style="padding: 1rem; color: var(--text-secondary); text-align: center;">
                <p>Failed to load sessions</p>
                <p style="font-size: 0.85rem; margin-top: 0.5rem;">${error.message}</p>
            </div>
        `;
    }
}

// ===== UI Rendering =====
function renderSessionList() {
    const container = document.getElementById('session-items');
    const countEl = document.querySelector('.session-count');

    countEl.textContent = `${state.sessions.length} sessions`;

    if (state.sessions.length === 0) {
        container.innerHTML = `
            <div style="padding: 1rem; color: var(--text-secondary); text-align: center;">
                No sessions yet
            </div>
        `;
        return;
    }

    container.innerHTML = state.sessions.map((session, index) => {
        const metrics = session.metrics || {};
        const isActive = state.selectedSession?.session_id === session.session_id;

        return `
            <div class="session-item ${isActive ? 'active' : ''}"
                 onclick="selectSession(${index})">
                <div class="session-item-name">${escapeHtml(session.name)}</div>
                <div class="session-item-meta">
                    <span>${formatDate(session.created_at)}</span>
                    <span>${metrics.total_turns || 0} turns</span>
                </div>
                <div class="session-item-metrics">
                    <span class="metric-badge">TTL: ${formatLatency(metrics.avg_ttl)}</span>
                    <span class="metric-badge">${session.timeline?.length || 0} events</span>
                </div>
            </div>
        `;
    }).join('');
}

function selectSession(index) {
    console.log('selectSession called:', index);
    state.selectedSession = state.sessions[index];
    state.isLiveSession = false;
    state.sessionState = 'stopped';

    renderSessionList();
    renderSessionDetail();
    renderTimeline();
    updateLiveSessionUI();
    updateHistoricalSessionPreview();
}

function renderSessionDetail() {
    if (!state.selectedSession) {
        document.getElementById('chat-history').innerHTML = `
            <div class="empty-state">
                <p>👈 Select a session from the sidebar</p>
            </div>
        `;
        return;
    }

    // Render config
    renderConfig();

    // Render chat history
    renderChatHistory();

    // Update timeline metrics
    renderTimelineMetrics();
}

function updateConfigPanelState() {
    const panel = document.getElementById('config-panel');
    if (!panel) return;
    if (state.isLiveSession) {
        panel.classList.add('config-panel--editable');
    } else {
        panel.classList.remove('config-panel--editable');
    }
}

function renderConfig() {
    updateConfigPanelState();
    const contentEl = document.getElementById('config-tab-content');
    const tab = state.activeConfigTab;
    // Device tab uses 'devices' key in config
    const configKey = tab === 'device' ? 'devices' : tab;

    // If in live session mode, show editable forms
    if (state.isLiveSession) {
        contentEl.innerHTML = renderEditableConfigForm(tab, currentConfig[configKey], false);
        if (tab === 'llm') setTimeout(() => fetchLLMModels(currentConfig.llm.api_base || (currentConfig.llm.ollama_url && currentConfig.llm.ollama_url.replace(/\/v1$/, '') + '/v1')), 0);
        if (tab === 'asr' && (currentConfig.asr.backend === 'riva' || currentConfig.asr.scheme === 'riva')) setTimeout(() => fetchASRModels(currentConfig.asr.server || currentConfig.asr.riva_server || 'localhost:50051'), 0);
        if (tab === 'device') setTimeout(populateAllDeviceDropdowns, 0);
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        return;
    }

    // If historical session, show read-only forms with session data
    const session = state.selectedSession;
    if (!session) {
        contentEl.innerHTML = '<p class="empty-state">Select a session or start a new one</p>';
        return;
    }

    const config = session.config || {};
    let tabConfig = { ...defaultConfig[configKey], ...(config[configKey] || {}) };

    contentEl.innerHTML = renderEditableConfigForm(tab, tabConfig, true);
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

// Default configuration state for new sessions
const defaultConfig = {
    asr: {
        backend: 'riva',
        server: 'localhost:50051',
        riva_server: 'localhost:50051', // legacy / display compat
        model: '',
        language: 'en-US',
        enable_vad: true,
        vad_start_threshold: 0.5,
        vad_stop_threshold: 0.3,
        speech_pad_ms: 500,
        speech_timeout_ms: 700,
        interim_results: true
    },
    llm: {
        backend: 'openai',
        api_base: 'http://localhost:11434/v1',
        ollama_url: 'http://localhost:11434',
        api_key: '',
        model: 'llama3.2:3b',
        temperature: 0.7,
        max_tokens: 2048,
        minimal_output: false,
        stream: true,
        system_prompt: 'You are a helpful AI assistant.'
    },
    tts: {
        backend: 'riva',
        riva_server: 'localhost:50051',
        voice: '',
        language: 'en-US',
        sample_rate: 22050,
        quality: 'high'
    },
    devices: {
        camera: 'browser',
        microphone: 'browser',
        speaker: 'browser'
    },
    app: {
        auto_start_recording: false,
        show_interim_asr: true,
        enable_timeline: true,
        log_level: 'info'
    }
};

// Current editable configuration (for new session)
let currentConfig = JSON.parse(JSON.stringify(defaultConfig));

function renderConfigSection(title, data) {
    const rows = Object.entries(data).map(([key, value]) => {
        // Format key nicely
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

        // Format value
        let formattedValue = value;
        if (typeof value === 'boolean') {
            formattedValue = value ? '✓ Enabled' : '✗ Disabled';
        } else if (typeof value === 'object') {
            formattedValue = JSON.stringify(value, null, 2);
        } else if (value === null || value === undefined || value === '') {
            formattedValue = '(default)';
        }

        return `
            <div class="config-row">
                <span class="config-label">${label}</span>
                <span class="config-value">${escapeHtml(String(formattedValue))}</span>
            </div>
        `;
    }).join('');

    return `
        <div class="config-group">
            ${rows}
        </div>
    `;
}

// New: Render configuration forms (editable or read-only)
function renderEditableConfigForm(tab, config, readonly = false) {
    switch (tab) {
        case 'asr':
            return renderASRConfig(config, readonly);
        case 'llm':
            return renderLLMConfig(config, readonly);
        case 'tts':
            return renderTTSConfig(config, readonly);
        case 'device':
            return renderDeviceConfig(config, readonly);
        case 'app':
            return renderAppConfig(config, readonly);
        default:
            return '<p>Unknown configuration tab</p>';
    }
}

function renderASRConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}

            <!-- Backend Tabs -->
            <div class="backend-tabs ${readonly ? 'disabled' : ''}">
                <button class="backend-tab ${config.backend === 'riva' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('asr', 'backend', 'riva')">NVIDIA Riva</button>
                <button class="backend-tab ${config.backend === 'openai' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('asr', 'backend', 'openai')">OpenAI REST API</button>
                <button class="backend-tab ${config.backend === 'openai-realtime' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('asr', 'backend', 'openai-realtime')">OpenAI Realtime API</button>
            </div>

            <!-- Riva Settings (Live RIVA WebUI format) -->
            <div class="backend-content" style="display: ${config.backend === 'riva' ? 'block' : 'none'}">
                <div class="form-group">
                    <label for="asr-riva-server"><i data-lucide="server" class="lucide-inline"></i> Server Address</label>
                    <input type="text" id="asr-riva-server" ${disabled} value="${config.server || config.riva_server || 'localhost:50051'}"
                           onchange="updateConfig('asr', 'server', this.value); updateConfig('asr', 'riva_server', this.value); if(!${readonly}) fetchASRModels(this.value);">
                    <div class="input-hint">gRPC endpoint for RIVA services</div>
                </div>

                <div class="form-group">
                    <label for="asrLanguage"><i data-lucide="globe" class="lucide-inline"></i> ASR Language</label>
                    <input type="text" id="asrLanguage" value="${config.language || 'en-US'}" readonly
                           class="readonly-config-input" title="ASR language is set during Riva deployment (config.sh)">
                    <small class="config-deployment-hint"><i data-lucide="info" class="lucide-inline"></i> Set during Riva deployment</small>
                </div>

                <div class="form-group">
                    <label for="asr-model-select"><i data-lucide="cpu" class="lucide-inline"></i> ASR Model</label>
                    ${readonly
                        ? `<input type="text" id="asrModel" value="${config.model || 'Server default'}" readonly class="readonly-config-input">`
                        : `<select id="asr-model-select" class="config-select" onchange="updateConfig('asr', 'model', this.value)">
                               <option value="">Server default</option>
                           </select>`
                    }
                    <small class="config-deployment-hint"><i data-lucide="info" class="lucide-inline"></i> ${readonly ? 'Set during session' : 'Queried from Riva server'}</small>
                </div>

                <!-- Advanced ASR Settings (Expandable) -->
                <div class="nested-panel" style="margin-top: 16px; margin-bottom: 16px;">
                    <div class="nested-panel-header" onclick="toggleNestedPanel('asrAdvanced')">
                        <div class="nested-panel-title"><i data-lucide="sliders" class="lucide-inline"></i> Advanced ASR Settings</div>
                        <span class="nested-panel-toggle" id="asrAdvancedToggle"><i data-lucide="chevron-right" class="lucide-inline"></i></span>
                    </div>
                    <div class="nested-panel-content" id="asrAdvanced" style="display: none;">
                        <div class="config-section">
                            <div class="config-section-label">Model Configuration</div>
                            <div class="config-grid">
                                <div class="config-item">
                                    <span class="config-label">Sample Rate:</span>
                                    <span class="config-value">16000 Hz</span>
                                </div>
                                <div class="config-item">
                                    <span class="config-label">Streaming:</span>
                                    <span class="config-value">Yes</span>
                                </div>
                            </div>
                        </div>

                        <div class="config-section">
                            <div class="config-section-label">Voice Activity Detection (VAD)</div>
                            <div class="config-grid">
                                <div class="config-item">
                                    <span class="config-label">VAD Enabled:</span>
                                    <span class="config-value">Yes</span>
                                </div>
                                <div class="config-item">
                                    <span class="config-label">VAD Type:</span>
                                    <span class="config-value">silero</span>
                                </div>
                            </div>
                        </div>

                        <!-- VAD Tuning (runtime adjustments) -->
                        <div class="config-section">
                            <div class="config-section-label" style="display: flex; justify-content: space-between; align-items: center;">
                                <span>VAD Tuning</span>
                                <span style="font-size: 10px; font-weight: normal; color: var(--text-muted); text-transform: none;"><i data-lucide="zap" class="lucide-inline"></i> Runtime adjustments (takes effect on next session - no restart needed!)</span>
                            </div>
                            <div class="vad-tuning-container">
                                <div class="vad-slider-group">
                                    <div class="vad-slider-label">
                                        <span>Speech Pad</span>
                                        <span class="vad-slider-value" id="vadSpeechPadValue">${config.speech_pad_ms ?? 500}</span>
                                        <span title="Higher values help capture the start of speech (e.g. if beginnings like 'Tell me' get cut off)">ⓘ</span>
                                    </div>
                                    <input type="range" ${disabled} id="vadSpeechPadSlider" class="vad-slider" min="0" max="1000" step="50" value="${config.speech_pad_ms ?? 500}"
                                           oninput="currentConfig.asr.speech_pad_ms = parseInt(this.value); document.getElementById('vadSpeechPadValue').textContent = this.value">
                                </div>
                                <div class="vad-slider-group">
                                    <div class="vad-slider-label">
                                        <span>Silence Duration</span>
                                        <span class="vad-slider-value" id="vadSilenceDurationValue">${config.speech_timeout_ms ?? 700}</span>
                                    </div>
                                    <input type="range" ${disabled} id="vadSilenceDurationSlider" class="vad-slider" min="100" max="2000" step="100" value="${config.speech_timeout_ms ?? 700}"
                                           oninput="currentConfig.asr.speech_timeout_ms = parseInt(this.value); document.getElementById('vadSilenceDurationValue').textContent = this.value">
                                </div>
                                <div class="vad-slider-group">
                                    <div class="vad-slider-label">
                                        <span>Threshold</span>
                                        <span class="vad-slider-value" id="vadThresholdValue">${(config.vad_start_threshold ?? config.vad_threshold ?? 0.5)}</span>
                                    </div>
                                    <input type="range" ${disabled} id="vadThresholdSlider" class="vad-slider" min="0.1" max="0.9" step="0.05" value="${config.vad_start_threshold ?? config.vad_threshold ?? 0.5}"
                                           oninput="currentConfig.asr.vad_start_threshold = parseFloat(this.value); document.getElementById('vadThresholdValue').textContent = this.value">
                                </div>
                                <div class="vad-presets">
                                    <button type="button" class="vad-preset-btn" ${disabled} onclick="applyVADPreset('aggressive')"><i data-lucide="zap" class="lucide-inline"></i> Aggressive</button>
                                    <button type="button" class="vad-preset-btn" ${disabled} onclick="applyVADPreset('balanced')"><i data-lucide="scale" class="lucide-inline"></i> Balanced</button>
                                    <button type="button" class="vad-preset-btn" ${disabled} onclick="applyVADPreset('conservative')"><i data-lucide="shield" class="lucide-inline"></i> Conservative</button>
                                </div>
                            </div>
                        </div>

                        <div class="config-section">
                            <div class="config-section-label">Performance Settings</div>
                            <div style="margin-top: 8px; padding: 8px; background: var(--bg-secondary); border-radius: 4px; font-size: 11px; color: var(--text-muted);">
                                These settings are configured during RIVA deployment (config.sh) and cannot be changed at runtime.
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- OpenAI REST Settings -->
            <div class="backend-content" style="display: ${config.backend === 'openai' ? 'block' : 'none'}">
                <div class="form-group">
                    <label>API Endpoint</label>
                    <input type="text" ${disabled} value="${config.openai_url || 'https://api.openai.com/v1'}"
                           onchange="updateConfig('asr', 'openai_url', this.value)">
                </div>

                <div class="form-group">
                    <label>Model</label>
                    <input type="text" ${disabled} value="${config.model || 'whisper-1'}"
                           onchange="updateConfig('asr', 'model', this.value)">
                    ${!readonly ? '<span class="input-hint">e.g., whisper-1</span>' : ''}
                </div>
            </div>

            <!-- OpenAI Realtime Settings -->
            <div class="backend-content" style="display: ${config.backend === 'openai-realtime' ? 'block' : 'none'}">
                <div class="form-group">
                    <label>WebSocket URL</label>
                    <input type="text" ${disabled} value="${config.realtime_url || 'wss://api.openai.com/v1/realtime'}"
                           onchange="updateConfig('asr', 'realtime_url', this.value)">
                </div>

                <div class="form-group">
                    <label>Model</label>
                    <input type="text" ${disabled} value="${config.model || 'gpt-4o-realtime-preview'}"
                           onchange="updateConfig('asr', 'model', this.value)">
                    ${!readonly ? '<span class="input-hint">e.g., gpt-4o-realtime-preview</span>' : ''}
                </div>
            </div>

            <!-- Common Settings (for non-Riva backends) -->
            <div class="form-group" style="display: ${config.backend === 'riva' ? 'none' : 'block'}">
                <label>Language</label>
                <select ${disabled} value="${config.language}" onchange="updateConfig('asr', 'language', this.value)">
                    <option value="en-US" ${config.language === 'en-US' ? 'selected' : ''}>English (US)</option>
                    <option value="en-GB" ${config.language === 'en-GB' ? 'selected' : ''}>English (UK)</option>
                    <option value="es-ES" ${config.language === 'es-ES' ? 'selected' : ''}>Spanish</option>
                    <option value="fr-FR" ${config.language === 'fr-FR' ? 'selected' : ''}>French</option>
                    <option value="de-DE" ${config.language === 'de-DE' ? 'selected' : ''}>German</option>
                    <option value="ja-JP" ${config.language === 'ja-JP' ? 'selected' : ''}>Japanese</option>
                    <option value="zh-CN" ${config.language === 'zh-CN' ? 'selected' : ''}>Chinese (Simplified)</option>
                </select>
            </div>

            <div class="form-group" style="display: ${config.backend === 'riva' ? 'none' : 'block'}">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} ${config.enable_vad ? 'checked' : ''}
                           onchange="updateConfig('asr', 'enable_vad', this.checked)">
                    Enable Voice Activity Detection (VAD)
                </label>
            </div>

            <div class="form-group" style="display: ${(config.backend === 'riva' || !config.enable_vad) ? 'none' : 'block'}">
                <label>VAD Threshold</label>
                <input type="range" ${disabled} min="0" max="1" step="0.05" value="${config.vad_start_threshold ?? config.vad_threshold ?? 0.5}"
                       oninput="updateConfig('asr', 'vad_start_threshold', parseFloat(this.value)); const el = document.getElementById('vad-threshold-value'); if(el) el.textContent = this.value;">
                <span id="vad-threshold-value" class="range-value">${config.vad_start_threshold ?? config.vad_threshold ?? 0.5}</span>
            </div>

            <div class="form-group" style="display: ${config.backend === 'riva' ? 'none' : 'block'}">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} ${config.interim_results ? 'checked' : ''}
                           onchange="updateConfig('asr', 'interim_results', this.checked)">
                    Show Interim Results
                </label>
            </div>
        </div>
    `;
}

function renderLLMConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';
    const apiBase = config.api_base || (config.ollama_url ? config.ollama_url.replace(/\/v1$/, '') + '/v1' : 'http://localhost:11434/v1');
    const showApiKey = !readonly && (apiBase.includes('openai.com') || apiBase.includes('nvidia.com'));

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}
            <div class="form-group">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <label style="margin: 0;">API Base URL</label>
                    ${!readonly ? `<div style="position: relative;">
                        <button type="button" class="icon-btn" onclick="togglePresetsMenu(event)" title="Select preset"><i data-lucide="list" class="lucide-inline"></i></button>
                        <div class="api-presets-menu" id="presetsMenu" style="display: none;">
                            <div class="api-preset-item" onclick="selectLLMPreset('http://localhost:11434/v1', 'Ollama')"><strong>Ollama</strong><span>http://localhost:11434/v1</span></div>
                            <div class="api-preset-item" onclick="selectLLMPreset('http://localhost:8000/v1', 'vLLM')"><strong>vLLM</strong><span>http://localhost:8000/v1</span></div>
                            <div class="api-preset-item" onclick="selectLLMPreset('http://localhost:8003/v1', 'vLLM (8003)')"><strong>vLLM (8003)</strong><span>http://localhost:8003/v1</span></div>
                            <div class="api-preset-item" onclick="selectLLMPreset('http://localhost:30000/v1', 'SGLang')"><strong>SGLang</strong><span>http://localhost:30000/v1</span></div>
                            <div style="border-top: 1px solid var(--border-color);"></div>
                            <div class="api-preset-item" onclick="selectLLMPreset('https://api.openai.com/v1', 'OpenAI')"><strong>OpenAI</strong><span>https://api.openai.com/v1</span></div>
                            <div class="api-preset-item" onclick="selectLLMPreset('https://integrate.api.nvidia.com/v1', 'NVIDIA')"><strong>NVIDIA API Catalog</strong><span>https://integrate.api.nvidia.com/v1</span></div>
                        </div>
                    </div>` : ''}
                </div>
                <input type="text" ${disabled} id="llm-api-base" value="${apiBase}" placeholder="http://localhost:11434/v1"
                       onchange="updateConfig('llm', 'api_base', this.value); if(!${readonly}) fetchLLMModels(this.value);">
                <div class="input-hint">OpenAI-compatible API endpoint (Ollama, vLLM, SGLang, OpenAI, etc.)</div>
            </div>

            <div class="form-group" id="llm-api-key-group" style="display: ${showApiKey ? 'block' : 'none'}">
                <label>API Key</label>
                <input type="password" ${disabled} id="llm-api-key" value="${config.api_key || ''}" placeholder="Optional for local; required for OpenAI/NVIDIA"
                       onchange="updateConfig('llm', 'api_key', this.value)">
            </div>

            <div class="form-group">
                <label>Model</label>
                <select ${disabled} id="llm-model-select" onchange="updateConfig('llm', 'model', this.value)">
                    <option value="${escapeHtml(config.model)}">${readonly ? escapeHtml(config.model) : 'Loading...'}</option>
                </select>
                ${!readonly ? '<button type="button" class="btn-secondary" style="margin-top: 6px;" onclick="refreshLLMModels()">Refresh models</button>' : ''}
                <div class="input-hint">Fetched from API Base URL; use Refresh if you added a model</div>
            </div>

            <div class="form-group">
                <label>Temperature</label>
                <input type="range" ${disabled} id="llm-temperature" min="0" max="2" step="0.1" value="${config.temperature}"
                       oninput="updateConfig('llm', 'temperature', parseFloat(this.value)); document.getElementById('temp-value').textContent = this.value;">
                <span id="temp-value" class="range-value">${config.temperature}</span>
                ${!readonly ? '<span class="input-hint">Lower = more focused, Higher = more creative</span>' : ''}
            </div>

            <div class="form-group">
                <label>Max Tokens</label>
                <input type="number" ${disabled} id="llm-max-tokens" min="1" max="8192" value="${config.max_tokens}"
                       onchange="updateConfig('llm', 'max_tokens', parseInt(this.value))">
                ${!readonly ? '<span class="input-hint">Use 1–4 for minimal output (e.g. single number, Nemotron without reasoning)</span>' : ''}
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} id="llm-minimal-output" ${config.minimal_output ? 'checked' : ''}
                           onchange="updateConfig('llm', 'minimal_output', this.checked)">
                    Minimal output (no reasoning)
                </label>
                ${!readonly ? '<span class="input-hint">For Nemotron etc.: answer with only a number or few tokens; system prompt + max_tokens are adjusted</span>' : ''}
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} id="llm-stream" ${config.stream ? 'checked' : ''}
                           onchange="updateConfig('llm', 'stream', this.checked)">
                    Enable Streaming Responses
                </label>
            </div>

            <div class="form-group">
                <label>System Prompt</label>
                <textarea id="llm-system-prompt" ${disabled} rows="3"
                          onchange="updateConfig('llm', 'system_prompt', this.value)">${config.system_prompt}</textarea>
            </div>
        </div>
    `;
}

function renderTTSConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}

            <!-- Backend Tabs -->
            <div class="backend-tabs ${readonly ? 'disabled' : ''}">
                <button class="backend-tab ${config.backend === 'riva' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('tts', 'backend', 'riva')">NVIDIA Riva</button>
                <button class="backend-tab ${config.backend === 'openai' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('tts', 'backend', 'openai')">OpenAI TTS API</button>
            </div>

            <!-- Riva Settings -->
            <div class="backend-content" style="display: ${config.backend === 'riva' ? 'block' : 'none'}">
                <div class="form-group">
                    <label>Riva Server</label>
                    <input type="text" ${disabled} value="${config.riva_server || 'localhost:50051'}"
                           onchange="updateConfig('tts', 'riva_server', this.value)">
                </div>

                <div class="form-group">
                    <label>Voice</label>
                    <input type="text" ${disabled} value="${config.voice}"
                           onchange="updateConfig('tts', 'voice', this.value)">
                    ${!readonly ? '<span class="input-hint">Leave empty for default voice</span>' : ''}
                </div>
            </div>

            <!-- OpenAI TTS Settings -->
            <div class="backend-content" style="display: ${config.backend === 'openai' ? 'block' : 'none'}">
                <div class="form-group">
                    <label>API Endpoint</label>
                    <input type="text" ${disabled} value="${config.openai_url || 'https://api.openai.com/v1'}"
                           onchange="updateConfig('tts', 'openai_url', this.value)">
                </div>

                <div class="form-group">
                    <label>Voice</label>
                    <select ${disabled} value="${config.voice || 'alloy'}" onchange="updateConfig('tts', 'voice', this.value)">
                        <option value="alloy" ${config.voice === 'alloy' ? 'selected' : ''}>Alloy</option>
                        <option value="echo" ${config.voice === 'echo' ? 'selected' : ''}>Echo</option>
                        <option value="fable" ${config.voice === 'fable' ? 'selected' : ''}>Fable</option>
                        <option value="onyx" ${config.voice === 'onyx' ? 'selected' : ''}>Onyx</option>
                        <option value="nova" ${config.voice === 'nova' ? 'selected' : ''}>Nova</option>
                        <option value="shimmer" ${config.voice === 'shimmer' ? 'selected' : ''}>Shimmer</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Model</label>
                    <select ${disabled} value="${config.model || 'tts-1'}" onchange="updateConfig('tts', 'model', this.value)">
                        <option value="tts-1" ${config.model === 'tts-1' ? 'selected' : ''}>TTS-1 (Faster)</option>
                        <option value="tts-1-hd" ${config.model === 'tts-1-hd' ? 'selected' : ''}>TTS-1-HD (Higher Quality)</option>
                    </select>
                </div>
            </div>

            <!-- Common Settings -->
            <div class="form-group">
                <label>Language</label>
                <select ${disabled} value="${config.language}" onchange="updateConfig('tts', 'language', this.value)">
                    <option value="en-US" ${config.language === 'en-US' ? 'selected' : ''}>English (US)</option>
                    <option value="en-GB" ${config.language === 'en-GB' ? 'selected' : ''}>English (UK)</option>
                    <option value="es-ES" ${config.language === 'es-ES' ? 'selected' : ''}>Spanish</option>
                    <option value="fr-FR" ${config.language === 'fr-FR' ? 'selected' : ''}>French</option>
                    <option value="de-DE" ${config.language === 'de-DE' ? 'selected' : ''}>German</option>
                    <option value="ja-JP" ${config.language === 'ja-JP' ? 'selected' : ''}>Japanese</option>
                </select>
            </div>

            <div class="form-group">
                <label>Sample Rate (Hz)</label>
                <select ${disabled} value="${config.sample_rate}" onchange="updateConfig('tts', 'sample_rate', parseInt(this.value))">
                    <option value="16000" ${config.sample_rate === 16000 ? 'selected' : ''}>16000</option>
                    <option value="22050" ${config.sample_rate === 22050 ? 'selected' : ''}>22050</option>
                    <option value="24000" ${config.sample_rate === 24000 ? 'selected' : ''}>24000</option>
                    <option value="44100" ${config.sample_rate === 44100 ? 'selected' : ''}>44100</option>
                    <option value="48000" ${config.sample_rate === 48000 ? 'selected' : ''}>48000</option>
                </select>
            </div>

            <div class="form-group">
                <label>Quality</label>
                <select ${disabled} value="${config.quality}" onchange="updateConfig('tts', 'quality', this.value)">
                    <option value="low" ${config.quality === 'low' ? 'selected' : ''}>Low (Faster)</option>
                    <option value="medium" ${config.quality === 'medium' ? 'selected' : ''}>Medium</option>
                    <option value="high" ${config.quality === 'high' ? 'selected' : ''}>High (Better)</option>
                </select>
            </div>
        </div>
    `;
}

function deviceLabelSuffix(label) {
    if (!label) return '(Browser)';
    var l = label.toLowerCase();
    if (l.indexOf('usb') !== -1 || l.indexOf('jabra') !== -1 || l.indexOf('webcam') !== -1) return '(USB)';
    return '(Browser)';
}

function renderDeviceConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';
    const micValue = config.microphone === 'none' ? 'none' : (state.selectedBrowserMicId || '');
    const camValue = config.camera === 'none' ? 'none' : (config.camera === 'browser' || config.camera === '' ? '' : config.camera);
    const spkValue = config.speaker === 'none' ? 'none' : (state.selectedBrowserSpeakerId || '');

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}
            <div class="form-group">
                <label>Camera devices</label>
                <select id="device-camera-list" ${disabled} data-device-type="camera" onchange="onDeviceListChange('camera', this.value)">
                    <option value="none" ${camValue === 'none' ? 'selected' : ''}>None</option>
                    <option value="" ${camValue === '' || camValue === 'browser' ? 'selected' : ''}>Default (browser)</option>
                </select>
                <div class="input-hint input-hint-camera">Lists cameras on this device (browser) and USB cameras attached to the Jetson (server). Default uses the browser’s default camera (e.g. Dell Integrated webcam).</div>
            </div>
            <div class="form-group">
                <label>Microphone devices</label>
                <select id="device-microphone-list" ${disabled} data-device-type="microphone" onchange="onDeviceListChange('microphone', this.value)">
                    <option value="none" ${micValue === 'none' ? 'selected' : ''}>None (Text Only)</option>
                    <option value="" ${micValue === '' ? 'selected' : ''}>Default (browser choice)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Speaker devices</label>
                <select id="device-speaker-list" ${disabled} data-device-type="speaker" onchange="onDeviceListChange('speaker', this.value)">
                    <option value="none" ${spkValue === 'none' ? 'selected' : ''}>None (Text Only)</option>
                    <option value="" ${spkValue === '' ? 'selected' : ''}>Default (browser choice)</option>
                </select>
            </div>
            ${!readonly ? '<div class="form-group"><button type="button" class="btn-secondary" onclick="requestDevicesAndPopulateAll()">Allow & list devices</button><div class="input-hint">Lists <strong>microphone, speaker, and cameras</strong> on this device (browser). Also loads USB cameras attached to the Jetson (server).</div></div>' : ''}
        </div>
    `;
}

function onDeviceListChange(type, value) {
    if (type === 'camera') {
        if (value === 'none') {
            currentConfig.devices.camera = 'none';
            state.selectedBrowserCameraId = null;
        } else if (value === '') {
            currentConfig.devices.camera = 'browser';
            state.selectedBrowserCameraId = null;
        } else if (value.indexOf('/dev/') === 0) {
            currentConfig.devices.camera = value;
            state.selectedBrowserCameraId = null;
        } else {
            currentConfig.devices.camera = value;
            state.selectedBrowserCameraId = value;
        }
    } else if (type === 'microphone') {
        currentConfig.devices.microphone = value === 'none' ? 'none' : 'browser';
        state.selectedBrowserMicId = value === 'none' || value === '' ? null : value;
    } else if (type === 'speaker') {
        currentConfig.devices.speaker = value === 'none' ? 'none' : 'browser';
        state.selectedBrowserSpeakerId = value === 'none' || value === '' ? null : value;
    }
    updateDeviceIndicators();
    updateChatInputVisibility();
    if (state.isLiveSession && state.sessionState === 'setup') startPreviewStream();
}

function renderAppConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}
            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} id="app-auto-start" ${config.auto_start_recording ? 'checked' : ''}
                           onchange="updateConfig('app', 'auto_start_recording', this.checked)">
                    Auto-start Recording After Device Setup
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} id="app-show-interim" ${config.show_interim_asr ? 'checked' : ''}
                           onchange="updateConfig('app', 'show_interim_asr', this.checked)">
                    Show Interim ASR Results
                </label>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} id="app-enable-timeline" ${config.enable_timeline ? 'checked' : ''}
                           onchange="updateConfig('app', 'enable_timeline', this.checked)">
                    Enable Timeline Visualization
                </label>
            </div>

            <div class="form-group">
                <label>Log Level</label>
                <select id="app-log-level" ${disabled} value="${config.log_level}" onchange="updateConfig('app', 'log_level', this.value)">
                    <option value="debug" ${config.log_level === 'debug' ? 'selected' : ''}>Debug</option>
                    <option value="info" ${config.log_level === 'info' ? 'selected' : ''}>Info</option>
                    <option value="warning" ${config.log_level === 'warning' ? 'selected' : ''}>Warning</option>
                    <option value="error" ${config.log_level === 'error' ? 'selected' : ''}>Error</option>
                </select>
                ${!readonly ? '<span class="input-hint">Timeline and UI display options are in UI Settings (<i data-lucide="settings" class="lucide-inline"></i> in header)</span>' : ''}
            </div>
        </div>
    `;
}

/**
 * Show a short-lived message when microphone permission is denied so the user knows how to fix it.
 */
function showMicrophonePermissionDeniedHint() {
    var hint = document.querySelector('#device-microphone-list')?.closest('.config-form')?.querySelector('.input-hint');
    if (hint) {
        hint.innerHTML = 'Microphone blocked. Click the <strong>lock/info icon</strong> in the address bar → Site settings → set <strong>Microphone</strong> to Allow → reload this page.';
        hint.style.color = 'var(--color-warning, #e67700)';
    }
    var toast = document.getElementById('microphone-permission-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'microphone-permission-toast';
        toast.setAttribute('role', 'alert');
        toast.style.cssText = 'position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); max-width: 90%; padding: 12px 20px; background: var(--color-warning, #e67700); color: #fff; border-radius: 8px; font-size: 14px; z-index: 10000; box-shadow: 0 4px 12px rgba(0,0,0,0.2);';
        document.body.appendChild(toast);
    }
    toast.textContent = 'Microphone denied. Allow mic in the address bar (lock icon → Site settings) then reload.';
    toast.style.display = 'block';
    setTimeout(function () { toast.style.display = 'none'; }, 8000);
}

/**
 * Request camera + microphone (user gesture), then enumerate and fill all device dropdowns.
 */
function requestDevicesAndPopulateAll() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(function (stream) {
            stream.getTracks().forEach(function (t) { t.stop(); });
            populateAllDeviceDropdowns();
        })
        .catch(function (err) {
            console.warn('[Devices] getUserMedia failed:', err);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                showMicrophonePermissionDeniedHint();
            }
        });
}

function populateAllDeviceDropdowns() {
    populateCameraDeviceDropdown();
    populateMicrophoneDeviceDropdown();
    populateSpeakerDeviceDropdown();
}

/** Populate camera dropdown: browser cameras (this device) + Jetson USB cameras (server). Default (browser) uses browser default, e.g. Dell Integrated webcam. */
function populateCameraDeviceDropdown() {
    var select = document.getElementById('device-camera-list');
    if (!select) return;
    var havePermission = navigator.mediaDevices && navigator.mediaDevices.enumerateDevices;
    var browserPromise = havePermission
        ? navigator.mediaDevices.enumerateDevices().then(function (devices) {
            return (devices.filter(function (d) { return d.kind === 'videoinput'; }) || []);
        })
        : Promise.resolve([]);
    var jetsonPromise = fetch(getApiBase() + '/api/devices/cameras')
        .then(function (r) { return r.json(); })
        .then(function (data) { return data.cameras || []; })
        .catch(function (err) { console.warn('[Devices] Fetch Jetson cameras failed:', err); return []; });

    Promise.all([browserPromise, jetsonPromise]).then(function (results) {
        var browserCams = results[0];
        var jetsonCams = results[1];
        select.innerHTML = '';
        select.appendChild(newOption('none', 'None'));
        select.appendChild(newOption('', 'Default (browser)'));
        browserCams.forEach(function (d) {
            var label = (d.label || 'Camera ' + (select.options.length)) + ' (browser)';
            select.appendChild(newOption(d.deviceId, label));
        });
        jetsonCams.forEach(function (c) {
            var label = (c.label || c.id);
            if (label.indexOf('(USB)') === -1 && label.indexOf('(Jetson)') === -1) label = label + ' (USB)';
            select.appendChild(newOption(c.id, label));
        });
        var cam = currentConfig.devices.camera;
        var val = (cam === 'none' || cam === null || cam === undefined) ? 'none' : (cam === 'browser' || cam === '' ? '' : cam);
        try { select.value = val; } catch (e) { select.value = ''; }
        updateDeviceIndicators();
    });
}

function newOption(value, text) {
    var opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    return opt;
}

/**
 * Enumerate audio input devices and fill the combined "Microphone devices" dropdown.
 */
function populateMicrophoneDeviceDropdown() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    var select = document.getElementById('device-microphone-list');
    if (!select) return;
    navigator.mediaDevices.enumerateDevices()
        .then(function (devices) {
            var audioInputs = devices.filter(function (d) { return d.kind === 'audioinput'; });
            select.innerHTML = '';
            select.appendChild(newOption('none', 'None (Text Only)'));
            select.appendChild(newOption('', 'Default (browser choice)'));
            audioInputs.forEach(function (d) {
                var label = (d.label || 'Microphone ' + (select.options.length)) + ' ' + deviceLabelSuffix(d.label);
                select.appendChild(newOption(d.deviceId, label));
            });
            select.value = currentConfig.devices.microphone === 'none' ? 'none' : (state.selectedBrowserMicId || '');
        })
        .catch(function (err) { console.warn('[Devices] enumerateDevices failed:', err); });
}

function populateSpeakerDeviceDropdown() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    var select = document.getElementById('device-speaker-list');
    if (!select) return;
    navigator.mediaDevices.enumerateDevices()
        .then(function (devices) {
            var outputs = devices.filter(function (d) { return d.kind === 'audiooutput'; });
            select.innerHTML = '';
            select.appendChild(newOption('none', 'None (Text Only)'));
            select.appendChild(newOption('', 'Default (browser choice)'));
            outputs.forEach(function (d) {
                var label = (d.label || 'Speaker ' + (select.options.length)) + ' ' + deviceLabelSuffix(d.label);
                select.appendChild(newOption(d.deviceId, label));
            });
            select.value = currentConfig.devices.speaker === 'none' ? 'none' : (state.selectedBrowserSpeakerId || '');
            updateDeviceIndicators();
        })
        .catch(function (err) { console.warn('[Devices] enumerateDevices failed:', err); });
}

// Update configuration value
function updateConfig(section, key, value) {
    console.log(`Config updated: ${section}.${key} = ${value}`);
    currentConfig[section][key] = value;

    // Re-render the config panel to show/hide conditional fields
    const contentEl = document.getElementById('config-tab-content');
    const configKey = state.activeConfigTab === 'device' ? 'devices' : state.activeConfigTab;
    contentEl.innerHTML = renderEditableConfigForm(state.activeConfigTab, currentConfig[configKey]);

    // After form is in DOM, fetch models to populate dropdowns
    if (state.activeConfigTab === 'llm') {
        setTimeout(() => fetchLLMModels(currentConfig.llm.api_base || (currentConfig.llm.ollama_url && currentConfig.llm.ollama_url.replace(/\/v1$/, '') + '/v1')), 0);
    }
    if (state.activeConfigTab === 'asr' && (currentConfig.asr.backend === 'riva' || currentConfig.asr.scheme === 'riva')) {
        setTimeout(() => fetchASRModels(currentConfig.asr.server || currentConfig.asr.riva_server || 'localhost:50051'), 0);
    }
    if (section === 'devices') {
        updateDeviceIndicators();
        updateChatInputVisibility();
        setTimeout(function () {
            if (document.getElementById('device-microphone-list')) populateAllDeviceDropdowns();
        }, 0);
        if (state.isLiveSession && state.sessionState === 'setup') {
            startPreviewStream();
        }
    }
    if (section === 'app' && key === 'enable_timeline') {
        updateTimelinePanelVisibility();
    }
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();

    // Visual feedback
    const formEl = contentEl.querySelector('.config-form');
    if (formEl) {
        formEl.classList.add('config-updated');
        setTimeout(() => formEl.classList.remove('config-updated'), 300);
    }
}

// --- LLM API Base presets and model list (Live RIVA WebUI-style) ---
function togglePresetsMenu(ev) {
    ev.preventDefault();
    const menu = document.getElementById('presetsMenu');
    if (!menu) return;
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}
function selectLLMPreset(url, label) {
    currentConfig.llm.api_base = url;
    const input = document.getElementById('llm-api-base');
    if (input) input.value = url;
    document.getElementById('presetsMenu').style.display = 'none';
    updateConfig('llm', 'api_base', url);
}
async function fetchLLMModels(apiBase) {
    if (!apiBase) return;
    const select = document.getElementById('llm-model-select');
    if (!select) return;
    const apiKey = (document.getElementById('llm-api-key') || {}).value || currentConfig.llm.api_key || '';
    select.innerHTML = '<option value="">Loading...</option>';
    try {
        const q = 'api_base=' + encodeURIComponent(apiBase) + (apiKey ? '&api_key=' + encodeURIComponent(apiKey) : '');
        const r = await fetch('/api/llm/models?' + q);
        const data = await r.json();
        const models = (data && data.models) ? data.models : [];
        const current = currentConfig.llm.model || '';
        select.innerHTML = models.length
            ? models.map(m => `<option value="${escapeHtml(m)}" ${m === current ? 'selected' : ''}>${escapeHtml(m)}</option>`).join('')
            : '<option value="">No models found</option>';
        if (models.length && !models.includes(current)) {
            currentConfig.llm.model = models[0];
            select.value = models[0];
        }
    } catch (e) {
        select.innerHTML = '<option value="">Error loading models</option>';
        console.error('fetchLLMModels failed:', e);
    }
}
function refreshLLMModels() {
    const apiBase = (document.getElementById('llm-api-base') || {}).value || currentConfig.llm.api_base;
    if (apiBase) fetchLLMModels(apiBase);
}

async function fetchASRModels(server) {
    if (!server) return;
    const select = document.getElementById('asr-model-select');
    if (!select) return;
    select.innerHTML = '<option value="">Loading...</option>';
    try {
        const r = await fetch('/api/asr/models?server=' + encodeURIComponent(server));
        const data = await r.json();
        const models = (data && data.models) ? data.models : [];
        const current = currentConfig.asr.model || '';
        select.innerHTML = '<option value="">Server default</option>' +
            (models.length ? models.map(m => '<option value="' + escapeHtml(m) + '"' + (m === current ? ' selected' : '') + '>' + escapeHtml(m) + '</option>').join('') : '');
        if (models.length && current && !models.includes(current)) {
            currentConfig.asr.model = models[0];
            select.value = models[0];
        } else if (models.length && !current) {
            select.value = '';
        }
    } catch (e) {
        select.innerHTML = '<option value="">Server default</option><option value="">Error loading models</option>';
        console.error('fetchASRModels failed:', e);
    }
}

// Toggle Advanced ASR Settings panel (Live RIVA WebUI-style)
function toggleNestedPanel(panelId) {
    const content = document.getElementById(panelId);
    const toggle = document.getElementById(panelId + 'Toggle');
    if (!content || !toggle) return;
    const isExpanded = content.style.display !== 'none';
    content.style.display = isExpanded ? 'none' : 'block';
    toggle.innerHTML = isExpanded ? '<i data-lucide="chevron-right" class="lucide-inline"></i>' : '<i data-lucide="chevron-down" class="lucide-inline"></i>';
    toggle.classList.toggle('expanded', !isExpanded);
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

// Apply VAD preset (Live RIVA WebUI-style: aggressive / balanced / conservative)
function applyVADPreset(preset) {
    let speech_pad_ms, speech_timeout_ms, threshold;
    switch (preset) {
        case 'aggressive':
            speech_pad_ms = 100; speech_timeout_ms = 400; threshold = 0.3;
            break;
        case 'balanced':
            speech_pad_ms = 300; speech_timeout_ms = 700; threshold = 0.5;
            break;
        case 'conservative':
            speech_pad_ms = 500; speech_timeout_ms = 1200; threshold = 0.7;
            break;
        default:
            return;
    }
    currentConfig.asr.speech_pad_ms = speech_pad_ms;
    currentConfig.asr.speech_timeout_ms = speech_timeout_ms;
    currentConfig.asr.vad_start_threshold = threshold;
    const padEl = document.getElementById('vadSpeechPadSlider');
    const silenceEl = document.getElementById('vadSilenceDurationSlider');
    const threshEl = document.getElementById('vadThresholdSlider');
    if (padEl) { padEl.value = speech_pad_ms; document.getElementById('vadSpeechPadValue').textContent = speech_pad_ms; }
    if (silenceEl) { silenceEl.value = speech_timeout_ms; document.getElementById('vadSilenceDurationValue').textContent = speech_timeout_ms; }
    if (threshEl) { threshEl.value = threshold; document.getElementById('vadThresholdValue').textContent = threshold; }
}

/** Build chat list from session.chat and, when present, add AI messages from timeline tts_complete (data.text). Sorted by timestamp. */
function getMergedChatMessages(session) {
    const chat = session.chat || [];
    const timeline = session.timeline || [];
    const aiFromTimeline = timeline
        .filter(e => e.event_type === 'tts_complete' && (e.data && e.data.text))
        .map(e => ({ role: 'assistant', content: String(e.data.text).trim(), timestamp: e.timestamp }))
        .filter(m => m.content.length > 0);
    if (aiFromTimeline.length === 0) return chat;
    const combined = [...chat.map(m => ({ ...m, timestamp: m.timestamp != null ? m.timestamp : 0 })), ...aiFromTimeline];
    combined.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    return combined;
}

function renderChatHistory() {
    const session = state.selectedSession;
    const chatEl = document.getElementById('chat-history');

    // Prefer merged chat (session.chat + AI from tts_complete), else session.turns (legacy)
    const chat = getMergedChatMessages(session);
    const turns = session.turns || [];

    if (chat.length > 0) {
        // Render: one bubble per message (user / assistant by role)
        chatEl.innerHTML = chat.map(msg => {
            const role = (msg.role || 'user').toLowerCase();
            const isUser = role === 'user';
            const ts = msg.timestamp != null && uiSettings.showTimestamps
                ? `<span class="chat-meta">${formatTimestamp(msg.timestamp)}</span>`
                : '';
            return `
                <div class="chat-message ${isUser ? 'user' : 'ai'}">
                    <div class="chat-avatar">${isUser ? '<i data-lucide="user" class="lucide-inline"></i>' : '<i data-lucide="bot" class="lucide-inline"></i>'}</div>
                    <div class="chat-bubble">
                        <div class="chat-text">${escapeHtml(msg.content || '...')}</div>
                        ${ts}
                    </div>
                </div>
            `;
        }).join('');
    } else if (turns.length > 0) {
        // Legacy: session.turns with user_transcript + ai_response per turn
        chatEl.innerHTML = turns.map(turn => {
            const userConfidence = turn.user_confidence ?
                `<span class="chat-meta">Confidence: ${(turn.user_confidence * 100).toFixed(0)}%</span>` : '';
            const turnMetrics = turn.latencies ?
                `<span class="chat-meta">TTL: ${formatLatency(turn.latencies.ttl)}</span>` : '';

            return `
                <div class="chat-message user">
                    <div class="chat-avatar"><i data-lucide="user" class="lucide-inline"></i></div>
                    <div class="chat-bubble">
                        <div class="chat-text">${escapeHtml(turn.user_transcript || '...')}</div>
                        ${userConfidence}
                    </div>
                </div>
                <div class="chat-message ai">
                    <div class="chat-avatar"><i data-lucide="bot" class="lucide-inline"></i></div>
                    <div class="chat-bubble">
                        <div class="chat-text">${escapeHtml(turn.ai_response || '...')}</div>
                        ${turnMetrics}
                    </div>
                </div>
            `;
        }).join('');
    } else {
        chatEl.innerHTML = `
            <div class="empty-state">
                <p>No conversation yet</p>
            </div>
        `;
        return;
    }

    // Auto-scroll to bottom
    chatEl.scrollTop = chatEl.scrollHeight;
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

function formatTimestamp(seconds) {
    if (seconds == null || typeof seconds !== 'number') return '';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return m > 0 ? `${m}:${s.padStart(4, '0')}s` : `${s}s`;
}

function renderTimelineMetrics() {
    const metricsEl = document.getElementById('timeline-metrics');
    if (!metricsEl) return;
    if (!state.selectedSession) {
        metricsEl.innerHTML = '<div class="timeline-metric"><span class="timeline-metric-label">—</span><span class="timeline-metric-value">No session</span></div>';
        return;
    }
    const session = state.selectedSession;
    const metrics = session.metrics || {};

    metricsEl.innerHTML = `
        <div class="timeline-metric">
            <span class="timeline-metric-label">Avg TTL</span>
            <span class="timeline-metric-value" style="color: var(--ttl-highlight)">
                ${formatLatency(metrics.avg_ttl)}
            </span>
        </div>
        <div class="timeline-metric">
            <span class="timeline-metric-label">TTL Range</span>
            <span class="timeline-metric-value">
                ${formatLatency(metrics.min_ttl)} - ${formatLatency(metrics.max_ttl)}
            </span>
        </div>
        <div class="timeline-metric">
            <span class="timeline-metric-label">ASR</span>
            <span class="timeline-metric-value" style="color: var(--timeline-speech)">
                ${formatLatency(metrics.avg_asr_latency)}
            </span>
        </div>
        <div class="timeline-metric">
            <span class="timeline-metric-label">LLM</span>
            <span class="timeline-metric-value" style="color: var(--timeline-llm)">
                ${formatLatency(metrics.avg_llm_latency)}
            </span>
        </div>
        <div class="timeline-metric">
            <span class="timeline-metric-label">TTS</span>
            <span class="timeline-metric-value" style="color: var(--timeline-tts)">
                ${formatLatency(metrics.avg_tts_latency)}
            </span>
        </div>
        <div class="timeline-metric">
            <span class="timeline-metric-label">Duration</span>
            <span class="timeline-metric-value">
                ${formatDuration(metrics.session_duration)}
            </span>
        </div>
    `;
}

/** Minimize or restore timeline panel based on Configuration > App > Enable Timeline Visualization. */
function updateTimelinePanelVisibility() {
    const panel = document.getElementById('timeline-panel');
    if (!panel) return;
    const enabled = currentConfig.app && currentConfig.app.enable_timeline !== false;
    state.ui.timelinePanelCollapsed = !enabled;
    if (enabled) {
        panel.classList.remove('timeline-panel--minimized');
    } else {
        panel.classList.add('timeline-panel--minimized');
    }
}

/** Clear timeline when starting New Voice Chat (no selected session). */
function initTimeline() {
    state.timelineZoom = 1.0;
    state.timelineOffset = 0;
    state.timelineDuration = 0;

    const canvas = document.getElementById('timeline-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        const width = rect.width;
        const height = rect.height;
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary').trim() || '#0a0a0a';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-tertiary').trim() || '#808080';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('New session — no timeline yet', width / 2, height / 2);
    }

    renderTimelineMetrics();
    const scrollBar = document.getElementById('timeline-scroll-bar');
    if (scrollBar) scrollBar.style.display = 'none';
}

// ===== Timeline Canvas Rendering =====
function renderTimeline() {
    const inLive = state.isLiveSession && state.sessionState === 'live' && state.liveTimelineEvents;
    const hasStoppedLiveData = state.isLiveSession && state.sessionState === 'stopped' && state.liveTimelineEvents && state.liveTimelineEvents.length > 0;
    const rawTimeline = inLive ? state.liveTimelineEvents : (hasStoppedLiveData ? state.liveTimelineEvents : (state.selectedSession && (state.selectedSession.timeline && state.selectedSession.timeline.events || state.selectedSession.timeline)));
    const timeline = Array.isArray(rawTimeline) ? rawTimeline : (rawTimeline && rawTimeline.events) || [];
    if (!inLive && !hasStoppedLiveData && !state.selectedSession) return;

    const canvas = document.getElementById('timeline-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Set canvas size
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary');
    ctx.fillRect(0, 0, width, height);

    if (timeline.length === 0) {
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        const liveHint = (state.isLiveSession && state.sessionState === 'live')
            ? 'Live — waiting for events…'
            : 'No timeline data';
        ctx.fillText(liveHint, width / 2, height / 2);
        return;
    }

    // Timeline rendering constants
    const combineSpeechLanes = uiSettings.combineSpeechLanes; // Use global UI setting
    const lanes = combineSpeechLanes
        ? ['audio', 'speech', 'llm', 'ttl', 'system']
        : ['audio', 'speech', 'llm', 'tts', 'ttl', 'system'];

    const PADDING_TOP = 20;
    const PADDING_LEFT = 100;
    const PADDING_RIGHT = 20;
    const TIME_LABEL_HEIGHT = 25;

    const laneColors = {
        system: getComputedStyle(document.documentElement).getPropertyValue('--timeline-system').trim() || '#9C27B0',
        audio: getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio').trim() || '#76B900',
        speech: getComputedStyle(document.documentElement).getPropertyValue('--timeline-speech').trim() || '#1976D2',
        speechPartial: getComputedStyle(document.documentElement).getPropertyValue('--timeline-speech-partial').trim() || '#64B5F6',
        llm: getComputedStyle(document.documentElement).getPropertyValue('--timeline-llm').trim() || '#FF9800',
        tts: getComputedStyle(document.documentElement).getPropertyValue('--timeline-tts-lane').trim() || '#EC407A',
        ttl: getComputedStyle(document.documentElement).getPropertyValue('--timeline-ttl').trim() || '#FFEB3B',
    };

    const laneLabels = {
        system: 'CPU / GPU',
        audio: 'AUDIO',
        speech: 'ASR',      // Changed from 'SPEECH' to 'ASR'
        llm: 'LLM',
        tts: 'TTS',
        ttl: 'TTL'
    };

    // Calculate lane dimensions: AUDIO 2x, CPU/GPU (system) 1.5x, others 1x
    const laneAreaHeight = height - PADDING_TOP - TIME_LABEL_HEIGHT;
    const LANE_GAP = 8;
    const totalGaps = (lanes.length - 1) * LANE_GAP;
    const laneUnits = lanes.map(lane => lane === 'audio' ? 2 : (lane === 'system' ? 1.5 : 1));
    const totalUnits = laneUnits.reduce((a, b) => a + b, 0);
    const baseHeight = Math.max(20, (laneAreaHeight - totalGaps) / totalUnits);
    const LANE_HEIGHTS = laneUnits.map(u => u * baseHeight);

    // Calculate time range with zoom
    const maxTimeFromEvents = timeline.length ? Math.max(0.1, ...timeline.map(e => e.timestamp || e.end_time || 0)) : 0.1;
    const viewportWidth = width - PADDING_LEFT - PADDING_RIGHT;
    const LIVE_WINDOW_SEC = 15;
    let baseTimeScale, timeScale, visibleTimeWindow, maxTime;
    if (inLive && state.liveSessionStartTime > 0) {
        // Live: sliding 15-sec window driven by current session time so it scrolls automatically
        const currentSessionTime = (Date.now() / 1000) - state.liveSessionStartTime;
        maxTime = Math.max(maxTimeFromEvents, currentSessionTime, 0.1);
        state.timelineDuration = maxTime;
        state.timelineOffset = Math.max(0, currentSessionTime - LIVE_WINDOW_SEC);
        state.timelineZoom = Math.max(0.1, Math.min(10, maxTime / LIVE_WINDOW_SEC));
        baseTimeScale = viewportWidth / LIVE_WINDOW_SEC;
        timeScale = baseTimeScale;
        visibleTimeWindow = LIVE_WINDOW_SEC;
    } else {
        maxTime = maxTimeFromEvents;
        state.timelineDuration = maxTime;
        baseTimeScale = viewportWidth / maxTime;
        timeScale = baseTimeScale * state.timelineZoom;
        visibleTimeWindow = viewportWidth / timeScale;
    }

    // Draw vertical grid lines (1 second intervals)
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.5; // Increased from 0.3 for better visibility

    const gridStartTime = Math.floor(state.timelineOffset);
    const gridEndTime = Math.ceil(state.timelineOffset + visibleTimeWindow);

    const totalLanesHeight = LANE_HEIGHTS.reduce((a, b) => a + b, 0) + (lanes.length - 1) * LANE_GAP;
    for (let t = Math.max(0, gridStartTime); t <= Math.min(maxTime, gridEndTime); t += 1) {
        const x = PADDING_LEFT + (t - state.timelineOffset) * timeScale;
        if (x >= PADDING_LEFT && x <= width - PADDING_RIGHT) {
            ctx.beginPath();
            ctx.moveTo(x, PADDING_TOP);
            ctx.lineTo(x, PADDING_TOP + totalLanesHeight);
            ctx.stroke();
        }
    }

    ctx.globalAlpha = 1.0;

    // Cumulative Y offset per lane (for variable lane heights)
    const laneYOffsets = [];
    let acc = 0;
    for (let i = 0; i < lanes.length; i++) {
        laneYOffsets.push(acc);
        acc += LANE_HEIGHTS[i] + LANE_GAP;
    }
    const getLaneY = (i) => PADDING_TOP + laneYOffsets[i];
    const getLaneHeight = (i) => LANE_HEIGHTS[i];

    // Draw lane labels
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'right';
    lanes.forEach((lane, i) => {
        const y = getLaneY(i) + getLaneHeight(i) / 2;
        ctx.fillText(laneLabels[lane], PADDING_LEFT - 10, y + 4);
    });

    // Draw lane backgrounds (semi-transparent so grid lines show through)
    lanes.forEach((lane, i) => {
        const y = getLaneY(i);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-tertiary');
        ctx.globalAlpha = 0.3; // Semi-transparent lanes
        ctx.fillRect(PADDING_LEFT, y, width - PADDING_LEFT - PADDING_RIGHT, getLaneHeight(i));
        ctx.globalAlpha = 1.0;
    });

    // Draw timeline events with enhanced rendering (rectangles, waveforms, points)
    const liveAmplitude = (inLive || hasStoppedLiveData) ? state.liveAudioAmplitudeHistory : null;
    const liveTtsSegments = (inLive || hasStoppedLiveData) ? state.liveTtsAmplitudeHistory : null;
    const liveSystemStats = (inLive || hasStoppedLiveData) ? state.liveSystemStats : (state.selectedSession && state.selectedSession.system_stats) || null;
    const replayTtsSegments = (!inLive && !hasStoppedLiveData && state.selectedSession && state.selectedSession.tts_playback_segments) ? state.selectedSession.tts_playback_segments : null;
    const liveSessionTime = (state.liveSessionStartTime > 0) ? (Date.now() / 1000 - state.liveSessionStartTime) : null;
    drawTimelineEvents(ctx, timeline, lanes, LANE_HEIGHTS, laneYOffsets, LANE_GAP, PADDING_TOP, PADDING_LEFT,
                       PADDING_RIGHT, width, timeScale, state.timelineOffset, laneColors, combineSpeechLanes,
                       inLive, hasStoppedLiveData, liveAmplitude, liveTtsSegments, liveSystemStats, replayTtsSegments, liveSessionTime);

    // Draw time axis at bottom of lanes
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 1;
    const axisY = PADDING_TOP + totalLanesHeight;
    ctx.beginPath();
    ctx.moveTo(PADDING_LEFT, axisY);
    ctx.lineTo(width - PADDING_RIGHT, axisY);
    ctx.stroke();

    // Draw time labels (every second with zoom)
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let t = gridStartTime; t <= gridEndTime; t += 1) {
        const x = PADDING_LEFT + (t - state.timelineOffset) * timeScale;
        if (x >= PADDING_LEFT && x <= width - PADDING_RIGHT) {
            ctx.fillText(`${t}s`, x, axisY + 5); // Closer to axis line
        }
    }

    // Update horizontal scrollbar
    updateTimelineScrollbar(maxTime, visibleTimeWindow);
}

// Get amplitude at time t from live history (nearest sample or linear interpolate)
function getAmplitudeAtTime(history, t) {
    if (!history.length) return 0;
    const getT = (s) => s.timestamp != null ? s.timestamp : s[0];
    const getA = (s) => s.amplitude != null ? s.amplitude : s[1];
    if (t <= getT(history[0])) return getA(history[0]);
    if (t >= getT(history[history.length - 1])) return getA(history[history.length - 1]);
    for (let i = 0; i < history.length - 1; i++) {
        const t0 = getT(history[i]), t1 = getT(history[i + 1]);
        if (t >= t0 && t <= t1) {
            const a0 = getA(history[i]), a1 = getA(history[i + 1]);
            const frac = (t - t0) / (t1 - t0 || 1);
            return a0 + frac * (a1 - a0);
        }
    }
    return 0;
}

// Get TTS (AI) amplitude at time t from segment list { startTime, endTime, amplitude }
function getTtsAmplitudeAtTime(segments, t) {
    if (!segments.length) return 0;
    for (let i = 0; i < segments.length; i++) {
        const s = segments[i];
        if (t >= s.startTime && t <= s.endTime) return s.amplitude || 0;
    }
    return 0;
}

// Draw timeline events with support for rectangles, waveforms, and points
function drawTimelineEvents(ctx, timeline, lanes, LANE_HEIGHTS, laneYOffsets, LANE_GAP, PADDING_TOP,
                            PADDING_LEFT, PADDING_RIGHT, width, timeScale, timelineOffset,
                            laneColors, combineSpeechLanes, inLive, hasStoppedLiveData, liveAmplitudeHistory, liveTtsSegments, liveSystemStats, replayTtsSegments, liveSessionTime) {
    if (replayTtsSegments === undefined) replayTtsSegments = null;
    if (hasStoppedLiveData === undefined) hasStoppedLiveData = false;
    if (liveSessionTime === undefined) liveSessionTime = null;
    const drawLiveWaveforms = inLive || hasStoppedLiveData;
    const getLaneY = (i) => PADDING_TOP + laneYOffsets[i];
    const getLaneHeight = (i) => LANE_HEIGHTS[i];

    // Infer rectangle events from point event sequences (handles multiple turns)
    const inferredRectangles = [];

    // Find all VAD/ASR/TTS events across the entire timeline
    const speechStarts = timeline.filter(e =>
        e.event_type === 'user_speech_start' || e.event_type === 'vad_start'
    );
    const speechEnds = timeline.filter(e =>
        e.event_type === 'user_speech_end' || e.event_type === 'vad_end'
    );
    const vadSpeechStarts = timeline.filter(e => e.event_type === 'vad_speech_start').sort((a, b) => a.timestamp - b.timestamp);
    const vadSpeechEnds = timeline.filter(e => e.event_type === 'vad_speech_end').sort((a, b) => a.timestamp - b.timestamp);
    const asrPartials = timeline.filter(e => e.event_type === 'asr_partial');
    const asrFinals = timeline.filter(e => e.event_type === 'asr_final').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const ttsStarts = timeline.filter(e => e.event_type === 'tts_start').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const ttsFirstAudios = timeline.filter(e => e.event_type === 'tts_first_audio');
    const ttsCompletes = timeline.filter(e => e.event_type === 'tts_complete');
    const llmStarts = timeline.filter(e => e.event_type === 'llm_start').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const llmFirstTokens = timeline.filter(e => e.event_type === 'llm_first_token').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const llmCompletes = timeline.filter(e => e.event_type === 'llm_complete').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

    // LLM: prefill (start → first token) and generate (first token → complete). Like Live RIVA WebUI; first-token boundary from pipeline.
    llmStarts.forEach((startEv, i) => {
        const firstToken = llmFirstTokens[i];
        const complete = llmCompletes[i];
        const startTime = startEv.timestamp || 0;
        const endTime = (complete && (complete.timestamp || 0)) || startTime;
        if (firstToken && (firstToken.timestamp || 0) > startTime) {
            inferredRectangles.push({
                event_type: 'llm_prefill',
                lane: 'llm',
                start_time: startTime,
                end_time: firstToken.timestamp,
                timestamp: startTime,
                phase: 'prefill',
                inferred: true
            });
        }
        const genStart = firstToken ? (firstToken.timestamp || 0) : startTime;
        if (complete && endTime > genStart) {
            inferredRectangles.push({
                event_type: 'llm_generate',
                lane: 'llm',
                start_time: genStart,
                end_time: endTime,
                timestamp: genStart,
                phase: 'generate',
                inferred: true
            });
        }
        if (!firstToken && complete && endTime > startTime) {
            inferredRectangles.push({
                event_type: 'llm_segment',
                lane: 'llm',
                start_time: startTime,
                end_time: endTime,
                timestamp: startTime,
                inferred: true
            });
        }
    });

    // Infer ASR from partials + final when no/few speech starts: blue = first partial to last partial, light blue = last partial to final
    const useAsrFinalOnly = (speechStarts.length === 0 || speechStarts.length < asrFinals.length) && asrFinals.length > 0;
    if (useAsrFinalOnly) {
        asrFinals.forEach((finalEv, i) => {
            const prevFinalTime = i === 0 ? 0 : (asrFinals[i - 1].timestamp || 0);
            const finalTime = finalEv.timestamp || 0;
            const turnPartials = asrPartials.filter(p => {
                const t = p.timestamp || 0;
                return t > prevFinalTime && t <= finalTime;
            }).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
            const firstPartial = turnPartials[0];
            const lastPartial = turnPartials[turnPartials.length - 1];
            if (firstPartial && lastPartial && firstPartial.timestamp !== undefined && lastPartial.timestamp !== undefined) {
                const tFirst = firstPartial.timestamp;
                const tLast = lastPartial.timestamp;
                if (tLast > tFirst) {
                    inferredRectangles.push({
                        event_type: 'asr_active',
                        lane: 'speech',
                        start_time: tFirst,
                        end_time: tLast,
                        timestamp: tFirst,
                        phase: 'active-asr',
                        inferred: true
                    });
                }
                if (finalTime > tLast) {
                    inferredRectangles.push({
                        event_type: 'asr_finalizing',
                        lane: 'speech',
                        start_time: tLast,
                        end_time: finalTime,
                        timestamp: tLast,
                        phase: 'post-asr',
                        inferred: true
                    });
                }
            }
        });
    }

    // For each turn, create multi-phase ASR visualization (only when we have matching speechStarts / VAD)
    if (!useAsrFinalOnly) speechStarts.forEach((speechStart, turnIdx) => {
        const speechEnd = speechEnds[turnIdx];
        const asrFinal = asrFinals[turnIdx];

        // Find first and last partial for this turn
        const turnPartials = asrPartials.filter(p =>
            p.timestamp >= speechStart.timestamp &&
            (!asrFinal || p.timestamp <= asrFinal.timestamp)
        );
        const firstPartial = turnPartials[0];
        const lastPartial = turnPartials[turnPartials.length - 1];

        // Pre-ASR start: use vad_speech_start for this turn if present and before first partial, else speechStart (handles both with/without VAD events)
        const prevTurnEnd = turnIdx === 0 ? 0 : (asrFinals[turnIdx - 1]?.timestamp ?? speechStart.timestamp - 0.001);
        const vadStartInRange = firstPartial ? vadSpeechStarts.filter(v =>
            v.timestamp > prevTurnEnd && v.timestamp < firstPartial.timestamp
        ) : [];
        const preAsrStart = (firstPartial && vadStartInRange.length > 0)
            ? Math.min(...vadStartInRange.map(v => v.timestamp))
            : speechStart.timestamp;

        if (speechStart && firstPartial && preAsrStart < firstPartial.timestamp) {
            // Phase 1: Pre-ASR (45° shaded) — VAD/speech start → first partial_transcript
            inferredRectangles.push({
                event_type: 'asr_pre',
                lane: 'speech',
                start_time: preAsrStart,
                end_time: firstPartial.timestamp,
                timestamp: preAsrStart,
                phase: 'pre-asr',
                inferred: true
            });
        }

        // Speech end for this turn: optional vad_speech_end if present, else speechEnd (with/without VAD events)
        const turnSpeechEnd = (() => {
            if (asrFinal && vadSpeechEnds.length > 0) {
                const inRange = vadSpeechEnds.filter(v =>
                    v.timestamp >= (firstPartial?.timestamp ?? 0) && v.timestamp <= (asrFinal.timestamp + 0.001)
                );
                if (inRange.length > 0) {
                    const v = inRange[inRange.length - 1];
                    if (!speechEnd || v.timestamp <= (speechEnd.timestamp + 0.001)) return { timestamp: v.timestamp };
                }
            }
            return speechEnd;
        })();

        if (firstPartial && (lastPartial || turnSpeechEnd)) {
            // Phase 2: Active ASR (partials coming in, user still speaking)
            const activeEnd = turnSpeechEnd ?
                Math.min(turnSpeechEnd.timestamp, (lastPartial || firstPartial).timestamp) :
                (lastPartial || firstPartial).timestamp;

            if (activeEnd > firstPartial.timestamp) {
                inferredRectangles.push({
                    event_type: 'asr_active',
                    lane: 'speech',
                    start_time: firstPartial.timestamp,
                    end_time: activeEnd,
                    timestamp: firstPartial.timestamp,
                    phase: 'active-asr',  // For styling
                    inferred: true
                });
            }
        }

        // Phase 3: Last partial → final transcript (light blue)
        if (lastPartial && asrFinal && asrFinal.timestamp > lastPartial.timestamp) {
            inferredRectangles.push({
                event_type: 'asr_finalizing',
                lane: 'speech',
                start_time: lastPartial.timestamp,
                end_time: asrFinal.timestamp,
                timestamp: lastPartial.timestamp,
                phase: 'post-asr',  // Light blue: last partial to final
                inferred: true
            });
        }
        // AUDIO lane: no VAD rectangles or green dots (waveform only)
    });

    // TTS lane: one magenta rectangle per turn from tts_start to tts_complete (tts_first_audio shown as thin vertical line later)
    ttsStarts.forEach((startEv, i) => {
        const complete = ttsCompletes[i];
        if (complete && (complete.timestamp || 0) > (startEv.timestamp || 0)) {
            inferredRectangles.push({
                event_type: 'tts_segment',
                lane: 'tts',
                start_time: startEv.timestamp,
                end_time: complete.timestamp,
                timestamp: startEv.timestamp,
                inferred: true
            });
        }
    });
    // Live: growing TTS segment from last tts_start to now until tts_complete arrives (use last start if we have more starts than completes)
    if (inLive && liveSessionTime != null && ttsStarts.length > 0 && ttsStarts.length > ttsCompletes.length) {
        const lastStart = ttsStarts[ttsStarts.length - 1];
        const endTime = Math.max(lastStart.timestamp + 0.1, liveSessionTime);
        if (endTime > lastStart.timestamp) {
            inferredRectangles.push({
                event_type: 'tts_segment',
                lane: 'tts',
                start_time: lastStart.timestamp,
                end_time: endTime,
                timestamp: lastStart.timestamp,
                inferred: true,
                growing: true
            });
        }
    }

    // Group events by render type for efficient rendering
    const waveformEvents = [];
    const rectangleEvents = [...inferredRectangles]; // Start with inferred rectangles
    const pointEvents = [];

    timeline.forEach(event => {
        // Skip if outside visible range (approximate check with buffer)
        const x = PADDING_LEFT + (event.timestamp - timelineOffset) * timeScale;
        if (x < PADDING_LEFT - 100 || x > width - PADDING_RIGHT + 100) return;

        // Categorize by data: waveform if amplitude present, else rectangle if start/end (render_type is UI preference, not in session)
        if (event.amplitude !== undefined || event.render_type === 'waveform') {
            waveformEvents.push(event);
        } else if (event.start_time !== undefined && event.end_time !== undefined) {
            // Only add if not already inferred
            if (!event.inferred) {
                rectangleEvents.push(event);
            }
        } else {
            pointEvents.push(event);
        }
    });

    // Debug logging
    console.log('Timeline rendering:', {
        totalEvents: timeline.length,
        inferredRectangles: inferredRectangles.length,
        explicitRectangles: rectangleEvents.length - inferredRectangles.length,
        totalRectangles: rectangleEvents.length,
        waveforms: waveformEvents.length,
        points: pointEvents.length,
        sampleInferred: inferredRectangles[0]
    });

    // 1. Draw rectangles first (background layer)
    rectangleEvents.forEach(event => {
        let targetLane = event.lane;

        // AUDIO lane: waveform only (no green/vad rectangles)
        if (targetLane === 'audio') return;

        // Map TTS events to speech lane if combining
        if (combineSpeechLanes && targetLane === 'tts') {
            targetLane = 'speech';
        }

        const laneIndex = lanes.indexOf(targetLane);
        if (laneIndex === -1) return;

        const x1 = PADDING_LEFT + (event.start_time - timelineOffset) * timeScale;
        const x2 = PADDING_LEFT + (event.end_time - timelineOffset) * timeScale;
        const laneY = getLaneY(laneIndex);
        const laneH = getLaneHeight(laneIndex);

        // Determine vertical position based on sub-lane (for ASR/TTS separation)
        let barY = laneY + laneH * 0.1;
        let barHeight = laneH * 0.8;

        if (combineSpeechLanes && targetLane === 'speech') {
            // ASR in top half, TTS in bottom half when combined
            if (event.event_type?.includes('asr') || event.event_type?.includes('vad')) {
                barY = laneY + laneH * 0.05;
                barHeight = laneH * 0.4;
            } else if (event.event_type?.includes('tts')) {
                barY = laneY + laneH * 0.55;
                barHeight = laneH * 0.4;
            }
        }

        // Color and style based on event type and phase
        let fillColor = laneColors[event.lane] || laneColors[targetLane] || '#888';
        let fillAlpha = event.alpha || 0.7;
        let fillPattern = null;

        // Special styling for ASR phases
        if (event.phase === 'pre-asr') {
            // Phase 1: Pre-ASR (transparent/shaded blue)
            fillColor = '#2196F3';  // Light blue
            fillAlpha = 0.3;
            // Create diagonal stripe pattern
            fillPattern = 'diagonal-stripes';
        } else if (event.phase === 'active-asr') {
            // Phase 2: Active ASR (solid blue)
            fillColor = '#1976D2';  // Darker blue
            fillAlpha = 0.8;
        } else if (event.phase === 'post-asr') {
            // Phase 3: Post-speech ASR (light blue)
            fillColor = '#64B5F6';  // Lighter blue
            fillAlpha = 0.6;
        } else if (event.phase === 'prefill' || event.event_type === 'llm_prefill') {
            fillColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-llm-prefill').trim() || '#FF9800';
            fillAlpha = 0.85;
        } else if (event.phase === 'generate' || event.event_type === 'llm_generate') {
            fillColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-llm-generate').trim() || '#FFB74D';
            fillAlpha = 0.85;
        } else {
            // Special colors for other event types
            if (event.event_type?.includes('vad')) {
                fillColor = '#76B900'; // Green for VAD (user speaking)
                fillAlpha = 0.5;
            } else if (event.event_type?.includes('asr')) {
                fillColor = '#2196F3'; // Light blue for generic ASR
            } else if (event.event_type?.includes('tts')) {
                fillColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-tts-rect').trim() || getComputedStyle(document.documentElement).getPropertyValue('--timeline-tts-lane').trim() || '#EC407A'; // Magenta for TTS rectangle (Live RIVA style)
            }
        }

        // Draw the rectangle
        if (fillPattern === 'diagonal-stripes') {
            // Pre-ASR: diagonal stripes clipped to rectangle
            const rx = Math.max(PADDING_LEFT, x1);
            const rw = Math.min(width - PADDING_RIGHT, x2) - rx;
            ctx.save();
            ctx.globalAlpha = fillAlpha;
            ctx.fillStyle = fillColor;
            ctx.fillRect(rx, barY, rw, barHeight);
            ctx.beginPath();
            ctx.rect(rx, barY, rw, barHeight);
            ctx.clip();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
            ctx.lineWidth = 2;
            const stripeSpacing = 8;
            ctx.beginPath();
            for (let i = -barHeight; i < rw + barHeight; i += stripeSpacing) {
                ctx.moveTo(rx + i, barY);
                ctx.lineTo(rx + i + barHeight, barY + barHeight);
            }
            ctx.stroke();
            ctx.restore();
        } else {
            // Solid fill
            ctx.fillStyle = fillColor;
            ctx.globalAlpha = fillAlpha;
            ctx.fillRect(
                Math.max(PADDING_LEFT, x1),
                barY,
                Math.min(width - PADDING_RIGHT, x2) - Math.max(PADDING_LEFT, x1),
                barHeight
            );
            ctx.globalAlpha = 1.0;
        }
    });

    // 1b. TTS lane: thin light purple vertical line at each tts_first_audio (FFTA boundary)
    const ttsLaneIndex = lanes.indexOf(combineSpeechLanes ? 'speech' : 'tts');
    if (ttsLaneIndex !== -1 && ttsFirstAudios.length > 0) {
        const lineColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-tts-first-audio-line').trim() || '#CE93D8';
        const lineWidth = 2;
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = lineWidth;
        const laneY = getLaneY(ttsLaneIndex);
        const laneH = getLaneHeight(ttsLaneIndex);
        ttsFirstAudios.forEach(function (ev) {
            const t = ev.timestamp != null ? ev.timestamp : 0;
            const x = PADDING_LEFT + (t - timelineOffset) * timeScale;
            if (x < PADDING_LEFT - 2 || x > width - PADDING_RIGHT + 2) return;
            ctx.beginPath();
            ctx.moveTo(x, laneY);
            ctx.lineTo(x, laneY + laneH);
            ctx.stroke();
        });
    }

    // 2. Draw waveforms (audio lane visualization)
    // Replay: build TTS segments and ASR final times for smarter user (green) waveform visibility
    const ttsStartsSorted = timeline.filter(e => e.event_type === 'tts_start').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const ttsCompletesSorted = timeline.filter(e => e.event_type === 'tts_complete').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const ttsTimeRanges = ttsStartsSorted.map((s, i) => ({ start: s.timestamp || 0, end: (ttsCompletesSorted[i] && (ttsCompletesSorted[i].timestamp || 0)) || (s.timestamp || 0) }));
    const USER_HIDE_DURING_TTS_THRESHOLD = 5;   // only hide user bar during TTS if amplitude is below this (0-100); show if >= so interruptions are visible
    function isInsideTtsSegment(t) {
        for (let i = 0; i < ttsTimeRanges.length; i++) {
            const r = ttsTimeRanges[i];
            if (t >= r.start && t <= r.end) return true;
        }
        return false;
    }

    waveformEvents.forEach(event => {
        const laneIndex = lanes.indexOf('audio');
        if (laneIndex === -1) return;

        const src = event.source || '';
        const isUser = (src !== 'tts' && src !== 'ai');
        const t = event.timestamp || 0;
        const amp = event.amplitude != null ? event.amplitude : 0;
        const inTts = ttsTimeRanges.length > 0 && isInsideTtsSegment(t);

        if (!inLive && !hasStoppedLiveData && isUser) {
            // During TTS: hide only low user amplitude (ambient); show significant level (possible interruption)
            if (inTts && amp < USER_HIDE_DURING_TTS_THRESHOLD) return;
            // NOTE: Near-full user amplitude with no nearby asr_final is NOT hidden here — it indicates a bug
            // (e.g. mic artifact, client sending wrong data, or server RMS). See docs/INVESTIGATE_USER_AMPLITUDE_ARTIFACT.md
        }

        const x = PADDING_LEFT + (event.timestamp - timelineOffset) * timeScale;
        const laneY = getLaneY(laneIndex);
        const laneH = getLaneHeight(laneIndex);

        // Waveform bar height based on amplitude (0-100)
        const amplitude = event.amplitude || 50;
        const barHeight = (amplitude / 100) * (laneH * 0.9);
        const y = laneY + laneH / 2 - barHeight / 2;

        // Color by source: green for user, purple for AI
        ctx.fillStyle = (event.source === 'tts' || event.source === 'ai') ? '#9C27B0' : '#76B900';
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x, y, 2, barHeight);
        ctx.globalAlpha = 1.0;
    });

    // 2b. Live or just-stopped: draw AUDIO lane — green mic waveform then purple TTS (AI) waveform (with UI gain)
    if (drawLiveWaveforms) {
        const laneIndex = lanes.indexOf('audio');
        if (laneIndex !== -1) {
            const laneY = getLaneY(laneIndex);
            const laneH = getLaneHeight(laneIndex);
            const centerY = laneY + laneH / 2;
            const maxBarHalf = (laneH * 0.65);
            const visibleLeft = PADDING_LEFT;
            const visibleRight = width - PADDING_RIGHT;
            const visibleW = visibleRight - visibleLeft;
            const visibleTimeWindow = visibleW / timeScale;
            const visibleStart = timelineOffset - 0.05;
            const visibleEnd = timelineOffset + visibleTimeWindow + 0.05;
            const getSampleT = (s) => s.timestamp != null ? s.timestamp : s[0];
            const getSampleA = (s) => s.amplitude != null ? s.amplitude : s[1];
            // Green: mic (user) waveform — fine tStep (0.025) so it matches purple AI voice visual density and doesn't look coarser
            if (liveAmplitudeHistory && liveAmplitudeHistory.length > 0) {
                const audioColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio').trim() || '#76B900';
                ctx.fillStyle = audioColor;
                ctx.globalAlpha = 0.85;
                const userGain = (typeof uiSettings.userAudioGain === 'number' ? uiSettings.userAudioGain : 1);
                const barWidthPx = 2;
                const tStep = 0.025; // ~40 Hz so user waveform matches purple smoothness (was 0.05 and looked coarser)
                for (let t = visibleStart; t <= visibleEnd; t += tStep) {
                    const tLo = t - tStep / 2;
                    const tHi = t + tStep / 2;
                    let maxAmp = 0;
                    for (let i = 0; i < liveAmplitudeHistory.length; i++) {
                        const s = liveAmplitudeHistory[i];
                        const st = getSampleT(s);
                        if (st >= tLo && st <= tHi) {
                            const a = (getSampleA(s) || 0) * userGain;
                            if (a > maxAmp) maxAmp = a;
                        }
                    }
                    if (maxAmp <= 0) continue;
                    const amp = Math.min(100, maxAmp);
                    const x = visibleLeft + (t - timelineOffset) * timeScale;
                    const halfH = (Math.min(100, Math.max(0, amp)) / 100) * maxBarHalf;
                    const y1 = centerY - halfH;
                    const barHeight = Math.max(1, halfH * 2);
                    ctx.fillRect(x, y1, barWidthPx, barHeight);
                }
                ctx.globalAlpha = 1.0;
            }
            // Purple: TTS (AI) waveform — coarse strips (~20 Hz), Live RIVA style; use purple (--timeline-ai) for AI voice
            if (liveTtsSegments && liveTtsSegments.length > 0) {
                const ttsColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-ai').trim() || '#9C27B0';
                ctx.fillStyle = ttsColor;
                ctx.globalAlpha = 0.9;
                const aiGain = (typeof uiSettings.aiAudioGain === 'number' ? uiSettings.aiAudioGain : 1);
                const barWidthPx = 2;
                const tStep = 0.025; // match user waveform density (~40 Hz)
                liveTtsSegments.forEach(function (seg) {
                    if (seg.endTime < visibleStart || seg.startTime > visibleEnd) return;
                    const amp = Math.min(100, (seg.amplitude || 0) * aiGain);
                    if (amp <= 0) return;
                    const halfH = (Math.min(100, Math.max(0, amp)) / 100) * maxBarHalf;
                    const barHeight = Math.max(1, halfH * 2);
                    for (let t = Math.max(visibleStart, seg.startTime); t <= Math.min(visibleEnd, seg.endTime); t += tStep) {
                        const x = visibleLeft + (t - timelineOffset) * timeScale;
                        if (x < visibleLeft - barWidthPx || x > visibleRight + barWidthPx) continue;
                        const y1 = centerY - halfH;
                        ctx.fillRect(x, y1, barWidthPx, barHeight);
                    }
                });
                ctx.globalAlpha = 1.0;
            }
        }
    }

    // 2b2. Replay (saved session): draw persisted TTS playback segments on AUDIO lane (full playback duration, same as live)
    if (!inLive && replayTtsSegments && replayTtsSegments.length > 0) {
        const laneIndex = lanes.indexOf('audio');
        if (laneIndex !== -1) {
            const laneY = getLaneY(laneIndex);
            const laneH = getLaneHeight(laneIndex);
            const centerY = laneY + laneH / 2;
            const maxBarHalf = (laneH * 0.65);
            const visibleLeft = PADDING_LEFT;
            const visibleRight = width - PADDING_RIGHT;
            const visibleW = visibleRight - visibleLeft;
            const ttsColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-ai').trim() || '#9C27B0';
            ctx.fillStyle = ttsColor;
            ctx.globalAlpha = 0.9;
            const aiGain = (typeof uiSettings.aiAudioGain === 'number' ? uiSettings.aiAudioGain : 1);
            for (let px = 0; px < visibleW; px += 1) {
                const x = visibleLeft + px;
                const t = timelineOffset + px / timeScale;
                const amp = Math.min(100, getTtsAmplitudeAtTime(replayTtsSegments, t) * aiGain);
                if (amp <= 0) continue;
                const halfH = (Math.min(100, Math.max(0, amp)) / 100) * maxBarHalf;
                const y1 = centerY - halfH;
                const y2 = centerY + halfH;
                ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
            }
            ctx.globalAlpha = 1.0;
        }
    }

    // 2c. Replay fallback: draw TTS segments from tts_first_audio+tts_complete only for saved sessions (not when we have live TTS data)
    if (!inLive && !liveTtsSegments && (!replayTtsSegments || replayTtsSegments.length === 0)) {
        const laneIndex = lanes.indexOf('audio');
        if (laneIndex !== -1) {
            const ttsSegments = rectangleEvents.filter(e => e.event_type === 'tts_segment');
            if (ttsSegments.length > 0) {
                const laneY = getLaneY(laneIndex);
                const laneH = getLaneHeight(laneIndex);
                const barY = laneY + laneH * 0.15;
                const barHeight = laneH * 0.7;
                const ttsColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-ai').trim() || '#9C27B0';
                ctx.fillStyle = ttsColor;
                ctx.globalAlpha = 0.85;
                ttsSegments.forEach(seg => {
                    const x1 = PADDING_LEFT + (seg.start_time - timelineOffset) * timeScale;
                    const x2 = PADDING_LEFT + (seg.end_time - timelineOffset) * timeScale;
                    const w = Math.max(2, x2 - x1);
                    ctx.fillRect(x1, barY, w, barHeight);
                });
                ctx.globalAlpha = 1.0;
            }
        }
    }

    // Helper: truncate text for timeline labels
    function truncateTimelineLabel(str, maxLen) {
        if (!str || typeof str !== 'string') return '';
        const s = str.trim();
        if (s.length <= maxLen) return s;
        return s.slice(0, maxLen) + '\u2026';
    }

    // 3. Draw point events: ASR partial = light blue small dot, ASR final = blue dot + transcript text; llm_complete = dot + response text; speech lane only draws dots for asr_partial/asr_final (no phantom blue)
    pointEvents.forEach(event => {
        let targetLane = event.lane;
        if (targetLane === 'audio') return;
        // Draw user_speech_end on speech lane so TTL boundary is visible (backend sends it as system)
        if (targetLane === 'system' && event.event_type !== 'user_speech_end') return;
        if (targetLane === 'system' && event.event_type === 'user_speech_end') targetLane = 'speech';

        if (combineSpeechLanes && targetLane === 'tts') targetLane = 'speech';

        const laneIndex = lanes.indexOf(targetLane);
        if (laneIndex === -1) return;

        const x = PADDING_LEFT + (event.timestamp - timelineOffset) * timeScale;
        const laneY = getLaneY(laneIndex);
        const laneH = getLaneHeight(laneIndex);
        const centerY = laneY + laneH / 2;

        const et = event.event_type || event.eventType || '';
        // Speech lane: only draw dots for partial (light blue) and final (blue); no phantom blue dots for other events
        let dotColor = laneColors[event.lane] || laneColors[targetLane] || '#888';
        let dotRadius = 4;
        if (targetLane === 'speech') {
            if (et === 'asr_partial') {
                dotColor = laneColors.speechPartial || '#64B5F6';
                dotRadius = 2.5;
            } else if (et === 'asr_final') {
                dotColor = laneColors.speech || '#1976D2';
                dotRadius = 4;
            } else {
                dotRadius = 0; // e.g. user_speech_end: no dot, only TTL ring below
            }
        } else {
            if (et === 'asr_partial') {
                dotColor = laneColors.speechPartial || '#64B5F6';
                dotRadius = 2.5;
            } else if (et === 'asr_final') {
                dotColor = laneColors.speech || '#1976D2';
                dotRadius = 4;
            }
        }

        if (dotRadius > 0) {
            ctx.fillStyle = dotColor;
            ctx.beginPath();
            ctx.arc(x, centerY, dotRadius, 0, Math.PI * 2);
            ctx.fill();
        }

        // TTL highlight: ring around the two TTL boundaries (user stopped speaking → first AI audio)
        if (et === 'user_speech_end' || et === 'tts_first_audio') {
            ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--ttl-highlight') || '#FFEB3B';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, centerY, 7, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Final transcript label: draw text to the right of asr_final dot (final = blue dot)
        if (et === 'asr_final') {
            const text = (event.data && event.data.text) ? String(event.data.text) : '';
            if (text) {
                const label = truncateTimelineLabel(text, 60);
                ctx.font = '11px sans-serif';
                ctx.fillStyle = laneColors.speech || '#1976D2';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const textX = x + (dotRadius > 0 ? dotRadius : 4) + 6;
                if (textX < width - PADDING_RIGHT - 2) {
                    ctx.fillText(label, textX, centerY);
                }
            }
        }

        // LLM response label: draw generated response text to the right of llm_complete dot
        if (et === 'llm_complete') {
            const text = (event.data && event.data.text) ? String(event.data.text) : '';
            if (text) {
                const label = truncateTimelineLabel(text, 60);
                ctx.font = '11px sans-serif';
                ctx.fillStyle = laneColors.llm || '#FF9800';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const textX = x + (dotRadius > 0 ? dotRadius : 4) + 6;
                if (textX < width - PADDING_RIGHT - 2) {
                    ctx.fillText(label, textX, centerY);
                }
            }
        }
    });

    // 4. System lane: CPU/GPU utilization as overlapping area chart (transparent fill below each line; overlap = darker)
    const systemLaneIndex = lanes.indexOf('system');
    const useSmoothCurves = true;
    if (liveSystemStats && liveSystemStats.length > 0 && systemLaneIndex !== -1) {
        const laneY = getLaneY(systemLaneIndex);
        const laneH = getLaneHeight(systemLaneIndex);
        const baseline = laneY + laneH; // 0% at bottom, 100% at top
        const visibleLeft = PADDING_LEFT;
        const visibleRight = width - PADDING_RIGHT;
        const samples = liveSystemStats.filter(function (s) {
            const x = PADDING_LEFT + (s.t - timelineOffset) * timeScale;
            return x >= visibleLeft - 2 && x <= visibleRight + 2;
        });
        if (samples.length > 0) {
            function buildPoints(getPoint) {
                var points = [];
                for (var i = 0; i < samples.length; i++) {
                    var s = samples[i];
                    var x = PADDING_LEFT + (s.t - timelineOffset) * timeScale;
                    var val = getPoint(s);
                    if (val == null) continue;
                    points.push({ x: x, y: val });
                }
                return points;
            }
            function drawAreaAndLine(points, fillStyle, strokeStyle, fillAlpha) {
                if (points.length === 0) return;
                var x0 = points[0].x, xLast = points[points.length - 1].x;
                // Area: baseline -> line -> baseline (transparent fill)
                ctx.beginPath();
                ctx.moveTo(x0, baseline);
                if (useSmoothCurves && points.length > 1) {
                    for (var k = 0; k < points.length; k++) {
                        if (k === 0) ctx.lineTo(points[0].x, points[0].y);
                        else {
                            var p0 = points[k - 1], p1 = points[k];
                            var cpx = (p0.x + p1.x) / 2, cpy = (p0.y + p1.y) / 2;
                            ctx.quadraticCurveTo(cpx, cpy, p1.x, p1.y);
                        }
                    }
                } else {
                    for (var k = 0; k < points.length; k++) ctx.lineTo(points[k].x, points[k].y);
                }
                ctx.lineTo(xLast, baseline);
                ctx.closePath();
                ctx.fillStyle = fillStyle;
                ctx.globalAlpha = fillAlpha;
                ctx.fill();
                ctx.globalAlpha = 1.0;
                // Line on top (full opacity so curve is visible)
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                if (useSmoothCurves && points.length > 1) {
                    for (var k = 1; k < points.length; k++) {
                        var p0 = points[k - 1], p1 = points[k];
                        var cpx = (p0.x + p1.x) / 2, cpy = (p0.y + p1.y) / 2;
                        ctx.quadraticCurveTo(cpx, cpy, p1.x, p1.y);
                    }
                } else {
                    for (var k = 1; k < points.length; k++) ctx.lineTo(points[k].x, points[k].y);
                }
                ctx.strokeStyle = strokeStyle;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            // Full height scale: 0% = bottom (baseline), 100% = top (laneY)
            var cpuPoints = buildPoints(function (s) {
                var cpu = s.cpu != null ? Math.max(0, Math.min(100, s.cpu)) : null;
                return cpu != null ? laneY + laneH * (1 - cpu / 100) : null;
            });
            var gpuPoints = buildPoints(function (s) {
                var gpu = s.gpu != null ? Math.max(0, Math.min(100, s.gpu)) : null;
                return gpu != null ? laneY + laneH * (1 - gpu / 100) : null;
            });
            // Draw CPU area first (transparent blue), then GPU area (transparent green); overlap blends darker
            drawAreaAndLine(cpuPoints, '#2196F3', '#2196F3', 0.45);
            drawAreaAndLine(gpuPoints, '#4CAF50', '#4CAF50', 0.45);
        }
    }
}

function updateTimelineScrollbar(maxTime, visibleTimeWindow) {
    const scrollBar = document.getElementById('timeline-scroll-bar');
    const scrollThumb = document.getElementById('timeline-scroll-thumb');

    // Show/hide scrollbar based on zoom
    if (state.timelineZoom <= 1.0) {
        scrollBar.style.display = 'none';
        return;
    }

    scrollBar.style.display = 'flex';

    // Calculate thumb width and position
    const thumbWidth = Math.max(30, (visibleTimeWindow / maxTime) * 100);
    const thumbPosition = (state.timelineOffset / maxTime) * 100;

    scrollThumb.style.width = `${thumbWidth}%`;
    scrollThumb.style.left = `${thumbPosition}%`;
}

// ===== Live Session Management =====
function deviceValueToLabel(value, type) {
    if (value === 'browser') return 'Browser WebRTC';
    if (value === 'none') return type === 'camera' ? 'None' : 'None (text only)';
    return value || '—';
}

/** Prefer actual device name from stream or selected device dropdown when config is browser. */
function getDeviceDisplayLabel(kind) {
    var d = currentConfig.devices || {};
    if (kind === 'camera') {
        if ((d.camera === 'browser' || d.camera === '') && state.previewStream) {
            var videoTracks = state.previewStream.getVideoTracks();
            if (videoTracks.length > 0 && videoTracks[0].label) return videoTracks[0].label;
        }
        if (d.camera === 'browser' || d.camera === '') return 'Default (browser)';
        if (d.camera && d.camera.indexOf('/dev/') === 0) return d.camera; // Jetson path; could resolve label from API if needed
        var camSel = document.getElementById('device-camera-list');
        if (camSel && d.camera && camSel.value === d.camera) {
            var opt = camSel.options[camSel.selectedIndex];
            if (opt && opt.textContent) return opt.textContent;
        }
        return deviceValueToLabel(d.camera, 'camera');
    }
    if (kind === 'mic') {
        if (d.microphone === 'browser') {
            if (state.voiceMicStream) {
                var vt = state.voiceMicStream.getAudioTracks();
                if (vt.length > 0 && vt[0].label) return vt[0].label;
            }
            var sel = document.getElementById('device-microphone-list');
            if (sel && state.selectedBrowserMicId && sel.value === state.selectedBrowserMicId) {
                var opt = sel.options[sel.selectedIndex];
                if (opt && opt.textContent && opt.value) return opt.textContent;
            }
            if (state.previewStream) {
                var audioTracks = state.previewStream.getAudioTracks();
                if (audioTracks.length > 0 && audioTracks[0].label) return audioTracks[0].label;
            }
            return 'Browser WebRTC';
        }
        return deviceValueToLabel(d.microphone, 'mic');
    }
    if (kind === 'speaker') {
        if (d.speaker === 'browser') {
            var sel = document.getElementById('device-speaker-list');
            if (sel && sel.value !== 'none') {
                var opt = sel.options[sel.selectedIndex];
                if (opt && opt.textContent) return opt.textContent;
            }
            return 'Browser WebRTC';
        }
        return deviceValueToLabel(d.speaker, 'speaker');
    }
    return '—';
}

function updateDeviceIndicators() {
    var cam = document.getElementById('device-indicator-camera');
    var mic = document.getElementById('device-indicator-mic');
    var spk = document.getElementById('device-indicator-speaker');
    if (cam) cam.textContent = getDeviceDisplayLabel('camera');
    if (mic) mic.textContent = getDeviceDisplayLabel('mic');
    if (spk) spk.textContent = getDeviceDisplayLabel('speaker');
}

function updateChatInputVisibility() {
    const container = document.getElementById('chat-input-container');
    if (!container) return;
    const mic = (currentConfig.devices || {}).microphone;
    const isTextOnly = mic === 'none';
    container.style.display = isTextOnly && state.isLiveSession ? 'flex' : 'none';
}

/** Stop mic waveform overlay and release AudioContext. */
function stopMicWaveform() {
    if (state.micWaveformAnimId != null) {
        cancelAnimationFrame(state.micWaveformAnimId);
        state.micWaveformAnimId = null;
    }
    if (state.micAudioContext) {
        try { state.micAudioContext.close(); } catch (e) {}
        state.micAudioContext = null;
    }
    state.micAnalyser = null;
    state.micAmplitudeBuffer = [];
    var overlay = document.getElementById('mic-waveform-overlay');
    if (overlay) overlay.style.display = 'none';
}

/** Draw last 2000ms of mic as symmetric dotted waveform (above and below center), timeline-style. */
function drawMicWaveform() {
    var overlay = document.getElementById('mic-waveform-overlay');
    var canvas = document.getElementById('mic-waveform-canvas');
    if (!overlay || !canvas || !state.micAnalyser || !state.previewStream) return;

    var buf = new Uint8Array(state.micAnalyser.fftSize);
    state.micAnalyser.getByteTimeDomainData(buf);
    var max = 0;
    for (var i = 0; i < buf.length; i++) {
        var v = Math.abs(buf[i] - 128);
        if (v > max) max = v;
    }

    var ring = state.micAmplitudeBuffer;
    var cap = 120; /* 2000ms at ~60fps */
    if (ring.length >= cap) ring.shift();
    ring.push(max);

    var w = canvas.width = canvas.offsetWidth;
    var h = canvas.height = canvas.offsetHeight;
    if (w <= 0 || h <= 0) { state.micWaveformAnimId = requestAnimationFrame(drawMicWaveform); return; }
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    if (ring.length < 2) { state.micWaveformAnimId = requestAnimationFrame(drawMicWaveform); return; }

    var margin = 30;
    var drawW = Math.max(0, w - margin * 2);
    var centerY = h / 2;
    var scale = (h / 2) * 0.8 / 128;
    var radius = 6;

    ctx.save();
    var bgX = margin;
    var bgY = 0;
    var bgW = drawW;
    var bgH = h;
    ctx.beginPath();
    ctx.moveTo(bgX + radius, bgY);
    ctx.lineTo(bgX + bgW - radius, bgY);
    ctx.quadraticCurveTo(bgX + bgW, bgY, bgX + bgW, bgY + radius);
    ctx.lineTo(bgX + bgW, bgY + bgH - radius);
    ctx.quadraticCurveTo(bgX + bgW, bgY + bgH, bgX + bgW - radius, bgY + bgH);
    ctx.lineTo(bgX + radius, bgY + bgH);
    ctx.quadraticCurveTo(bgX, bgY + bgH, bgX, bgY + bgH - radius);
    ctx.lineTo(bgX, bgY + radius);
    ctx.quadraticCurveTo(bgX, bgY, bgX + radius, bgY);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fill();
    ctx.restore();

    var green = (getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio') || '').trim() || '#76B900';
    ctx.strokeStyle = green;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 3]);
    var step = ring.length > 1 ? drawW / (ring.length - 1) : 0;
    for (var j = 0; j < ring.length; j++) {
        var x = margin + j * step;
        var yTop = centerY - ring[j] * scale;
        var yBottom = centerY + ring[j] * scale;
        ctx.beginPath();
        ctx.moveTo(x, yTop);
        ctx.lineTo(x, yBottom);
        ctx.stroke();
    }
    state.micWaveformAnimId = requestAnimationFrame(drawMicWaveform);
}

/** Start mic waveform overlay when we have a stream with audio. Uses last 2000ms (120 samples at 60fps). */
function startMicWaveform(stream) {
    stopMicWaveform();
    if (!stream || stream.getAudioTracks().length === 0) return;
    var overlay = document.getElementById('mic-waveform-overlay');
    var canvas = document.getElementById('mic-waveform-canvas');
    if (!overlay || !canvas) return;
    try {
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        state.micAudioContext = ctx;
        var src = ctx.createMediaStreamSource(stream);
        var analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.5;
        src.connect(analyser);
        state.micAnalyser = analyser;
        state.micAmplitudeBuffer = [];
        overlay.style.display = 'block';
        drawMicWaveform();
    } catch (e) {
        console.warn('Mic waveform failed:', e);
    }
}

/** Stop camera/mic preview stream and clear video element. Call on STOP or when leaving live session. */
function stopPreviewStream() {
    stopMicWaveform();
    if (state.previewStream) {
        state.previewStream.getTracks().forEach(function (t) { t.stop(); });
        state.previewStream = null;
    }
    const videoFeed = document.getElementById('video-feed');
    if (videoFeed) {
        videoFeed.srcObject = null;
        videoFeed.src = '';
    }
}

/**
 * Start preview: browser camera (getUserMedia) and/or Jetson camera (server MJPEG stream), plus browser mic.
 */
function startPreviewStream() {
    if (!state.isLiveSession || state.sessionState !== 'setup') return;
    const d = currentConfig.devices || {};
    const isJetsonCamera = (d.camera && typeof d.camera === 'string' && d.camera.indexOf('/dev/') === 0);
    const wantJetsonVideo = (d.camera !== 'none' && d.camera != null && d.camera !== undefined && isJetsonCamera);
    const wantBrowserVideo = (d.camera !== 'none' && d.camera != null && d.camera !== undefined && !isJetsonCamera);
    const wantAudio = d.microphone !== 'none' && d.microphone != null;
    if (!wantJetsonVideo && !wantBrowserVideo && !wantAudio) {
        stopPreviewStream();
        const videoFeed = document.getElementById('video-feed');
        const imagePlaceholder = document.getElementById('image-placeholder');
        if (videoFeed) {
            videoFeed.src = '';
            videoFeed.srcObject = null;
            videoFeed.style.display = 'none';
        }
        if (imagePlaceholder) imagePlaceholder.style.display = 'flex';
        return;
    }

    stopPreviewStream();
    const videoFeed = document.getElementById('video-feed');
    const imagePlaceholder = document.getElementById('image-placeholder');

    if (wantJetsonVideo && videoFeed) {
        var deviceParam = (d.camera && d.camera !== '') ? encodeURIComponent(d.camera) : '';
        videoFeed.src = getApiBase() + '/api/camera/stream?device=' + deviceParam;
        videoFeed.style.display = 'block';
        if (imagePlaceholder) imagePlaceholder.style.display = 'none';
    } else if (!wantBrowserVideo && videoFeed) {
        videoFeed.src = '';
        videoFeed.srcObject = null;
        videoFeed.style.display = 'none';
        if (imagePlaceholder) imagePlaceholder.style.display = 'flex';
    }

    var needGetUserMedia = wantBrowserVideo || wantAudio;
    if (!needGetUserMedia) {
        updateDeviceIndicators();
        if (wantAudio) {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                updateDeviceIndicators();
                return;
            }
            var audioOnlyConstraint = state.selectedBrowserMicId ? { deviceId: { exact: state.selectedBrowserMicId } } : true;
            navigator.mediaDevices.getUserMedia({ video: false, audio: audioOnlyConstraint })
                .then(function (stream) {
                    if (!state.isLiveSession || state.sessionState !== 'setup') { stream.getTracks().forEach(function (t) { t.stop(); }); return; }
                    state.previewStream = stream;
                    updateDeviceIndicators();
                    if (stream.getAudioTracks().length > 0) startMicWaveform(stream);
                })
                .catch(function (err) {
                    console.error('getUserMedia (mic) failed:', err);
                    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') showMicrophonePermissionDeniedHint();
                    updateDeviceIndicators();
                });
        }
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('getUserMedia not available (need HTTPS or localhost)');
        updateDeviceIndicators();
        return;
    }

    var videoConstraint = wantBrowserVideo ? (state.selectedBrowserCameraId ? { deviceId: { exact: state.selectedBrowserCameraId } } : true) : false;
    var audioConstraint = wantAudio ? (state.selectedBrowserMicId ? { deviceId: { exact: state.selectedBrowserMicId } } : true) : false;
    navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: audioConstraint })
        .then(function (stream) {
            if (!state.isLiveSession || state.sessionState !== 'setup') {
                stream.getTracks().forEach(function (t) { t.stop(); });
                return;
            }
            state.previewStream = stream;
            if (wantBrowserVideo && videoFeed && stream.getVideoTracks().length > 0) {
                videoFeed.src = '';
                videoFeed.srcObject = stream;
                videoFeed.style.display = 'block';
                if (imagePlaceholder) imagePlaceholder.style.display = 'none';
            }
            updateDeviceIndicators();
            if (stream.getAudioTracks().length > 0) startMicWaveform(stream);
        })
        .catch(function (err) {
            console.error('getUserMedia failed:', err);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                showMicrophonePermissionDeniedHint();
            }
            updateDeviceIndicators();
        });
}

function updateLiveSessionUI() {
    updateConfigPanelState();
    const deviceControls = document.getElementById('device-controls-container');
    const deviceTags = document.getElementById('device-tags');
    const videoFeed = document.getElementById('video-feed');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const startOverlay = document.getElementById('start-session-overlay');
    const previewImage = document.getElementById('preview-image');
    const sessionTitle = document.getElementById('session-title');
    const sessionStats = document.getElementById('session-stats');
    const startBtn = document.getElementById('start-session-btn');
    const stopBtn = document.getElementById('stop-session-btn');

    const sessionImageEl = document.getElementById('session-image');
    if (sessionImageEl && state.isLiveSession) sessionImageEl.style.display = '';

    if (state.isLiveSession) {
        deviceControls.style.display = 'flex';
        updateDeviceIndicators();
        if (deviceTags) deviceTags.style.display = 'flex';

        const now = new Date();
        sessionTitle.textContent = `Live Session - ${now.toLocaleTimeString()}`;

        if (state.sessionState === 'setup') {
            document.getElementById('new-session-btn')?.classList.add('new-session-btn--highlight');
            document.getElementById('config-panel')?.classList.add('config-panel--start-ready');
            if (sessionStats) sessionStats.innerHTML = '';
            if (startOverlay) startOverlay.style.display = 'flex';
            if (startBtn) startBtn.style.display = 'flex';
            if (stopBtn) stopBtn.style.display = 'none';
            startPreviewStream();
            var cam = (currentConfig.devices || {}).camera;
            var hasVideo = (cam !== 'none' && cam != null && cam !== undefined);
            if (imagePlaceholder) imagePlaceholder.style.display = hasVideo ? 'none' : 'flex';
            if (videoFeed) videoFeed.style.display = hasVideo ? 'block' : 'none';
        } else if (state.sessionState === 'live') {
            document.getElementById('new-session-btn')?.classList.remove('new-session-btn--highlight');
            document.getElementById('config-panel')?.classList.remove('config-panel--start-ready');
            if (sessionStats) sessionStats.innerHTML = '<span class="stat-value" style="color: #ef4444;"><i data-lucide="circle" class="lucide-inline" style="fill: currentColor;"></i> RECORDING</span>';
            if (imagePlaceholder) imagePlaceholder.style.display = 'none';
            if (videoFeed) videoFeed.style.display = 'block';
            if (startOverlay) startOverlay.style.display = 'none';
            if (startBtn) startBtn.style.display = 'flex';
            if (stopBtn) stopBtn.style.display = 'flex';
            renderTimeline();
            updateVoiceDebugPanel();
        } else if (state.sessionState === 'stopped') {
            document.getElementById('new-session-btn')?.classList.remove('new-session-btn--highlight');
            document.getElementById('config-panel')?.classList.remove('config-panel--start-ready');
            if (sessionStats) sessionStats.innerHTML = '<span class="stat-value" style="color: var(--text-secondary);"><i data-lucide="check-circle" class="lucide-inline"></i> Session recorded</span>';
            if (imagePlaceholder) imagePlaceholder.style.display = 'flex';
            if (videoFeed) videoFeed.style.display = 'none';
            if (startOverlay) startOverlay.style.display = 'none';
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'none';
            renderTimeline();
            if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        }
        previewImage.style.display = 'none';
        updateChatInputVisibility();
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
    } else {
        document.getElementById('new-session-btn')?.classList.remove('new-session-btn--highlight');
        document.getElementById('config-panel')?.classList.remove('config-panel--start-ready');
        stopPreviewStream();
        deviceControls.style.display = 'none';
        if (videoFeed) videoFeed.style.display = 'none';
        if (startOverlay) startOverlay.style.display = 'none';
        if (deviceTags) deviceTags.style.display = 'none';
        updateChatInputVisibility();
    }
}

function updateHistoricalSessionPreview() {
    const sessionMeta = document.getElementById('session-meta');
    const sessionTitle = document.getElementById('session-title');
    const sessionStats = document.getElementById('session-stats');
    const previewImage = document.getElementById('preview-image');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const videoFeed = document.getElementById('video-feed');

    console.log('updateHistoricalSessionPreview called', {
        isLiveSession: state.isLiveSession,
        hasSelectedSession: !!state.selectedSession,
    });

    // Show historical session data when:
    // - Not a live session
    // - A session is selected
    if (!state.isLiveSession && state.selectedSession) {
        console.log('Showing historical session preview');

        const session = state.selectedSession;
        const metrics = session.metrics || {};

        // Update session title
        sessionTitle.textContent = session.name || 'Unnamed Session';

        // Update session stats (horizontal: Date | Turns | TTL) — parse as UTC, show local
        const dateObj = parseSessionDate(session.created_at);
        const date = dateObj ? dateObj.toLocaleDateString(undefined, { dateStyle: 'medium' }) : '';
        const turns = metrics.total_turns || 0;
        const ttl = formatLatency(metrics.avg_ttl);

        sessionStats.innerHTML = `
            <span class="session-stat-item">
                <span class="stat-label">Date:</span>
                <span class="stat-value">${date}</span>
            </span>
            <span class="session-stat-item">
                <span class="stat-label">Turns:</span>
                <span class="stat-value">${turns}</span>
            </span>
            <span class="session-stat-item">
                <span class="stat-label">Avg TTL:</span>
                <span class="stat-value">${ttl}</span>
            </span>
        `;

        // Hide session image area for recorded session to maximize chat history
        videoFeed.style.display = 'none';
        const sessionImageEl = document.getElementById('session-image');
        if (sessionImageEl) sessionImageEl.style.display = 'none';

        if (session.thumbnail) {
            previewImage.src = session.thumbnail;
            previewImage.style.display = 'block';
            imagePlaceholder.style.display = 'none';
        } else {
            previewImage.style.display = 'none';
            imagePlaceholder.style.display = 'none';
        }

        sessionMeta.style.display = 'flex';
    } else if (!state.selectedSession) {
        sessionTitle.textContent = 'Select a session';
        sessionStats.innerHTML = '';
        previewImage.style.display = 'none';
        videoFeed.style.display = 'none';
        imagePlaceholder.style.display = 'flex';
        const sessionImageEl = document.getElementById('session-image');
        if (sessionImageEl) sessionImageEl.style.display = '';
    } else {
        // Live session or other: ensure session image area is visible
        const sessionImageEl = document.getElementById('session-image');
        if (sessionImageEl) sessionImageEl.style.display = '';
    }
}

function startNewSession() {
    // Create new live session
    state.isLiveSession = true;
    state.sessionState = 'setup';
    state.selectedSession = null;

    // Reset config to defaults
    currentConfig = JSON.parse(JSON.stringify(defaultConfig));

    // Clear chat history
    document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p>Configure your session and click START to begin</p>
        </div>
    `;

    // Initialize timeline so it doesn't keep showing the previous session's data
    initTimeline();

    // Show Devices tab by default so user can pick mic/camera and start right away
    state.activeConfigTab = 'device';
    renderSessionList();
    renderConfig();
    document.querySelectorAll('.config-tab').forEach(tab => {
        tab.setAttribute('aria-selected', tab.dataset.tab === 'device');
        tab.classList.toggle('active', tab.dataset.tab === 'device');
    });
    setTimeout(function () {
        if (document.getElementById('device-microphone-list')) populateAllDeviceDropdowns();
    }, 0);

    updateLiveSessionUI();

    console.log('New session created - Setup mode');
}

function scheduleLiveTimelineTick() {
    if (state.isLiveSession && state.sessionState === 'live') {
        state.liveTimelineRafId = requestAnimationFrame(function () {
            renderTimeline();
            scheduleLiveTimelineTick();
        });
    }
}

const LIVE_SYSTEM_STATS_POLL_MS = 100;
const LIVE_SYSTEM_STATS_MAX_SAMPLES = 900; // ~90 s at 100ms

function startLiveSystemStatsPoll() {
    stopLiveSystemStatsPoll();
    state.liveSystemStatsPollIntervalId = setInterval(function () {
        if (state.liveSessionStartTime <= 0 || !state.liveSystemStats) return;
        fetch(getApiBase() + '/api/system/stats')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var t = (Date.now() / 1000) - state.liveSessionStartTime;
                state.liveSystemStats.push({
                    t: t,
                    cpu: data.cpu_percent != null ? data.cpu_percent : null,
                    gpu: data.gpu_percent != null ? data.gpu_percent : null
                });
                if (state.liveSystemStats.length > LIVE_SYSTEM_STATS_MAX_SAMPLES)
                    state.liveSystemStats = state.liveSystemStats.slice(-LIVE_SYSTEM_STATS_MAX_SAMPLES);
            })
            .catch(function () {});
    }, LIVE_SYSTEM_STATS_POLL_MS);
}

function stopLiveSystemStatsPoll() {
    if (state.liveSystemStatsPollIntervalId != null) {
        clearInterval(state.liveSystemStatsPollIntervalId);
        state.liveSystemStatsPollIntervalId = null;
    }
}

function getApiBase() {
    return window.location.origin;
}

function startSessionRecording() {
    if (state.sessionState !== 'setup') return;

    console.log('[Voice] Start session recording (WebSocket + mic)');
    state.liveTimelineEvents = [];
    state.voiceMessageLog = [];
    state.liveChatTurns = [];
    state.ttsNextStartTime = 0;
    state.liveAudioAmplitudeHistory = [];
    state.liveTtsAmplitudeHistory = [];
    state.liveSystemStats = [];
    state.liveTimelineInitialZoomSet = false;
    state.sessionState = 'live';
    scheduleLiveTimelineTick();
    updateLiveSessionUI();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + window.location.host + '/ws/voice';
    console.log('[Voice] Connecting to', wsUrl);
    const ws = new WebSocket(wsUrl);
    state.voiceWs = ws;

    ws.onopen = function () {
        console.log('[Voice] WebSocket connected, sending config');
        const config = {
            asr: { ...currentConfig.asr },
            llm: { ...currentConfig.llm },
            tts: { ...currentConfig.tts },
            devices: currentConfig.devices ? { ...currentConfig.devices } : {},
            app: currentConfig.app ? { ...currentConfig.app } : {}
        };
        if (config.asr.riva_server === undefined && config.asr.server) config.asr.riva_server = config.asr.server;
        if (config.tts.riva_server === undefined && config.tts.server) config.tts.riva_server = config.tts.server;
        if (config.llm.ollama_url === undefined && config.llm.api_base) config.llm.ollama_url = (config.llm.api_base || '').replace(/\/v1\/?$/, '');
        ws.send(JSON.stringify({ type: 'config', config: config }));
        console.log('[Voice] Config sent, starting mic stream');
        startVoiceMicStream();
    };

    ws.onmessage = function (ev) {
        if (typeof ev.data === 'string') {
            try {
                const msg = JSON.parse(ev.data);
                // Debug: log every message type (and event_type for events), show as JSON in UI
                const eventType = (msg.type === 'event' && msg.event && msg.event.event_type) ? msg.event.event_type : '';
                console.log('[Voice] 📥', msg.type, eventType ? ' event_type=' + eventType : '', msg);
                // For UI log, use summary for large payloads (tts_audio) so JSON stays readable
                const toLog = (msg.type === 'tts_audio' && msg.data)
                    ? { type: 'tts_audio', sample_rate: msg.sample_rate, data_length: msg.data.length }
                    : msg;
                pushVoiceMessageLog(toLog);

                if (msg.type === 'event' && msg.event) {
                    const ev = msg.event;
                    if (ev.event_type === 'session_start') {
                        state.liveSessionStartTime = Date.now() / 1000;
                        startLiveSystemStatsPoll();
                    }
                    if (ev.lane === undefined || ev.lane === null) {
                        if (ev.event_type && ev.event_type.startsWith('asr_')) ev.lane = 'speech';
                        else if (ev.event_type && ev.event_type.startsWith('llm_')) ev.lane = 'llm';
                        else if (ev.event_type && ev.event_type.startsWith('tts_')) ev.lane = 'tts';
                        else if (ev.event_type === 'session_start') ev.lane = 'system';
                    } else if (typeof ev.lane === 'string') {
                        ev.lane = ev.lane.toLowerCase();
                    }
                    state.liveTimelineEvents.push(ev);
                    renderTimeline();
                    if (ev.event_type === 'asr_final') {
                        var userText = (ev.data && ev.data.text != null) ? String(ev.data.text).trim() : '';
                        if (userText) {
                            state.liveChatTurns.push({ user: userText, assistant: '' });
                            renderLiveChat();
                        }
                    } else if (ev.event_type === 'chat') {
                        var userText = ev.user != null ? String(ev.user) : (ev.data && ev.data.user != null ? String(ev.data.user) : null);
                        var assistantText = ev.assistant != null ? String(ev.assistant) : (ev.data && ev.data.assistant != null ? String(ev.data.assistant) : '');
                        if (userText != null) {
                            var last = state.liveChatTurns[state.liveChatTurns.length - 1];
                            if (last && last.assistant === '' && last.user === userText) {
                                last.assistant = assistantText;
                            } else {
                                state.liveChatTurns.push({ user: userText, assistant: assistantText });
                            }
                            renderLiveChat();
                        }
                    }
                } else if (msg.type === 'tts_start') {
                    if (state.ttsAudioContext) {
                        if (state.ttsAudioContext.state === 'suspended') state.ttsAudioContext.resume();
                    }
                    // Do not reset ttsNextStartTime here: schedule new TTS after previous playback ends to avoid overlap
                } else if (msg.type === 'tts_audio' && msg.data) {
                    playTtsChunk(msg.data, msg.sample_rate || 24000);
                } else if (msg.type === 'session_saved' && msg.session_id) {
                    state.lastSavedSessionId = msg.session_id;
                    loadSessions();
                } else if (msg.type === 'error') {
                    console.error('Voice pipeline error:', msg.error);
                    appendLiveChatError(msg.error);
                }
            } catch (e) {
                console.error('Parse WS message error:', e);
            }
        }
    };

    ws.onclose = function (ev) {
        console.log('[Voice] WebSocket closed: code=' + (ev && ev.code) + ' reason=' + (ev && ev.reason) + ' clean=' + (ev && ev.wasClean));
        state.voiceWs = null;
        stopVoiceMicStream();
        if (state.sessionState === 'live') {
            stopLiveSystemStatsPoll();
            if (state.liveTimelineRafId != null) {
                cancelAnimationFrame(state.liveTimelineRafId);
                state.liveTimelineRafId = null;
            }
            state.sessionState = 'stopped';
            // Keep isLiveSession true so timeline keeps showing waveforms/ASR until user selects another session
            updateLiveSessionUI();
        }
    };

    ws.onerror = function () {
        console.error('[Voice] WebSocket error');
    };

    updateVoiceDebugPanel();
    document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p><i data-lucide="circle" class="lucide-inline" style="fill: #ef4444;"></i> Session is LIVE - Start talking!</p>
        </div>
    `;
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

function appendLiveChatError(text) {
    const chatEl = document.getElementById('chat-history');
    if (!chatEl) return;
    const wrap = document.createElement('div');
    wrap.className = 'chat-message ai';
    wrap.innerHTML = '<div class="chat-avatar"><i data-lucide="bot" class="lucide-inline"></i></div><div class="chat-bubble"><div class="chat-text error">' + escapeHtml(text) + '</div></div>';
    chatEl.appendChild(wrap);
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

function renderLiveChat() {
    const chatEl = document.getElementById('chat-history');
    if (!chatEl) return;
    if (state.liveChatTurns.length === 0) {
        chatEl.innerHTML = '<div class="empty-state"><p><i data-lucide="circle" class="lucide-inline" style="fill: #ef4444;"></i> Session is LIVE - Start talking!</p></div>';
    } else {
        chatEl.innerHTML = state.liveChatTurns.map(t => {
            var assistantDisplay = t.assistant ? escapeHtml(t.assistant) : '<span class="chat-placeholder">…</span>';
            return `
            <div class="chat-message user">
                <div class="chat-avatar"><i data-lucide="user" class="lucide-inline"></i></div>
                <div class="chat-bubble"><div class="chat-text">${escapeHtml(t.user)}</div></div>
            </div>
            <div class="chat-message ai">
                <div class="chat-avatar"><i data-lucide="bot" class="lucide-inline"></i></div>
                <div class="chat-bubble"><div class="chat-text">${assistantDisplay}</div></div>
            </div>
        `;
        }).join('');
    }
    chatEl.scrollTop = chatEl.scrollHeight;
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

const TARGET_SAMPLE_RATE = 16000;

function startVoiceMicStream() {
    // Request a fresh microphone stream. Use selected device from device list if set, else browser default.
    stopVoiceMicStream();
    var audioConstraint = state.selectedBrowserMicId
        ? { deviceId: { exact: state.selectedBrowserMicId } }
        : true;
    console.log('[Voice] Requesting getUserMedia({ audio: ... })', state.selectedBrowserMicId ? 'deviceId: ' + state.selectedBrowserMicId : 'default');
    navigator.mediaDevices.getUserMedia({ audio: audioConstraint }).then(function (s) {
        state.voiceMicStream = s;
        const label = s.getAudioTracks()[0]?.label || 'default';
        console.log('[Voice] getUserMedia ok, mic:', label, ', connecting PCM to WS');
        updateDeviceIndicators();
        connectPcmToWs(s);
    }).catch(function (e) {
        console.error('[Voice] getUserMedia for voice failed:', e);
    });
}

const VOICE_MESSAGE_LOG_MAX = 20;

function pushVoiceMessageLog(obj) {
    try {
        const line = JSON.stringify(obj, null, 2);
        state.voiceMessageLog.push(line);
        if (state.voiceMessageLog.length > VOICE_MESSAGE_LOG_MAX) state.voiceMessageLog.shift();
        updateVoiceDebugPanel();
    } catch (e) {
        state.voiceMessageLog.push('[stringify error] ' + String(e));
        updateVoiceDebugPanel();
    }
}

function updateVoiceDebugPanel() {
    const panel = document.getElementById('voice-debug-panel');
    const logEl = document.getElementById('voice-debug-log');
    if (!panel || !logEl) return;
    const show = (state.isLiveSession && state.sessionState === 'live') || uiSettings.showDebugInfo;
    panel.style.display = show ? 'flex' : 'none';
    logEl.textContent = state.voiceMessageLog.length ? state.voiceMessageLog.join('\n\n---\n\n') : '(no messages yet)';
}

function connectPcmToWs(stream) {
    const ws = state.voiceWs;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.warn('[Voice] connectPcmToWs: WS not open, skipping');
        return;
    }
    // Capture at 16 kHz directly (no JS resampling). Browser resamples device to context rate if needed.
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SAMPLE_RATE });
    console.log('[Voice] AudioContext state:', ctx.state, 'sampleRate:', ctx.sampleRate);
    const src = ctx.createMediaStreamSource(stream);
    const bufferLen = 2048; // 2048 @ 16 kHz = 128 ms per chunk
    const processor = ctx.createScriptProcessor(bufferLen, 1, 1);
    let pcmChunkCount = 0;

    if (ctx.state === 'suspended') {
        ctx.resume().then(function () {
            console.log('[Voice] AudioContext resumed, mic streaming to server');
        }).catch(function (e) {
            console.warn('[Voice] AudioContext resume failed:', e);
        });
    }

    var lastClientAmpLogTime = 0;
    processor.onaudioprocess = function (e) {
        if (ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const v = Math.max(-1, Math.min(1, input[i]));
            pcmData[i] = v < 0 ? v * 0x8000 : v * 0x7FFF;
        }
        // Debug: log client-side mic RMS ~every 1s (INVESTIGATE_USER_AMPLITUDE_ARTIFACT.md)
        var nowSec = Date.now() / 1000;
        var sumSq = 0;
        for (var i = 0; i < input.length; i++) sumSq += input[i] * input[i];
        var clientRms = input.length ? Math.sqrt(sumSq / input.length) : 0;
        var clientAmpScaled = Math.min(100, clientRms * 400);
        if (state.liveSessionStartTime > 0 && nowSec - lastClientAmpLogTime >= 1.0) {
            console.warn('[user_amplitude] client: buffer_len=' + input.length + ' float_rms=' + clientRms.toFixed(4) + ' amp_0_100=' + clientAmpScaled.toFixed(2));
            lastClientAmpLogTime = nowSec;
        }
        // Log every high amplitude on client so we can compare with server (false green on replay = high values in saved timeline)
        if (state.liveSessionStartTime > 0 && clientAmpScaled >= 20) {
            var sessionT = nowSec - state.liveSessionStartTime;
            console.warn('[user_amplitude_high] client: session_t=' + sessionT.toFixed(2) + 's amp_0_100=' + clientAmpScaled.toFixed(2) + ' (same buffer we send to server)');
        }
        ws.send(pcmData.buffer);
        pcmChunkCount++;
        if (pcmChunkCount % 50 === 0) console.log('[Voice] Sent', pcmChunkCount, 'PCM chunks');
        // Feed live AUDIO lane waveform: RMS amplitude 0–100, timestamp relative to session start
        if (state.liveSessionStartTime > 0 && state.liveAudioAmplitudeHistory) {
            let sumSq = 0;
            for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i];
            const rms = Math.min(100, Math.sqrt(sumSq / input.length) * 400);
            const ts = (Date.now() / 1000) - state.liveSessionStartTime;
            state.liveAudioAmplitudeHistory.push({ timestamp: ts, amplitude: rms });
            const maxSamples = 400;
            if (state.liveAudioAmplitudeHistory.length > maxSamples)
                state.liveAudioAmplitudeHistory = state.liveAudioAmplitudeHistory.slice(-maxSamples);
        }
    };
    src.connect(processor);
    processor.connect(ctx.destination);
    state.voicePcmNode = { processor, src, ctx };
}

function stopVoiceMicStream() {
    if (state.voicePcmNode) {
        try {
            state.voicePcmNode.src.disconnect();
            state.voicePcmNode.processor.disconnect();
        } catch (e) {}
        state.voicePcmNode = null;
    }
    if (state.voiceMicStream && state.voiceMicStream !== state.previewStream) {
        state.voiceMicStream.getTracks().forEach(function (t) { t.stop(); });
        state.voiceMicStream = null;
    }
}

function playTtsChunk(base64Data, sampleRate) {
    const binary = atob(base64Data);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    const samples = new Int16Array(bytes.buffer);
    if (!state.ttsAudioContext) {
        state.ttsAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        state.ttsNextStartTime = 0;
    }
    const ctx = state.ttsAudioContext;
    if (ctx.state === 'suspended') ctx.resume();
    const numSamples = samples.length;
    const buffer = ctx.createBuffer(1, numSamples, sampleRate);
    const ch = buffer.getChannelData(0);
    for (let i = 0; i < numSamples; i++) ch[i] = samples[i] / (samples[i] < 0 ? 0x8000 : 0x7FFF);
    const duration = numSamples / sampleRate;
    let startTime = state.ttsNextStartTime;
    if (startTime < ctx.currentTime) startTime = ctx.currentTime;
    state.ttsNextStartTime = startTime + duration;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start(startTime);
    // Record TTS segment for purple AI waveform on AUDIO lane using actual playback time (when speaker plays)
    if (state.liveSessionStartTime > 0 && state.liveTtsAmplitudeHistory) {
        var nowSec = Date.now() / 1000;
        var sessionStart = state.liveSessionStartTime;
        var delayUntilPlay = Math.max(0, startTime - ctx.currentTime);
        var actualStartSession = (nowSec - sessionStart) + delayUntilPlay;
        var actualEndSession = actualStartSession + duration;
        var sumSq = 0;
        for (var i = 0; i < numSamples; i++) sumSq += ch[i] * ch[i];
        var rms = Math.min(100, Math.sqrt(sumSq / numSamples) * 400);
        state.liveTtsAmplitudeHistory.push({
            startTime: actualStartSession,
            endTime: actualEndSession,
            amplitude: rms
        });
        var nowSession = nowSec - sessionStart;
        while (state.liveTtsAmplitudeHistory.length > 0 && state.liveTtsAmplitudeHistory[0].endTime < nowSession - 20)
            state.liveTtsAmplitudeHistory.shift();
    }
}

function stopSessionRecording() {
    if (state.sessionState !== 'live') return;

    stopLiveSystemStatsPoll();

    if (state.liveTimelineRafId != null) {
        cancelAnimationFrame(state.liveTimelineRafId);
        state.liveTimelineRafId = null;
    }
    if (state.voiceWs && state.voiceWs.readyState === WebSocket.OPEN) {
        state.voiceWs.send(JSON.stringify({
            type: 'stop',
            system_stats: state.liveSystemStats || [],
            tts_playback_segments: state.liveTtsAmplitudeHistory || [],
            audio_amplitude_history: state.liveAudioAmplitudeHistory || []
        }));
        state.voiceWs.close();
        state.voiceWs = null;
    }
    stopVoiceMicStream();
    stopPreviewStream();
    state.sessionState = 'stopped';
    // Keep isLiveSession true so timeline still shows and can be zoomed/panned

    document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p>✅ Session saved! Check the session list.</p>
        </div>
    `;

    updateLiveSessionUI();
    loadSessions(); // Refresh session list so new recording appears
    renderTimeline(); // One frame so zoom/scroll is available
    // Reload session list again after server has had time to write the file (client may close before session_saved is received)
    setTimeout(function () { loadSessions(); }, 1500);
}

// ===== Event Handlers =====
function setupEventHandlers() {
    console.log('setupEventHandlers() called');

    // Theme: Auto / Light / Dark (matches Live RIVA WebUI – Lucide icons)
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    function applyTheme(theme) {
        const html = document.documentElement;
        if (theme === 'light') {
            html.setAttribute('data-theme', 'light');
            if (themeIcon) themeIcon.innerHTML = '<i data-lucide="sun"></i>';
            if (themeText) themeText.textContent = 'Light';
        } else if (theme === 'dark') {
            html.setAttribute('data-theme', 'dark');
            if (themeIcon) themeIcon.innerHTML = '<i data-lucide="moon"></i>';
            if (themeText) themeText.textContent = 'Dark';
        } else {
            const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
            html.setAttribute('data-theme', prefersLight ? 'light' : 'dark');
            if (themeIcon) themeIcon.innerHTML = '<i data-lucide="monitor"></i>';
            if (themeText) themeText.textContent = 'Auto';
        }
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        if (state.selectedSession) renderTimeline();
    }

    try {
        themeToggle.addEventListener('click', () => {
            const currentTheme = localStorage.getItem('theme') || 'auto';
            const nextTheme = currentTheme === 'auto' || !currentTheme ? 'light' : currentTheme === 'light' ? 'dark' : 'auto';
            if (nextTheme === 'auto') localStorage.removeItem('theme');
            else localStorage.setItem('theme', nextTheme);
            applyTheme(nextTheme);
        });
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light' || savedTheme === 'dark') applyTheme(savedTheme);
        else applyTheme('auto');
        const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: light)');
        colorSchemeQuery.addEventListener('change', () => {
            if (!localStorage.getItem('theme')) applyTheme('auto');
        });
        console.log('✓ Theme toggle (Auto/Light/Dark) attached');
    } catch (e) {
        console.error('✗ Error attaching theme toggle:', e);
    }

    // Config tabs (NVIDIA-style: aria-selected for accessibility)
    document.querySelectorAll('.config-tab').forEach(tabEl => {
        tabEl.addEventListener('click', () => {
            document.querySelectorAll('.config-tab').forEach(t => {
                t.classList.remove('active');
                t.setAttribute('aria-selected', 'false');
            });
            tabEl.classList.add('active');
            tabEl.setAttribute('aria-selected', 'true');
            state.activeConfigTab = tabEl.dataset.tab;
            renderConfig();
        });
    });

    // Timeline zoom controls
    document.getElementById('timeline-zoom-in').addEventListener('click', () => {
        state.timelineZoom = Math.min(10, state.timelineZoom * 1.5); // Increased zoom factor
        renderTimeline();
    });

    document.getElementById('timeline-zoom-out').addEventListener('click', () => {
        state.timelineZoom = Math.max(0.5, state.timelineZoom / 1.5);
        // Reset offset when zooming out fully
        if (state.timelineZoom <= 1.0) {
            state.timelineZoom = 1.0;
            state.timelineOffset = 0;
        }
        renderTimeline();
    });

    document.getElementById('timeline-reset').addEventListener('click', () => {
        state.timelineZoom = 1.0;
        state.timelineOffset = 0;
        renderTimeline();
    });

    // Audio gain (UI-only) for timeline waveform
    const userGainEl = document.getElementById('timeline-user-audio-gain');
    const aiGainEl = document.getElementById('timeline-ai-audio-gain');
    if (userGainEl) {
        userGainEl.value = String(uiSettings.userAudioGain);
        userGainEl.addEventListener('change', () => {
            uiSettings.userAudioGain = parseInt(userGainEl.value, 10);
            saveUISettings();
            renderTimeline();
        });
    }
    if (aiGainEl) {
        aiGainEl.value = String(uiSettings.aiAudioGain);
        aiGainEl.addEventListener('change', () => {
            uiSettings.aiAudioGain = parseInt(aiGainEl.value, 10);
            saveUISettings();
            renderTimeline();
        });
    }

    // Timeline canvas scroll/pan when zoomed (allowed for selected session or just-stopped live session)
    const canvas = document.getElementById('timeline-canvas');
    canvas.addEventListener('wheel', (e) => {
        const hasRecordedTimeline = state.isLiveSession && state.sessionState === 'stopped' && state.liveTimelineEvents && state.liveTimelineEvents.length > 0;
        const hasSelectedTimeline = state.selectedSession && (state.selectedSession.timeline && state.selectedSession.timeline.events || state.selectedSession.timeline);
        const timelineArray = hasRecordedTimeline ? state.liveTimelineEvents : (hasSelectedTimeline ? (Array.isArray(hasSelectedTimeline) ? hasSelectedTimeline : (hasSelectedTimeline.events || [])) : []);
        if (timelineArray.length === 0 || state.timelineZoom <= 1.0) return;

        e.preventDefault();
        const maxTime = state.timelineDuration || (timelineArray.length ? Math.max(0.1, ...timelineArray.map(ev => ev.timestamp || ev.end_time || 0)) : 0.1);
        const rect = canvas.getBoundingClientRect();
        const viewportWidth = rect.width - 100 - 20; // PADDING_LEFT, PADDING_RIGHT
        const visibleTimeWindow = viewportWidth / (viewportWidth / maxTime * state.timelineZoom);

        const scrollSpeed = 0.5;
        state.timelineOffset += (e.deltaY > 0 ? scrollSpeed : -scrollSpeed);
        state.timelineOffset = Math.max(0, Math.min(maxTime - visibleTimeWindow, state.timelineOffset));

        renderTimeline();
    });

    // Window resize
    window.addEventListener('resize', () => {
        if (state.selectedSession) {
            renderTimeline();
        }
    });

    // New session button
    document.getElementById('new-session-btn').addEventListener('click', () => {
        startNewSession();
    });

    // Start session recording
    document.getElementById('start-session-btn').addEventListener('click', () => {
        startSessionRecording();
    });

    // Stop session recording
    document.getElementById('stop-session-btn').addEventListener('click', () => {
        stopSessionRecording();
    });

    // Chat text input (when Mic = None - Text Only): send on button or Enter
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    if (chatInput && chatSendBtn) {
        function sendChatText() {
            const text = (chatInput.value || '').trim();
            if (!text) return;
            const historyEl = document.getElementById('chat-history');
            const emptyState = historyEl.querySelector('.empty-state');
            if (emptyState) emptyState.remove();
            const msgEl = document.createElement('div');
            msgEl.className = 'chat-message user';
            msgEl.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
            historyEl.appendChild(msgEl);
            historyEl.scrollTop = historyEl.scrollHeight;
            chatInput.value = '';
            if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
            // TODO: send to LLM and append assistant reply
        }
        chatSendBtn.addEventListener('click', sendChatText);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatText();
            }
        });
    }
}

// ===== Timeline Panel Resizing =====
function initTimelineResize() {
    const handle = document.getElementById('timeline-resize-handle');
    const panel = document.getElementById('timeline-panel');
    const mainPanel = panel.parentElement; // .main-panel

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = panel.offsetHeight;
        handle.classList.add('dragging');
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        // Calculate new height (drag up = increase height, drag down = decrease height)
        const deltaY = startY - e.clientY; // Inverted because timeline is at bottom
        const newHeight = Math.max(200, Math.min(600, startHeight + deltaY));

        panel.style.height = `${newHeight}px`;

        // Re-render timeline to adjust canvas size
        if (state.selectedSession) {
            renderTimeline();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            handle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

// ===== Timeline Panel Resizing =====
function initTimelineResize() {
    const handle = document.getElementById('timeline-resize-handle');
    const panel = document.getElementById('timeline-panel');
    const mainPanel = panel.parentElement; // .main-panel

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = panel.offsetHeight;
        handle.classList.add('dragging');
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        // Calculate new height (drag up = increase height, drag down = decrease height)
        const deltaY = startY - e.clientY; // Inverted because timeline is at bottom
        const newHeight = Math.max(200, Math.min(600, startHeight + deltaY));

        panel.style.height = `${newHeight}px`;

        // Re-render timeline to adjust canvas size
        if (state.selectedSession) {
            renderTimeline();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            handle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

// ===== Timeline Canvas Drag-to-Pan =====
function initTimelineCanvasPan() {
    const canvas = document.getElementById('timeline-canvas');
    if (!canvas) return;

    let isPanning = false;
    let startX = 0;
    let startOffset = 0;

    canvas.addEventListener('mousedown', (e) => {
        // Only pan if zoomed in and timeline exists
        if (state.timelineZoom <= 1.0 || !state.timelineDuration) return;

        isPanning = true;
        startX = e.clientX;
        startOffset = state.timelineOffset;
        canvas.style.cursor = 'grabbing';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isPanning || !state.timelineDuration) return;

        // Calculate how far the mouse moved (in pixels)
        const deltaX = e.clientX - startX;

        // Convert pixels to timeline seconds based on current zoom
        const rect = canvas.getBoundingClientRect();
        const PADDING_LEFT = 100;
        const PADDING_RIGHT = 20;
        const visibleWidth = rect.width - PADDING_LEFT - PADDING_RIGHT;

        // Calculate time scale: how many pixels per second
        const baseTimeScale = visibleWidth / state.timelineDuration;
        const timeScale = baseTimeScale * state.timelineZoom;

        // Convert pixel movement to time delta (negative for natural scrolling)
        const deltaTime = -deltaX / timeScale;

        // Calculate new offset with proper clamping
        const visibleDuration = state.timelineDuration / state.timelineZoom;
        const maxOffset = Math.max(0, state.timelineDuration - visibleDuration);
        const newOffset = startOffset + deltaTime;

        state.timelineOffset = Math.max(0, Math.min(maxOffset, newOffset));

        // Re-render only if offset actually changed
        if (state.timelineOffset !== startOffset) {
            renderTimeline();
            updateTimelineScrollbar();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isPanning) {
            isPanning = false;
            canvas.style.cursor = state.timelineZoom > 1.0 ? 'grab' : 'default';
        }
    });

    // Update cursor based on zoom level
    canvas.addEventListener('mouseenter', () => {
        if (state.timelineZoom > 1.0 && !isPanning && state.timelineDuration) {
            canvas.style.cursor = 'grab';
        }
    });

    canvas.addEventListener('mouseleave', () => {
        if (!isPanning) {
            canvas.style.cursor = 'default';
        }
    });
}

// ===== Timeline Scrollbar Interaction =====
function initTimelineScrollbar() {
    const scrollTrack = document.getElementById('timeline-scroll-track');
    const scrollThumb = document.getElementById('timeline-scroll-thumb');

    let isDragging = false;
    let startX = 0;
    let startOffset = 0;

    // Drag the thumb
    scrollThumb.addEventListener('mousedown', (e) => {
        if (!state.selectedSession || state.timelineZoom <= 1.0) return;

        isDragging = true;
        startX = e.clientX;
        startOffset = state.timelineOffset;
        e.preventDefault();
        e.stopPropagation();
    });

    // Click on track to jump
    scrollTrack.addEventListener('mousedown', (e) => {
        if (!state.selectedSession || state.timelineZoom <= 1.0) return;
        if (e.target !== scrollTrack) return; // Ignore if clicking thumb

        const rect = scrollTrack.getBoundingClientRect();
        const clickRatio = (e.clientX - rect.left) / rect.width;
        const timeline = state.selectedSession.timeline || [];
        const maxTime = Math.max(...timeline.map(ev => ev.timestamp));

        const canvas = document.getElementById('timeline-canvas');
        const canvasRect = canvas.getBoundingClientRect();
        const viewportWidth = canvasRect.width - 100 - 20; // PADDING_LEFT - PADDING_RIGHT
        const visibleTimeWindow = viewportWidth / (viewportWidth / maxTime * state.timelineZoom);

        state.timelineOffset = Math.max(0, Math.min(maxTime - visibleTimeWindow, clickRatio * maxTime));
        renderTimeline();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const timeline = state.selectedSession.timeline || [];
        const maxTime = Math.max(...timeline.map(ev => ev.timestamp));
        const rect = scrollTrack.getBoundingClientRect();
        const deltaX = e.clientX - startX;
        const deltaRatio = deltaX / rect.width;

        const canvas = document.getElementById('timeline-canvas');
        const canvasRect = canvas.getBoundingClientRect();
        const viewportWidth = canvasRect.width - 100 - 20;
        const visibleTimeWindow = viewportWidth / (viewportWidth / maxTime * state.timelineZoom);

        state.timelineOffset = Math.max(0, Math.min(maxTime - visibleTimeWindow, startOffset + deltaRatio * maxTime));
        renderTimeline();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
        }
    });
}

// ===== Utility Functions =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatLatency(seconds) {
    if (!seconds && seconds !== 0) return 'N/A';
    return `${(seconds * 1000).toFixed(0)}ms`;
}

function formatDuration(seconds) {
    if (!seconds && seconds !== 0) return 'N/A';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
}

/** Parse session created_at (UTC). If string has no timezone, treat as UTC so UI shows correct local time. */
function parseSessionDate(dateString) {
    if (!dateString) return null;
    if (typeof dateString === 'string' && !/Z|[+-]\d{2}:?\d{2}$/.test(dateString)) {
        dateString = dateString + 'Z';
    }
    return new Date(dateString);
}

function formatDate(dateString) {
    const date = parseSessionDate(dateString);
    if (!date || isNaN(date.getTime())) return '';
    return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', hour12: true });
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded fired!');

    console.log('Loading UI settings...');
    try {
        loadUISettings();
        console.log('UI settings loaded:', uiSettings);
    } catch (error) {
        console.error('Error loading UI settings:', error);
    }

    console.log('Setting up event handlers...');
    try {
        setupEventHandlers();
        console.log('Event handlers set up successfully');
    } catch (error) {
        console.error('Error setting up event handlers:', error);
    }

    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();

    console.log('Setting up modal handlers...');
    try {
        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', openSettingsModal);
        document.getElementById('settings-modal-close').addEventListener('click', closeSettingsModal);
        document.getElementById('settings-save-btn').addEventListener('click', saveSettingsFromModal);
        document.getElementById('settings-reset-btn').addEventListener('click', resetUISettings);

        // Close modal on backdrop click
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target.id === 'settings-modal') {
                closeSettingsModal();
            }
        });

        // Voice debug panel collapse
        const voiceDebugToggle = document.getElementById('voice-debug-toggle');
        if (voiceDebugToggle) {
            voiceDebugToggle.addEventListener('click', () => {
                const panel = document.getElementById('voice-debug-panel');
                if (panel) {
                    panel.classList.toggle('collapsed');
                    voiceDebugToggle.textContent = panel.classList.contains('collapsed') ? '+' : '−';
                    voiceDebugToggle.title = panel.classList.contains('collapsed') ? 'Expand' : 'Collapse';
                }
            });
        }

        // Close modal on ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.getElementById('settings-modal');
                if (modal.classList.contains('show')) {
                    closeSettingsModal();
                }
            }
        });

        console.log('Modal handlers set up successfully');
    } catch (error) {
        console.error('Error setting up modal handlers:', error);
    }

    console.log('Initializing timeline resize...');
    try {
        initTimelineResize();
        console.log('Timeline resize initialized');
    } catch (error) {
        console.error('Error initializing timeline resize:', error);
    }

    console.log('Initializing timeline scrollbar...');
    try {
        initTimelineScrollbar();
        console.log('Timeline scrollbar initialized');
    } catch (error) {
        console.error('Error initializing timeline scrollbar:', error);
    }

    console.log('Initializing timeline canvas panning...');
    try {
        initTimelineCanvasPan();
        console.log('Timeline canvas panning initialized');
    } catch (error) {
        console.error('Error initializing timeline canvas panning:', error);
    }

    console.log('Loading sessions...');
    try {
        loadSessions();
    } catch (error) {
        console.error('Error loading sessions:', error);
    }

    updateTimelinePanelVisibility();

    // Initial state: start in "new voice chat" setup so Config is active and user can START right away
    startNewSession();
});

// ===== Future: UI State Persistence =====
// Uncomment when implementing localStorage persistence

// function saveUIState() {
//     localStorage.setItem('multi-modal-ui-state', JSON.stringify(state.ui));
// }

// function loadUIState() {
//     const saved = localStorage.getItem('multi-modal-ui-state');
//     if (saved) {
//         try {
//             const savedState = JSON.parse(saved);
//             state.ui = { ...state.ui, ...savedState };
//             applyUIState();
//         } catch (e) {
//             console.error('Failed to load UI state:', e);
//         }
//     }
// }

// function applyUIState() {
//     if (state.ui.configPanelCollapsed) {
//         toggleConfigPanel(true);
//     }
//     if (state.ui.timelinePanelCollapsed) {
//         // toggleTimelinePanel(true);
//     }
// }

// ===== Future: Timeline Panel Collapse =====
// Uncomment when implementing timeline collapse

// function toggleTimelinePanel(collapsed) {
//     const timelinePanel = document.getElementById('timeline-panel');
//     const toggleBtn = document.getElementById('timeline-collapse-btn');
//
//     if (collapsed) {
//         timelinePanel.classList.add('collapsed');
//         toggleBtn.textContent = '▲';
//         toggleBtn.title = 'Expand timeline';
//         state.ui.timelinePanelCollapsed = true;
//     } else {
//         timelinePanel.classList.remove('collapsed');
//         toggleBtn.textContent = '▼';
//         toggleBtn.title = 'Collapse timeline';
//         state.ui.timelinePanelCollapsed = false;
//     }
//
//     saveUIState();
// }

// ===== Future: Mobile Session List Toggle =====
// Uncomment when implementing mobile hamburger menu

// function toggleSessionList() {
//     const sessionList = document.querySelector('.session-list');
//     const backdrop = document.querySelector('.session-list-backdrop');
//
//     sessionList.classList.toggle('open');
//     backdrop.classList.toggle('visible');
//     state.ui.sessionListVisible = sessionList.classList.contains('open');
// }

// ===== UI Settings Modal =====
function openSettingsModal() {
    const modal = document.getElementById('settings-modal');

    // Load current settings into checkboxes
    document.getElementById('ui-combine-speech-lanes').checked = uiSettings.combineSpeechLanes;
    document.getElementById('ui-show-session-thumbnails').checked = uiSettings.showSessionThumbnails;
    document.getElementById('ui-auto-scroll-chat').checked = uiSettings.autoScrollChat;
    document.getElementById('ui-show-timestamps').checked = uiSettings.showTimestamps;
    document.getElementById('ui-show-debug-info').checked = uiSettings.showDebugInfo;

    modal.classList.add('show');
}

function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    modal.classList.remove('show');
}

function saveSettingsFromModal() {
    // Read values from checkboxes
    uiSettings.combineSpeechLanes = document.getElementById('ui-combine-speech-lanes').checked;
    uiSettings.showSessionThumbnails = document.getElementById('ui-show-session-thumbnails').checked;
    uiSettings.autoScrollChat = document.getElementById('ui-auto-scroll-chat').checked;
    uiSettings.showTimestamps = document.getElementById('ui-show-timestamps').checked;
    uiSettings.showDebugInfo = document.getElementById('ui-show-debug-info').checked;

    // Save to localStorage
    saveUISettings();

    // Re-render timeline if speech lanes setting changed
    if (state.selectedSession) {
        renderTimeline();
    }
    updateVoiceDebugPanel();

    // Close modal
    closeSettingsModal();
}

function resetUISettings() {
    if (confirm('Reset all UI settings to defaults?')) {
        uiSettings.combineSpeechLanes = false;
        uiSettings.showSessionThumbnails = true;
        uiSettings.autoScrollChat = true;
        uiSettings.showTimestamps = false;
        uiSettings.showDebugInfo = false;
        uiSettings.userAudioGain = 2;
        uiSettings.aiAudioGain = 2;

        saveUISettings();

        // Update checkboxes and gain selects
        document.getElementById('ui-combine-speech-lanes').checked = uiSettings.combineSpeechLanes;
        document.getElementById('ui-show-session-thumbnails').checked = uiSettings.showSessionThumbnails;
        document.getElementById('ui-auto-scroll-chat').checked = uiSettings.autoScrollChat;
        document.getElementById('ui-show-timestamps').checked = uiSettings.showTimestamps;
        document.getElementById('ui-show-debug-info').checked = uiSettings.showDebugInfo;
        const ug = document.getElementById('timeline-user-audio-gain');
        const ag = document.getElementById('timeline-ai-audio-gain');
        if (ug) ug.value = String(uiSettings.userAudioGain);
        if (ag) ag.value = String(uiSettings.aiAudioGain);

        // Re-render timeline
        if (state.selectedSession || (state.isLiveSession && state.sessionState === 'stopped')) {
            renderTimeline();
        }
    }
}
