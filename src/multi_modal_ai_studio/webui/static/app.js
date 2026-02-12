// Multi-modal AI Studio - WebUI App
// Handles session loading, display, and timeline visualization
// Mic waveform debug: in console set localStorage.setItem('micWaveformDebug','1') and reload to see [MicWaveform] logs
if (typeof localStorage !== 'undefined' && localStorage.getItem('micWaveformDebug')) window._micWaveformDebug = true;

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
    /** When true, preview waveform is fed by server user_amplitude (no AnalyserNode) */
    micWaveformFromServer: false,
    /** WebSocket for mic level preview only (setup, before START); closed when starting session or stopping preview */
    micPreviewWs: null,

    /** Voice pipeline WebSocket (live session) */
    voiceWs: null,
    /** WebRTC peer connection and signaling WS for server camera (preview) */
    cameraWebrtcPc: null,
    cameraWebrtcWs: null,
    /** Server camera device currently in use for preview (e.g. '/dev/video0'); used to avoid reconnecting when only mic/speaker changes */
    previewServerCameraDevice: null,
    /** Live session: timeline events streamed from backend */
    liveTimelineEvents: [],
    /** Live session: chat turns [{ user, assistant }] for display */
    liveChatTurns: [],
    /** Live session: current partial/interim ASR text (updated on every asr_partial; cleared on asr_final) */
    liveAsrInterimText: '',
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
    /** Running buffer of last N client-side RMS values for smoothing green waveform (avoids saw-tooth from chunk boundaries) */
    _userAmplitudeSmoothBuf: [],
    /** Server USB: amplitude samples received before session_start; flushed into liveAudioAmplitudeHistory when session starts */
    pendingServerMicAmplitude: [],
    /** Live session: TTS (AI) segments for purple waveform on AUDIO lane: { startTime, endTime, amplitude } */
    liveTtsAmplitudeHistory: [],
    /** Live session: TTL bands from JS end-of-speech to first sound out: [{ start, end, ttlMs }] (times in session sec) */
    liveTtlBands: [],
    /** Live session: session time (sec) when JS confirmed silence (amplitude < threshold for 150ms); cleared when band closes */
    liveTtlBandStartTime: null,
    /** True when we have seen asr_partial this turn (gate for silence detection) */
    voiceTurnActive: false,
    /** Session time (sec) of last asr_partial; used as fallback for TTL band start when JS silence not detected */
    lastAsrPartialTime: null,
    /** Earliest TTS segment start (session sec) this response; band end = this so we lock onto first audio, not a later chunk */
    firstTtsPlayTimeThisResponse: null,
    /** Earliest TTS segment start this response with amplitude > 0 (any signal = AI voice start) */
    earliestTtsPlayTimeAboveThreshold: null,
    /** Session time (sec) when amplitude first went below threshold; used to confirm 150ms silence */
    voiceSilenceCandidate: null,
    /** Consecutive user_amplitude samples below threshold (Server USB 20 Hz path only); used to confirm silence without wall-clock. */
    voiceSilenceConsecutiveCount: 0,
    /** Live session: CPU/GPU samples for bottom timeline lane: { t, cpu, gpu } (t = session-relative sec) */
    liveSystemStats: [],
    /** Live session: interval id for system stats polling; cleared on disconnect */
    liveSystemStatsPollIntervalId: null,
    /** Live session: have we set initial 15s zoom once */
    liveTimelineInitialZoomSet: false,

    /** Server health: { llm: null | { ok, error? }, riva: null | { ok, error? } } from /api/health/llm and /api/health/riva */
    serverHealth: { llm: null, riva: null },

    /** Cached OpenAI API key from env (GET /api/config/prefills); re-applied when starting a new chat */
    envOpenaiApiKey: '',

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
    /** Show pipeline (ASR | LLM | TTS) badge in each session item in the left Sessions list */
    showPipelineInSessionList: false,
    /** Show "+ New Chat with Default Config" as the default action button (else "+ New Voice Chat") */
    showNewChatWithDefaultConfig: false,
    /** When false, do not record/save preview camera image in session history data */
    recordPreviewInSessionHistory: true,
    /** When true, timeline panel height follows layout (auto); when false, use timelineHeightPx */
    timelineHeightAuto: true,
    /** Default timeline view height in px when timelineHeightAuto is false (200–600) */
    timelineHeightPx: 400,
    /** UI-only gain for mic preview bar (setup + live); 1–4, default 2 for quiet mics (e.g. EMEET) */
    micPreviewGain: 2,
    /** UI-only gain for user (mic) waveform on timeline: 1, 2, or 4 */
    userAudioGain: 2,
    /** UI-only gain for AI (TTS) waveform on timeline: 1, 2, or 4 */
    aiAudioGain: 2,
    /** User (mic) voice threshold 0–100: amplitude >= this = voice, below = silence (TTL end-of-speech, hide low user bars during TTS). Default 5. */
    userVoiceThreshold: 5,
    /** Session directory override: '' = default (sessions), 'mock_sessions' = sample data. Sent to server via PATCH /api/app/session-dir. */
    sessionDirOverride: ''
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
/** Load config prefills from server (e.g. OPENAI_API_KEY from env/.env) and apply to defaultConfig/currentConfig. Caches key in state.envOpenaiApiKey for re-apply on new chat. */
async function loadConfigPrefills() {
    try {
        const r = await fetch(getApiBase() + '/api/config/prefills');
        if (!r.ok) return;
        const data = await r.json();
        const key = (data && data.openai_api_key) ? String(data.openai_api_key).trim() : '';
        state.envOpenaiApiKey = key || '';
        if (!key) return;
        if (defaultConfig.asr) defaultConfig.asr.api_key = key;
        if (defaultConfig.llm) defaultConfig.llm.api_key = key;
        if (currentConfig.asr) currentConfig.asr.api_key = key;
        if (currentConfig.llm) currentConfig.llm.api_key = key;
        const asrKeyEl = document.getElementById('asr-realtime-api-key');
        if (asrKeyEl) asrKeyEl.value = key;
        const llmKeyEl = document.getElementById('llm-api-key');
        if (llmKeyEl) llmKeyEl.value = key;
    } catch (e) {
        // ignore (e.g. offline or old server without /api/config/prefills)
    }
}

/** Re-apply env prefills (e.g. OPENAI_API_KEY) to currentConfig and DOM. Call after resetting currentConfig (e.g. startNewSession). */
function applyEnvPrefillsToCurrentConfig() {
    const key = (state.envOpenaiApiKey || '').trim();
    if (!key) return;
    if (currentConfig.asr) currentConfig.asr.api_key = key;
    if (currentConfig.llm) currentConfig.llm.api_key = key;
    const asrKeyEl = document.getElementById('asr-realtime-api-key');
    if (asrKeyEl) asrKeyEl.value = key;
    const llmKeyEl = document.getElementById('llm-api-key');
    if (llmKeyEl) llmKeyEl.value = key;
}

async function loadSessions() {
    console.log('loadSessions() called');
    try {
        await loadConfigPrefills();
        // Apply saved session directory override so server uses the right dir (e.g. mock_sessions).
        // If PATCH is not supported (e.g. old server), continue and load sessions anyway.
        const override = (typeof uiSettings.sessionDirOverride === 'string' && uiSettings.sessionDirOverride) ? uiSettings.sessionDirOverride : '';
        const payload = override ? { session_dir: override } : { session_dir: null };
        try {
            const dirRes = await fetch(getApiBase() + '/api/app/session-dir', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!dirRes.ok) { /* ignore; e.g. 404 on older server */ }
        } catch (_) { /* ignore */ }

        console.log('Fetching from /api/sessions...');
        const response = await fetch(getApiBase() + '/api/sessions');
        console.log('Response received:', response.status);
        if (!response.ok) {
            throw new Error('Server returned ' + response.status + (response.statusText ? ' ' + response.statusText : ''));
        }
        const sessions = await response.json();
        console.log('Sessions loaded:', sessions.length);
        state.sessions = Array.isArray(sessions) ? sessions : [];
        renderSessionList();
    } catch (error) {
        console.error('Failed to load sessions:', error);
        state.sessions = [];
        var countEl = document.querySelector('.session-count');
        if (countEl) countEl.textContent = 'Error';
        var container = document.getElementById('session-items');
        if (container) {
            container.innerHTML = `
                <div style="padding: 1rem; color: var(--text-secondary); text-align: center;">
                    <p>Failed to load sessions</p>
                    <p style="font-size: 0.85rem; margin-top: 0.5rem;">${(error && error.message) ? String(error.message) : 'Unknown error'}</p>
                </div>
            `;
        }
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
        const sid = escapeHtml(session.session_id);
        const safeName = escapeHtml(session.name);
        const pipelineSummary = uiSettings.showPipelineInSessionList && session.config ? getPipelineSummaryHtml(session.config) : '';

        return `
            <div class="session-item ${isActive ? 'active' : ''}" data-session-index="${index}">
                <div class="session-item-body" onclick="selectSession(${index})">
                    <div class="session-item-name">${safeName}</div>
                    <div class="session-item-meta">
                        <span>${formatSessionDateOnly(session.created_at)}</span>
                        <span>${formatDate(session.created_at)}</span>
                        <span>${metrics.total_turns || 0} turns</span>
                    </div>
                    <div class="session-item-metrics">
                        <span class="metric-badge">TTL: ${formatLatency(metrics.avg_ttl)}</span>
                        <span class="metric-badge">${session.timeline?.length || 0} events</span>
                    </div>
                    ${pipelineSummary ? `<div class="session-item-pipeline">${pipelineSummary}</div>` : ''}
                </div>
                <button type="button" class="session-item-menu-btn" data-session-id="${sid}" data-session-index="${index}" title="Session menu" aria-label="Session menu">
                    <i data-lucide="ellipsis-vertical" class="lucide-inline"></i>
                </button>
                <div class="session-item-dropdown" data-session-id="${sid}" role="menu" aria-hidden="true">
                    <button type="button" role="menuitem" data-action="rename"><i data-lucide="pencil" class="lucide-inline"></i> Rename</button>
                    <button type="button" role="menuitem" data-action="delete" class="session-menu-delete"><i data-lucide="trash-2" class="lucide-inline"></i> Delete</button>
                </div>
            </div>
        `;
    }).join('');

    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

function toggleSessionMenu(ev, sessionId, index) {
    ev.preventDefault();
    ev.stopPropagation();
    var menu = document.querySelector('.session-item-dropdown[data-session-id="' + sessionId + '"]');
    var open = document.querySelector('.session-item-dropdown.open');
    if (open && open !== menu) open.classList.remove('open');
    if (menu) {
        menu.classList.toggle('open');
        menu.setAttribute('aria-hidden', menu.classList.contains('open') ? 'false' : 'true');
    }
}

function closeSessionMenus() {
    document.querySelectorAll('.session-item-dropdown.open').forEach(function (el) {
        el.classList.remove('open');
        el.setAttribute('aria-hidden', 'true');
    });
}

function renameSession(sessionId) {
    closeSessionMenus();
    var session = state.sessions.find(function (s) { return s.session_id === sessionId; });
    if (!session) return;
    var name = window.prompt('Rename session', session.name || '');
    if (name == null) return;
    name = (name || '').trim();
    if (!name) return;
    fetch('/api/sessions/' + encodeURIComponent(sessionId), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name })
    })
        .then(function (r) {
            if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || 'Rename failed'); });
            return r.json();
        })
        .then(function (updated) {
            session.name = updated.name;
            if (state.selectedSession && state.selectedSession.session_id === sessionId) state.selectedSession.name = updated.name;
            renderSessionList();
            renderSessionDetail();
        })
        .catch(function (e) {
            alert('Could not rename: ' + e.message);
        });
}

function deleteSession(sessionId) {
    closeSessionMenus();
    if (!window.confirm('Delete this session? This cannot be undone.')) return;
    fetch('/api/sessions/' + encodeURIComponent(sessionId), { method: 'DELETE' })
        .then(function (r) {
            if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || 'Delete failed'); });
            if (state.selectedSession && state.selectedSession.session_id === sessionId) {
                state.selectedSession = null;
                state.isLiveSession = false;
                state.sessionState = 'stopped';
            }
            return loadSessions();
        })
        .then(function () {
            renderSessionList();
            renderSessionDetail();
            renderTimeline();
            updateLiveSessionUI();
        })
        .catch(function (e) {
            alert('Could not delete: ' + e.message);
        });
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
    const saveDefaultBtn = document.getElementById('save-default-config-btn');
    if (state.isLiveSession) {
        panel.classList.add('config-panel--editable');
        if (saveDefaultBtn) saveDefaultBtn.disabled = false;
    } else {
        panel.classList.remove('config-panel--editable');
        if (saveDefaultBtn) saveDefaultBtn.disabled = true;
    }
}

function renderConfig() {
    updateConfigPanelState();
    updateConfigTabStates();
    const contentEl = document.getElementById('config-tab-content');
    const tab = state.activeConfigTab;
    // Device tab uses 'devices' key in config
    const configKey = tab === 'device' ? 'devices' : tab;

    // If in live session mode, show editable forms
    if (state.isLiveSession) {
        contentEl.innerHTML = renderEditableConfigForm(tab, currentConfig[configKey], false);
        if (tab === 'llm' && !isRealtimeFullVoiceLock()) setTimeout(() => fetchLLMModels(currentConfig.llm.api_base || (currentConfig.llm.ollama_url && currentConfig.llm.ollama_url.replace(/\/v1$/, '') + '/v1')), 0);
        if (tab === 'asr' && (currentConfig.asr.backend === 'riva' || currentConfig.asr.scheme === 'riva')) setTimeout(() => fetchASRModels(currentConfig.asr.server || currentConfig.asr.riva_server || 'localhost:50051'), 0);
        if (tab === 'tts' && (currentConfig.tts.backend === 'riva' || currentConfig.tts.scheme === 'riva')) setTimeout(() => fetchTTSVoices(currentConfig.tts.riva_server || currentConfig.tts.server || 'localhost:50051'), 0);
        if (tab === 'device') setTimeout(populateAllDeviceDropdowns, 0);
        // Preload ASR/TTS model names so pipeline shows them even when user hasn't opened those tabs
        setTimeout(function () { preloadASRModelName(); preloadTTSModelName(); }, 0);
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

    contentEl.innerHTML = renderEditableConfigForm(tab, tabConfig, true, config);
    if (tab === 'device') setTimeout(populateAllDeviceDropdowns, 0);
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
        interim_results: true,
        realtime_transport: 'websocket',
        realtime_session_type: 'transcription',
        realtime_url: 'wss://api.openai.com/v1/realtime',
        api_key: ''
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
        system_prompt: 'You are a helpful AI assistant.',
        extra_request_body: '',
        enable_vision: false,
        vision_system_prompt: 'You are a vision assistant. Give ONE short sentence answers only. Be direct. No explanations.',
        vision_detail: 'auto',
        vision_frames: 4,
        vision_quality: 0.7,
        vision_max_width: 640,
        vision_buffer_fps: 3.0
    },
    tts: {
        backend: 'riva',
        riva_server: 'localhost:50051',
        voice: '',
        language: 'en-US',
        sample_rate: 22050,
        quality: 'high',
        realtime_transport: 'websocket'
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
        llm_warmup_while_preview: true,
        barge_in_enabled: false,
        barge_in_trigger: 'final',
        barge_in_partial_count: 3,
        log_level: 'info'
    }
};

// Current editable configuration (for new session)
let currentConfig = JSON.parse(JSON.stringify(defaultConfig));

// Default config saved by user (for "New Voice Chat with Default Configuration" later)
const DEFAULT_VOICE_CHAT_CONFIG_KEY = 'defaultVoiceChatConfig';

function getDefaultConfig() {
    try {
        const saved = localStorage.getItem(DEFAULT_VOICE_CHAT_CONFIG_KEY);
        if (!saved) return null;
        const parsed = JSON.parse(saved);
        return { ...defaultConfig, ...parsed };
    } catch (e) {
        console.warn('Failed to load saved default config:', e);
        return null;
    }
}

function saveDefaultConfig() {
    if (!state.isLiveSession) return;
    try {
        localStorage.setItem(DEFAULT_VOICE_CHAT_CONFIG_KEY, JSON.stringify(currentConfig));
        const btn = document.getElementById('save-default-config-btn');
        if (btn) {
            const origHTML = btn.innerHTML;
            btn.innerHTML = 'Saved';
            btn.disabled = true;
            setTimeout(() => {
                btn.innerHTML = origHTML;
                btn.disabled = false;
                if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
            }, 1500);
        }
    } catch (e) {
        console.error('Failed to save default config:', e);
    }
}

/** Preset system prompts for voice AI. [1] [2] [3] in the UI apply these. */
var SYSTEM_PROMPT_PRESETS = [
    'You are a helpful voice assistant.',
    'You are a helpful AI assistant.',
    'You are a concise voice assistant. Keep replies brief and natural for conversation.'
];

function applySystemPromptPreset(index) {
    var text = SYSTEM_PROMPT_PRESETS[index];
    if (text == null) return;
    updateConfig('llm', 'system_prompt', text);
    var el = document.getElementById('llm-system-prompt');
    if (el) el.value = text;
}

/** Pin an LLM field (e.g. system_prompt, extra_request_body) to the saved default so it is used in other sessions. */
function pinLlmFieldToDefault(fieldName) {
    try {
        const saved = localStorage.getItem(DEFAULT_VOICE_CHAT_CONFIG_KEY);
        const base = saved ? JSON.parse(saved) : {};
        const merged = {
            ...base,
            llm: {
                ...(base.llm || defaultConfig.llm || {}),
                [fieldName]: currentConfig.llm[fieldName]
            }
        };
        localStorage.setItem(DEFAULT_VOICE_CHAT_CONFIG_KEY, JSON.stringify(merged));
    } catch (e) {
        console.warn('Failed to pin LLM field to default:', e);
    }
}

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
// sessionConfig: when readonly, full session.config (used for device_labels on Devices tab)
function renderEditableConfigForm(tab, config, readonly = false, sessionConfig = null) {
    switch (tab) {
        case 'asr':
            return renderASRConfig(config, readonly);
        case 'llm':
            return renderLLMConfig(config, readonly);
        case 'tts':
            return renderTTSConfig(config, readonly);
        case 'device':
            return renderDeviceConfig(config, readonly, readonly && sessionConfig ? sessionConfig.device_labels : null);
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

            <!-- Backend Tabs (traditional style: RIVA | REST API | Realtime API) -->
            <div class="backend-tabs speech-api-tabs ${readonly ? 'disabled' : ''}">
                <button type="button" class="backend-tab speech-api-tab ${config.backend === 'riva' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('asr', 'backend', 'riva')"><span class="riva-tab-inner"><i data-lucide="bird" class="lucide-inline riva-tab-icon"></i><span class="riva-tab-text">NVIDIA<br>RIVA</span></span></button>
                <button type="button" class="backend-tab speech-api-tab ${config.backend === 'openai' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('asr', 'backend', 'openai')">OpenAI<br>REST API</button>
                <button type="button" class="backend-tab speech-api-tab ${config.backend === 'openai-realtime' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('asr', 'backend', 'openai-realtime')">OpenAI<br>Realtime API</button>
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
                        ? `<input type="text" id="asrModel" value="${config.model || 'Default'}" readonly class="readonly-config-input">`
                        : `<select id="asr-model-select" class="config-select" onchange="updateConfig('asr', 'model', this.value)">
                               <option value="">Loading...</option>
                           </select>`
                    }
                    <small class="config-deployment-hint"><i data-lucide="info" class="lucide-inline"></i> ${readonly ? 'Set during session' : 'Queried from Riva server'}</small>
                    ${!readonly ? '<div id="asr-model-default-hint" class="input-hint" style="margin-top: 4px; color: var(--text-secondary);"></div>' : ''}
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

            <!-- OpenAI Realtime Settings: Connection (WebRTC | WebSocket | SIP) + Mode (full | transcript only) -->
            <div class="backend-content" style="display: ${config.backend === 'openai-realtime' ? 'block' : 'none'}">
                <div class="config-section-label" style="margin-bottom: 8px;"><i data-lucide="radio" class="lucide-inline"></i> Connection</div>
                <div class="backend-tabs speech-api-tabs realtime-transport-tabs" style="margin-bottom: 12px;">
                    <button type="button" class="backend-tab speech-api-tab ${(config.realtime_transport || 'websocket') === 'webrtc' ? 'active' : ''} ${!readonly ? '' : 'disabled'}"
                            ${disabled} title="Not supported yet"
                            onclick="if(!${readonly}) updateConfig('asr', 'realtime_transport', 'webrtc')">WebRTC</button>
                    <button type="button" class="backend-tab speech-api-tab ${(config.realtime_transport || 'websocket') === 'websocket' ? 'active' : ''}"
                            ${disabled}
                            onclick="if(!${readonly}) updateConfig('asr', 'realtime_transport', 'websocket')">WebSocket</button>
                    <button type="button" class="backend-tab speech-api-tab ${(config.realtime_transport || 'websocket') === 'sip' ? 'active' : ''} ${!readonly ? '' : 'disabled'}"
                            ${disabled} title="Not supported yet"
                            onclick="if(!${readonly}) updateConfig('asr', 'realtime_transport', 'sip')">SIP</button>
                </div>
                ${((config.realtime_transport || 'websocket') === 'webrtc' || (config.realtime_transport || 'websocket') === 'sip') ? '<p class="input-hint" style="margin: 0 0 12px 0;"><i data-lucide="info" class="lucide-inline"></i> Not supported yet. Use WebSocket.</p>' : ''}

                <div class="config-section-label" style="margin-bottom: 8px;"><i data-lucide="mic-2" class="lucide-inline"></i> Mode</div>
                <div class="form-group" style="margin-bottom: 12px;">
                    <label class="radio-label">
                        <input type="radio" ${disabled} name="asr-realtime-mode" value="full" ${(config.realtime_session_type || 'transcription') === 'full' ? 'checked' : ''}
                               onchange="if(!${readonly}) updateConfig('asr', 'realtime_session_type', 'full'); applyRealtimeLock();">
                        Full voice (speech-to-speech)
                    </label>
                    <label class="radio-label">
                        <input type="radio" ${disabled} name="asr-realtime-mode" value="transcription" ${(config.realtime_session_type || 'transcription') === 'transcription' ? 'checked' : ''}
                               onchange="if(!${readonly}) updateConfig('asr', 'realtime_session_type', 'transcription'); applyRealtimeLock();">
                        Transcript only
                    </label>
                </div>

                <div class="form-group" style="display: ${(config.realtime_transport || 'websocket') === 'websocket' ? 'block' : 'none'};">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                        <label style="margin: 0;">WebSocket URL</label>
                        ${!readonly ? `<div style="position: relative;">
                            <button type="button" class="icon-btn" onclick="toggleRealtimePresetsMenu(event)" title="Select preset"><i data-lucide="list" class="lucide-inline"></i></button>
                            <div class="api-presets-menu" id="realtimePresetsMenu" style="display: none;">
                                <div class="api-preset-item" onclick="selectRealtimePreset('wss://api.openai.com/v1/realtime', 'gpt-realtime')"><strong>OpenAI</strong><span>wss://api.openai.com/v1/realtime</span><span class="preset-model-hint">gpt-realtime</span></div>
                                <div class="api-preset-item" onclick="selectRealtimePreset('wss://api.openai.com/v1/realtime', 'gpt-4o-realtime-preview')"><strong>OpenAI (preview)</strong><span>wss://api.openai.com/v1/realtime</span><span class="preset-model-hint">gpt-4o-realtime-preview</span></div>
                            </div>
                        </div>` : ''}
                    </div>
                    <input type="text" ${disabled} id="asr-realtime-url" value="${config.realtime_url || 'wss://api.openai.com/v1/realtime'}"
                           onchange="updateConfig('asr', 'realtime_url', this.value); syncRealtimeApiKeyVisibility();">
                </div>

                <div class="form-group" style="display: ${(config.realtime_transport || 'websocket') === 'websocket' ? 'block' : 'none'};">
                    <label>Model</label>
                    <input type="text" ${disabled} id="asr-realtime-model" value="${config.model || 'gpt-realtime'}"
                           onchange="updateConfig('asr', 'model', this.value)">
                    ${!readonly ? '<span class="input-hint">e.g. gpt-realtime, gpt-4o-realtime-preview</span>' : ''}
                </div>

                <div class="form-group" id="asr-realtime-api-key-group" style="display: ${(config.realtime_transport || 'websocket') === 'websocket' && (config.realtime_url || '').indexOf('openai.com') !== -1 ? 'block' : 'none'};">
                    <label>API Key</label>
                    <input type="password" ${disabled} id="asr-realtime-api-key" value="${escapeHtml((config.api_key || (typeof currentConfig !== 'undefined' && currentConfig.llm && currentConfig.llm.api_key) || ''))}" placeholder="Same as LLM; required for OpenAI"
                           onchange="updateConfig('asr', 'api_key', this.value); updateConfig('llm', 'api_key', this.value);">
                    <div class="input-hint">Shared with Configuration &gt; LLM. Required for OpenAI Realtime.</div>
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
    const realtimeFullVoice = !readonly && isRealtimeFullVoiceConfig(config);
    const disabled = readonly ? 'disabled' : '';
    const disableApiAndModel = readonly || realtimeFullVoice;
    const roClass = readonly ? 'readonly' : '';
    const apiBase = config.api_base || (config.ollama_url ? config.ollama_url.replace(/\/v1$/, '') + '/v1' : 'http://localhost:11434/v1');
    const showApiKey = !readonly && (apiBase.includes('openai.com') || apiBase.includes('integrate.api.nvidia.com'));

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}
            ${realtimeFullVoice ? '<p class="config-note"><i data-lucide="message-circle" class="lucide-inline"></i> Realtime full-voice: only <strong>System Prompt</strong> is used (sent as Realtime instructions). API Base and Model are fixed.</p>' : ''}
            <div class="form-group">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <label style="margin: 0;">API Base URL</label>
                    ${!disableApiAndModel ? `<div style="position: relative;">
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
                <input type="text" ${disableApiAndModel ? 'disabled' : ''} id="llm-api-base" value="${apiBase}" placeholder="http://localhost:11434/v1"
                       onchange="updateConfig('llm', 'api_base', this.value); if(!${readonly}) fetchLLMModels(this.value);">
                <div class="input-hint">${realtimeFullVoice ? 'Fixed when using Realtime full-voice (not used).' : 'OpenAI-compatible API endpoint (Ollama, vLLM, SGLang, OpenAI, etc.)'}</div>
            </div>

            <div class="form-group llm-api-key-group" id="llm-api-key-group" style="display: ${showApiKey ? 'block' : 'none'}">
                <label>API Key ${showApiKey ? '(required for OpenAI / NVIDIA API Catalog)' : ''}</label>
                <input type="password" ${disabled} id="llm-api-key" value="${config.api_key || ''}" placeholder="${showApiKey ? 'Paste your API key' : 'Optional for local'}"
                       onchange="updateConfig('llm', 'api_key', this.value)">
            </div>

            <div class="form-group">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <label style="margin: 0;">Model</label>
                    ${!disableApiAndModel ? '<button type="button" class="icon-btn" onclick="refreshLLMModels()" title="Refresh models"><i data-lucide="refresh-cw" class="lucide-inline"></i></button>' : ''}
                </div>
                <select ${disableApiAndModel ? 'disabled' : ''} id="llm-model-select" onchange="updateConfig('llm', 'model', this.value)">
                    <option value="${escapeHtml(config.model)}">${readonly ? escapeHtml(config.model) : (realtimeFullVoice ? 'gpt-realtime (fixed)' : 'Loading...')}</option>
                </select>
                <div class="input-hint">${realtimeFullVoice ? 'Fixed when using Realtime full-voice.' : 'Fetched from API Base URL; use reload icon if you added a model'}</div>
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
                           onchange="onLLMMinimalOutputChange(this.checked)">
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
                <label class="checkbox-label">
                    <input type="checkbox" ${disabled} id="llm-enable-vision" ${config.enable_vision ? 'checked' : ''}
                           onchange="updateConfig('llm', 'enable_vision', this.checked); toggleVlmSettings();">
                    Enable Vision (VLM)
                </label>
                ${!readonly ? '<span class="input-hint">Send camera frames to VLM during speech (e.g. Cosmos-Reason, GPT-4V)</span>' : ''}
            </div>

            <div id="vlm-settings" class="form-group" style="display: ${config.enable_vision ? 'block' : 'none'}; padding-left: 20px; border-left: 2px solid var(--border-color);">
                <label>Frames per Turn</label>
                <input type="range" ${disabled} id="llm-vision-frames" min="1" max="10" step="1" value="${config.vision_frames || 4}"
                       oninput="updateConfig('llm', 'vision_frames', parseInt(this.value)); document.getElementById('vision-frames-value').textContent = this.value;">
                <span id="vision-frames-value" class="range-value">${config.vision_frames || 4}</span>
                ${!readonly ? '<span class="input-hint">1 = single frame at end, 2-10 = multiple frames during speech</span>' : ''}

                <label style="margin-top: 10px;">Frame Quality</label>
                <input type="range" ${disabled} id="llm-vision-quality" min="0.3" max="1.0" step="0.1" value="${config.vision_quality || 0.7}"
                       oninput="updateConfig('llm', 'vision_quality', parseFloat(this.value)); document.getElementById('vision-quality-value').textContent = this.value;">
                <span id="vision-quality-value" class="range-value">${config.vision_quality || 0.7}</span>
                ${!readonly ? '<span class="input-hint">JPEG quality: lower = smaller, higher = better</span>' : ''}

                <label style="margin-top: 10px;">Max Frame Width</label>
                <input type="range" ${disabled} id="llm-vision-max-width" min="320" max="1280" step="64" value="${config.vision_max_width || 640}"
                       oninput="updateConfig('llm', 'vision_max_width', parseInt(this.value)); document.getElementById('vision-max-width-value').textContent = this.value + 'px';">
                <span id="vision-max-width-value" class="range-value">${config.vision_max_width || 640}px</span>
            </div>

            <div class="form-group">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <label style="margin: 0;">System Prompt</label>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        ${!readonly ? SYSTEM_PROMPT_PRESETS.map((_, i) => '<button type="button" class="system-prompt-preset-btn" onclick="applySystemPromptPreset(' + i + ')" title="Preset ' + (i + 1) + '">[' + (i + 1) + ']</button>').join('') : ''}
                        ${!readonly ? '<span class="system-prompt-preset-sep" aria-hidden="true">|</span>' : ''}
                        ${!readonly ? '<button type="button" class="icon-btn" onclick="var el = document.getElementById(\'llm-system-prompt\'); if(el) { updateConfig(\'llm\', currentConfig.llm.enable_vision ? \'vision_system_prompt\' : \'system_prompt\', el.value); pinLlmFieldToDefault(currentConfig.llm.enable_vision ? \'vision_system_prompt\' : \'system_prompt\'); }" title="Pin to use in other sessions"><i data-lucide="pin" class="lucide-inline"></i></button>' : ''}
                    </div>
                </div>
                <textarea id="llm-system-prompt" ${disabled} rows="3"
                          onchange="updateConfig('llm', currentConfig.llm.enable_vision ? 'vision_system_prompt' : 'system_prompt', this.value)">${escapeHtml(config.enable_vision ? (config.vision_system_prompt || 'You are a vision assistant. Give ONE short sentence answers only. Be direct. No explanations.') : (config.system_prompt || ''))}</textarea>
                ${!readonly ? `<span class="input-hint">${config.enable_vision ? 'Vision system prompt (used when Enable Vision is checked)' : 'Text LLM system prompt'}</span>` : ''}
            </div>

            <div class="form-group">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <label style="margin: 0;">Extra request body (JSON)</label>
                    ${!readonly ? '<button type="button" class="icon-btn" onclick="var el = document.getElementById(\'llm-extra-request-body\'); if(el) { updateConfig(\'llm\', \'extra_request_body\', el.value); pinLlmFieldToDefault(\'extra_request_body\'); }" title="Pin to use in other sessions"><i data-lucide="pin" class="lucide-inline"></i></button>' : ''}
                </div>
                <textarea id="llm-extra-request-body" ${disabled} rows="4" placeholder='{"chat_template_kwargs": {"enable_thinking": false}}'
                          onchange="updateConfig('llm', 'extra_request_body', this.value)">${escapeHtml(config.extra_request_body || '')}</textarea>
                ${!readonly ? '<span class="input-hint">Merged into the chat completion request. Leave empty or valid JSON.</span>' : ''}
            </div>
        </div>
    `;
}

function onLLMMinimalOutputChange(checked) {
    updateConfig('llm', 'minimal_output', checked);
    if (checked) {
        const base = { chat_template_kwargs: { enable_thinking: false } };
        let current = {};
        try {
            const raw = (currentConfig.llm.extra_request_body || '').trim();
            if (raw) current = JSON.parse(raw);
        } catch (_) {}
        const merged = { ...current };
        merged.chat_template_kwargs = { ...(merged.chat_template_kwargs || {}), enable_thinking: false };
        const jsonStr = JSON.stringify(merged, null, 2);
        currentConfig.llm.extra_request_body = jsonStr;
        const el = document.getElementById('llm-extra-request-body');
        if (el) { el.value = jsonStr; }
    }
}

function toggleVlmSettings() {
    const vlmSettings = document.getElementById('vlm-settings');
    const enableVision = document.getElementById('llm-enable-vision');
    const systemPrompt = document.getElementById('llm-system-prompt');
    
    if (vlmSettings && enableVision) {
        vlmSettings.style.display = enableVision.checked ? 'block' : 'none';
    }
    
    // Swap system prompt content based on vision state
    if (systemPrompt && currentConfig.llm) {
        if (enableVision.checked) {
            // Switching to vision mode - show vision_system_prompt
            systemPrompt.value = currentConfig.llm.vision_system_prompt || 'You are a vision assistant. Give ONE short sentence answers only. Be direct. No explanations.';
        } else {
            // Switching to text mode - show system_prompt
            systemPrompt.value = currentConfig.llm.system_prompt || 'You are a helpful voice assistant.';
        }
    }
}


function renderTTSConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}

            <!-- Backend Tabs (traditional style: RIVA | REST API | Realtime API) -->
            <div class="backend-tabs speech-api-tabs ${readonly ? 'disabled' : ''}">
                <button type="button" class="backend-tab speech-api-tab ${config.backend === 'riva' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('tts', 'backend', 'riva')"><span class="riva-tab-inner"><i data-lucide="bird" class="lucide-inline riva-tab-icon"></i><span class="riva-tab-text">NVIDIA<br>RIVA</span></span></button>
                <button type="button" class="backend-tab speech-api-tab ${config.backend === 'openai' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('tts', 'backend', 'openai')">OpenAI<br>REST API</button>
                <button type="button" class="backend-tab speech-api-tab ${config.backend === 'openai-realtime' ? 'active' : ''}"
                        ${disabled}
                        onclick="updateConfig('tts', 'backend', 'openai-realtime')">OpenAI<br>Realtime API</button>
            </div>

            <!-- Riva Settings -->
            <div class="backend-content" style="display: ${config.backend === 'riva' ? 'block' : 'none'}">
                <div class="form-group">
                    <label>RIVA server</label>
                    <input type="text" ${disabled} id="tts-riva-server" value="${config.riva_server || 'localhost:50051'}"
                           onchange="updateConfig('tts', 'riva_server', this.value); if(!${readonly}) fetchTTSVoices(this.value);">
                </div>

                <div class="form-group">
                    <label>TTS Model</label>
                    ${readonly
                        ? `<input type="text" ${disabled} value="${config.riva_model_name || 'Default'}" readonly class="readonly-config-input">`
                        : `<select id="tts-model-select" class="config-select" onchange="updateConfig('tts', 'riva_model_name', this.value); if(!${readonly}) refreshPipelineDisplay();">
                               <option value="">Default</option>
                               ${(config.riva_model_names || []).map(m => '<option value="' + escapeHtml(m) + '"' + (m === (config.riva_model_name || '') ? ' selected' : '') + '>' + escapeHtml(m) + '</option>').join('')}
                           </select>`
                    }
                    ${!readonly ? '<span class="input-hint">Queried from RIVA server</span>' : ''}
                </div>

                <div class="form-group">
                    <label>Language</label>
                    <select ${disabled} id="tts-riva-language" onchange="updateConfig('tts', 'language', this.value); if(!${readonly}) fetchTTSVoices(document.getElementById('tts-riva-server')?.value);">
                        <option value="en-US" ${config.language === 'en-US' ? 'selected' : ''}>English (US)</option>
                        <option value="en-GB" ${config.language === 'en-GB' ? 'selected' : ''}>English (UK)</option>
                        <option value="es-ES" ${config.language === 'es-ES' ? 'selected' : ''}>Spanish</option>
                        <option value="fr-FR" ${config.language === 'fr-FR' ? 'selected' : ''}>French</option>
                        <option value="de-DE" ${config.language === 'de-DE' ? 'selected' : ''}>German</option>
                        <option value="ja-JP" ${config.language === 'ja-JP' ? 'selected' : ''}>Japanese</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Voice</label>
                    ${readonly
                        ? `<input type="text" ${disabled} value="${config.voice || 'Default'}" readonly class="readonly-config-input">`
                        : `<select id="tts-voice-select" class="config-select" onchange="updateConfig('tts', 'voice', this.value); updateTTSVoiceDefaultHint();">
                               <option value="">Default</option>
                           </select>`
                    }
                    ${!readonly ? '<span class="input-hint">Queried from RIVA (language-specific voices)</span>' : ''}
                    ${!readonly ? '<div id="tts-voice-default-hint" class="input-hint" style="margin-top: 4px; color: var(--text-secondary);"></div>' : ''}
                </div>

                ${!readonly ? `
                <div class="form-group">
                    <button type="button" class="btn-secondary tts-riva-reload-btn" onclick="var s=document.getElementById(\'tts-riva-server\'); var l=document.getElementById(\'tts-riva-language\'); if(s) fetchTTSVoices(s.value, l?l.value:null);" title="Reload TTS model and voices from RIVA server">
                        <i data-lucide="refresh-cw" class="lucide-inline"></i> Reload from RIVA server
                    </button>
                    <div id="tts-riva-reload-hint" class="input-hint" style="color: var(--text-secondary); margin-top: 4px;"></div>
                </div>
                ` : ''}
            </div>

            <!-- OpenAI REST TTS Settings -->
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

            <!-- OpenAI Realtime TTS (same session as Realtime ASR): Connection tabs + optional overrides -->
            <div class="backend-content" style="display: ${config.backend === 'openai-realtime' ? 'block' : 'none'}">
                <div class="form-group">
                    <p class="input-hint" style="margin: 0;">TTS for OpenAI Realtime uses the same session as Realtime ASR. Voice and model are typically configured in the Realtime session.</p>
                </div>
                <div class="config-section-label" style="margin-bottom: 8px;"><i data-lucide="radio" class="lucide-inline"></i> Connection</div>
                <div class="backend-tabs speech-api-tabs realtime-transport-tabs" style="margin-bottom: 12px;">
                    <button type="button" class="backend-tab speech-api-tab ${(config.realtime_transport || 'websocket') === 'webrtc' ? 'active' : ''} ${!readonly ? '' : 'disabled'}"
                            ${disabled} title="Not supported yet"
                            onclick="if(!${readonly}) updateConfig('tts', 'realtime_transport', 'webrtc')">WebRTC</button>
                    <button type="button" class="backend-tab speech-api-tab ${(config.realtime_transport || 'websocket') === 'websocket' ? 'active' : ''}"
                            ${disabled}
                            onclick="if(!${readonly}) updateConfig('tts', 'realtime_transport', 'websocket')">WebSocket</button>
                    <button type="button" class="backend-tab speech-api-tab ${(config.realtime_transport || 'websocket') === 'sip' ? 'active' : ''} ${!readonly ? '' : 'disabled'}"
                            ${disabled} title="Not supported yet"
                            onclick="if(!${readonly}) updateConfig('tts', 'realtime_transport', 'sip')">SIP</button>
                </div>
                ${((config.realtime_transport || 'websocket') === 'webrtc' || (config.realtime_transport || 'websocket') === 'sip') ? '<p class="input-hint" style="margin: 0 0 12px 0;"><i data-lucide="info" class="lucide-inline"></i> Not supported yet. Use WebSocket.</p>' : ''}
                <div class="form-group" style="display: ${(config.realtime_transport || 'websocket') === 'websocket' ? 'block' : 'none'};">
                    <label>WebSocket / API base</label>
                    <input type="text" ${disabled} value="${config.realtime_url || config.openai_url || 'wss://api.openai.com/v1/realtime'}"
                           onchange="updateConfig('tts', 'realtime_url', this.value); updateConfig('tts', 'openai_url', this.value)">
                </div>
            </div>

            <!-- Common Settings (Language only for non-RIVA backends) -->
            <div class="form-group" style="display: ${config.backend === 'riva' ? 'none' : 'block'}">
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

/** Suffix for devices enumerated by the browser (this PC, including USB attached to it). */
function deviceLabelSuffix(label) {
    return '(Browser)';
}

/** Base label for deduplication: strip (Browser), (Server USB), (USB) suffix so we can match same device across lists. */
function deviceLabelBase(label) {
    return (label || '').replace(/\s*\((?:Server USB|Browser|USB)\)\s*$/i, '').trim();
}

function renderDeviceConfig(config, readonly = false, deviceLabels = null) {
    // When viewing a recorded session with saved device names, show plain read-only fields (no dropdowns)
    if (readonly && deviceLabels && (deviceLabels.camera != null || deviceLabels.mic != null || deviceLabels.speaker != null)) {
        const camName = (deviceLabels.camera != null && deviceLabels.camera !== '') ? escapeHtml(String(deviceLabels.camera)) : '—';
        const micName = (deviceLabels.mic != null && deviceLabels.mic !== '') ? escapeHtml(String(deviceLabels.mic)) : '—';
        const spkName = (deviceLabels.speaker != null && deviceLabels.speaker !== '') ? escapeHtml(String(deviceLabels.speaker)) : '—';
        return `
        <div class="config-form readonly">
            <p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> Recorded devices for this session (read-only)</p>
            <div class="form-group">
                <label><i data-lucide="video" class="lucide-inline"></i> Camera device</label>
                <div class="config-value config-value--device" aria-readonly="true">${camName}</div>
            </div>
            <div class="form-group">
                <label><i data-lucide="mic" class="lucide-inline"></i> Microphone device</label>
                <div class="config-value config-value--device" aria-readonly="true">${micName}</div>
            </div>
            <div class="form-group">
                <label><i data-lucide="volume-2" class="lucide-inline"></i> Speaker device</label>
                <div class="config-value config-value--device" aria-readonly="true">${spkName}</div>
            </div>
        </div>
    `;
    }

    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';
    const micValue = config.microphone === 'none' ? 'none' : (state.selectedBrowserMicId || '');
    const camValue = config.camera === 'none' ? 'none' : (config.camera === 'browser' || config.camera === '' ? '' : config.camera);
    const spkValue = config.speaker === 'none' ? 'none' : (state.selectedBrowserSpeakerId || '');

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}
            <p class="input-hint" style="margin-bottom: 1rem;">Select the device for your chat session. Select &#128683;None if you don&apos;t plan to use the device or go text based. <strong>Microphone:</strong> (Browser) = mic on this PC; Server USB = mic attached to the server (e.g. EMEET). <strong>Speaker:</strong> Server USB not yet wired; use (Browser) for playback.</p>
            <div class="form-group">
                <label><i data-lucide="video" class="lucide-inline"></i> Camera device</label>
                <select id="device-camera-list" ${disabled} data-device-type="camera" onchange="onDeviceListChange('camera', this.value)">
                    <option value="none" ${camValue === 'none' ? 'selected' : ''}>&#128683;None (No vision-modality)</option>
                    <option value="" ${camValue === '' || camValue === 'browser' ? 'selected' : ''}>Default (Browser)</option>
                </select>
                <div class="input-hint input-hint-camera">Lists cameras on this PC (Browser) and USB cameras attached to the server (Server USB). Default uses the browser’s default camera.</div>
            </div>
            <div class="form-group">
                <label><i data-lucide="mic" class="lucide-inline"></i> Microphone device</label>
                <select id="device-microphone-list" ${disabled} data-device-type="microphone" onchange="onDeviceListChange('microphone', this.value)">
                    <option value="none" ${micValue === 'none' ? 'selected' : ''}>&#128683;None (Text Only)</option>
                    <option value="" ${micValue === '' ? 'selected' : ''}>Default (Browser)</option>
                </select>
                <div class="input-hint">Preview uses browser mic. Live session: select a (Browser) device to send mic from this PC, or a Server USB device to use the mic attached to the server (e.g. EMEET).</div>
            </div>
            <div class="form-group">
                <label><i data-lucide="volume-2" class="lucide-inline"></i> Speaker device</label>
                <select id="device-speaker-list" ${disabled} data-device-type="speaker" onchange="onDeviceListChange('speaker', this.value)">
                    <option value="none" ${spkValue === 'none' ? 'selected' : ''}>&#128683;None (Text Only)</option>
                    <option value="" ${spkValue === '' ? 'selected' : ''}>Default (Browser)</option>
                </select>
            </div>
            ${!readonly ? '<div class="form-group"><button type="button" class="btn-secondary" onclick="requestDevicesAndPopulateAll()">Allow & list devices</button><div class="input-hint">Lists <strong>microphone, speaker, and cameras</strong> on this PC (Browser) and USB devices attached to the server (Server USB).</div></div>' : ''}
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
        if (value === 'none') {
            currentConfig.devices.microphone = 'none';
            state.selectedBrowserMicId = null;
        } else if (value === '') {
            currentConfig.devices.microphone = 'browser';
            state.selectedBrowserMicId = null;
        } else if (value.startsWith('pyaudio:') || value.startsWith('alsa:')) {
            currentConfig.devices.microphone = value;
            state.selectedBrowserMicId = null;
        } else {
            currentConfig.devices.microphone = 'browser';
            state.selectedBrowserMicId = value;
        }
    } else if (type === 'speaker') {
        if (value === 'none') {
            currentConfig.devices.speaker = 'none';
            state.selectedBrowserSpeakerId = null;
        } else if (value === '') {
            currentConfig.devices.speaker = 'browser';
            state.selectedBrowserSpeakerId = null;
        } else if (value.startsWith('pyaudio:') || value.startsWith('alsa:')) {
            currentConfig.devices.speaker = value;
            state.selectedBrowserSpeakerId = null;
        } else {
            currentConfig.devices.speaker = 'browser';
            state.selectedBrowserSpeakerId = value;
        }
    }
    updateDeviceIndicators();
    updateChatInputVisibility();
    refreshPipelineDisplay();
    if (state.isLiveSession && state.sessionState === 'setup') {
        // When microphone changes to (possibly different) Server USB device, close voice WS so server gets new device; otherwise we'd keep capturing from the first USB mic.
        if (type === 'microphone' && (value.startsWith('alsa:') || value.startsWith('pyaudio:'))) {
            stopMicWaveform();
        }
        // When only mic/speaker changes, keep server camera WebRTC open to avoid release/reopen race
        var keepServerCamera = (type === 'microphone' || type === 'speaker');
        startPreviewStream(keepServerCamera ? { keepServerCamera: true } : undefined);
    }
}

function renderAppConfig(config, readonly = false) {
    const disabled = readonly ? 'disabled' : '';
    const roClass = readonly ? 'readonly' : '';
    const overrideVal = (typeof uiSettings.sessionDirOverride === 'string' && uiSettings.sessionDirOverride) ? uiSettings.sessionDirOverride : '';
    const bargeInEnabled = !!config.barge_in_enabled;
    const bargeInTrigger = config.barge_in_trigger === 'partial' ? 'partial' : 'final';
    const partialCount = Math.max(1, Math.min(20, parseInt(config.barge_in_partial_count, 10) || 3));

    return `
        <div class="config-form ${roClass}">
            ${readonly ? '<p class="config-note"><i data-lucide="clipboard-list" class="lucide-inline"></i> This is a historical session configuration (read-only)</p>' : ''}

            <div class="form-group form-group--barge-in">
                <div class="form-group-row form-group-row--toggle">
                    <label class="label-text" for="app-enable-barge-in">
                        <i data-lucide="ship" class="lucide-inline" aria-hidden="true"></i>
                        <span>Barge-in</span>
                    </label>
                    <label class="toggle-switch">
                        <input type="checkbox" ${disabled} id="app-enable-barge-in" ${bargeInEnabled ? 'checked' : ''}
                               onchange="updateConfig('app', 'barge_in_enabled', this.checked); appConfigRefresh();">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="input-hint">Interrupt AI speech by speaking. When on, TTS stops when your speech is detected.</div>
            </div>

            ${bargeInEnabled ? `
            <div class="form-group app-barge-in-options" id="app-barge-in-options">
                <div class="app-config-subheading">Stop frontend TTS on</div>
                <div class="app-radio-group">
                    <label class="radio-label">
                        <input type="radio" ${disabled} name="app-barge-in-trigger" value="final" ${bargeInTrigger === 'final' ? 'checked' : ''}
                               onchange="updateConfig('app', 'barge_in_trigger', 'final'); appConfigRefresh();">
                        <span><strong>Final transcript</strong> (stable, ~0.5–1s delay)</span>
                    </label>
                    <label class="radio-label">
                        <input type="radio" ${disabled} name="app-barge-in-trigger" value="partial" ${bargeInTrigger === 'partial' ? 'checked' : ''}
                               onchange="updateConfig('app', 'barge_in_trigger', 'partial'); appConfigRefresh();">
                        <span><strong>Partial transcript</strong> (fast, may have false positive)</span>
                    </label>
                </div>
            </div>
            ${bargeInTrigger === 'partial' ? `
            <div class="form-group" id="app-barge-in-partial-count-row">
                <label class="app-config-subheading">Number of partial</label>
                <div class="app-number-stepper">
                    <button type="button" class="btn-stepper" ${disabled} aria-label="Decrease" onclick="var n=Math.max(1,(currentConfig.app.barge_in_partial_count||3)-1); updateConfig('app','barge_in_partial_count',n); var el=document.getElementById('app-barge-in-partial-value'); if(el){el.value=n;el.textContent=n;}">−</button>
                    <span class="app-number-stepper-value" id="app-barge-in-partial-value">${partialCount}</span>
                    <button type="button" class="btn-stepper" ${disabled} aria-label="Increase" onclick="var n=Math.min(20,(currentConfig.app.barge_in_partial_count||3)+1); updateConfig('app','barge_in_partial_count',n); var el=document.getElementById('app-barge-in-partial-value'); if(el){el.value=n;el.textContent=n;}">+</button>
                </div>
                <div class="input-hint">Stop TTS after this many partial transcripts (1–20).</div>
            </div>
            ` : ''}
            ` : ''}

            ${!readonly ? `
            <div class="form-group">
                <label for="app-session-dir-override"><i data-lucide="folder" class="lucide-inline"></i> Session directory</label>
                <select id="app-session-dir-override" class="config-select" onchange="applySessionDirOverride(this.value)">
                    <option value="" ${overrideVal === '' ? 'selected' : ''}>Default (sessions)</option>
                    <option value="mock_sessions" ${overrideVal === 'mock_sessions' ? 'selected' : ''}>mock_sessions (sample data)</option>
                </select>
                <div class="input-hint">Where to load and save session JSON files. Change applies immediately; session list will refresh.</div>
            </div>
            ` : ''}

            <div class="form-group">
                <div class="form-group-row form-group-row--toggle">
                    <label class="label-text" for="app-enable-timeline"><i data-lucide="chart-gantt" class="lucide-inline" aria-hidden="true"></i><span>Show Timeline Visualization</span></label>
                    <label class="toggle-switch">
                        <input type="checkbox" ${disabled} id="app-enable-timeline" ${config.enable_timeline ? 'checked' : ''}
                               onchange="updateConfig('app', 'enable_timeline', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>

            <div class="form-group">
                <div class="form-group-row form-group-row--toggle">
                    <label class="label-text" for="app-llm-warmup-while-preview"><i data-lucide="heater" class="lucide-inline" aria-hidden="true"></i><span>Enable LLM warmup while preview</span></label>
                    <label class="toggle-switch">
                        <input type="checkbox" ${disabled} id="app-llm-warmup-while-preview" ${config.llm_warmup_while_preview ? 'checked' : ''}
                               onchange="updateConfig('app', 'llm_warmup_while_preview', this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                <div class="input-hint">Send dummy request to pre-load model and reduce first-turn latency.</div>
            </div>

            <div class="form-group">
                <label><i data-lucide="scroll-text" class="lucide-inline"></i> Log Level</label>
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

function appConfigRefresh() {
    if (state.activeConfigTab === 'app') renderConfig();
}

async function applySessionDirOverride(value) {
    const payload = (value === '' || value === 'mock_sessions') ? (value ? { session_dir: value } : { session_dir: null }) : null;
    if (payload === null) return;
    uiSettings.sessionDirOverride = value;
    saveUISettings();
    try {
        const r = await fetch('/api/app/session-dir', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            console.error('Session dir override failed:', err);
            return;
        }
        await loadSessions();
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
    } catch (e) {
        console.error('applySessionDirOverride failed:', e);
    }
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

var _populateDeviceDropdownsTimer = null;
var POPULATE_DEVICE_DROPDOWNS_DEBOUNCE_MS = 180;

function populateAllDeviceDropdowns() {
    if (_populateDeviceDropdownsTimer != null) clearTimeout(_populateDeviceDropdownsTimer);
    _populateDeviceDropdownsTimer = setTimeout(function () {
        _populateDeviceDropdownsTimer = null;
        populateCameraDeviceDropdown();
        populateMicrophoneDeviceDropdown();
        populateSpeakerDeviceDropdown();
    }, POPULATE_DEVICE_DROPDOWNS_DEBOUNCE_MS);
}

/** Populate camera dropdown: browser cameras (Browser) + server USB cameras (Server USB). Default (Browser) uses browser default. Uses allSettled so one failure doesn't block the other list. */
function populateCameraDeviceDropdown() {
    var select = document.getElementById('device-camera-list');
    if (!select) return;
    var havePermission = navigator.mediaDevices && navigator.mediaDevices.enumerateDevices;
    var browserPromise = havePermission
        ? navigator.mediaDevices.enumerateDevices().then(function (devices) {
            return Array.isArray(devices) ? devices.filter(function (d) { return d.kind === 'videoinput'; }) : [];
        }).catch(function (err) { console.warn('[Devices] enumerateDevices (camera) failed:', err); return []; })
        : Promise.resolve([]);
    var jetsonPromise = fetch(getApiBase() + '/api/devices/cameras')
        .then(function (r) { return r.json(); })
        .then(function (data) { return Array.isArray(data.cameras) ? data.cameras : []; })
        .catch(function (err) { console.warn('[Devices] Fetch server cameras failed:', err); return []; });

    Promise.allSettled([browserPromise, jetsonPromise]).then(function (outcomes) {
        var browserCams = (outcomes[0].status === 'fulfilled' && Array.isArray(outcomes[0].value)) ? outcomes[0].value : [];
        var jetsonCams = (outcomes[1].status === 'fulfilled' && Array.isArray(outcomes[1].value)) ? outcomes[1].value : [];
        if (!select.parentNode) return;
        select.innerHTML = '';
        select.appendChild(newOption('none', '\uD83D\uDEABNone (No vision-modality)'));
        select.appendChild(newOption('', 'Default (Browser)'));
        browserCams.forEach(function (d) {
            var label = (d.label || 'Camera ' + (select.options.length)) + ' (Browser)';
            select.appendChild(newOption(d.deviceId || '', label));
        });
        var browserCamBases = browserCams.map(function (d) { return deviceLabelBase(d.label || d.deviceId || ''); });
        jetsonCams.forEach(function (c) {
            var base = deviceLabelBase(c.label || c.id || '');
            if (browserCamBases.indexOf(base) !== -1) return;
            var label = (c.label || c.id || '');
            if (label.indexOf('(Server USB)') === -1) label = label + ' (Server USB)';
            select.appendChild(newOption(c.id || '', label));
        });
        var cam = (state.selectedSession && !state.isLiveSession && state.selectedSession.config && state.selectedSession.config.devices)
            ? (state.selectedSession.config.devices.camera != null ? state.selectedSession.config.devices.camera : state.selectedSession.config.devices.video_device)
            : currentConfig.devices.camera;
        var val = (cam === 'none' || cam === null || cam === undefined) ? 'none' : (cam === 'browser' || cam === '') ? '' : cam;
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
 * Includes browser devices (Browser) and server USB devices from /api/devices/audio-inputs.
 * Uses allSettled so one failure (e.g. browser permission or server fetch) doesn't block the other list.
 */
function populateMicrophoneDeviceDropdown() {
    var select = document.getElementById('device-microphone-list');
    if (!select) return;
    var browserPromise = (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices)
        ? navigator.mediaDevices.enumerateDevices().then(function (devices) {
            return Array.isArray(devices) ? devices.filter(function (d) { return d.kind === 'audioinput'; }) : [];
        }).catch(function (err) { console.warn('[Devices] enumerateDevices (mic) failed:', err); return []; })
        : Promise.resolve([]);
    var jetsonPromise = fetch(getApiBase() + '/api/devices/audio-inputs')
        .then(function (r) { return r.json(); })
        .then(function (data) { return Array.isArray(data.devices) ? data.devices : []; })
        .catch(function (err) { console.warn('[Devices] Fetch server audio inputs failed:', err); return []; });

    Promise.allSettled([browserPromise, jetsonPromise]).then(function (outcomes) {
        var browserInputs = (outcomes[0].status === 'fulfilled' && Array.isArray(outcomes[0].value)) ? outcomes[0].value : [];
        var jetsonInputs = (outcomes[1].status === 'fulfilled' && Array.isArray(outcomes[1].value)) ? outcomes[1].value : [];
        if (!select.parentNode) return; // select was removed by a re-render
        select.innerHTML = '';
        select.appendChild(newOption('none', '\uD83D\uDEABNone (Text Only)'));
        select.appendChild(newOption('', 'Default (Browser)'));
        browserInputs.forEach(function (d) {
            var label = (d.label || 'Microphone ' + (select.options.length)) + ' (Browser)';
            select.appendChild(newOption(d.deviceId || '', label));
        });
        var browserBases = browserInputs.map(function (d) { return deviceLabelBase(d.label || d.deviceId || ''); });
        jetsonInputs.forEach(function (d) {
            var base = deviceLabelBase(d.label || d.id || '');
            if (browserBases.indexOf(base) !== -1) return; // same device already shown as (Browser), skip server duplicate
            var label = (d.label || d.id || '');
            if (label.indexOf('(Server USB)') === -1) label = label + ' (Server USB)';
            select.appendChild(newOption(d.id || '', label));
        });
        var mic = (state.selectedSession && !state.isLiveSession && state.selectedSession.config && state.selectedSession.config.devices)
            ? (state.selectedSession.config.devices.microphone != null ? state.selectedSession.config.devices.microphone : state.selectedSession.config.devices.audio_input_device)
            : currentConfig.devices.microphone;
        var val = (mic === 'none' || mic === null || mic === undefined) ? 'none' : (mic === 'browser' || mic === '') ? (state.selectedBrowserMicId || '') : mic;
        try { select.value = val; } catch (e) { select.value = ''; }
        updateDeviceIndicators();
    });
}

/**
 * Enumerate audio output devices and fill the "Speaker devices" dropdown.
 * Includes browser devices (Browser) and server USB devices from /api/devices/audio-outputs.
 * Uses allSettled so one failure doesn't block the other list.
 */
function populateSpeakerDeviceDropdown() {
    var select = document.getElementById('device-speaker-list');
    if (!select) return;
    var browserPromise = (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices)
        ? navigator.mediaDevices.enumerateDevices().then(function (devices) {
            return Array.isArray(devices) ? devices.filter(function (d) { return d.kind === 'audiooutput'; }) : [];
        }).catch(function (err) { console.warn('[Devices] enumerateDevices (speaker) failed:', err); return []; })
        : Promise.resolve([]);
    var jetsonPromise = fetch(getApiBase() + '/api/devices/audio-outputs')
        .then(function (r) { return r.json(); })
        .then(function (data) { return Array.isArray(data.devices) ? data.devices : []; })
        .catch(function (err) { console.warn('[Devices] Fetch server audio outputs failed:', err); return []; });

    Promise.allSettled([browserPromise, jetsonPromise]).then(function (outcomes) {
        var browserOutputs = (outcomes[0].status === 'fulfilled' && Array.isArray(outcomes[0].value)) ? outcomes[0].value : [];
        var jetsonOutputs = (outcomes[1].status === 'fulfilled' && Array.isArray(outcomes[1].value)) ? outcomes[1].value : [];
        if (!select.parentNode) return;
        select.innerHTML = '';
        select.appendChild(newOption('none', '\uD83D\uDEABNone (Text Only)'));
        select.appendChild(newOption('', 'Default (Browser)'));
        browserOutputs.forEach(function (d) {
            var label = (d.label || 'Speaker ' + (select.options.length)) + ' (Browser)';
            select.appendChild(newOption(d.deviceId || '', label));
        });
        var browserBases = browserOutputs.map(function (d) { return deviceLabelBase(d.label || d.deviceId || ''); });
        jetsonOutputs.forEach(function (d) {
            var base = deviceLabelBase(d.label || d.id || '');
            if (browserBases.indexOf(base) !== -1) return;
            var label = (d.label || d.id || '');
            if (label.indexOf('(Server USB)') === -1) label = label + ' (Server USB)';
            select.appendChild(newOption(d.id || '', label));
        });
        var spk = (state.selectedSession && !state.isLiveSession && state.selectedSession.config && state.selectedSession.config.devices)
            ? (state.selectedSession.config.devices.speaker != null ? state.selectedSession.config.devices.speaker : state.selectedSession.config.devices.audio_output_device)
            : currentConfig.devices.speaker;
        var val = (spk === 'none' || spk === null || spk === undefined) ? 'none' : (spk === 'browser' || spk === '') ? (state.selectedBrowserSpeakerId || '') : spk;
        try { select.value = val; } catch (e) { select.value = ''; }
        updateDeviceIndicators();
    });
}

/** True when config has ASR = OpenAI Realtime (WebSocket, Full voice) and TTS = OpenAI Realtime. Used for pipeline display. */
function isRealtimeFullVoiceConfig(config) {
    if (!config || !config.asr || !config.tts) return false;
    const asr = config.asr;
    const tts = config.tts;
    const asrFull = (asr.backend === 'openai-realtime' || asr.scheme === 'openai-realtime') &&
        (asr.realtime_transport || 'websocket') === 'websocket' &&
        (asr.realtime_session_type || 'transcription') === 'full';
    const ttsRealtime = tts.backend === 'openai-realtime' || tts.scheme === 'openai-realtime';
    return asrFull && ttsRealtime;
}

/** True when ASR is Realtime WebSocket Full-voice and TTS is also Realtime: LLM tab disabled (single Realtime session). Unlock when ASR is transcript-only or TTS is not Realtime. Default mode is transcript-only. */
function isRealtimeFullVoiceLock() {
    const asr = currentConfig.asr;
    const tts = currentConfig.tts;
    const asrFull = (asr.backend === 'openai-realtime' || asr.scheme === 'openai-realtime') &&
        (asr.realtime_transport || 'websocket') === 'websocket' &&
        (asr.realtime_session_type || 'transcription') === 'full';
    const ttsRealtime = tts.backend === 'openai-realtime' || tts.scheme === 'openai-realtime';
    return asrFull && ttsRealtime;
}

/** When user selects ASR Realtime WebSocket Full-voice: set TTS to Realtime (WebSocket) and disable LLM tab. When they switch to transcript-only or change TTS away, unlock. */
function applyRealtimeLock() {
    if (isRealtimeFullVoiceLock()) {
        currentConfig.tts.backend = 'openai-realtime';
        currentConfig.tts.scheme = 'openai-realtime';
        currentConfig.tts.realtime_transport = 'websocket';
    }
    updateConfigTabStates();
}

/** When Realtime full-voice is selected: LLM tab stays enabled (system prompt = Realtime instructions); tab title explains API Base/Model are fixed. */
function updateConfigTabStates() {
    const llmTab = document.querySelector('.config-tab[data-tab="llm"]');
    if (!llmTab) return;
    const realtimeFullVoice = isRealtimeFullVoiceLock();
    llmTab.classList.remove('config-tab--disabled');
    llmTab.setAttribute('aria-disabled', 'false');
    llmTab.title = realtimeFullVoice ? 'System prompt is used as Realtime instructions; API Base and Model are fixed.' : '';
}

function toggleRealtimePresetsMenu(ev) {
    ev.preventDefault();
    const menu = document.getElementById('realtimePresetsMenu');
    if (!menu) return;
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}

function selectRealtimePreset(url, model) {
    currentConfig.asr.realtime_url = url;
    currentConfig.asr.model = model;
    const urlEl = document.getElementById('asr-realtime-url');
    const modelEl = document.getElementById('asr-realtime-model');
    if (urlEl) urlEl.value = url;
    if (modelEl) modelEl.value = model;
    const menu = document.getElementById('realtimePresetsMenu');
    if (menu) menu.style.display = 'none';
    updateConfig('asr', 'realtime_url', url);
    updateConfig('asr', 'model', model);
    syncRealtimeApiKeyVisibility();
}

function syncRealtimeApiKeyVisibility() {
    const group = document.getElementById('asr-realtime-api-key-group');
    if (!group) return;
    const url = (document.getElementById('asr-realtime-url') || {}).value || currentConfig.asr.realtime_url || '';
    group.style.display = url.indexOf('openai.com') !== -1 ? 'block' : 'none';
}

// Update configuration value
function updateConfig(section, key, value) {
    console.log(`Config updated: ${section}.${key} = ${value}`);
    currentConfig[section][key] = value;

    // Sync backend -> scheme for pipeline
    if (section === 'asr' && key === 'backend') currentConfig.asr.scheme = value;
    if (section === 'tts' && key === 'backend') currentConfig.tts.scheme = value;
    // Keep ASR Realtime API key in sync with LLM when using Realtime WebSocket
    if (section === 'llm' && key === 'api_key' && (currentConfig.asr.backend === 'openai-realtime' || currentConfig.asr.scheme === 'openai-realtime') && (currentConfig.asr.realtime_transport || 'websocket') === 'websocket') {
        currentConfig.asr.api_key = value;
    }

    // Realtime lock: when ASR is openai-realtime + websocket + full-voice, force TTS to openai-realtime so lock applies
    if (section === 'asr' && (key === 'backend' || key === 'realtime_transport' || key === 'realtime_session_type')) {
        const asrFullVoice = (currentConfig.asr.backend === 'openai-realtime' || currentConfig.asr.scheme === 'openai-realtime') &&
            (currentConfig.asr.realtime_transport || 'websocket') === 'websocket' &&
            (currentConfig.asr.realtime_session_type || 'transcription') === 'full';
        if (asrFullVoice) {
            currentConfig.tts.backend = 'openai-realtime';
            currentConfig.tts.scheme = 'openai-realtime';
            currentConfig.tts.realtime_transport = 'websocket';
        }
        // When switching to ASR > OpenAI Realtime API (WebSocket), set defaults: Transcript only, URL, model, API key shown (blank)
        if (section === 'asr' && key === 'backend' && value === 'openai-realtime') {
            if (!currentConfig.asr.realtime_transport || currentConfig.asr.realtime_transport !== 'websocket') {
                currentConfig.asr.realtime_transport = 'websocket';
            }
            if (!(currentConfig.asr.realtime_session_type || '').trim()) {
                currentConfig.asr.realtime_session_type = 'transcription';
            }
            if (!(currentConfig.asr.realtime_url || '').trim()) {
                currentConfig.asr.realtime_url = 'wss://api.openai.com/v1/realtime';
            }
            var m = (currentConfig.asr.model || '').trim();
            if (!m || m.indexOf('realtime') === -1) {
                currentConfig.asr.model = 'gpt-realtime';
            }
            if (currentConfig.asr.api_key === undefined) {
                currentConfig.asr.api_key = '';
            }
        }
    }
    // When TTS is changed to something other than OpenAI Realtime and ASR was full-voice, force ASR to transcript-only
    if (section === 'tts' && key === 'backend' && value !== 'openai-realtime') {
        const asrFullVoice = (currentConfig.asr.backend === 'openai-realtime' || currentConfig.asr.scheme === 'openai-realtime') &&
            (currentConfig.asr.realtime_transport || 'websocket') === 'websocket' &&
            (currentConfig.asr.realtime_session_type || 'transcription') === 'full';
        if (asrFullVoice) {
            currentConfig.asr.realtime_session_type = 'transcription';
        }
    }
    updateConfigTabStates();

    // Persist LLM system prompt and extra request body so they survive reload without "Save as default"
    if (section === 'llm' && (key === 'system_prompt' || key === 'extra_request_body') && state.isLiveSession) {
        try {
            localStorage.setItem(DEFAULT_VOICE_CHAT_CONFIG_KEY, JSON.stringify(JSON.parse(JSON.stringify(currentConfig))));
        } catch (e) {
            console.warn('Failed to persist config to localStorage:', e);
        }
    }

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
    if (state.activeConfigTab === 'tts' && (currentConfig.tts.backend === 'riva' || currentConfig.tts.scheme === 'riva')) {
        setTimeout(() => fetchTTSVoices(currentConfig.tts.riva_server || currentConfig.tts.server || 'localhost:50051'), 0);
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
    refreshPipelineDisplay();
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
    const hint = document.getElementById('asr-model-default-hint');
    if (!select) return;
    select.innerHTML = '<option value="">Loading...</option>';
    if (hint) hint.textContent = '';
    try {
        const r = await fetch('/api/asr/models?server=' + encodeURIComponent(server));
        const data = await r.json().catch(function () { return {}; });
        const models = (data && data.models) ? data.models : [];
        const defaultModel = (data && data.default_model) || (models[0] || '');

        if (!r.ok) {
            const errMsg = (data && data.error) ? data.error : ('Request failed: ' + r.status);
            select.innerHTML = '<option value="">' + escapeHtml(errMsg) + '</option>';
            if (hint) hint.textContent = errMsg;
            return;
        }

        if (!models.length) {
            select.innerHTML = '<option value="">No models found</option>';
            currentConfig.asr.model = '';
            if (hint) hint.textContent = 'No ASR models returned from RIVA server.';
            return;
        }

        var current = currentConfig.asr.model || '';
        if (!current || !models.includes(current)) {
            current = defaultModel && models.includes(defaultModel) ? defaultModel : models[0];
            currentConfig.asr.model = current;
        }
        select.innerHTML = models.map(m => '<option value="' + escapeHtml(m) + '"' + (m === current ? ' selected' : '') + '>' + escapeHtml(m) + '</option>').join('');
        select.value = current;
        if (hint && current === defaultModel) hint.textContent = 'Default model from RIVA server.';
        refreshPipelineDisplay();
    } catch (e) {
        select.innerHTML = '<option value="">Error loading models</option>';
        if (hint) hint.textContent = 'Network error: ' + (e.message || String(e));
        console.error('fetchASRModels failed:', e);
    }
}

async function fetchTTSVoices(server, language) {
    if (!server) return;
    language = language || currentConfig.tts.language || 'en-US';
    const voiceSelect = document.getElementById('tts-voice-select');
    const modelSelect = document.getElementById('tts-model-select');
    const reloadHint = document.getElementById('tts-riva-reload-hint');
    if (!voiceSelect) return;
    voiceSelect.innerHTML = '<option value="">Loading...</option>';
    if (modelSelect) modelSelect.innerHTML = '<option value="">Loading...</option>';
    if (reloadHint) reloadHint.textContent = '';
    try {
        const r = await fetch('/api/tts/voices?server=' + encodeURIComponent(server) + '&language=' + encodeURIComponent(language));
        const data = await r.json().catch(function () { return {}; });
        const voices = (data && data.voices) ? data.voices : [];

        if (!r.ok) {
            const errMsg = (data && data.error) ? data.error : ('Request failed: ' + r.status);
            voiceSelect.innerHTML = '<option value="">Default</option><option value="">' + escapeHtml(errMsg) + '</option>';
            if (modelSelect) modelSelect.innerHTML = '<option value="">Default</option><option value="">' + escapeHtml(errMsg) + '</option>';
            if (reloadHint) reloadHint.textContent = errMsg;
            return;
        }

        // Derive model from first voice if API didn't return model_name (e.g. older Riva or different proto)
        var modelName = data.model_name != null ? data.model_name : (data.model_names && data.model_names[0]) || null;
        var modelNames = (data.model_names && data.model_names.length) ? data.model_names : (modelName ? [modelName] : []);
        if (!modelName && voices.length && voices[0]) {
            var v0 = voices[0];
            modelName = (typeof v0 === 'object' && v0.model) ? v0.model : null;
            if (modelName) modelNames = [modelName];
        }

        if (modelName != null || modelNames.length) {
            currentConfig.tts.riva_model_name = modelName || modelNames[0] || null;
            currentConfig.tts.riva_model_names = modelNames.length ? modelNames : (modelName ? [modelName] : []);
        }

        if (modelSelect && currentConfig.tts.riva_model_names && currentConfig.tts.riva_model_names.length) {
            const currentModel = currentConfig.tts.riva_model_name || '';
            modelSelect.innerHTML = '<option value="">Default</option>' +
                currentConfig.tts.riva_model_names.map(m => '<option value="' + escapeHtml(m) + '"' + (m === currentModel ? ' selected' : '') + '>' + escapeHtml(m) + '</option>').join('');
        } else if (modelSelect) {
            modelSelect.innerHTML = '<option value="">Default</option>' +
                (currentConfig.tts.riva_model_name ? '<option value="' + escapeHtml(currentConfig.tts.riva_model_name) + '" selected>' + escapeHtml(currentConfig.tts.riva_model_name) + '</option>' : '');
        }
        const current = currentConfig.tts.voice || '';
        voiceSelect.innerHTML = '<option value="">Default</option>' +
            voices.map(v => {
                const name = (typeof v === 'string') ? v : (v.name || v.id || String(v));
                return '<option value="' + escapeHtml(name) + '"' + (name === current ? ' selected' : '') + '>' + escapeHtml(name) + '</option>';
            }).join('');
        if (voices.length && current && !voices.some(v => ((typeof v === 'string') ? v : v.name) === current)) {
            voiceSelect.value = '';
            currentConfig.tts.voice = '';
        } else if (!current) {
            voiceSelect.value = '';
        }
        updateTTSVoiceDefaultHint();
        refreshPipelineDisplay();
    } catch (e) {
        voiceSelect.innerHTML = '<option value="">Default</option><option value="">Error loading voices</option>';
        if (modelSelect) modelSelect.innerHTML = '<option value="">Default</option><option value="">Error loading models</option>';
        if (reloadHint) reloadHint.textContent = 'Network error: ' + (e.message || String(e));
        console.error('fetchTTSVoices failed:', e);
    }
}

/** Check Ollama (LLM) and Riva (ASR/TTS) server health; updates state.serverHealth and UI. */
async function checkServersHealth() {
    var apiBase = (currentConfig.llm && (currentConfig.llm.api_base || (currentConfig.llm.ollama_url && (currentConfig.llm.ollama_url.replace(/\/v1\/?$/, '') + '/v1')))) || 'http://localhost:11434/v1';
    var rivaServer = (currentConfig.asr && (currentConfig.asr.server || currentConfig.asr.riva_server)) || (currentConfig.tts && (currentConfig.tts.server || currentConfig.tts.riva_server)) || 'localhost:50051';
    state.serverHealth = { llm: null, riva: null };
    updateServerHealthUI();
    try {
        var llmRes = await fetch(getApiBase() + '/api/health/llm?api_base=' + encodeURIComponent(apiBase));
        var llmData = await llmRes.json().catch(function () { return { ok: false, error: 'Invalid response' }; });
        state.serverHealth.llm = llmData.ok ? { ok: true } : { ok: false, error: llmData.error || 'Unknown error' };
    } catch (e) {
        state.serverHealth.llm = { ok: false, error: e.message || String(e) };
    }
    try {
        var rivaRes = await fetch(getApiBase() + '/api/health/riva?server=' + encodeURIComponent(rivaServer));
        var rivaData = await rivaRes.json().catch(function () { return { ok: false, error: 'Invalid response' }; });
        state.serverHealth.riva = rivaData.ok ? { ok: true } : { ok: false, error: rivaData.error || 'Unknown error' };
    } catch (e) {
        state.serverHealth.riva = { ok: false, error: e.message || String(e) };
    }
    updateServerHealthUI();
}

function updateServerHealthUI() {
    var row = document.getElementById('server-health-row');
    var statusEl = document.getElementById('server-health-status');
    if (!row || !statusEl) return;
    var llm = state.serverHealth.llm;
    var riva = state.serverHealth.riva;
    var parts = [];
    if (llm === null) parts.push('LLM: …');
    else if (llm.ok) parts.push('LLM: ✓');
    else parts.push('LLM: ✗ ' + (llm.error || '').slice(0, 40));
    if (riva === null) parts.push('Riva: …');
    else if (riva.ok) parts.push('Riva: ✓');
    else parts.push('Riva: ✗ ' + (riva.error || '').slice(0, 40));
    statusEl.textContent = parts.join('  ');
}

/** Preload ASR model name from API into currentConfig so pipeline shows it without opening ASR tab. */
async function preloadASRModelName() {
    if (!state.isLiveSession || !currentConfig.asr) return;
    if (currentConfig.asr.backend !== 'riva' && currentConfig.asr.scheme !== 'riva') return;
    var server = currentConfig.asr.server || currentConfig.asr.riva_server || 'localhost:50051';
    if (!server) return;
    try {
        var r = await fetch('/api/asr/models?server=' + encodeURIComponent(server));
        var data = await r.json().catch(function () { return {}; });
        var models = (data && data.models) ? data.models : [];
        var defaultModel = (data && data.default_model) || (models[0] || '');
        if (!r.ok || !models.length) return;
        var current = currentConfig.asr.model || '';
        if (!current || !models.includes(current)) {
            current = defaultModel && models.includes(defaultModel) ? defaultModel : models[0];
            currentConfig.asr.model = current;
        }
        refreshPipelineDisplay();
    } catch (e) {
        // Silent; pipeline will show fallback label
    }
}

/** Preload TTS model name from API into currentConfig so pipeline shows it without opening TTS tab. */
async function preloadTTSModelName() {
    if (!state.isLiveSession || !currentConfig.tts) return;
    if (currentConfig.tts.backend !== 'riva' && currentConfig.tts.scheme !== 'riva') return;
    var server = currentConfig.tts.riva_server || currentConfig.tts.server || 'localhost:50051';
    if (!server) return;
    var language = currentConfig.tts.language || 'en-US';
    try {
        var r = await fetch('/api/tts/voices?server=' + encodeURIComponent(server) + '&language=' + encodeURIComponent(language));
        var data = await r.json().catch(function () { return {}; });
        var voices = (data && data.voices) ? data.voices : [];
        if (!r.ok) return;
        var modelName = data.model_name != null ? data.model_name : (data.model_names && data.model_names[0]) || null;
        var modelNames = (data.model_names && data.model_names.length) ? data.model_names : (modelName ? [modelName] : []);
        if (!modelName && voices.length && voices[0]) {
            var v0 = voices[0];
            modelName = (typeof v0 === 'object' && v0.model) ? v0.model : null;
            if (modelName) modelNames = [modelName];
        }
        if (modelName != null || modelNames.length) {
            currentConfig.tts.riva_model_name = modelName || modelNames[0] || null;
            currentConfig.tts.riva_model_names = modelNames.length ? modelNames : (modelName ? [modelName] : []);
            refreshPipelineDisplay();
        }
    } catch (e) {
        // Silent; pipeline will show fallback label
    }
}

/** Show which voice is used when "Default" is selected, or "Using: X" when a specific voice is chosen. */
function updateTTSVoiceDefaultHint() {
    const sel = document.getElementById('tts-voice-select');
    const hint = document.getElementById('tts-voice-default-hint');
    if (!sel || !hint) return;
    const val = sel.value;
    if (val === '') {
        var firstVoice = sel.options.length > 1 ? sel.options[1] : null;
        hint.textContent = firstVoice ? 'Default voice: ' + firstVoice.text : '';
    } else {
        var chosen = sel.options[sel.selectedIndex];
        hint.textContent = chosen ? 'Using: ' + chosen.text : '';
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
                <div class="chat-bubble ${isUser ? 'user' : 'ai'}">
                    <div class="chat-avatar">${isUser ? '<i data-lucide="user" class="lucide-inline"></i>' : '<i data-lucide="bot" class="lucide-inline"></i>'}</div>
                    <div class="chat-content"><div class="chat-text">${escapeHtml(msg.content || '...')}</div>${ts}</div>
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
                <div class="chat-bubble user">
                    <div class="chat-avatar"><i data-lucide="user" class="lucide-inline"></i></div>
                    <div class="chat-content"><div class="chat-text">${escapeHtml(turn.user_transcript || '...')}</div>${userConfidence}</div>
                </div>
                <div class="chat-bubble ai">
                    <div class="chat-avatar"><i data-lucide="bot" class="lucide-inline"></i></div>
                    <div class="chat-content"><div class="chat-text">${escapeHtml(turn.ai_response || '...')}</div>${turnMetrics}</div>
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

/** Minimize or restore timeline panel based on Configuration > App > Show Timeline Visualization. */
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
    const TIME_LABEL_HEIGHT = 30; /* Space for x-axis line + time labels (0s, 1s, …) so they are not cut off */

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

    // Calculate time range with zoom (when stopped, include waveform data so we don't cut after last tts_complete)
    const maxTimeFromEvents = timeline.length ? Math.max(0.1, ...timeline.map(e => e.timestamp || e.end_time || 0)) : 0.1;
    let maxTimeFromWaveforms = maxTimeFromEvents;
    if (hasStoppedLiveData) {
        const ah = state.liveAudioAmplitudeHistory || [];
        const th = state.liveTtsAmplitudeHistory || [];
        const fromAh = ah.length ? Math.max(...ah.map(s => s.timestamp != null ? s.timestamp : s[0])) : 0;
        const fromTh = th.length ? Math.max(...th.map(s => s.endTime != null ? s.endTime : 0)) : 0;
        maxTimeFromWaveforms = Math.max(maxTimeFromEvents, fromAh, fromTh, 0.1);
    }
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
        maxTime = maxTimeFromWaveforms;
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
    // Replay: use timeline as single source for waveforms (simplifies session JSON). Fallback to client-sent lists for old sessions.
    const timelineUserAmp = (!inLive && !hasStoppedLiveData && timeline && timeline.length) ? buildUserAmplitudeFromTimeline(timeline) : [];
    const replayAudioAmplitudeHistory = (timelineUserAmp.length > 0) ? timelineUserAmp : ((!inLive && !hasStoppedLiveData && state.selectedSession && state.selectedSession.audio_amplitude_history) ? state.selectedSession.audio_amplitude_history : null);
    const timelineTtsSegments = (!inLive && !hasStoppedLiveData && timeline && timeline.length) ? buildTtsSegmentsFromTimeline(timeline) : [];
    const replayTtsSegments = (timelineTtsSegments.length > 0) ? timelineTtsSegments : ((!inLive && !hasStoppedLiveData && state.selectedSession && state.selectedSession.tts_playback_segments) ? state.selectedSession.tts_playback_segments : null);
    const replayUserAmplitudeForTtl = replayAudioAmplitudeHistory;
    const liveSessionTime = (state.liveSessionStartTime > 0) ? (Date.now() / 1000 - state.liveSessionStartTime) : null;
    drawTimelineEvents(ctx, timeline, lanes, LANE_HEIGHTS, laneYOffsets, LANE_GAP, PADDING_TOP, PADDING_LEFT,
                       PADDING_RIGHT, width, timeScale, state.timelineOffset, laneColors, combineSpeechLanes,
                       inLive, hasStoppedLiveData, liveAmplitude, liveTtsSegments, liveSystemStats, replayTtsSegments, replayAudioAmplitudeHistory, replayUserAmplitudeForTtl, liveSessionTime);

    // Overlay over label strip (0..PADDING_LEFT): graph shows faintly behind; labels drawn once on top
    const labelFadeWidth = PADDING_LEFT;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark' ||
        (!document.documentElement.getAttribute('data-theme') && !window.matchMedia('(prefers-color-scheme: light)').matches);
    const grad = ctx.createLinearGradient(0, 0, labelFadeWidth, 0);
    if (isDark) {
        grad.addColorStop(0, 'rgba(12,12,12,0.96)');
        grad.addColorStop(0.75, 'rgba(18,18,18,0.92)');
        grad.addColorStop(1, 'rgba(0,0,0,0.5)');
    } else {
        grad.addColorStop(0, 'rgba(248,248,248,0.94)');
        grad.addColorStop(0.75, 'rgba(252,252,252,0.88)');
        grad.addColorStop(1, 'rgba(255,255,255,0.5)');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, PADDING_TOP, labelFadeWidth, totalLanesHeight);
    // Draw lane labels once (white on dark theme, dark on light theme); extra padding so transition looks natural
    const labelRightPadding = 24;
    ctx.fillStyle = isDark ? 'rgba(255,255,255,0.95)' : 'rgba(28,28,28,0.92)';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    lanes.forEach((lane, i) => {
        const y = getLaneY(i) + getLaneHeight(i) / 2;
        ctx.fillText(laneLabels[lane], PADDING_LEFT - labelRightPadding, y);
    });

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

// Single source: build user amplitude list from timeline (server already has every user_amplitude). Fallback to session.audio_amplitude_history for old sessions.
function buildUserAmplitudeFromTimeline(timeline) {
    if (!timeline || !timeline.length) return [];
    const userEvents = timeline.filter(function (e) {
        return e.event_type === 'audio_amplitude' && e.source !== 'tts' && e.source !== 'ai';
    });
    var out = userEvents.map(function (e) {
        var a = e.amplitude != null ? Number(e.amplitude) : 0;
        return { timestamp: e.timestamp != null ? Number(e.timestamp) : 0, amplitude: a };
    }).sort(function (a, b) { return a.timestamp - b.timestamp; });
    // Server stores 0–100. Only normalize 0–1 → 0–100 when data looks like 0–1 (max ≤ 1).
    if (out.length) {
        var maxA = Math.max.apply(null, out.map(function (s) { return s.amplitude; }));
        if (maxA > 0 && maxA <= 1) { out.forEach(function (s) { s.amplitude = s.amplitude * 100; }); }
    }
    return out;
}

// Build TTS segments from timeline (tts_start / tts_complete pairs). Fallback to session.tts_playback_segments for old sessions.
function buildTtsSegmentsFromTimeline(timeline) {
    if (!timeline || !timeline.length) return [];
    const starts = timeline.filter(function (e) { return e.event_type === 'tts_start'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
    const completes = timeline.filter(function (e) { return e.event_type === 'tts_complete'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
    if (!starts.length) return [];
    return starts.map(function (s, i) {
        const startTime = s.timestamp != null ? Number(s.timestamp) : 0;
        const endEvent = completes[i];
        const endTime = endEvent && endEvent.timestamp != null ? Number(endEvent.timestamp) : startTime + 0.1;
        return { startTime: startTime, endTime: endTime, amplitude: 50 };
    });
}

// Get amplitude at time t from live history (nearest sample or linear interpolate).
// If maxGapSec is set and t falls between two samples more than maxGapSec apart, return 0 (don't draw in gaps between turns).
function getAmplitudeAtTime(history, t, maxGapSec) {
    if (!history.length) return 0;
    const getT = (s) => s.timestamp != null ? s.timestamp : s[0];
    const getA = (s) => s.amplitude != null ? s.amplitude : s[1];
    if (t <= getT(history[0])) return getA(history[0]);
    if (t >= getT(history[history.length - 1])) return getA(history[history.length - 1]);
    for (let i = 0; i < history.length - 1; i++) {
        const t0 = getT(history[i]), t1 = getT(history[i + 1]);
        if (t >= t0 && t <= t1) {
            if (typeof maxGapSec === 'number' && (t1 - t0) > maxGapSec) return 0;
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

/** Compute TTL bands from stored amplitude + TTS playback (for recorded session replay). End-of-speech constrained to between first partial and final (before LLM). AI voice start = first TTS segment with amplitude > 0 (any signal). */
function computeTtlBandsFromReplay(amplitudeHistory, ttsSegments, timeline) {
    if (!ttsSegments || !ttsSegments.length) return [];
    const getT = (s) => (s.timestamp != null ? s.timestamp : s[0]);
    const getA = (s) => (s.amplitude != null ? s.amplitude : s[1]);
    const userTh = (typeof uiSettings.userVoiceThreshold === 'number' && !isNaN(uiSettings.userVoiceThreshold)) ? uiSettings.userVoiceThreshold : 5;
    const SILENCE_CONFIRM_SEC = 0.15;
    const silenceStarts = [];
    if (amplitudeHistory && amplitudeHistory.length) {
        let lastVoiceTime = -1;
        let runStartIdx = -1;
        for (let i = 0; i < amplitudeHistory.length; i++) {
            const t = getT(amplitudeHistory[i]);
            const a = getA(amplitudeHistory[i]);
            if (a >= userTh) {
                lastVoiceTime = t;
                runStartIdx = -1;
                continue;
            }
            if (runStartIdx === -1) runStartIdx = i;
            const runStartT = getT(amplitudeHistory[runStartIdx]);
            if (t - runStartT >= SILENCE_CONFIRM_SEC && lastVoiceTime >= 0 && runStartT > lastVoiceTime) {
                silenceStarts.push(runStartT);
                lastVoiceTime = -1;
                runStartIdx = -1;
            }
        }
    }
    // First playback time per turn (one per response): group by gap, then apply amplitude threshold
    const segments = ttsSegments.slice().filter(function (s) { return typeof s.startTime === 'number' && typeof s.endTime === 'number'; }).sort(function (a, b) { return a.startTime - b.startTime; });
    const GAP_BETWEEN_RESPONSES_SEC = 1.5;
    const responseGroups = [];
    let currentGroup = [];
    for (let i = 0; i < segments.length; i++) {
        const start = segments[i].startTime;
        const prevEnd = i > 0 ? segments[i - 1].endTime : -1;
        if (i === 0 || (start - prevEnd) >= GAP_BETWEEN_RESPONSES_SEC) {
            if (currentGroup.length) { responseGroups.push(currentGroup); currentGroup = []; }
        }
        currentGroup.push(segments[i]);
    }
    if (currentGroup.length) responseGroups.push(currentGroup);
    // Per-turn window: end-of-speech must be between first partial and final (before LLM / first AI voice)
    const turnWindows = [];
    if (timeline && timeline.length) {
        const finals = timeline.filter(function (e) { return e.event_type === 'asr_final'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        const partials = timeline.filter(function (e) { return e.event_type === 'asr_partial'; });
        for (let i = 0; i < finals.length; i++) {
            const asrFinalTime = finals[i].timestamp != null ? Number(finals[i].timestamp) : null;
            if (asrFinalTime == null || isNaN(asrFinalTime)) continue;
            const prevFinalTime = i > 0 && finals[i - 1].timestamp != null ? Number(finals[i - 1].timestamp) : 0;
            const turnPartials = partials.filter(function (p) {
                const pt = p.timestamp != null ? Number(p.timestamp) : null;
                return pt != null && !isNaN(pt) && pt > prevFinalTime && pt <= asrFinalTime;
            });
            const firstPartialTime = turnPartials.length ? Math.min.apply(null, turnPartials.map(function (p) { return Number(p.timestamp); })) : (asrFinalTime - 0.3);
            turnWindows.push({ firstPartialTime: firstPartialTime, asrFinalTime: asrFinalTime });
        }
    }
    // First TTS sound per turn: prefer timeline's first audio_amplitude (source tts) at 6.69s; else first segment with signal or first chunk
    const firstTtsAmplitudeByTurn = [];
    if (timeline && timeline.length && turnWindows.length) {
        const ttsAmpEvents = timeline.filter(function (e) { return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai'); }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        for (var ti = 0; ti < turnWindows.length; ti++) {
            const win = turnWindows[ti];
            const nextFinal = turnWindows[ti + 1] ? turnWindows[ti + 1].asrFinalTime : Infinity;
            const firstInWindow = ttsAmpEvents.find(function (e) {
                var t = e.timestamp != null ? Number(e.timestamp) : NaN;
                return !isNaN(t) && t > win.asrFinalTime && t < nextFinal;
            });
            firstTtsAmplitudeByTurn.push(firstInWindow ? Number(firstInWindow.timestamp) : null);
        }
    }
    // Segment-based fallback: first chunk with amplitude > 0 or first chunk
    const firstPlayPerTurn = responseGroups.map(function (group, k) {
        var firstChunkTime = group[0].startTime;
        const withSignal = group.filter(function (s) { return (s.amplitude != null ? s.amplitude : 0) > 0; });
        var segmentTime = withSignal.length ? Math.min(firstChunkTime, Math.min.apply(null, withSignal.map(function (s) { return s.startTime; }))) : firstChunkTime;
        var timelineFirst = firstTtsAmplitudeByTurn[k];
        return (timelineFirst != null && timelineFirst > 0) ? Math.min(timelineFirst, segmentTime) : segmentTime;
    });
    // One band per turn: S must be in (firstPartialTime, asrFinalTime]; if no timeline, use any S < T
    const bands = [];
    const usedSilence = {};
    for (let k = 0; k < firstPlayPerTurn.length; k++) {
        const T = firstPlayPerTurn[k];
        const win = turnWindows[k];
        const minS = win ? win.firstPartialTime : 0;
        const maxS = win ? win.asrFinalTime : T;
        let bestS = -1;
        let bestJ = -1;
        for (let j = 0; j < silenceStarts.length; j++) {
            if (usedSilence[j]) continue;
            const S = silenceStarts[j];
            if (S < T && S > bestS && S > minS && S <= maxS) { bestS = S; bestJ = j; }
        }
        if (bestJ >= 0) {
            usedSilence[bestJ] = true;
            bands.push({ start: bestS, end: T, ttlMs: Math.round((T - bestS) * 1000) });
        } else if (win && maxS < T) {
            // Fallback: place band start between partial and final (e.g. 150ms before final)
            const fallbackS = Math.max(minS, maxS - 0.15);
            bands.push({ start: fallbackS, end: T, ttlMs: Math.round((T - fallbackS) * 1000) });
        } else {
            // No turn window for this index (e.g. fewer asr_finals than TTS response groups) or maxS >= T: still show band so 2nd+ turns get a red band
            const fallbackS = Math.max(0, T - 0.5);
            bands.push({ start: fallbackS, end: T, ttlMs: Math.round((T - fallbackS) * 1000) });
        }
    }
    // If we have more turns (asr_finals) than TTS segment groups (e.g. first turn's segments were trimmed), add bands for the missing early turns from timeline so every turn gets a red band
    if (turnWindows.length > bands.length && firstTtsAmplitudeByTurn.length >= turnWindows.length) {
        var missingCount = turnWindows.length - bands.length;
        for (var ti = 0; ti < missingCount; ti++) {
            const T = firstTtsAmplitudeByTurn[ti];
            const win = turnWindows[ti];
            if (T == null || T <= 0 || !win) continue;
            const fallbackS = Math.max(win.firstPartialTime, win.asrFinalTime - 0.15);
            bands.unshift({ start: fallbackS, end: T, ttlMs: Math.round((T - fallbackS) * 1000) });
        }
    }
    return bands;
}

// Draw timeline events with support for rectangles, waveforms, and points
function drawTimelineEvents(ctx, timeline, lanes, LANE_HEIGHTS, laneYOffsets, LANE_GAP, PADDING_TOP,
                            PADDING_LEFT, PADDING_RIGHT, width, timeScale, timelineOffset,
                            laneColors, combineSpeechLanes, inLive, hasStoppedLiveData, liveAmplitudeHistory, liveTtsSegments, liveSystemStats, replayTtsSegments, replayAudioAmplitudeHistory, replayUserAmplitudeForTtl, liveSessionTime) {
    if (replayTtsSegments === undefined) replayTtsSegments = null;
    if (replayAudioAmplitudeHistory === undefined) replayAudioAmplitudeHistory = null;
    if (replayUserAmplitudeForTtl === undefined) replayUserAmplitudeForTtl = replayAudioAmplitudeHistory;
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

        // Find first and last partial for this turn (sort by timestamp so first/last are correct)
        const turnPartials = asrPartials.filter(p =>
            p.timestamp >= speechStart.timestamp &&
            (!asrFinal || p.timestamp <= asrFinal.timestamp)
        ).sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
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

        // Phase 2: Blue = first partial → last partial (active ASR; partials coming in)
        if (firstPartial && lastPartial && lastPartial.timestamp > firstPartial.timestamp) {
            inferredRectangles.push({
                event_type: 'asr_active',
                lane: 'speech',
                start_time: firstPartial.timestamp,
                end_time: lastPartial.timestamp,
                timestamp: firstPartial.timestamp,
                phase: 'active-asr',
                inferred: true
            });
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

    // Debug logging only when debug panel is on (avoid 60fps log spam during live session which blocks main thread and prevents ASR/timeline from updating)
    if (uiSettings.showDebugInfo) {
        console.log('Timeline rendering:', {
            totalEvents: timeline.length,
            inferredRectangles: inferredRectangles.length,
            explicitRectangles: rectangleEvents.length - inferredRectangles.length,
            totalRectangles: rectangleEvents.length,
            waveforms: waveformEvents.length,
            points: pointEvents.length,
            sampleInferred: inferredRectangles[0]
        });
    }

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

    // --- AUDIO lane: two rendering paths (same for live and saved session) ---
    // Dense: bars at fixed tStep (0.025s) from amplitude history (user) or segment list (AI). Source: live = state
    //   .liveAudioAmplitudeHistory / .liveTtsAmplitudeHistory; replay = session.audio_amplitude_history /
    //   session.tts_playback_segments. User gain (×1/×2/×4) and AI gain apply to all dense and sparse bars.
    // Sparse: one bar per timeline event that has amplitude or render_type === 'waveform' (waveformEvents).
    // AI/TTS is primarily drawn from the dense segment list (live or replayed); sparse waveform events are an
    // additional layer when the timeline contains point-style waveform events.
    // 2. Draw waveforms (audio lane visualization)
    // Replay: build TTS segments and ASR final times for smarter user (green) waveform visibility
    const ttsStartsSorted = timeline.filter(e => e.event_type === 'tts_start').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const ttsCompletesSorted = timeline.filter(e => e.event_type === 'tts_complete').sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
    const ttsTimeRanges = ttsStartsSorted.map((s, i) => ({ start: s.timestamp || 0, end: (ttsCompletesSorted[i] && (ttsCompletesSorted[i].timestamp || 0)) || (s.timestamp || 0) }));
    const userVoiceTh = (typeof uiSettings.userVoiceThreshold === 'number' && !isNaN(uiSettings.userVoiceThreshold)) ? uiSettings.userVoiceThreshold : 5;
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
            // Replay: skip sparse user bars when we draw the dense user waveform (2b1). Otherwise we get
            // near-duplicate bars (sparse at e.g. 0.04021 and dense at 0.025/0.05 grid) with 3–5 ms offset.
            if (replayAudioAmplitudeHistory && replayAudioAmplitudeHistory.length > 0) return;
            // During TTS: hide only low user amplitude (ambient); show significant level (possible interruption)
            if (inTts && amp < userVoiceTh) return;
            // NOTE: Near-full user amplitude with no nearby asr_final is NOT hidden here — it indicates a bug (e.g. mic artifact, client sending wrong data, or server RMS). See docs/INVESTIGATE_USER_AMPLITUDE_ARTIFACT.md
        }
        if (!inLive && !hasStoppedLiveData && !isUser) {
            // Replay: skip sparse TTS bars when we draw dense TTS from timeline amplitude (2b replay); avoids massive duplicate purple.
            if (timeline && timeline.some(function (e) { return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai'); })) return;
        }

        const x = PADDING_LEFT + (event.timestamp - timelineOffset) * timeScale;
        const laneY = getLaneY(laneIndex);
        const laneH = getLaneHeight(laneIndex);

        // Waveform bar: amplitude 0–100 with UI gain (User ×1/×2/×4, AI same). Same gain as dense live/replay.
        const userGain = (typeof uiSettings.userAudioGain === 'number' ? uiSettings.userAudioGain : 1);
        const aiGain = (typeof uiSettings.aiAudioGain === 'number' ? uiSettings.aiAudioGain : 1);
        const rawAmplitude = event.amplitude != null ? event.amplitude : 50;
        const amplitude = isUser ? Math.min(100, rawAmplitude * userGain) : Math.min(100, rawAmplitude * aiGain);
        const barHeight = (amplitude / 100) * (laneH * 0.9);
        const y = laneY + laneH / 2 - barHeight / 2;

        const audioColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio').trim() || '#76B900';
        const ttsColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-ai').trim() || '#9C27B0';
        ctx.fillStyle = isUser ? audioColor : ttsColor;
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
            // Green: mic (user) waveform. One amplitude bar = 25ms (tStep = 0.025). User and AI data match this.
            if (liveAmplitudeHistory && liveAmplitudeHistory.length > 0) {
                const audioColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio').trim() || '#76B900';
                ctx.fillStyle = audioColor;
                ctx.globalAlpha = 0.85;
                const userGain = (typeof uiSettings.userAudioGain === 'number' ? uiSettings.userAudioGain : 1);
                const barWidthPx = 2;
                const tStep = 0.025; // same as TTS (~40 Hz) for matching visual density
                const lastUserTs = liveAmplitudeHistory.length ? (liveAmplitudeHistory[liveAmplitudeHistory.length - 1].timestamp != null ? liveAmplitudeHistory[liveAmplitudeHistory.length - 1].timestamp : liveAmplitudeHistory[liveAmplitudeHistory.length - 1][0]) : 0;
                for (let t = visibleStart; t <= visibleEnd; t += tStep) {
                    const rawAmp = (inLive && t > lastUserTs) ? 0 : getAmplitudeAtTime(liveAmplitudeHistory, t);
                    const amp = Math.min(100, rawAmp * userGain);
                    if (amp <= 0) continue;
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
                    const ampForBar = Math.max(amp, 2); // small threshold so first/low-amplitude AI audio still shows purple
                    const halfH = (Math.min(100, Math.max(0, ampForBar)) / 100) * maxBarHalf;
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

    // 2b1. Replay (saved session): draw persisted user (mic) waveform — only when t in data range and hide during TTS if below threshold (no green over AI when user silent)
    if (!inLive && !hasStoppedLiveData && replayAudioAmplitudeHistory && replayAudioAmplitudeHistory.length > 0) {
        const laneIndex = lanes.indexOf('audio');
        if (laneIndex !== -1) {
            const getT = (s) => s.timestamp != null ? s.timestamp : s[0];
            const userT0 = getT(replayAudioAmplitudeHistory[0]);
            const userT1 = getT(replayAudioAmplitudeHistory[replayAudioAmplitudeHistory.length - 1]);
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
            const audioColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio').trim() || '#76B900';
            ctx.fillStyle = audioColor;
            ctx.globalAlpha = 0.85;
            const userGain = (typeof uiSettings.userAudioGain === 'number' ? uiSettings.userAudioGain : 1);
            const barWidthPx = 2;
            const tStep = 0.025;
            for (let t = visibleStart; t <= visibleEnd; t += tStep) {
                if (t < userT0 || t > userT1) continue;
                const amp = Math.min(100, getAmplitudeAtTime(replayAudioAmplitudeHistory, t) * userGain);
                if (amp <= 0) continue;
                const x = visibleLeft + (t - timelineOffset) * timeScale;
                const halfH = (Math.min(100, Math.max(0, amp)) / 100) * maxBarHalf;
                const y1 = centerY - halfH;
                const barHeight = Math.max(1, halfH * 2);
                ctx.fillRect(x, y1, barWidthPx, barHeight);
            }
            ctx.globalAlpha = 1.0;
        }
    }

    // 2b2. Replay (saved session): draw TTS only from segments when we have them (no timeline fill = no strange decay). When no segments, fill from timeline in [ttsT0, ttsT1] only.
    if (!inLive && !hasStoppedLiveData && ((replayTtsSegments && replayTtsSegments.length > 0) || (timeline && timeline.length))) {
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
            const ttsColor = getComputedStyle(document.documentElement).getPropertyValue('--timeline-ai').trim() || '#9C27B0';
            ctx.fillStyle = ttsColor;
            ctx.globalAlpha = 0.9;
            const aiGain = (typeof uiSettings.aiAudioGain === 'number' ? uiSettings.aiAudioGain : 1);
            const barWidthPx = 2;
            const tStep = 0.025;
            // Replay: prefer actual amplitude from timeline; only use segment fill when no timeline TTS amplitude (old sessions).
            var hasTimelineTtsAmp = timeline && timeline.some(function (e) { return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai'); });
            if (hasTimelineTtsAmp) {
                var ttsFromTimeline = [];
                timeline.filter(function (e) { return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai'); }).forEach(function (e) {
                    var a = e.amplitude != null ? Number(e.amplitude) : 0;
                    var ts = e.timestamp != null ? Number(e.timestamp) : 0;
                    ttsFromTimeline.push({ timestamp: ts, amplitude: a });
                });
                ttsFromTimeline.sort(function (a, b) { return a.timestamp - b.timestamp; });
                // Server stores 0–100 (RMS scale). Only normalize 0–1 → 0–100 when data looks like 0–1 (max ≤ 1).
                var ttsMaxAmp = ttsFromTimeline.length ? Math.max.apply(null, ttsFromTimeline.map(function (s) { return s.amplitude; })) : 0;
                if (ttsMaxAmp > 0 && ttsMaxAmp <= 1) {
                    ttsFromTimeline.forEach(function (s) { s.amplitude = s.amplitude * 100; });
                }
                var ttsDataT0 = ttsFromTimeline.length ? ttsFromTimeline[0].timestamp : 0;
                var ttsDataT1 = ttsFromTimeline.length ? ttsFromTimeline[ttsFromTimeline.length - 1].timestamp : 0;
                for (var t = visibleStart; t <= visibleEnd; t += tStep) {
                    if (t < ttsDataT0 || t > ttsDataT1) continue;
                    // maxGapSec=0.25: interpolate within one TTS response (chunk gaps ~50–100ms) but don't draw in silence between two responses (avoids bridge of tiny dots)
                    var ampTl = getAmplitudeAtTime(ttsFromTimeline, t, 0.25);
                    var amp = Math.min(100, ampTl * aiGain);
                    if (amp <= 0) continue;
                    var ampForBar = Math.max(amp, 2);
                    var halfH = (Math.min(100, Math.max(0, ampForBar)) / 100) * maxBarHalf;
                    var x = visibleLeft + (t - timelineOffset) * timeScale;
                    if (x < visibleLeft - barWidthPx || x > visibleRight + barWidthPx) continue;
                    ctx.fillRect(x, centerY - halfH, barWidthPx, Math.max(1, halfH * 2));
                }
            } else if (replayTtsSegments && replayTtsSegments.length > 0) {
                replayTtsSegments.forEach(function (seg) {
                    if (seg.endTime < visibleStart || seg.startTime > visibleEnd) return;
                    const amp = Math.min(100, (seg.amplitude || 0) * aiGain);
                    const ampForBar = Math.max(amp, 2);
                    const halfH = (Math.min(100, Math.max(0, ampForBar)) / 100) * maxBarHalf;
                    for (let t = Math.max(visibleStart, seg.startTime); t <= Math.min(visibleEnd, seg.endTime); t += tStep) {
                        const x = visibleLeft + (t - timelineOffset) * timeScale;
                        if (x < visibleLeft - barWidthPx || x > visibleRight + barWidthPx) continue;
                        const y1 = centerY - halfH;
                        ctx.fillRect(x, y1, barWidthPx, Math.max(1, halfH * 2));
                    }
                });
            }
            ctx.globalAlpha = 1.0;
        }
    }

    // 2c. Replay fallback: when no dense TTS data and no timeline TTS amplitude, draw flat TTS blocks from tts_start→tts_complete.
    var hasTimelineTts = timeline && timeline.some(function (e) { return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai'); });
    if (!inLive && !hasStoppedLiveData && !liveTtsSegments && (!replayTtsSegments || replayTtsSegments.length === 0) && !hasTimelineTts) {
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

    // 3. Draw point events: ASR partial = light blue small dot, ASR final = blue dot + transcript text; LLM/TTS lanes = no dots (tts_first_audio shown as thin pale magenta line); llm_complete = response label only; Realtime = system + TTS output transcript
    pointEvents.forEach(event => {
        let targetLane = event.lane;
        if (targetLane === 'audio') return;
        // Draw user_speech_end on speech lane so TTL boundary is visible (backend sends it as system)
        if (targetLane === 'system' && event.event_type === 'user_speech_end') targetLane = 'speech';
        // Realtime session_ready and error stay on system lane and are drawn below
        if (targetLane === 'system' && event.event_type !== 'user_speech_end' && event.event_type !== 'realtime_session_ready' && event.event_type !== 'error') return;

        if (combineSpeechLanes && targetLane === 'tts') targetLane = 'speech';

        const laneIndex = lanes.indexOf(targetLane);
        if (laneIndex === -1) return;

        const x = PADDING_LEFT + (event.timestamp - timelineOffset) * timeScale;
        const laneY = getLaneY(laneIndex);
        const laneH = getLaneHeight(laneIndex);
        // When ASR and TTS share one lane: top half = ASR, bottom half = TTS (avoids overlap)
        let centerY = laneY + laneH / 2;
        if (combineSpeechLanes && targetLane === 'speech') {
            const et0 = event.event_type || event.eventType || '';
            if (et0 === 'asr_partial' || et0 === 'asr_final' || et0 === 'user_speech_end') {
                centerY = laneY + laneH * 0.25;
            } else if (et0 === 'realtime_output_partial' || et0 === 'realtime_output_final' || et0 === 'tts_complete') {
                centerY = laneY + laneH * 0.75;
            }
        }

        const et = event.event_type || event.eventType || '';
        // System lane: Realtime session_ready (green), error (red)
        if (targetLane === 'system') {
            let dotColor = '#888';
            let dotRadius = 0;
            let label = '';
            if (et === 'realtime_session_ready') {
                dotColor = '#4CAF50';
                dotRadius = 4;
                label = 'Realtime ready';
            } else if (et === 'error') {
                dotColor = '#D32F2F';
                dotRadius = 4;
                label = (event.data && event.data.message) ? String(event.data.message) : 'Error';
            }
            if (dotRadius > 0) {
                ctx.fillStyle = dotColor;
                ctx.beginPath();
                ctx.arc(x, centerY, dotRadius, 0, Math.PI * 2);
                ctx.fill();
            }
            if (label) {
                const shortLabel = truncateTimelineLabel(label, 40);
                ctx.font = '10px sans-serif';
                ctx.fillStyle = dotColor;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const textX = x + dotRadius + 6;
                if (textX < width - PADDING_RIGHT - 2) {
                    ctx.fillText(shortLabel, textX, centerY);
                }
            }
            return;
        }
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
                dotRadius = 0; // e.g. user_speech_end: no dot
            }
        } else {
            if (et === 'asr_partial') {
                dotColor = laneColors.speechPartial || '#64B5F6';
                dotRadius = 2.5;
            } else if (et === 'asr_final') {
                dotColor = laneColors.speech || '#1976D2';
                dotRadius = 4;
            } else if (et === 'realtime_output_partial') {
                dotColor = laneColors.tts || '#9C27B0';
                dotRadius = 2.5;
            } else if (et === 'realtime_output_final') {
                dotColor = laneColors.tts || '#9C27B0';
                dotRadius = 4;
            } else if (targetLane === 'llm' || targetLane === 'tts') {
                dotRadius = 0; // No dots on LLM/TTS lanes; TTS first audio is shown as thin pale magenta line
            }
        }

        if (dotRadius > 0) {
            ctx.fillStyle = dotColor;
            ctx.beginPath();
            ctx.arc(x, centerY, dotRadius, 0, Math.PI * 2);
            ctx.fill();
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
        // Realtime AI output: draw transcript to the right of realtime_output_final (and tts_complete when it has text)
        if ((et === 'realtime_output_final' || (et === 'tts_complete' && event.data && event.data.text)) && targetLane === (combineSpeechLanes ? 'speech' : 'tts')) {
            const text = (event.data && event.data.text) ? String(event.data.text) : '';
            if (text) {
                const label = truncateTimelineLabel(text, 60);
                ctx.font = '11px sans-serif';
                ctx.fillStyle = laneColors.tts || '#9C27B0';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const textX = x + (dotRadius > 0 ? dotRadius : 4) + 6;
                if (textX < width - PADDING_RIGHT - 2) {
                    ctx.fillText(label, textX, centerY);
                }
            }
        }
    });

    // 3b. TTL bands: transparent red from JS end-of-speech to first sound out (browser); ms label on TTL lane. Live: from state; replay: always recompute from segments when we have them so band end = first AI sound (~6.6s), not stored bands from old logic (~8s).
    const ttlLaneIndex = lanes.indexOf('ttl');
    const replayBands = (!inLive && !hasStoppedLiveData && state.selectedSession)
        ? (replayTtsSegments && replayTtsSegments.length
            ? computeTtlBandsFromReplay(replayUserAmplitudeForTtl || [], replayTtsSegments, timeline)
            : (state.selectedSession.ttl_bands || []))
        : [];
    const hasLiveBands = drawLiveWaveforms && (state.liveTtlBands && state.liveTtlBands.length > 0 || state.liveTtlBandStartTime != null);
    if (ttlLaneIndex !== -1 && (hasLiveBands || replayBands.length > 0)) {
        const bandTop = getLaneY(0);
        const bandBottom = getLaneY(lanes.length - 1) + getLaneHeight(lanes.length - 1);
        const ttlLaneY = getLaneY(ttlLaneIndex);
        const ttlLaneH = getLaneHeight(ttlLaneIndex);
        const ttlLaneCenterY = ttlLaneY + ttlLaneH / 2;
        const redFill = getComputedStyle(document.documentElement).getPropertyValue('--ttl-band-fill').trim() || 'rgba(200, 50, 50, 0.12)';
        const bandsToDraw = [];
        if (hasLiveBands) {
            if (state.liveTtlBands) bandsToDraw.push(...state.liveTtlBands);
            if (state.liveTtlBandStartTime != null && liveSessionTime != null)
                bandsToDraw.push({ start: state.liveTtlBandStartTime, end: liveSessionTime, ttlMs: null });
        }
        if (replayBands.length > 0) bandsToDraw.push(...replayBands);
        bandsToDraw.forEach(function (band) {
            const x1 = PADDING_LEFT + (band.start - timelineOffset) * timeScale;
            const x2 = PADDING_LEFT + (band.end - timelineOffset) * timeScale;
            if (x2 <= PADDING_LEFT || x1 >= width - PADDING_RIGHT) return;
            const drawX1 = Math.max(PADDING_LEFT, x1);
            const drawX2 = Math.min(width - PADDING_RIGHT, x2);
            ctx.fillStyle = redFill;
            ctx.fillRect(drawX1, bandTop, drawX2 - drawX1, bandBottom - bandTop);
            if (band.ttlMs != null && (drawX2 - drawX1) > 30) {
                ctx.font = '11px sans-serif';
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--ttl-band-text').trim() || 'rgba(180, 40, 40, 0.95)';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const midX = (drawX1 + drawX2) / 2;
                ctx.fillText(band.ttlMs + ' ms', midX, ttlLaneCenterY);
            }
        });
    }

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
            // Monotonic quadratic control point: smooth but never goes backward in time or overshoots (no hooks/lobes)
            function smoothQuadraticCtrl(p0, p1, pPrev) {
                if (pPrev == null) return { x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2 };
                var c = { x: p0.x + (p1.x - pPrev.x) * 0.25, y: p0.y + (p1.y - pPrev.y) * 0.25 };
                c.x = Math.max(p0.x, Math.min(p1.x, c.x));
                var yMin = Math.min(p0.y, p1.y), yMax = Math.max(p0.y, p1.y);
                c.y = Math.max(yMin, Math.min(yMax, c.y));
                return c;
            }
            // Monotonic cubic Bézier control points (Catmull-Rom style, clamped): smoother curve, no time-reversal or lobes
            function smoothCubicBezierCtrl(pPrev, p0, p1, pNext) {
                var tension = 1 / 3; // standard cubic Bézier from Catmull-Rom
                var m0x = pPrev != null ? (p1.x - pPrev.x) * 0.5 : (p1.x - p0.x);
                var m0y = pPrev != null ? (p1.y - pPrev.y) * 0.5 : (p1.y - p0.y);
                var m1x = pNext != null ? (pNext.x - p0.x) * 0.5 : (p1.x - p0.x);
                var m1y = pNext != null ? (pNext.y - p0.y) * 0.5 : (p1.y - p0.y);
                var cp1 = { x: p0.x + m0x * tension, y: p0.y + m0y * tension };
                var cp2 = { x: p1.x - m1x * tension, y: p1.y - m1y * tension };
                var xMin = Math.min(p0.x, p1.x), xMax = Math.max(p0.x, p1.x);
                var yMin = Math.min(p0.y, p1.y), yMax = Math.max(p0.y, p1.y);
                cp1.x = Math.max(xMin, Math.min(xMax, cp1.x));
                cp1.y = Math.max(yMin, Math.min(yMax, cp1.y));
                cp2.x = Math.max(xMin, Math.min(xMax, cp2.x));
                cp2.y = Math.max(yMin, Math.min(yMax, cp2.y));
                return { cp1: cp1, cp2: cp2 };
            }
            var useCubicBezier = true; // cubic Bézier = smoother; false = monotonic quadratic
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
                            var prev = k >= 2 ? points[k - 2] : null;
                            var next = k + 1 < points.length ? points[k + 1] : null;
                            if (useCubicBezier) {
                                var c = smoothCubicBezierCtrl(prev, p0, p1, next);
                                ctx.bezierCurveTo(c.cp1.x, c.cp1.y, c.cp2.x, c.cp2.y, p1.x, p1.y);
                            } else {
                                var c = smoothQuadraticCtrl(p0, p1, prev);
                                ctx.quadraticCurveTo(c.x, c.y, p1.x, p1.y);
                            }
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
                        var prev = k >= 2 ? points[k - 2] : null;
                        var next = k + 1 < points.length ? points[k + 1] : null;
                        if (useCubicBezier) {
                            var c = smoothCubicBezierCtrl(prev, p0, p1, next);
                            ctx.bezierCurveTo(c.cp1.x, c.cp1.y, c.cp2.x, c.cp2.y, p1.x, p1.y);
                        } else {
                            var c = smoothQuadraticCtrl(p0, p1, prev);
                            ctx.quadraticCurveTo(c.x, c.y, p1.x, p1.y);
                        }
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
            // Clip to lane bounds so quadratic curves never render below 0% (baseline)
            ctx.save();
            ctx.beginPath();
            ctx.rect(visibleLeft, laneY, visibleRight - visibleLeft, laneH);
            ctx.clip();
            // Draw CPU area first (transparent blue), then GPU area (transparent green); overlap blends darker
            drawAreaAndLine(cpuPoints, '#2196F3', '#2196F3', 0.45);
            drawAreaAndLine(gpuPoints, '#4CAF50', '#4CAF50', 0.45);
            ctx.restore();
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
    if (value === 'none') return type === 'camera' ? '\uD83D\uDEABNone (No vision-modality)' : '\uD83D\uDEABNone (Text Only)';
    return value || '—';
}

/** Prefer actual device name: from saved device_labels, stream/dropdown when config is browser, or dropdown option text for server devices (e.g. EMEET OfficeCore M0 Plus (Server USB)). */
function getDeviceDisplayLabel(kind) {
    var d = currentConfig.devices || {};
    var labels = currentConfig.device_labels || {};
    if (kind === 'camera') {
        if ((d.camera === 'browser' || d.camera === '') && state.previewStream) {
            var videoTracks = state.previewStream.getVideoTracks();
            if (videoTracks.length > 0 && videoTracks[0].label) return videoTracks[0].label;
        }
        if (d.camera === 'browser' || d.camera === '') return 'Default (Browser)';
        if (labels.camera) return labels.camera;
        var camSel = document.getElementById('device-camera-list');
        if (camSel && d.camera && camSel.value === d.camera) {
            var opt = camSel.options[camSel.selectedIndex];
            if (opt && opt.textContent) return opt.textContent;
        }
        if (d.camera && d.camera.indexOf('/dev/') === 0) return d.camera;
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
        if (d.microphone !== 'none' && labels.mic) return labels.mic;
        var micSel = document.getElementById('device-microphone-list');
        if (micSel && d.microphone && micSel.value === d.microphone) {
            var micOpt = micSel.options[micSel.selectedIndex];
            if (micOpt && micOpt.textContent) return micOpt.textContent;
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
        if (d.speaker !== 'none' && labels.speaker) return labels.speaker;
        var spkSel = document.getElementById('device-speaker-list');
        if (spkSel && d.speaker && spkSel.value === d.speaker) {
            var spkOpt = spkSel.options[spkSel.selectedIndex];
            if (spkOpt && spkOpt.textContent) return spkOpt.textContent;
        }
        return deviceValueToLabel(d.speaker, 'speaker');
    }
    return '—';
}

/** Device type for pipeline badge: 'browser' | 'usb' | null (null for n/a or text-only). */
function getDeviceDisplayType(kind) {
    var d = currentConfig.devices || {};
    var v;
    if (kind === 'mic') v = d.microphone ?? d.audio_input_source;
    else if (kind === 'camera') v = d.camera ?? d.video_source;
    else if (kind === 'speaker') v = d.speaker ?? d.audio_output_source;
    else return null;
    if (v == null || v === 'none') return null;
    if (v === 'browser' || v === '') return 'browser';
    return 'usb';
}

function updateDeviceIndicators() {
    var camTag = document.getElementById('device-tag-camera');
    var mic = document.getElementById('device-indicator-mic');
    var spk = document.getElementById('device-indicator-speaker');
    var camConfig = (currentConfig.devices || {}).camera;
    var hasCamera = camConfig !== 'none' && camConfig != null && camConfig !== undefined;
    if (camTag) {
        if (hasCamera) {
            camTag.classList.remove('device-tag--blank');
            camTag.innerHTML = '<i data-lucide="video" class="lucide-inline"></i> <span id="device-indicator-camera">' + escapeHtml(getDeviceDisplayLabel('camera')) + '</span>';
            if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons(camTag);
        } else {
            camTag.classList.add('device-tag--blank');
            camTag.innerHTML = '<span class="device-tag-placeholder" aria-hidden="true"></span>';
        }
    }
    if (mic) mic.textContent = getDeviceDisplayLabel('mic');
    if (spk) spk.textContent = getDeviceDisplayLabel('speaker');
}

/** Update image-placeholder content: when camera is None show "SOUND ONLY" (red) + video-off icon; otherwise Multi-modal Session + Vision + Voice. */
function updateImagePlaceholderContent(config) {
    var placeholder = document.getElementById('image-placeholder');
    if (!placeholder) return;
    var c = config || (state.isLiveSession ? currentConfig : (state.selectedSession && state.selectedSession.config ? state.selectedSession.config : currentConfig));
    var d = c.devices || {};
    var cam = d.camera ?? d.video_source;
    var noCamera = cam === 'none' || cam == null || cam === undefined;
    if (noCamera) {
        placeholder.className = 'image-placeholder placeholder--sound-only';
        placeholder.innerHTML = '<span class="placeholder-icon"><i data-lucide="video-off" class="lucide-inline" aria-hidden="true"></i></span><p class="placeholder-text">SOUND ONLY</p>';
    } else {
        placeholder.className = 'image-placeholder';
        placeholder.innerHTML = '<span class="placeholder-icon"><i data-lucide="video" class="lucide-inline" aria-hidden="true"></i></span><p class="placeholder-text">Multi-modal Session</p><p class="placeholder-subtext">Vision + Voice Analysis</p>';
    }
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons(placeholder);
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
    if (state.micPreviewWs) {
        try { state.micPreviewWs.close(); } catch (e) {}
        state.micPreviewWs = null;
    }
    if (state.voiceWs && state.sessionState === 'setup') {
        try { state.voiceWs.close(); } catch (e) {}
        state.voiceWs = null;
    }
    if (state.micWaveformAnimId != null) {
        cancelAnimationFrame(state.micWaveformAnimId);
        state.micWaveformAnimId = null;
    }
    if (state.micAudioContext) {
        try { state.micAudioContext.close(); } catch (e) {}
        state.micAudioContext = null;
    }
    state.micAnalyser = null;
    state.micWaveformFromServer = false;
    state.micAmplitudeBuffer = [];
    state._micWaveformDrawLogged = false;
    state._micWaveformSizeLogged = false;
    state._micWaveformFirstDrawLogged = false;
    var overlay = document.getElementById('mic-waveform-overlay');
    if (overlay) overlay.style.display = 'none';
}

/** Draw last 2000ms of mic as symmetric dotted waveform (above and below center), timeline-style. */
function drawMicWaveform() {
    var overlay = document.getElementById('mic-waveform-overlay');
    var canvas = document.getElementById('mic-waveform-canvas');
    if (!overlay || !canvas) return;
    var fromServer = state.micWaveformFromServer;
    if (!state.micAnalyser && !fromServer) return;
    if (window._micWaveformDebug && fromServer && !state._micWaveformDrawLogged) {
        state._micWaveformDrawLogged = true;
        var rect = overlay.getBoundingClientRect();
        console.log('[MicWaveform] drawMicWaveform (server): overlay display=' + overlay.style.display + ' rect=' + rect.width + 'x' + rect.height + ' parentDisplay=' + (overlay.parentElement ? getComputedStyle(overlay.parentElement).display : 'n/a'));
    }
    if (state.micAnalyser && !fromServer && !state.previewStream) return;

    var ring = state.micAmplitudeBuffer;
    if (!Array.isArray(ring)) ring = state.micAmplitudeBuffer = [];
    if (state.micAnalyser && !fromServer) {
        var buf = new Uint8Array(state.micAnalyser.fftSize);
        state.micAnalyser.getByteTimeDomainData(buf);
        var max = 0;
        for (var i = 0; i < buf.length; i++) {
            var v = Math.abs(buf[i] - 128);
            if (v > max) max = v;
        }
        var cap = 120; /* 2000ms at ~60fps */
        if (ring.length >= cap) ring.shift();
        ring.push(Math.min(128, max * getMicPreviewGain()));
    }

    var w = canvas.width = canvas.offsetWidth;
    var h = canvas.height = canvas.offsetHeight;
    if (w <= 0 || h <= 0) {
        if (fromServer && !state._micWaveformSizeLogged) {
            state._micWaveformSizeLogged = true;
            console.warn('[MicWaveform] Canvas size 0x0 – overlay may be hidden or not laid out yet');
        }
        state.micWaveformAnimId = requestAnimationFrame(drawMicWaveform);
        return;
    }
    if (fromServer && !state._micWaveformSizeLogged) {
        state._micWaveformSizeLogged = true;
        console.log('[MicWaveform] Canvas size ' + w + 'x' + h);
    }
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    var margin = 30;
    var innerPadding = 20;
    var drawW = Math.max(0, w - margin * 2);
    var centerY = h / 2;
    var scale = (h / 2) * 0.8 / 128;
    var radius = 6;

    /* Same rounded rect background for both browser-mic and server-mic (empty or with data). */
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

    if (ring.length < 2) { state.micWaveformAnimId = requestAnimationFrame(drawMicWaveform); return; }

    if (fromServer && !state._micWaveformFirstDrawLogged) {
        state._micWaveformFirstDrawLogged = true;
        console.log('[MicWaveform] Drawing server waveform (buffer len=' + ring.length + '); if you see this but no green bars, check canvas/overlay.');
    }
    var green = (getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio') || '').trim() || '#76B900';
    ctx.fillStyle = green;
    var waveformLeft = margin + innerPadding;
    var waveformWidth = Math.max(0, drawW - innerPadding * 2);
    var barWidthPx = 2;
    var step = ring.length > 1 ? Math.max(barWidthPx, (waveformWidth - barWidthPx) / (ring.length - 1)) : barWidthPx;
    for (var j = 0; j < ring.length; j++) {
        var x = waveformLeft + j * step;
        var halfH = ring[j] * scale;
        var yTop = centerY - halfH;
        var yBottom = centerY + halfH;
        var barHeight = Math.max(1, yBottom - yTop);
        ctx.fillRect(x, yTop, barWidthPx, barHeight);
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

/** Build the voice config object sent to the server (WS config message or start_session). Uses currentConfig and sets audio_output_* from devices.speaker so saved sessions have correct speaker. */
function buildVoiceConfig() {
    var config = {
        asr: { ...currentConfig.asr },
        llm: { ...currentConfig.llm },
        tts: { ...currentConfig.tts },
        devices: currentConfig.devices ? { ...currentConfig.devices } : {},
        app: currentConfig.app ? { ...currentConfig.app } : {},
        device_labels: { mic: getDeviceDisplayLabel('mic'), camera: getDeviceDisplayLabel('camera'), speaker: getDeviceDisplayLabel('speaker') },
        device_types: { mic: getDeviceDisplayType('mic'), camera: getDeviceDisplayType('camera'), speaker: getDeviceDisplayType('speaker') },
        asr_model_name: (currentConfig.asr && currentConfig.asr.model) ? String(currentConfig.asr.model).replace(/\(.*\)/, '').trim() : null,
        llm_model_name: (currentConfig.llm && currentConfig.llm.model) ? String(currentConfig.llm.model) : null,
        tts_model_name: (currentConfig.tts && (currentConfig.tts.riva_model_name || currentConfig.tts.voice || currentConfig.tts.model)) ? (currentConfig.tts.riva_model_name || currentConfig.tts.voice || currentConfig.tts.model) : null
    };
    if (config.asr.riva_server === undefined && config.asr.server) config.asr.riva_server = config.asr.server;
    if (config.tts.riva_server === undefined && config.tts.server) config.tts.riva_server = config.tts.server;
    // Ensure backend sends scheme + realtime fields so server picks Realtime path when UI is set to Realtime
    if ((config.asr.backend === 'openai-realtime' || config.asr.scheme === 'openai-realtime')) {
        config.asr.scheme = config.asr.scheme || config.asr.backend || 'openai-realtime';
        config.asr.realtime_transport = config.asr.realtime_transport || 'websocket';
        config.asr.realtime_session_type = config.asr.realtime_session_type || 'transcription';
    }
    if (config.tts.backend === 'openai-realtime' || config.tts.scheme === 'openai-realtime') {
        config.tts.scheme = config.tts.scheme || config.tts.backend || 'openai-realtime';
        config.tts.realtime_transport = config.tts.realtime_transport || 'websocket';
    }
    // When ASR is Realtime full-voice, server requires TTS=Realtime; force it in payload so server never sees asr=realtime + tts=riva
    if ((config.asr.backend === 'openai-realtime' || config.asr.scheme === 'openai-realtime') &&
        (config.asr.realtime_session_type || 'transcription') === 'full' &&
        (config.asr.realtime_transport || 'websocket') === 'websocket') {
        config.tts.scheme = 'openai-realtime';
        config.tts.backend = 'openai-realtime';
        config.tts.realtime_transport = 'websocket';
        if (!(config.tts.realtime_url || '').trim()) {
            config.tts.realtime_url = (config.asr.realtime_url || 'wss://api.openai.com/v1/realtime').trim();
        }
    }
    var spk = (config.devices && config.devices.speaker) ? String(config.devices.speaker) : '';
    if (spk.startsWith('alsa:')) {
        config.devices.audio_output_source = 'alsa';
        config.devices.audio_output_device = spk.slice(5) || 'default';
    } else if (spk.startsWith('pyaudio:')) {
        config.devices.audio_output_source = 'usb';
        config.devices.audio_output_device = spk.slice(8) || '';
    }
    // Device names (stable across reboots; session can resolve id by name when hw:N,M changes)
    var cam = (config.devices.camera || config.devices.video_device) ? String(config.devices.camera || config.devices.video_device) : '';
    if (cam && (cam.indexOf('/dev/') === 0 || config.devices.video_source === 'usb')) {
        config.devices.video_device_name = getDeviceDisplayLabel('camera') || undefined;
    }
    var mic = (config.devices.microphone) ? String(config.devices.microphone) : '';
    if (mic && (mic.indexOf('alsa:') === 0 || mic.indexOf('pyaudio:') === 0)) {
        config.devices.audio_input_device_name = getDeviceDisplayLabel('mic') || undefined;
    }
    if (spk && (spk.indexOf('alsa:') === 0 || spk.indexOf('pyaudio:') === 0)) {
        config.devices.audio_output_device_name = getDeviceDisplayLabel('speaker') || undefined;
    }
    var ttsModelSelect = document.getElementById('tts-model-select');
    var dropdownVal = (ttsModelSelect && ttsModelSelect.value && String(ttsModelSelect.value).trim()) || null;
    var sentRivaModelName = (currentConfig.tts && currentConfig.tts.riva_model_name) || dropdownVal || null;
    if (sentRivaModelName) {
        config.tts.riva_model_name = sentRivaModelName;
        config.tts_model_name = sentRivaModelName;
    }
    if (config.llm.ollama_url === undefined && config.llm.api_base) config.llm.ollama_url = (config.llm.api_base || '').replace(/\/v1\/?$/, '');
    return config;
}

/** Start preview waveform for Server USB mic: open /ws/voice and use user_amplitude for the green bar (same connection used for live). */
function startMicWaveformFromServer() {
    if (state.voiceWs && (state.voiceWs.readyState === WebSocket.OPEN || state.voiceWs.readyState === WebSocket.CONNECTING)) {
        var overlay = document.getElementById('mic-waveform-overlay');
        if (overlay) overlay.style.display = 'block';
        state.micWaveformFromServer = true;
        state.micAmplitudeBuffer = state.micAmplitudeBuffer || [];
        drawMicWaveform();
        return;
    }
    stopMicWaveform();
    var overlay = document.getElementById('mic-waveform-overlay');
    var canvas = document.getElementById('mic-waveform-canvas');
    if (!overlay || !canvas) {
        console.warn('[MicWaveform] startMicWaveformFromServer: overlay=' + !!overlay + ' canvas=' + !!canvas + ' (elements missing?)');
        return;
    }
    state.micWaveformFromServer = true;
    state.micAmplitudeBuffer = [];
    overlay.style.display = 'block';
    drawMicWaveform();

    // Single connection: use /ws/voice for preview (same 50 Hz user_amplitude). On START we send start_session on this WS instead of reopening.
    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    var wsUrl = protocol + '//' + window.location.host + '/ws/voice';
    if (window._micWaveformDebug) console.log('[MicWaveform] Opening voice WebSocket for Server USB preview:', wsUrl);
    var ws = new WebSocket(wsUrl);
    state.voiceWs = ws;
    ws.onopen = function () {
        var config = buildVoiceConfig();
        ws.send(JSON.stringify({ type: 'config', config: config }));
        if (window._micWaveformDebug) console.log('[MicWaveform] Voice config sent (preview); server will stream user_amplitude at 50 Hz');
    };
    ws.onmessage = handleVoiceWsMessage;
    ws.onclose = handleVoiceWsClose;
    ws.onerror = function () {
        console.error('[Voice] WebSocket error');
        if (state.voiceWs === ws) state.voiceWs = null;
    };
}

/** Stop camera/mic preview stream and clear video/img elements. Call on STOP or when leaving live session. */
function stopPreviewStream() {
    // Keep Server USB voice WS open when we're in setup with Server USB selected, so a refresh (e.g. updateLiveSessionUI) doesn't close and immediately reopen it and hit "Device or resource busy".
    if (!(state.sessionState === 'setup' && isServerMicSelected())) {
        stopMicWaveform();
    }
    if (state.previewStream) {
        state.previewStream.getTracks().forEach(function (t) { t.stop(); });
        state.previewStream = null;
    }
    const videoFeed = document.getElementById('video-feed');
    const mjpegFeed = document.getElementById('video-feed-mjpeg');
    if (videoFeed) {
        videoFeed.srcObject = null;
        videoFeed.src = '';
    }
    if (mjpegFeed) {
        mjpegFeed.src = '';
    }
    if (state.cameraWebrtcWs) {
        try { state.cameraWebrtcWs.close(); } catch (e) {}
        state.cameraWebrtcWs = null;
    }
    if (state.cameraWebrtcPc) {
        try { state.cameraWebrtcPc.close(); } catch (e) {}
        state.cameraWebrtcPc = null;
    }
    state.previewServerCameraDevice = null;
}

function isServerMicSelected() {
    var mic = (currentConfig.devices || {}).microphone;
    return mic && (String(mic).startsWith('alsa:') || String(mic).startsWith('pyaudio:'));
}

function isServerSpeakerSelected() {
    var d = currentConfig.devices || {};
    var spk = d.speaker;
    if (spk && (String(spk).startsWith('alsa:') || String(spk).startsWith('pyaudio:'))) return true;
    if (d.audio_output_source === 'alsa' || d.audio_output_source === 'usb') return !!d.audio_output_device;
    return false;
}

/** UI-only gain for mic preview waveform (1–4). Used for both browser and server USB mic bar. */
function getMicPreviewGain() {
    var g = (typeof uiSettings.micPreviewGain === 'number' && !isNaN(uiSettings.micPreviewGain)) ? uiSettings.micPreviewGain : 2;
    return Math.max(1, Math.min(4, g));
}

/**
 * Start preview: browser camera (getUserMedia) and/or server camera (Server USB MJPEG stream), plus mic.
 * Mic preview has two paths: (1) Browser device (e.g. AirPods) uses getUserMedia + AnalyserNode, no WebSocket.
 * (2) Server USB mic uses /ws/mic-preview to route amplitude from server capture to this client for the same waveform UI.
 * options.keepServerCamera: if true, do not tear down server camera WebRTC when it is already open for the same device (avoids release/reopen race when only mic/speaker changes).
 */
function startPreviewStream(options) {
    if (!state.isLiveSession || state.sessionState !== 'setup') {
        if (window._micWaveformDebug) console.log('[MicWaveform] startPreviewStream skipped: isLiveSession=' + state.isLiveSession + ' sessionState=' + state.sessionState);
        return;
    }
    const d = currentConfig.devices || {};
    const isJetsonCamera = (d.camera && typeof d.camera === 'string' && d.camera.indexOf('/dev/') === 0);
    const wantJetsonVideo = (d.camera !== 'none' && d.camera != null && d.camera !== undefined && isJetsonCamera);
    const wantBrowserVideo = (d.camera !== 'none' && d.camera != null && d.camera !== undefined && !isJetsonCamera);
    const wantAudio = d.microphone !== 'none' && d.microphone != null;
    const wantAudioForPreview = wantAudio && !isServerMicSelected();
    const serverCamDevice = (wantJetsonVideo && d.camera) ? d.camera : null;
    const keepServerCamera = options && options.keepServerCamera && wantJetsonVideo && state.cameraWebrtcPc && state.previewServerCameraDevice === serverCamDevice;
    if (!wantJetsonVideo && !wantBrowserVideo && !wantAudio) {
        stopPreviewStream();
        const videoFeed = document.getElementById('video-feed');
        const mjpegFeed = document.getElementById('video-feed-mjpeg');
        const imagePlaceholder = document.getElementById('image-placeholder');
        if (videoFeed) {
            videoFeed.src = '';
            videoFeed.srcObject = null;
            videoFeed.style.display = 'none';
        }
        if (mjpegFeed) {
            mjpegFeed.src = '';
            mjpegFeed.style.display = 'none';
        }
        if (imagePlaceholder) {
            imagePlaceholder.style.display = 'flex';
            updateImagePlaceholderContent();
        }
        return;
    }

    if (!keepServerCamera) stopPreviewStream();
    if (wantAudio && isServerMicSelected()) {
        if (window._micWaveformDebug) console.log('[MicWaveform] startPreviewStream: calling startMicWaveformFromServer (wantAudio, Server USB selected)');
        startMicWaveformFromServer();
    }
    const videoFeed = document.getElementById('video-feed');
    const mjpegFeed = document.getElementById('video-feed-mjpeg');
    const imagePlaceholder = document.getElementById('image-placeholder');

    if (wantJetsonVideo && !keepServerCamera) {
        var deviceParam = (d.camera && d.camera !== '') ? encodeURIComponent(d.camera) : '';
        var streamUrl = getApiBase() + '/api/camera/stream?device=' + deviceParam;
        var wsUrl = (getApiBase().replace(/^https/, 'wss').replace(/^http/, 'ws') || ('wss://' + window.location.host)) + '/ws/camera-webrtc?device=' + deviceParam;
        function fallbackToMjpeg() {
            if (mjpegFeed) {
                mjpegFeed.src = streamUrl;
                mjpegFeed.style.display = 'block';
            }
            if (videoFeed) {
                videoFeed.src = '';
                videoFeed.srcObject = null;
                videoFeed.style.display = 'none';
            }
            if (imagePlaceholder) imagePlaceholder.style.display = 'none';
        }
        var pc = new RTCPeerConnection();
        state.cameraWebrtcPc = pc;
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.ontrack = function (e) {
            // Always show video from server camera WebRTC - during setup AND live session
            if (e.streams && e.streams[0] && videoFeed) {
                videoFeed.srcObject = e.streams[0];
                videoFeed.style.display = 'block';
                if (mjpegFeed) { mjpegFeed.src = ''; mjpegFeed.style.display = 'none'; }
                if (imagePlaceholder) imagePlaceholder.style.display = 'none';
            }
        };
        state.previewServerCameraDevice = serverCamDevice;
        pc.createOffer().then(function (offer) {
            return pc.setLocalDescription(offer);
        }).then(function () {
            var ws = new WebSocket(wsUrl);
            state.cameraWebrtcWs = ws;
            ws.onopen = function () {
                ws.send(JSON.stringify({ type: 'offer', sdp: pc.localDescription.sdp }));
            };
            ws.onmessage = function (ev) {
                try {
                    var msg = JSON.parse(ev.data);
                    if (msg.type === 'answer' && msg.sdp) {
                        pc.setRemoteDescription({ type: msg.answerType || msg.type || 'answer', sdp: msg.sdp }).catch(function (err) {
                            console.warn('[Camera WebRTC] setRemoteDescription failed:', err);
                            fallbackToMjpeg();
                        });
                    } else if (msg.type === 'ice' && msg.candidate != null) {
                        pc.addIceCandidate(new RTCIceCandidate({ candidate: msg.candidate, sdpMid: msg.sdpMid, sdpMLineIndex: msg.sdpMLineIndex })).catch(function () {});
                    } else if (msg.type === 'ice' && msg.candidate === null) {
                        pc.addIceCandidate(null).catch(function () {});
                    } else if (msg.type === 'error') {
                        console.warn('[Camera WebRTC] server error:', msg.error);
                        fallbackToMjpeg();
                    }
                } catch (e) {
                    console.warn('[Camera WebRTC] message parse error:', e);
                }
            };
            ws.onerror = ws.onclose = function () {
                if (videoFeed && !videoFeed.srcObject) fallbackToMjpeg();
            };
            pc.onicecandidate = function (e) {
                if (ws.readyState === WebSocket.OPEN && e.candidate) {
                    ws.send(JSON.stringify({ type: 'ice', candidate: e.candidate.candidate, sdpMid: e.candidate.sdpMid, sdpMLineIndex: e.candidate.sdpMLineIndex }));
                } else if (ws.readyState === WebSocket.OPEN && e.candidate === null) {
                    ws.send(JSON.stringify({ type: 'ice', candidate: null }));
                }
            };
        }).catch(function (err) {
            console.warn('[Camera WebRTC] offer failed:', err);
            fallbackToMjpeg();
        });
        if (imagePlaceholder) imagePlaceholder.style.display = 'none';
    } else if (!wantBrowserVideo && !keepServerCamera) {
        if (videoFeed) {
            videoFeed.src = '';
            videoFeed.srcObject = null;
            videoFeed.style.display = 'none';
        }
        if (mjpegFeed) {
            mjpegFeed.src = '';
            mjpegFeed.style.display = 'none';
        }
        if (imagePlaceholder) imagePlaceholder.style.display = 'flex';
    }

    var needGetUserMedia = wantBrowserVideo || wantAudioForPreview;
    if (!needGetUserMedia) {
        updateDeviceIndicators();
        if (wantAudioForPreview) {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                updateDeviceIndicators();
                return;
            }
            var audioOnlyConstraint = state.selectedBrowserMicId ? { deviceId: { exact: state.selectedBrowserMicId } } : true;
            navigator.mediaDevices.getUserMedia({ video: false, audio: audioOnlyConstraint })
                .then(function (stream) {
                    if (!state.isLiveSession || state.sessionState !== 'setup') { stream.getTracks().forEach(function (t) { t.stop(); }); return; }
                    if (state.previewStream) state.previewStream.getTracks().forEach(function (t) { t.stop(); });
                    state.previewStream = stream;
                    updateDeviceIndicators();
                    if (stream.getAudioTracks().length > 0) {
                        if (isServerMicSelected()) startMicWaveformFromServer();
                        else startMicWaveform(stream);
                    }
                })
                .catch(function (err) {
                    console.error('getUserMedia (mic) failed:', err);
                    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') showMicrophonePermissionDeniedHint();
                    updateDeviceIndicators();
                });
        } else if (wantAudio && isServerMicSelected()) {
            startMicWaveformFromServer();
        }
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('getUserMedia not available (need HTTPS or localhost)');
        updateDeviceIndicators();
        return;
    }

    var videoConstraint = wantBrowserVideo ? (state.selectedBrowserCameraId ? { deviceId: { exact: state.selectedBrowserCameraId } } : true) : false;
    var audioConstraint = wantAudioForPreview ? (state.selectedBrowserMicId ? { deviceId: { exact: state.selectedBrowserMicId } } : true) : false;
    navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: audioConstraint })
        .then(function (stream) {
            if (!state.isLiveSession || state.sessionState !== 'setup') {
                stream.getTracks().forEach(function (t) { t.stop(); });
                return;
            }
            if (state.previewStream) state.previewStream.getTracks().forEach(function (t) { t.stop(); });
            state.previewStream = stream;
            if (wantBrowserVideo && videoFeed && stream.getVideoTracks().length > 0) {
                var mjpegEl = document.getElementById('video-feed-mjpeg');
                if (mjpegEl) { mjpegEl.src = ''; mjpegEl.style.display = 'none'; }
                videoFeed.src = '';
                videoFeed.srcObject = stream;
                videoFeed.style.display = 'block';
                if (imagePlaceholder) imagePlaceholder.style.display = 'none';
            }
            updateDeviceIndicators();
            if (stream.getAudioTracks().length > 0) {
                if (isServerMicSelected()) startMicWaveformFromServer();
                else startMicWaveform(stream);
            } else if (wantAudio && isServerMicSelected()) {
                startMicWaveformFromServer();
            }
        })
        .catch(function (err) {
            console.error('getUserMedia failed:', err);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                showMicrophonePermissionDeniedHint();
            }
            updateDeviceIndicators();
            if (wantAudio && isServerMicSelected()) startMicWaveformFromServer();
        });
}

/** Update the live ASR label (Listening: &lt;partial text&gt;) from state.liveAsrInterimText. Call on every asr_partial and on asr_final (to clear). */
function updateLiveAsrLabel() {
    const wrap = document.getElementById('live-asr-label-wrap');
    const textEl = document.getElementById('live-asr-text');
    if (!wrap || !textEl) return;
    textEl.textContent = state.liveAsrInterimText || '';
}

function updateLiveSessionUI() {
    updateConfigPanelState();
    const liveAsrWrap = document.getElementById('live-asr-label-wrap');
    if (liveAsrWrap) {
        const showInterim = (currentConfig.app && currentConfig.app.show_interim_asr !== false);
        liveAsrWrap.style.display = (state.isLiveSession && state.sessionState === 'live' && showInterim) ? 'block' : 'none';
        if (!state.isLiveSession || state.sessionState !== 'live') state.liveAsrInterimText = '';
        if (state.isLiveSession && state.sessionState === 'live' && showInterim) updateLiveAsrLabel();
    }
    const deviceControls = document.getElementById('device-controls-container');
    const deviceTags = document.getElementById('device-tags');
    const videoFeed = document.getElementById('video-feed');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const startOverlay = document.getElementById('start-session-overlay');
    const previewImage = document.getElementById('preview-image');
    const sessionTitle = document.getElementById('session-title');
    const sessionMetaLine2 = document.getElementById('session-meta-line2');
    const sessionStats = document.getElementById('session-stats');
    const sessionFilenameEl = document.getElementById('session-filename');
    const pipelineConfigEl = document.getElementById('pipeline-config');
    const startBtn = document.getElementById('start-session-btn');
    const stopBtn = document.getElementById('stop-session-btn');

    const sessionImageEl = document.getElementById('session-image');
    if (sessionImageEl && state.isLiveSession) sessionImageEl.style.display = '';

    if (state.isLiveSession) {
        deviceControls.style.display = 'flex';
        updateDeviceIndicators();
        if (deviceTags) deviceTags.style.display = 'none'; // Pipeline table shows devices when visible
        if (pipelineConfigEl) {
            pipelineConfigEl.style.display = 'block';
            pipelineConfigEl.classList.remove('condensed');
            var deviceLabels = { mic: getDeviceDisplayLabel('mic'), camera: getDeviceDisplayLabel('camera'), speaker: getDeviceDisplayLabel('speaker') };
            var deviceTypes = { mic: getDeviceDisplayType('mic'), camera: getDeviceDisplayType('camera'), speaker: getDeviceDisplayType('speaker') };
            pipelineConfigEl.innerHTML = getPipelineTableHtml(currentConfig, { condensed: false, deviceLabels: deviceLabels, deviceTypes: deviceTypes });
            if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
            requestAnimationFrame(function () { updatePipelineSegShapes(pipelineConfigEl); });
        }
        var serverHealthRow = document.getElementById('server-health-row');
        if (serverHealthRow) {
            serverHealthRow.style.display = (state.sessionState === 'setup') ? 'flex' : 'none';
            if (state.sessionState === 'setup') {
                updateServerHealthUI();
                if (state.serverHealth.llm === null && state.serverHealth.riva === null) {
                    setTimeout(function () { checkServersHealth(); }, 400);
                }
            }
        }

        const turnCount = (state.liveChatTurns && state.liveChatTurns.length) || 0;
        const dateTimeStr = state.liveSessionStartTime > 0 ? formatSessionDateTime(state.liveSessionStartTime) : new Date().toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
        sessionTitle.textContent = state.liveSessionStartTime > 0 ? ('Session – ' + dateTimeStr) : ('Live Session – ' + new Date().toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', hour12: true }));
        if (sessionMetaLine2) sessionMetaLine2.textContent = dateTimeStr + ', ' + turnCount + ' turn' + (turnCount !== 1 ? 's' : '');

        if (state.sessionState === 'setup') {
            document.getElementById('new-session-btn')?.classList.add('new-session-btn--highlight');
            document.getElementById('config-panel')?.classList.add('config-panel--start-ready');
            if (sessionStats) sessionStats.innerHTML = '';
            if (sessionFilenameEl) { sessionFilenameEl.innerHTML = ''; sessionFilenameEl.style.display = 'none'; }
            if (startOverlay) startOverlay.style.display = 'flex';
            if (startBtn) startBtn.style.display = 'flex';
            if (stopBtn) stopBtn.style.display = 'none';
            // Preview (camera + mic waveform slot) is shown in setup so user sees what they’ll get before clicking START
            startPreviewStream();
            if (isServerMicSelected()) startMicWaveformFromServer();
            var cam = (currentConfig.devices || {}).camera;
            var hasVideo = (cam !== 'none' && cam != null && cam !== undefined);
            if (imagePlaceholder) {
                imagePlaceholder.style.display = hasVideo ? 'none' : 'flex';
                if (!hasVideo) updateImagePlaceholderContent();
            }
            // Show video feed - either WebRTC (videoFeed) or MJPEG fallback (mjpegFeed)
            // Only show one to avoid overlap
            var mjpegFeedSetup = document.getElementById('video-feed-mjpeg');
            var hasWebRTCSetup = videoFeed && videoFeed.srcObject && videoFeed.srcObject.getVideoTracks().length > 0;
            var hasMjpegSetup = mjpegFeedSetup && mjpegFeedSetup.src && mjpegFeedSetup.src !== '';
            if (hasVideo) {
                if (hasWebRTCSetup) {
                    videoFeed.style.display = 'block';
                    if (mjpegFeedSetup) mjpegFeedSetup.style.display = 'none';
                } else if (hasMjpegSetup) {
                    if (videoFeed) videoFeed.style.display = 'none';
                    mjpegFeedSetup.style.display = 'block';
                } else if (videoFeed) {
                    videoFeed.style.display = 'block';
                }
            } else {
                if (videoFeed) videoFeed.style.display = 'none';
                if (mjpegFeedSetup) mjpegFeedSetup.style.display = 'none';
            }
        } else if (state.sessionState === 'live') {
            document.getElementById('new-session-btn')?.classList.remove('new-session-btn--highlight');
            document.getElementById('config-panel')?.classList.remove('config-panel--start-ready');
            if (sessionStats) sessionStats.innerHTML = '<span class="stat-value" style="color: #ef4444;"><i data-lucide="circle" class="lucide-inline" style="fill: currentColor;"></i> RECORDING</span>';
            if (sessionFilenameEl) { sessionFilenameEl.innerHTML = ''; sessionFilenameEl.style.display = 'none'; }
            if (imagePlaceholder) imagePlaceholder.style.display = 'none';
            // Show video feed - either WebRTC (videoFeed) or MJPEG fallback (mjpegFeed)
            // Only show one to avoid overlap (empty video covering the MJPEG img)
            var mjpegFeedLive = document.getElementById('video-feed-mjpeg');
            var hasWebRTC = videoFeed && videoFeed.srcObject && videoFeed.srcObject.getVideoTracks().length > 0;
            var hasMjpeg = mjpegFeedLive && mjpegFeedLive.src && mjpegFeedLive.src !== '';
            if (hasWebRTC) {
                videoFeed.style.display = 'block';
                if (mjpegFeedLive) mjpegFeedLive.style.display = 'none';
            } else if (hasMjpeg) {
                if (videoFeed) videoFeed.style.display = 'none';
                mjpegFeedLive.style.display = 'block';
            } else if (videoFeed) {
                videoFeed.style.display = 'block';
            }
            if (startOverlay) startOverlay.style.display = 'none';
            if (startBtn) startBtn.style.display = 'flex';
            if (stopBtn) stopBtn.style.display = 'flex';
            renderTimeline();
            updateVoiceDebugPanel();
        } else if (state.sessionState === 'stopped') {
            document.getElementById('new-session-btn')?.classList.remove('new-session-btn--highlight');
            document.getElementById('config-panel')?.classList.remove('config-panel--start-ready');
            if (sessionStats) sessionStats.innerHTML = '<span class="stat-value" style="color: var(--text-secondary);"><i data-lucide="check-circle" class="lucide-inline"></i> Session recorded</span>';
            if (sessionFilenameEl && state.lastSavedSessionId) {
                var sessionFileName = state.lastSavedSessionId + '.json';
                sessionFilenameEl.innerHTML = '<span class="session-filename-label">Session filename: <span class="session-filename-text">' + escapeHtml(sessionFileName) + '</span></span> <button type="button" class="session-filename-copy" aria-label="Copy filename" data-filename="' + escapeHtml(sessionFileName) + '"><i data-lucide="copy" class="lucide-inline" aria-hidden="true"></i></button>';
                sessionFilenameEl.style.display = '';
                if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons(sessionFilenameEl);
            } else if (sessionFilenameEl) {
                sessionFilenameEl.innerHTML = '';
                sessionFilenameEl.style.display = 'none';
            }
            if (imagePlaceholder) {
                imagePlaceholder.style.display = 'flex';
                updateImagePlaceholderContent();
            }
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
        if (pipelineConfigEl) pipelineConfigEl.style.display = 'none';
        var serverHealthRow = document.getElementById('server-health-row');
        if (serverHealthRow) serverHealthRow.style.display = 'none';
        updateChatInputVisibility();
    }
}

function updateHistoricalSessionPreview() {
    const sessionMeta = document.getElementById('session-meta');
    const sessionTitle = document.getElementById('session-title');
    const sessionMetaLine2 = document.getElementById('session-meta-line2');
    const sessionStats = document.getElementById('session-stats');
    const sessionFilenameEl = document.getElementById('session-filename');
    const pipelineConfigEl = document.getElementById('pipeline-config');
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
        const turns = metrics.total_turns || 0;
        const dateTimeStr = formatSessionDateTime(session.created_at);

        sessionTitle.textContent = session.name || 'Unnamed Session';
        if (sessionMetaLine2) sessionMetaLine2.textContent = dateTimeStr + ', ' + turns + ' turn' + (turns !== 1 ? 's' : '');

        const ttl = formatLatency(metrics.avg_ttl);
        const ttfa = metrics.avg_ttfa != null ? formatLatency(metrics.avg_ttfa) : '—';
        sessionStats.innerHTML = `
            <span class="session-stat-item">
                <span class="stat-label">Avg TTL:</span>
                <span class="stat-value">${ttl}</span>
            </span>
            <span class="session-stat-item">
                <span class="stat-label">TTFA:</span>
                <span class="stat-value">${ttfa}</span>
            </span>
        `;
        const sessionFileName = session.session_id ? (session.session_id + '.json') : '';
        if (sessionFilenameEl) {
            if (sessionFileName) {
                sessionFilenameEl.innerHTML = '<span class="session-filename-label">Session filename: <span class="session-filename-text">' + escapeHtml(sessionFileName) + '</span></span> <button type="button" class="session-filename-copy" aria-label="Copy filename" data-filename="' + escapeHtml(sessionFileName) + '"><i data-lucide="copy" class="lucide-inline" aria-hidden="true"></i></button>';
                sessionFilenameEl.style.display = '';
                if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons(sessionFilenameEl);
            } else {
                sessionFilenameEl.innerHTML = '';
                sessionFilenameEl.style.display = 'none';
            }
        }

        if (pipelineConfigEl && session.config) {
            pipelineConfigEl.style.display = 'block';
            pipelineConfigEl.classList.remove('condensed');
            var c = session.config;
            var opts = { condensed: false };
            if (c.device_labels && (c.device_labels.mic != null || c.device_labels.camera != null || c.device_labels.speaker != null)) opts.deviceLabels = c.device_labels;
            if (c.device_types && (c.device_types.mic != null || c.device_types.camera != null || c.device_types.speaker != null)) opts.deviceTypes = c.device_types;
            pipelineConfigEl.innerHTML = getPipelineTableHtml(c, opts);
            if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
            requestAnimationFrame(function () { updatePipelineSegShapes(pipelineConfigEl); });
        } else if (pipelineConfigEl) {
            pipelineConfigEl.style.display = 'none';
        }

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
        sessionTitle.textContent = 'New Session';
        if (sessionMetaLine2) sessionMetaLine2.textContent = '';
        sessionStats.innerHTML = '';
        if (sessionFilenameEl) { sessionFilenameEl.textContent = ''; sessionFilenameEl.style.display = 'none'; }
        const sessionImageEl = document.getElementById('session-image');
        if (state.isLiveSession) {
            if (sessionImageEl) sessionImageEl.style.display = '';
        } else {
            if (pipelineConfigEl) {
                pipelineConfigEl.style.display = 'none';
                pipelineConfigEl.innerHTML = '';
            }
            previewImage.style.display = 'none';
            videoFeed.style.display = 'none';
            imagePlaceholder.style.display = 'flex';
            if (sessionImageEl) sessionImageEl.style.display = '';
        }
    } else {
        if (sessionFilenameEl) { sessionFilenameEl.textContent = ''; sessionFilenameEl.style.display = 'none'; }
        const sessionImageEl = document.getElementById('session-image');
        if (sessionImageEl) sessionImageEl.style.display = '';
    }
}

function startNewSession() {
    // Create new live session
    state.isLiveSession = true;
    state.sessionState = 'setup';
    state.selectedSession = null;

    // Restore saved default config when present so system prompt and other edits persist across reloads
    const saved = getDefaultConfig();
    currentConfig = saved ? JSON.parse(JSON.stringify(saved)) : JSON.parse(JSON.stringify(defaultConfig));
    applyEnvPrefillsToCurrentConfig();

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
    // Preload ASR/TTS model names so pipeline shows them without opening ASR/TTS tabs
    setTimeout(function () {
        preloadASRModelName();
        preloadTTSModelName();
    }, 100);

    updateLiveSessionUI();
    updateHistoricalSessionPreview();

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

function stopLiveSystemStatsPoll() {
    if (state.liveSystemStatsPollIntervalId != null) {
        clearInterval(state.liveSystemStatsPollIntervalId);
        state.liveSystemStatsPollIntervalId = null;
    }
}

function getApiBase() {
    return window.location.origin;
}

/** Shared voice WebSocket message handler (preview and live). Used when Server USB opens /ws/voice for preview and when START opens it for browser mic. */
function handleVoiceWsMessage(ev) {
    if (typeof ev.data !== 'string') return;
    try {
        const msg = JSON.parse(ev.data);
        if (uiSettings.showDebugInfo) {
            const eventType = (msg.type === 'event' && msg.event && msg.event.event_type) ? msg.event.event_type : '';
            console.log('[Voice] 📥', msg.type, eventType ? ' event_type=' + eventType : '', msg);
        }
        if (msg.type !== 'user_amplitude') {
            const toLog = (msg.type === 'tts_audio' && msg.data)
                ? { type: 'tts_audio', sample_rate: msg.sample_rate, data_length: msg.data.length }
                : msg;
            pushVoiceMessageLog(toLog);
        }
        if (msg.type === 'system_stats') {
            var t = typeof msg.timestamp === 'number' ? msg.timestamp : (Date.now() / 1000 - state.liveSessionStartTime);
            if (Array.isArray(state.liveSystemStats)) {
                state.liveSystemStats.push({
                    t: t,
                    cpu: msg.cpu_percent != null ? msg.cpu_percent : null,
                    gpu: msg.gpu_percent != null ? msg.gpu_percent : null
                });
                renderTimeline();
            }
        } else if (msg.type === 'user_amplitude') {
            var amp = typeof msg.amplitude === 'number' ? msg.amplitude : 0;
            var serverTs = typeof msg.timestamp === 'number' ? msg.timestamp : 0;
            // Use server timestamp so amplitude history is monotonic and matches server 25ms spacing (no saw/block).
            var ts = serverTs;
            if (state.liveSessionStartTime > 0 && Array.isArray(state.liveAudioAmplitudeHistory)) {
                state.liveAudioAmplitudeHistory.push({ timestamp: ts, amplitude: amp });
                // Server USB: end-of-speech from sparse user_amplitude (20 Hz). Use consecutive sample count, not wall-clock,
                // so we don't need 0.15s of messages (asr_final would win). 3 consecutive below-threshold ≈ 100ms at 20 Hz.
                var userTh = (typeof uiSettings.userVoiceThreshold === 'number' && !isNaN(uiSettings.userVoiceThreshold)) ? uiSettings.userVoiceThreshold : 5;
                var effectiveTh = state.micWaveformFromServer ? Math.min(userTh, 2) : userTh;
                var SILENCE_CONSECUTIVE_SAMPLES = 3; // 20 Hz => ~100ms; enough to beat asr_final when possible
                if (state.micWaveformFromServer && state.voiceTurnActive && state.liveTtlBandStartTime == null) {
                    if (amp < effectiveTh) {
                        if (state.voiceSilenceCandidate == null) {
                            state.voiceSilenceCandidate = ts;
                            state.voiceSilenceConsecutiveCount = 1;
                        } else {
                            state.voiceSilenceConsecutiveCount = (state.voiceSilenceConsecutiveCount || 0) + 1;
                        }
                        if (state.voiceSilenceConsecutiveCount >= SILENCE_CONSECUTIVE_SAMPLES) {
                            state.liveTtlBandStartTime = state.voiceSilenceCandidate;
                            state.voiceSilenceCandidate = null;
                            state.voiceSilenceConsecutiveCount = 0;
                            if (window._micWaveformDebug || uiSettings.showDebugInfo) {
                                console.log('[TTL] liveTtlBandStartTime set from amplitude (Server USB silence)', state.liveTtlBandStartTime);
                            }
                        }
                    } else {
                        state.voiceSilenceCandidate = null;
                        state.voiceSilenceConsecutiveCount = 0;
                    }
                }
            } else if (state.micWaveformFromServer && Array.isArray(state.pendingServerMicAmplitude)) {
                // Buffer until session_start so we don't drop the first samples (message ordering).
                if (state.pendingServerMicAmplitude.length < 300) state.pendingServerMicAmplitude.push({ timestamp: serverTs, amplitude: amp });
            }
            if (state.micWaveformFromServer && Array.isArray(state.micAmplitudeBuffer)) {
                var cap = 120;
                var val = Math.min(128, (amp / 100) * 128 * getMicPreviewGain());
                // Server sends ~20 Hz (chunk-bound); browser mic gets ~60 fps. Push 3 samples per message so preview scrolls at similar visual rate.
                var samplesPerMessage = 3;
                for (var si = 0; si < samplesPerMessage; si++) {
                    if (state.micAmplitudeBuffer.length >= cap) state.micAmplitudeBuffer.shift();
                    state.micAmplitudeBuffer.push(val);
                }
                if (state.micAmplitudeBuffer.length <= samplesPerMessage && state.sessionState === 'live') {
                    console.log('[MicWaveform] First user_amplitude received (live); green waveform will show as more samples arrive.');
                }
            }
            renderTimeline();
        } else if (msg.type === 'event' && msg.event) {
            const evt = msg.event;
            if (evt.event_type === 'session_start') {
                if (state.liveSessionStartTime <= 0) {
                    state.liveSessionStartTime = Date.now() / 1000;
                    startLiveSystemStatsPoll();
                    state.liveAudioAmplitudeHistory = [];
                    state._userAmplitudeSmoothBuf = [];
                }
                // Flush Server USB amplitude received before session_start so AUDIO lane has data from the first sample.
                // When Server USB mic is used, client sets liveSessionStartTime before sending start_session so
                // user_amplitude may already be pushing into liveAudioAmplitudeHistory; do not clear it.
                if (state.pendingServerMicAmplitude && state.pendingServerMicAmplitude.length) {
                    state.liveAudioAmplitudeHistory = state.liveAudioAmplitudeHistory || [];
                    state.liveAudioAmplitudeHistory.push.apply(state.liveAudioAmplitudeHistory, state.pendingServerMicAmplitude);
                    state.pendingServerMicAmplitude = [];
                }
                if (state.micWaveformFromServer) state.micAmplitudeBuffer = [];
            }
            if (evt.lane === undefined || evt.lane === null) {
                if (evt.event_type && evt.event_type.startsWith('asr_')) evt.lane = 'speech';
                else if (evt.event_type && evt.event_type.startsWith('llm_')) evt.lane = 'llm';
                else if (evt.event_type && evt.event_type.startsWith('tts_')) evt.lane = 'tts';
                else if (evt.event_type === 'session_start') evt.lane = 'system';
                else if (evt.event_type === 'realtime_session_ready' || evt.event_type === 'error') evt.lane = 'system';
                else if (evt.event_type === 'realtime_output_partial' || evt.event_type === 'realtime_output_final') evt.lane = 'tts';
            } else if (typeof evt.lane === 'string') {
                evt.lane = evt.lane.toLowerCase();
            }
            state.liveTimelineEvents.push(evt);
            renderTimeline();
            if (evt.event_type === 'asr_partial') {
                state.voiceTurnActive = true;
                var pt = evt.timestamp != null ? Number(evt.timestamp) : null;
                if (typeof pt === 'number' && !isNaN(pt)) state.lastAsrPartialTime = pt;
                state.liveAsrInterimText = (evt.data && evt.data.text != null) ? String(evt.data.text).trim() : '';
                updateLiveAsrLabel();
            } else if (evt.event_type === 'asr_final') {
                if (state.voiceTurnActive && state.liveTtlBandStartTime == null) {
                    var ft = evt.timestamp != null ? Number(evt.timestamp) : null;
                    if (typeof ft === 'number' && !isNaN(ft))
                        state.liveTtlBandStartTime = state.lastAsrPartialTime != null ? state.lastAsrPartialTime : (ft - 0.2);
                }
                state.liveAsrInterimText = '';
                updateLiveAsrLabel();
                var userText = (evt.data && evt.data.text != null) ? String(evt.data.text).trim() : '';
                if (userText) {
                    var last = state.liveChatTurns[state.liveChatTurns.length - 1];
                    var sameUtterance = last && last.assistant === '' && last.user &&
                        (userText === last.user || userText.indexOf(last.user) === 0 || last.user.indexOf(userText) === 0);
                    if (sameUtterance) {
                        last.user = userText;
                    } else {
                        state.liveChatTurns.push({ user: userText, assistant: '' });
                    }
                    renderLiveChat();
                    requestAnimationFrame(function () { updateLiveSessionUI(); });
                }
            } else if (evt.event_type === 'chat') {
                var userText = evt.user != null ? String(evt.user) : (evt.data && evt.data.user != null ? String(evt.data.user) : null);
                var assistantText = evt.assistant != null ? String(evt.assistant) : (evt.data && evt.data.assistant != null ? String(evt.data.assistant) : '');
                if (userText != null) {
                    var last = state.liveChatTurns[state.liveChatTurns.length - 1];
                    if (last && last.assistant === '' && last.user === userText) {
                        last.assistant = assistantText;
                    } else {
                        state.liveChatTurns.push({ user: userText, assistant: assistantText });
                    }
                    renderLiveChat();
                    requestAnimationFrame(function () { updateLiveSessionUI(); });
                }
            } else if (evt.event_type === 'tts_complete' && evt.data && evt.data.text != null) {
                // Realtime: tts_complete carries the AI response transcript
                var last = state.liveChatTurns[state.liveChatTurns.length - 1];
                if (last && last.assistant === '') {
                    last.assistant = String(evt.data.text).trim();
                    renderLiveChat();
                    requestAnimationFrame(function () { updateLiveSessionUI(); });
                }
            }
        } else if (msg.type === 'tts_start') {
            state.firstTtsPlayTimeThisResponse = null;
            state.earliestTtsPlayTimeAboveThreshold = null;
            if (isServerSpeakerSelected()) state.ttsNextStartTime = -1;
            if (state.ttsAudioContext) {
                if (state.ttsAudioContext.state === 'suspended') state.ttsAudioContext.resume();
            }
        } else if (msg.type === 'tts_audio' && msg.data) {
            // Server may send 25ms amplitude_segments. For browser speaker we assign times when we schedule playback
            // (client session time) so purple aligns with audio. For server speaker we push server times (no client play time).
            var skipSegmentPush = false;
            var serverAmplitudeSegments = null;
            if (Array.isArray(msg.amplitude_segments) && msg.amplitude_segments.length && Array.isArray(state.liveTtsAmplitudeHistory)) {
                skipSegmentPush = true;
                serverAmplitudeSegments = msg.amplitude_segments;
                if (isServerSpeakerSelected()) {
                    msg.amplitude_segments.forEach(function (seg) {
                        state.liveTtsAmplitudeHistory.push({
                            startTime: seg.startTime,
                            endTime: seg.endTime,
                            amplitude: seg.amplitude != null ? seg.amplitude : 0
                        });
                    });
                    var firstSeg = msg.amplitude_segments[0];
                    if (firstSeg && typeof firstSeg.startTime === 'number') {
                        if (state.firstTtsPlayTimeThisResponse == null || firstSeg.startTime < state.firstTtsPlayTimeThisResponse)
                            state.firstTtsPlayTimeThisResponse = firstSeg.startTime;
                        if (firstSeg.amplitude > 0 && (state.earliestTtsPlayTimeAboveThreshold == null || firstSeg.startTime < state.earliestTtsPlayTimeAboveThreshold))
                            state.earliestTtsPlayTimeAboveThreshold = firstSeg.startTime;
                    }
                }
            }
            if (isServerSpeakerSelected()) {
                recordTtsSegmentOnly(msg.data, msg.sample_rate || 24000, skipSegmentPush);
            } else {
                playTtsChunk(msg.data, msg.sample_rate || 24000, skipSegmentPush, serverAmplitudeSegments);
            }
        } else if (msg.type === 'session_saved' && msg.session_id) {
            state.lastSavedSessionId = msg.session_id;
            loadSessions();
            updateLiveSessionUI();
        } else if (msg.type === 'error') {
            console.error('Voice pipeline error:', msg.error);
            appendLiveChatError(msg.error);
        } else if (msg.type === 'request_frame') {
            // VLM: Legacy single frame request (backwards compatibility)
            console.log('[VLM] Server requested single frame');
            captureAndSendVideoFrame();
        } else if (msg.type === 'vlm_start_capture') {
            // VLM: Start continuous frame capture into ring buffer
            const fps = msg.fps || 3.0;
            const quality = msg.quality || 0.7;
            const maxWidth = msg.max_width || 640;
            vlmStartCapture(fps, quality, maxWidth);
        } else if (msg.type === 'vlm_get_frames') {
            // VLM: Get frames from ring buffer
            const tStart = msg.t_start || 0;
            const tEnd = msg.t_end || 0;
            const nFrames = msg.n_frames || 4;
            vlmSendFrames(tStart, tEnd, nFrames);
        } else if (msg.type === 'vlm_stop_capture') {
            // VLM: Stop frame capture
            vlmStopCapture();
        }
    } catch (e) {
        console.error('Parse WS message error:', e);
    }
}

/**
 * VLM Ring Buffer: Continuous frame capture for multi-frame VLM requests.
 * Frames are stored with timestamps and selected based on speech timing.
 */
const vlmRingBuffer = {
    frames: [],           // Array of {data: dataUrl, timestamp: sessionTime}
    maxFrames: 60,        // ~20 seconds at 3 fps
    captureInterval: null,
    fps: 3.0,
    quality: 0.7,
    maxWidth: 640,
    isCapturing: false
};

/**
 * Start VLM frame capture into ring buffer.
 * Called by server when session starts with vlm_start_capture message.
 */
function vlmStartCapture(fps, quality, maxWidth) {
    if (vlmRingBuffer.isCapturing) {
        console.log('[VLM] Capture already running');
        return;
    }
    
    vlmRingBuffer.fps = fps || 3.0;
    vlmRingBuffer.quality = quality || 0.7;
    vlmRingBuffer.maxWidth = maxWidth || 640;
    vlmRingBuffer.frames = [];
    vlmRingBuffer.isCapturing = true;
    
    const intervalMs = 1000 / vlmRingBuffer.fps;
    vlmRingBuffer.captureInterval = setInterval(vlmCaptureFrame, intervalMs);
    
    console.log('[VLM] Started capture: fps=' + vlmRingBuffer.fps + ', quality=' + vlmRingBuffer.quality + ', maxWidth=' + vlmRingBuffer.maxWidth);
}

/**
 * Stop VLM frame capture.
 */
function vlmStopCapture() {
    if (vlmRingBuffer.captureInterval) {
        clearInterval(vlmRingBuffer.captureInterval);
        vlmRingBuffer.captureInterval = null;
    }
    vlmRingBuffer.isCapturing = false;
    vlmRingBuffer.frames = [];
    console.log('[VLM] Stopped capture');
}

/**
 * Capture a single frame into the ring buffer.
 */
function vlmCaptureFrame() {
    const videoFeed = document.getElementById('video-feed');
    if (!videoFeed || !videoFeed.srcObject || videoFeed.paused || videoFeed.ended || videoFeed.videoWidth === 0) {
        return; // No video available
    }
    
    try {
        const canvas = document.createElement('canvas');
        const aspectRatio = videoFeed.videoWidth / videoFeed.videoHeight;
        canvas.width = Math.min(videoFeed.videoWidth, vlmRingBuffer.maxWidth);
        canvas.height = Math.round(canvas.width / aspectRatio);
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        
        const dataUrl = canvas.toDataURL('image/jpeg', vlmRingBuffer.quality);
        const sessionTime = state.liveSessionStartTime > 0 ? (Date.now() / 1000) - state.liveSessionStartTime : 0;
        
        // Add to ring buffer
        vlmRingBuffer.frames.push({
            data: dataUrl,
            timestamp: sessionTime,
            width: canvas.width,
            height: canvas.height
        });
        
        // Trim if over max
        while (vlmRingBuffer.frames.length > vlmRingBuffer.maxFrames) {
            vlmRingBuffer.frames.shift();
        }
    } catch (e) {
        // Silently ignore capture errors
    }
}

/**
 * Get frames from ring buffer, evenly spaced between t_start and t_end.
 */
function vlmGetFrames(tStart, tEnd, nFrames) {
    const frames = vlmRingBuffer.frames;
    if (frames.length === 0 || nFrames <= 0) {
        return [];
    }
    
    // Filter frames within time range
    const inRange = frames.filter(f => f.timestamp >= tStart && f.timestamp <= tEnd);
    if (inRange.length === 0) {
        // If no frames in range, return latest frame(s)
        console.log('[VLM] No frames in range [' + tStart.toFixed(2) + ', ' + tEnd.toFixed(2) + '], using latest');
        const latest = frames.slice(-Math.min(nFrames, frames.length));
        return latest;
    }
    
    // If we have fewer frames than requested, return all
    if (inRange.length <= nFrames) {
        return inRange;
    }
    
    // Select evenly spaced frames
    const result = [];
    const duration = tEnd - tStart;
    for (let i = 0; i < nFrames; i++) {
        const targetTime = tStart + (i * duration / (nFrames - 1));
        // Find closest frame to target time
        let closest = inRange[0];
        let minDiff = Math.abs(inRange[0].timestamp - targetTime);
        for (const f of inRange) {
            const diff = Math.abs(f.timestamp - targetTime);
            if (diff < minDiff) {
                minDiff = diff;
                closest = f;
            }
        }
        // Avoid duplicates
        if (result.length === 0 || result[result.length - 1] !== closest) {
            result.push(closest);
        }
    }
    
    return result;
}

/**
 * Send frames to server in response to vlm_get_frames request.
 */
function vlmSendFrames(tStart, tEnd, nFrames) {
    const selectedFrames = vlmGetFrames(tStart, tEnd, nFrames);
    
    if (state.voiceWs && state.voiceWs.readyState === WebSocket.OPEN) {
        const payload = {
            type: 'vlm_frames',
            frames: selectedFrames.map(f => ({
                data: f.data,
                timestamp: f.timestamp,
                width: f.width,
                height: f.height
            })),
            t_start: tStart,
            t_end: tEnd,
            n_requested: nFrames
        };
        state.voiceWs.send(JSON.stringify(payload));
        console.log('[VLM] Sent ' + selectedFrames.length + ' frames (requested ' + nFrames + ') for t=[' + tStart.toFixed(2) + ', ' + tEnd.toFixed(2) + ']');
    }
}

/**
 * Legacy: Single frame capture for backwards compatibility.
 */
function captureAndSendVideoFrame() {
    // Use ring buffer if available, otherwise capture fresh
    if (vlmRingBuffer.frames.length > 0) {
        const latest = vlmRingBuffer.frames[vlmRingBuffer.frames.length - 1];
        if (state.voiceWs && state.voiceWs.readyState === WebSocket.OPEN) {
            state.voiceWs.send(JSON.stringify({
                type: 'vlm_frames',
                frames: [latest],
                t_start: latest.timestamp,
                t_end: latest.timestamp,
                n_requested: 1
            }));
        }
        return;
    }
    
    // Fallback: capture fresh frame
    const videoFeed = document.getElementById('video-feed');
    if (!videoFeed || !videoFeed.srcObject || videoFeed.paused || videoFeed.ended || videoFeed.videoWidth === 0) {
        if (state.voiceWs && state.voiceWs.readyState === WebSocket.OPEN) {
            state.voiceWs.send(JSON.stringify({
                type: 'vlm_frames',
                frames: [],
                t_start: 0,
                t_end: 0,
                n_requested: 1
            }));
        }
        return;
    }
    
    try {
        const canvas = document.createElement('canvas');
        const aspectRatio = videoFeed.videoWidth / videoFeed.videoHeight;
        canvas.width = Math.min(videoFeed.videoWidth, 640);
        canvas.height = Math.round(canvas.width / aspectRatio);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        const sessionTime = state.liveSessionStartTime > 0 ? (Date.now() / 1000) - state.liveSessionStartTime : 0;
        
        if (state.voiceWs && state.voiceWs.readyState === WebSocket.OPEN) {
            state.voiceWs.send(JSON.stringify({
                type: 'vlm_frames',
                frames: [{data: dataUrl, timestamp: sessionTime, width: canvas.width, height: canvas.height}],
                t_start: sessionTime,
                t_end: sessionTime,
                n_requested: 1
            }));
        }
    } catch (e) {
        console.error('[VLM] Frame capture error:', e);
    }
}

function handleVoiceWsClose(ev) {
    console.log('[Voice] WebSocket closed: code=' + (ev && ev.code) + ' reason=' + (ev && ev.reason) + ' clean=' + (ev && ev.wasClean));
    state.voiceWs = null;
    stopVoiceMicStream();
    vlmStopCapture();  // Stop VLM frame capture
    if (state.sessionState === 'live') {
        stopLiveSystemStatsPoll();
        if (state.liveTimelineRafId != null) {
            cancelAnimationFrame(state.liveTimelineRafId);
            state.liveTimelineRafId = null;
        }
        state.sessionState = 'stopped';
        updateLiveSessionUI();
    } else if (state.sessionState === 'setup') {
        state.micWaveformFromServer = false;
        state.micAmplitudeBuffer = [];
    }
}

function startSessionRecording() {
    if (state.sessionState !== 'setup') return;

    // Realtime ASR with non-Realtime TTS is not supported; fail fast with a clear message
    var asrRealtime = (currentConfig.asr.backend === 'openai-realtime' || currentConfig.asr.scheme === 'openai-realtime');
    var ttsRealtime = (currentConfig.tts.backend === 'openai-realtime' || currentConfig.tts.scheme === 'openai-realtime');
    if (asrRealtime && !ttsRealtime) {
        var chatEl = document.getElementById('chat-history');
        if (chatEl) {
            chatEl.innerHTML = '<div class="empty-state"><p class="error">OpenAI Realtime ASR with Riva TTS is not supported. Use <strong>Full voice</strong> (ASR + TTS Realtime) or set ASR to NVIDIA RIVA.</p></div>';
            if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        } else {
            alert('OpenAI Realtime ASR with Riva TTS is not supported. Use Full voice (ASR + TTS Realtime) or set ASR to NVIDIA RIVA.');
        }
        return;
    }

    // Server USB: we may already have the voice WS open for preview. Reuse it only if current config matches
    // what the server started with (first message = config). Realtime full-voice requires the first message
    // to be Realtime config, so if user switched to Realtime after opening preview, close and reopen.
    if (state.voiceWs && state.voiceWs.readyState === WebSocket.OPEN && isServerMicSelected() && !isRealtimeFullVoiceLock()) {
        console.log('[Voice] Already connected (Server USB preview); sending start_session');
        state.liveTimelineEvents = [];
        state.voiceMessageLog = [];
        state.liveChatTurns = [];
        state.ttsNextStartTime = 0;
        state.liveAudioAmplitudeHistory = [];
        state._userAmplitudeSmoothBuf = [];
        state.pendingServerMicAmplitude = [];
        state.liveTtsAmplitudeHistory = [];
        state.liveTtlBands = [];
        state.liveTtlBandStartTime = null;
        state.voiceTurnActive = false;
        state.voiceSilenceCandidate = null;
        state.voiceSilenceConsecutiveCount = 0;
        state.lastAsrPartialTime = null;
        state.firstTtsPlayTimeThisResponse = null;
        state.earliestTtsPlayTimeAboveThreshold = null;
        state.liveSystemStats = [];
        state.liveTimelineInitialZoomSet = false;
        state.sessionState = 'live';
        state.liveSessionStartTime = Date.now() / 1000;
        if (!state.ttsAudioContext) {
            state.ttsAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            state.ttsNextStartTime = 0;
        }
        if (state.ttsAudioContext.state === 'suspended') {
            state.ttsAudioContext.resume().catch(function (e) { console.warn('[Voice] TTS AudioContext resume on start:', e); });
        }
        startLiveSystemStatsPoll();
        scheduleLiveTimelineTick();
        updateLiveSessionUI();
        state.voiceWs.send(JSON.stringify({ type: 'start_session', config: buildVoiceConfig() }));
        updateVoiceDebugPanel();
        document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p><i data-lucide="circle" class="lucide-inline" style="fill: #ef4444;"></i> Session is LIVE - Start talking!</p>
        </div>
    `;
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        return;
    }

    console.log('[Voice] Start session recording (WebSocket + mic)');
    if (state.micPreviewWs) {
        try { state.micPreviewWs.close(); } catch (e) {}
        state.micPreviewWs = null;
    }
    var delayMs = 0;
    function connectVoiceAndStart() {
        if (state.voiceWs) {
            try { state.voiceWs.close(); } catch (e) {}
            state.voiceWs = null;
        }
        state.liveTimelineEvents = [];
        state.voiceMessageLog = [];
        state.liveChatTurns = [];
        state.ttsNextStartTime = 0;
        state.liveAudioAmplitudeHistory = [];
        state._userAmplitudeSmoothBuf = [];
        state.pendingServerMicAmplitude = [];
        state.liveTtsAmplitudeHistory = [];
        state.liveTtlBands = [];
        state.liveTtlBandStartTime = null;
        state.voiceTurnActive = false;
        state.voiceSilenceCandidate = null;
        state.voiceSilenceConsecutiveCount = 0;
        state.lastAsrPartialTime = null;
        state.firstTtsPlayTimeThisResponse = null;
        state.earliestTtsPlayTimeAboveThreshold = null;
        state.liveSystemStats = [];
        state.liveTimelineInitialZoomSet = false;
        state.sessionState = 'live';
        state.liveSessionStartTime = Date.now() / 1000;
        scheduleLiveTimelineTick();
        updateLiveSessionUI();

        // Create TTS AudioContext now (under user gesture) so playback is not blocked by autoplay policy
        if (!state.ttsAudioContext) {
            state.ttsAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            state.ttsNextStartTime = 0;
        }
        if (state.ttsAudioContext.state === 'suspended') {
            state.ttsAudioContext.resume().catch(function (e) { console.warn('[Voice] TTS AudioContext resume on start:', e); });
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = protocol + '//' + window.location.host + '/ws/voice';
        if (delayMs) console.log('[Voice] Connecting to', wsUrl, '(after', delayMs, 'ms release delay for Server USB mic)');
        else console.log('[Voice] Connecting to', wsUrl);
        const ws = new WebSocket(wsUrl);
        state.voiceWs = ws;

    ws.onopen = function () {
        console.log('[Voice] WebSocket connected, sending config and start_session');
        var config = buildVoiceConfig();
        ws.send(JSON.stringify({ type: 'config', config: config }));
        ws.send(JSON.stringify({ type: 'start_session', config: config }));
        console.log('[Voice] Config and start_session sent, starting mic stream');
        startVoiceMicStream();
    };

    ws.onmessage = handleVoiceWsMessage;
    ws.onclose = handleVoiceWsClose;
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
    };
    if (delayMs) setTimeout(connectVoiceAndStart, delayMs); else connectVoiceAndStart();
}

function appendLiveChatError(text) {
    const chatEl = document.getElementById('chat-history');
    if (!chatEl) return;
    const wrap = document.createElement('div');
    wrap.className = 'chat-bubble ai';
    wrap.innerHTML = '<div class="chat-avatar"><i data-lucide="bot" class="lucide-inline"></i></div><div class="chat-content"><div class="chat-text error">' + escapeHtml(text) + '</div></div>';
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
            <div class="chat-bubble user">
                <div class="chat-avatar"><i data-lucide="user" class="lucide-inline"></i></div>
                <div class="chat-content"><div class="chat-text">${escapeHtml(t.user)}</div></div>
            </div>
            <div class="chat-bubble ai">
                <div class="chat-avatar"><i data-lucide="bot" class="lucide-inline"></i></div>
                <div class="chat-content"><div class="chat-text">${assistantDisplay}</div></div>
            </div>
        `;
        }).join('');
    }
    chatEl.scrollTop = chatEl.scrollHeight;
    if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
}

const TARGET_SAMPLE_RATE = 16000;

function startVoiceMicStream() {
    // When user selected a Server USB microphone, the server captures from it; do not send browser PCM.
    var mic = (currentConfig.devices || {}).microphone;
    if (mic && (String(mic).startsWith('alsa:') || String(mic).startsWith('pyaudio:'))) {
        stopVoiceMicStream();
        console.log('[Voice] Using Server USB microphone; no browser mic stream (server captures from device).');
        updateDeviceIndicators();
        startMicWaveformFromServer();
        return;
    }
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
        // Debug-only: client mic RMS (enable with localStorage micWaveformDebug=1)
        var nowSec = Date.now() / 1000;
        var sumSq = 0;
        for (var i = 0; i < input.length; i++) sumSq += input[i] * input[i];
        var clientRms = input.length ? Math.sqrt(sumSq / input.length) : 0;
        var clientAmpScaled = Math.min(100, clientRms * 400);
        if (window._micWaveformDebug && state.liveSessionStartTime > 0 && nowSec - lastClientAmpLogTime >= 1.0) {
            console.log('[user_amplitude] client: buffer_len=' + input.length + ' float_rms=' + clientRms.toFixed(4) + ' amp_0_100=' + clientAmpScaled.toFixed(2));
            lastClientAmpLogTime = nowSec;
        }
        if (window._micWaveformDebug && state.liveSessionStartTime > 0 && clientAmpScaled >= 20) {
            var sessionT = nowSec - state.liveSessionStartTime;
            console.log('[user_amplitude_high] client: session_t=' + sessionT.toFixed(2) + 's amp_0_100=' + clientAmpScaled.toFixed(2));
        }
        ws.send(pcmData.buffer);
        pcmChunkCount++;
        if (pcmChunkCount % 50 === 0) console.log('[Voice] Sent', pcmChunkCount, 'PCM chunks');
        // AUDIO lane waveform: use only server user_amplitude (single source). Client used to push here too → double source caused saw/block shape with Realtime. RIVA looks correct because it only uses server.
        if (state.liveSessionStartTime > 0) {
            let sumSq = 0;
            for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i];
            const rms = Math.min(100, Math.sqrt(sumSq / input.length) * 400);
            const ts = (Date.now() / 1000) - state.liveSessionStartTime;
            // Browser-side end-of-speech for TTL band: only when we have seen voice this turn (voiceTurnActive set by asr_partial)
            const userTh = (typeof uiSettings.userVoiceThreshold === 'number' && !isNaN(uiSettings.userVoiceThreshold)) ? uiSettings.userVoiceThreshold : 5;
            const SILENCE_CONFIRM_SEC = 0.15;
            if (state.voiceTurnActive && state.liveTtlBandStartTime == null) {
                if (rms < userTh) {
                    if (state.voiceSilenceCandidate == null) state.voiceSilenceCandidate = ts;
                    else if ((ts - state.voiceSilenceCandidate) >= SILENCE_CONFIRM_SEC) {
                        state.liveTtlBandStartTime = state.voiceSilenceCandidate;
                        state.voiceSilenceCandidate = null;
                    }
                } else {
                    state.voiceSilenceCandidate = null;
                }
            }
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

/** Compute TTS amplitude segments at 25ms windows (match user waveform density) so live purple is not blocky.
 * ch: Float32Array or array of float samples in [-1,1]; sampleRate in Hz; chunkStartTime/chunkDuration in seconds.
 * Returns [{ startTime, endTime, amplitude }, ...] with amplitude 0–100. */
function ttsChunkToAmplitudeSegments(ch, sampleRate, chunkStartTime, chunkDuration) {
    var segments = [];
    var numSamples = ch.length;
    if (numSamples < 2) return segments;
    var windowSec = 0.025;
    var samplesPerWindow = Math.max(1, Math.floor(sampleRate * windowSec));
    for (var i = 0; i < numSamples; i += samplesPerWindow) {
        var endIdx = Math.min(i + samplesPerWindow, numSamples);
        var n = endIdx - i;
        var sumSq = 0;
        for (var j = i; j < endIdx; j++) sumSq += ch[j] * ch[j];
        var rms = n ? Math.min(100, Math.sqrt(sumSq / n) * 100) : 0;
        var t0 = chunkStartTime + (i / sampleRate);
        var t1 = chunkStartTime + (endIdx / sampleRate);
        segments.push({ startTime: t0, endTime: t1, amplitude: rms });
    }
    return segments;
}

/** When server speaker is selected: record TTS segment for purple waveform and saved session, without playing in browser.
 * skipSegmentPush: when true (e.g. server sent amplitude_segments), do not push; only run TTL band logic. */
function recordTtsSegmentOnly(base64Data, sampleRate, skipSegmentPush) {
    if (state.liveSessionStartTime <= 0 || !state.liveTtsAmplitudeHistory) return;
    if (skipSegmentPush) {
        // first/earliest/ttsNextStartTime already set by handler; just run TTL band close if applicable
        if (state.liveTtlBandStartTime != null && (state.earliestTtsPlayTimeAboveThreshold != null || state.firstTtsPlayTimeThisResponse != null)) {
            var bandStart = state.liveTtlBandStartTime;
            var firstChunk = state.firstTtsPlayTimeThisResponse;
            var firstAbove = state.earliestTtsPlayTimeAboveThreshold;
            var bandEnd = (firstChunk != null && firstAbove != null) ? Math.min(firstChunk, firstAbove) : (firstAbove != null ? firstAbove : firstChunk);
            state.liveTtlBandStartTime = null;
            state.voiceTurnActive = false;
            state.lastAsrPartialTime = null;
            state.firstTtsPlayTimeThisResponse = null;
            state.earliestTtsPlayTimeAboveThreshold = null;
            state.liveTtlBands.push({ start: bandStart, end: bandEnd, ttlMs: Math.round((bandEnd - bandStart) * 1000) });
        }
        return;
    }
    const binary = atob(base64Data);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    const samples = new Int16Array(bytes.buffer);
    const numSamples = samples.length;
    const duration = numSamples / sampleRate;
    let startTime = state.ttsNextStartTime;
    if (typeof startTime !== 'number' || startTime < 0) {
        startTime = (Date.now() / 1000) - state.liveSessionStartTime;
        state.ttsNextStartTime = startTime;
    }
    const endTime = startTime + duration;
    state.ttsNextStartTime = endTime;
    var ch = [];
    for (var i = 0; i < numSamples; i++) ch.push(samples[i] / (samples[i] < 0 ? 0x8000 : 0x7FFF));
    var segs = ttsChunkToAmplitudeSegments(ch, sampleRate, startTime, duration);
    for (var s = 0; s < segs.length; s++) state.liveTtsAmplitudeHistory.push(segs[s]);
    if (state.firstTtsPlayTimeThisResponse == null || startTime < state.firstTtsPlayTimeThisResponse)
        state.firstTtsPlayTimeThisResponse = startTime;
    for (var k = 0; k < segs.length; k++) {
        if (segs[k].amplitude > 0 && (state.earliestTtsPlayTimeAboveThreshold == null || segs[k].startTime < state.earliestTtsPlayTimeAboveThreshold))
            state.earliestTtsPlayTimeAboveThreshold = segs[k].startTime;
    }
    if (state.liveTtlBandStartTime != null && (state.earliestTtsPlayTimeAboveThreshold != null || state.firstTtsPlayTimeThisResponse != null)) {
        var bandStart = state.liveTtlBandStartTime;
        var firstChunk = state.firstTtsPlayTimeThisResponse;
        var firstAbove = state.earliestTtsPlayTimeAboveThreshold;
        var bandEnd = (firstChunk != null && firstAbove != null) ? Math.min(firstChunk, firstAbove) : (firstAbove != null ? firstAbove : firstChunk);
        state.liveTtlBandStartTime = null;
        state.voiceTurnActive = false;
        state.lastAsrPartialTime = null;
        state.firstTtsPlayTimeThisResponse = null;
        state.earliestTtsPlayTimeAboveThreshold = null;
        state.liveTtlBands.push({ start: bandStart, end: bandEnd, ttlMs: Math.round((bandEnd - bandStart) * 1000) });
    }
}

/** skipSegmentPush: when true (e.g. server sent amplitude_segments), do not push from PCM; first/earliest set below when serverAmplitudeSegments provided.
 * serverAmplitudeSegments: optional array of { startTime?, endTime?, amplitude } from server; we use amplitude but assign startTime/endTime from when we schedule playback (client session time) so purple aligns with audio. */
function playTtsChunk(base64Data, sampleRate, skipSegmentPush, serverAmplitudeSegments) {
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
    const numSamples = samples.length;
    const buffer = ctx.createBuffer(1, numSamples, sampleRate);
    const ch = buffer.getChannelData(0);
    for (let i = 0; i < numSamples; i++) ch[i] = samples[i] / (samples[i] < 0 ? 0x8000 : 0x7FFF);
    const duration = numSamples / sampleRate;
    var windowSec = 0.025;
    function schedulePlayback() {
        let startTime = state.ttsNextStartTime;
        if (typeof startTime !== 'number' || startTime < 0 || startTime < ctx.currentTime) startTime = ctx.currentTime;
        state.ttsNextStartTime = startTime + duration;
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        source.start(startTime);
        var nowSec = Date.now() / 1000;
        var sessionStart = state.liveSessionStartTime;
        var delayUntilPlay = Math.max(0, startTime - ctx.currentTime);
        var actualStartSession = (nowSec - sessionStart) + delayUntilPlay;
        // Use server amplitude data but place segments at client play time so purple aligns with audio
        if (state.liveSessionStartTime > 0 && state.liveTtsAmplitudeHistory && Array.isArray(serverAmplitudeSegments) && serverAmplitudeSegments.length) {
            for (var i = 0; i < serverAmplitudeSegments.length; i++) {
                var seg = serverAmplitudeSegments[i];
                var amp = (seg && typeof seg.amplitude === 'number') ? seg.amplitude : 0;
                var t0 = actualStartSession + i * windowSec;
                var t1 = t0 + windowSec;
                state.liveTtsAmplitudeHistory.push({ startTime: t0, endTime: t1, amplitude: amp });
            }
            if (state.firstTtsPlayTimeThisResponse == null || actualStartSession < state.firstTtsPlayTimeThisResponse)
                state.firstTtsPlayTimeThisResponse = actualStartSession;
            for (var ki = 0; ki < serverAmplitudeSegments.length; ki++) {
                var a = (serverAmplitudeSegments[ki] && typeof serverAmplitudeSegments[ki].amplitude === 'number') ? serverAmplitudeSegments[ki].amplitude : 0;
                if (a > 0 && (state.earliestTtsPlayTimeAboveThreshold == null || actualStartSession + ki * windowSec < state.earliestTtsPlayTimeAboveThreshold))
                    state.earliestTtsPlayTimeAboveThreshold = actualStartSession + ki * windowSec;
            }
            if (state.liveTtlBandStartTime != null && (state.earliestTtsPlayTimeAboveThreshold != null || state.firstTtsPlayTimeThisResponse != null)) {
                var bandStart = state.liveTtlBandStartTime;
                var firstChunk = state.firstTtsPlayTimeThisResponse;
                var firstAbove = state.earliestTtsPlayTimeAboveThreshold;
                var bandEnd = (firstChunk != null && firstAbove != null) ? Math.min(firstChunk, firstAbove) : (firstAbove != null ? firstAbove : firstChunk);
                state.liveTtlBandStartTime = null;
                state.voiceTurnActive = false;
                state.lastAsrPartialTime = null;
                state.firstTtsPlayTimeThisResponse = null;
                state.earliestTtsPlayTimeAboveThreshold = null;
                state.liveTtlBands.push({ start: bandStart, end: bandEnd, ttlMs: Math.round((bandEnd - bandStart) * 1000) });
            }
        } else if (state.liveSessionStartTime > 0 && state.liveTtsAmplitudeHistory && !skipSegmentPush) {
            var actualEndSession = actualStartSession + duration;
            var segs = ttsChunkToAmplitudeSegments(ch, sampleRate, actualStartSession, duration);
            for (var si = 0; si < segs.length; si++) state.liveTtsAmplitudeHistory.push(segs[si]);
            if (state.firstTtsPlayTimeThisResponse == null || actualStartSession < state.firstTtsPlayTimeThisResponse)
                state.firstTtsPlayTimeThisResponse = actualStartSession;
            for (var ki = 0; ki < segs.length; ki++) {
                if (segs[ki].amplitude > 0 && (state.earliestTtsPlayTimeAboveThreshold == null || segs[ki].startTime < state.earliestTtsPlayTimeAboveThreshold))
                    state.earliestTtsPlayTimeAboveThreshold = segs[ki].startTime;
            }
            if (state.liveTtlBandStartTime != null && (state.earliestTtsPlayTimeAboveThreshold != null || state.firstTtsPlayTimeThisResponse != null)) {
                var bandStart = state.liveTtlBandStartTime;
                var firstChunk = state.firstTtsPlayTimeThisResponse;
                var firstAbove = state.earliestTtsPlayTimeAboveThreshold;
                var bandEnd = (firstChunk != null && firstAbove != null) ? Math.min(firstChunk, firstAbove) : (firstAbove != null ? firstAbove : firstChunk);
                state.liveTtlBandStartTime = null;
                state.voiceTurnActive = false;
                state.lastAsrPartialTime = null;
                state.firstTtsPlayTimeThisResponse = null;
                state.earliestTtsPlayTimeAboveThreshold = null;
                state.liveTtlBands.push({ start: bandStart, end: bandEnd, ttlMs: Math.round((bandEnd - bandStart) * 1000) });
            }
        } else if (skipSegmentPush && state.liveTtlBandStartTime != null && (state.earliestTtsPlayTimeAboveThreshold != null || state.firstTtsPlayTimeThisResponse != null)) {
            var bandStart = state.liveTtlBandStartTime;
            var firstChunk = state.firstTtsPlayTimeThisResponse;
            var firstAbove = state.earliestTtsPlayTimeAboveThreshold;
            var bandEnd = (firstChunk != null && firstAbove != null) ? Math.min(firstChunk, firstAbove) : (firstAbove != null ? firstAbove : firstChunk);
            state.liveTtlBandStartTime = null;
            state.voiceTurnActive = false;
            state.lastAsrPartialTime = null;
            state.firstTtsPlayTimeThisResponse = null;
            state.earliestTtsPlayTimeAboveThreshold = null;
            state.liveTtlBands.push({ start: bandStart, end: bandEnd, ttlMs: Math.round((bandEnd - bandStart) * 1000) });
        }
    }
    if (ctx.state === 'suspended') {
        ctx.resume().then(schedulePlayback).catch(function (e) { console.warn('TTS AudioContext resume failed:', e); });
    } else {
        schedulePlayback();
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
            ttl_bands: state.liveTtlBands || []
        }));
        // Do not close the WebSocket here: the server may still send a synthetic asr_final (e.g. for
        // the 2nd turn when the stream ended with only partials) and session_saved. Let the server
        // close the connection when the pipeline finishes; ws.onclose will set state.voiceWs = null.
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
    // Reload again after server closes and sends session_saved (in case session_saved arrives after first load)
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
        renderTimeline();
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
            if (tabEl.classList.contains('config-tab--disabled')) return;
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
    // User voice threshold (silence detection, TTL band start, hide low user during TTS)
    const userVoiceThEl = document.getElementById('timeline-user-voice-threshold');
    const userVoiceThValueEl = document.getElementById('timeline-user-voice-threshold-value');
    if (userVoiceThEl) {
        var uv = (typeof uiSettings.userVoiceThreshold === 'number' && !isNaN(uiSettings.userVoiceThreshold)) ? uiSettings.userVoiceThreshold : 5;
        uv = Math.max(0, Math.min(30, uv));
        userVoiceThEl.value = String(uv);
        if (userVoiceThValueEl) userVoiceThValueEl.textContent = String(uv);
        userVoiceThEl.addEventListener('input', function () {
            var v = parseInt(userVoiceThEl.value, 10);
            if (!isNaN(v)) {
                v = Math.max(0, Math.min(30, v));
                uiSettings.userVoiceThreshold = v;
                if (userVoiceThValueEl) userVoiceThValueEl.textContent = String(v);
                saveUISettings();
                renderTimeline();
            }
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

    // Save as default config (enabled only when config panel is editable)
    const saveDefaultConfigBtn = document.getElementById('save-default-config-btn');
    if (saveDefaultConfigBtn) {
        saveDefaultConfigBtn.addEventListener('click', saveDefaultConfig);
        saveDefaultConfigBtn.disabled = !state.isLiveSession;
    }

    // Start session recording
    document.getElementById('start-session-btn').addEventListener('click', () => {
        startSessionRecording();
    });
    var serverHealthCheckBtn = document.getElementById('server-health-check-btn');
    if (serverHealthCheckBtn) {
        serverHealthCheckBtn.addEventListener('click', function () {
            serverHealthCheckBtn.disabled = true;
            checkServersHealth().then(function () {
                serverHealthCheckBtn.disabled = false;
            }).catch(function () {
                serverHealthCheckBtn.disabled = false;
            });
        });
    }

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
            msgEl.className = 'chat-bubble user';
            msgEl.innerHTML = `<div class="chat-content"><div class="chat-text">${escapeHtml(text)}</div></div>`;
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

/** Date only for session list meta (e.g. "Feb 12, 2025"). */
function formatSessionDateOnly(dateString) {
    const date = parseSessionDate(dateString);
    if (!date || isNaN(date.getTime())) return '';
    return date.toLocaleDateString(undefined, { dateStyle: 'medium' });
}

/** Full date and time for session meta line 2 (matches session list context). */
function formatSessionDateTime(dateString) {
    const date = typeof dateString === 'number' ? new Date(dateString * 1000) : parseSessionDate(dateString);
    if (!date || isNaN(date.getTime())) return '';
    return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

/** Truncate text for display; full text goes in title for tooltip. maxLen default 16. */
function truncateWithTitle(text, maxLen) {
    if (text == null || text === '') return { short: '', title: '' };
    var s = String(text);
    maxLen = maxLen || 16;
    if (s.length <= maxLen) return { short: s, title: '' };
    return { short: s.slice(0, maxLen - 3) + '...', title: s };
}

/**
 * Build pipeline config HTML: grid with device slots (icon + name) + connected pipeline bar ( > ASR > LLM > TTS > ).
 * @param {Object} config - Session config: { devices, asr, llm, tts }
 * @param {{ condensed?: boolean, deviceLabels?: { mic?: string, camera?: string, speaker?: string }, deviceTypes?: { mic?: 'browser'|'usb'|null, camera?: 'browser'|'usb'|null, speaker?: 'browser'|'usb'|null } }} options - deviceLabels: when provided use for slot text; deviceTypes: when provided show small icon at end of name (chromium=browser, usb=local)
 */
function getPipelineTableHtml(config, options) {
    if (!config) return '';
    const d = config.devices || {};
    const mic = d.microphone ?? d.audio_input_source;
    const cam = d.camera ?? d.video_source;
    const spk = d.speaker ?? d.audio_output_source;
    const isTextOnly = mic === 'none' || !mic;
    const hasCamera = cam && cam !== 'none';
    const hasSpeaker = spk && spk !== 'none';

    var deviceLabelsOpt = options && options.deviceLabels;
    function deviceLabel(v) {
        if (v == null || v === 'none') return '—';
        if (v === 'browser') return 'Browser';
        return String(v).slice(0, 12);
    }
    var rawMic = isTextOnly ? '(text)' : (deviceLabelsOpt && deviceLabelsOpt.mic != null ? deviceLabelsOpt.mic : deviceLabel(mic));
    var rawCam = hasCamera ? (deviceLabelsOpt && deviceLabelsOpt.camera != null ? deviceLabelsOpt.camera : deviceLabel(cam)) : '(n/a)';
    var rawSpk = hasSpeaker ? (deviceLabelsOpt && deviceLabelsOpt.speaker != null ? deviceLabelsOpt.speaker : deviceLabel(spk)) : '(text)';
    var micT = truncateWithTitle(rawMic, 16);
    var camT = truncateWithTitle(rawCam, 16);
    var spkT = truncateWithTitle(rawSpk, 16);

    const asrModel = config.asr_model_name != null && config.asr_model_name !== '' ? config.asr_model_name : ((config.asr && config.asr.model) ? String(config.asr.model).replace(/\(.*\)/, '').trim() : 'Parakeet');
    const llmModel = config.llm_model_name != null && config.llm_model_name !== '' ? config.llm_model_name : ((config.llm && config.llm.model) ? String(config.llm.model) : '—');
    // When RIVA is selected, show RIVA TTS model name in pipeline-seg; fallback to "RIVA" if not loaded yet
    const isRivaTts = config.tts && (config.tts.backend === 'riva' || config.tts.scheme === 'riva');
    const ttsLabel = config.tts_model_name != null && config.tts_model_name !== '' ? config.tts_model_name
        : (config.tts && (config.tts.riva_model_name || config.tts.voice || config.tts.model))
            ? (isRivaTts && config.tts.riva_model_name ? config.tts.riva_model_name : (config.tts.riva_model_name || config.tts.voice || config.tts.model))
            : (config.tts ? (isRivaTts ? 'RIVA' : 'Default') : '—');
    /* Segment label: model name only; tooltip shows "ASR: ..." / "VLM: ..." / "TTS: ..." on hover */
    const asrLabel = isTextOnly ? 'n/a' : asrModel;
    const midLabel = llmModel;
    const ttsLabelVal = hasSpeaker ? ttsLabel : 'n/a';
    const asrTooltip = isTextOnly ? 'n/a' : ('ASR: ' + asrModel);
    const midTooltip = hasCamera ? ('VLM: ' + llmModel) : ('LLM: ' + llmModel);
    const ttsTooltip = hasSpeaker ? ('TTS: ' + ttsLabel) : 'n/a';

    var deviceTypesOpt = options && options.deviceTypes;
    var fullLabel = function (short, full) { return full || short; };
    function slot(icon, shortLabel, fullTitle, typeIcon) {
        var titleAttr = fullTitle ? ' title="' + escapeHtml(fullTitle) + '"' : '';
        var dataFull = ' data-full-label="' + escapeHtml(fullLabel(shortLabel, fullTitle)) + '"';
        var typeIconHtml = typeIcon === 'browser' ? '<i data-lucide="chromium" class="lucide-inline pipeline-device-type-icon" aria-hidden="true"></i>' : (typeIcon === 'usb' ? '<i data-lucide="usb" class="lucide-inline pipeline-device-type-icon" aria-hidden="true"></i>' : '');
        return '<span class="pipeline-device-slot"' + titleAttr + dataFull + '><span class="pipeline-device-slot-icon"><i data-lucide="' + icon + '" class="lucide-inline"></i></span><span class="pipeline-device-slot-label"><span class="pipeline-device-slot-name">' + escapeHtml(shortLabel) + '</span>' + typeIconHtml + '</span></span>';
    }
    var micType = deviceTypesOpt && deviceTypesOpt.mic != null ? deviceTypesOpt.mic : null;
    var camType = deviceTypesOpt && deviceTypesOpt.camera != null ? deviceTypesOpt.camera : null;
    var spkType = deviceTypesOpt && deviceTypesOpt.speaker != null ? deviceTypesOpt.speaker : null;
    var emptySlot = '<span class="pipeline-device-slot pipeline-device-slot--placeholder" aria-hidden="true"><span class="pipeline-device-slot-icon"></span><span class="pipeline-device-slot-label"><span class="pipeline-device-slot-name"></span></span></span>';
    const slots = slot('mic', micT.short, micT.title, micType) + (hasCamera ? slot('video', camT.short, camT.title, camType) : emptySlot) + slot('volume-2', spkT.short, spkT.title, spkType);
    function seg(type, labelText, tooltip) {
        var titleAttr = tooltip ? ' title="' + escapeHtml(tooltip) + '"' : '';
        var dataFull = ' data-full-label="' + escapeHtml(labelText) + '"';
        return '<span class="pipeline-seg pipeline-seg-' + type + '"><svg class="pipeline-seg-shape" viewBox="0 0 200 40" preserveAspectRatio="none" aria-hidden="true"><polygon class="pipeline-seg-poly" points="0,0 180,0 200,20 180,40 0,40 20,20"/></svg><span class="pipeline-seg-label"' + titleAttr + dataFull + '>' + escapeHtml(labelText) + '</span></span>';
    }
    const arrowRow = hasCamera ? '<div class="pipeline-arrow-row" aria-hidden="true"><i data-lucide="chevron-down" class="lucide-inline pipeline-arrow-down"></i></div>' : '';
    var flowSegments;
    if (isRealtimeFullVoiceConfig(config)) {
        var realtimeLabel = 'OpenAI Realtime';
        var realtimeTitle = 'ASR + TTS Realtime (Full voice)';
        flowSegments = '<div class="pipeline-flow">' +
            '<span class="pipeline-seg pipeline-seg-realtime">' +
            '<svg class="pipeline-seg-shape" viewBox="0 0 200 40" preserveAspectRatio="none" aria-hidden="true">' +
            '<defs><linearGradient id="pipeline-realtime-stroke" x1="0%" y1="0%" x2="100%" y2="0%">' +
            '<stop offset="0%" stop-color="var(--timeline-asr, #1976D2)"/>' +
            '<stop offset="100%" stop-color="var(--timeline-tts, #EC407A)"/>' +
            '</linearGradient></defs>' +
            '<polygon class="pipeline-seg-poly" stroke="url(#pipeline-realtime-stroke)" points="0,0 180,0 200,20 180,40 0,40 20,20"/></svg>' +
            '<span class="pipeline-seg-label" title="' + escapeHtml(realtimeTitle) + '" data-full-label="' + escapeHtml(realtimeLabel) + '">' + escapeHtml(realtimeLabel) + '</span></span></div>';
    } else {
        flowSegments = '<div class="pipeline-flow">' + seg('asr', asrLabel, asrTooltip) + seg('llm', midLabel, midTooltip) + seg('tts', ttsLabelVal, ttsTooltip) + '</div>';
    }
    const flowRow = '<div class="pipeline-flow-row">' + flowSegments + '</div>';
    const cornerLeft = '<div class="pipeline-corner pipeline-corner-left" aria-hidden="true"><i data-lucide="corner-down-right" class="lucide-inline"></i></div>';
    const cornerRight = '<div class="pipeline-corner pipeline-corner-right" aria-hidden="true"><i data-lucide="corner-right-up" class="lucide-inline"></i></div>';
    return '<div class="pipeline-grid">' + slots + arrowRow + flowRow + cornerLeft + cornerRight + '</div>';
}

/** One-line pipeline summary for session list (e.g. "ASR: Parakeet | LLM: llama3:3b | TTS: Default"). */
function getPipelineSummaryHtml(config) {
    if (!config) return '';
    const d = config.devices || {};
    const mic = d.microphone ?? d.audio_input_source;
    const cam = d.camera ?? d.video_source;
    const spk = d.speaker ?? d.audio_output_source;
    const isTextOnly = mic === 'none' || !mic;
    const hasCamera = cam && cam !== 'none';
    const hasSpeaker = spk && spk !== 'none';
    const asrModel = config.asr_model_name != null && config.asr_model_name !== '' ? config.asr_model_name : ((config.asr && config.asr.model) ? String(config.asr.model).replace(/\(.*\)/, '').trim() : 'Parakeet');
    const llmModel = config.llm_model_name != null && config.llm_model_name !== '' ? config.llm_model_name : ((config.llm && config.llm.model) ? String(config.llm.model) : '—');
    const isRivaTtsSummary = config.tts && (config.tts.backend === 'riva' || config.tts.scheme === 'riva');
    const ttsLabel = config.tts_model_name != null && config.tts_model_name !== '' ? config.tts_model_name
        : (config.tts && (config.tts.riva_model_name || config.tts.voice || config.tts.model))
            ? (isRivaTtsSummary && config.tts.riva_model_name ? config.tts.riva_model_name : (config.tts.riva_model_name || config.tts.voice || config.tts.model))
            : (config.tts ? (isRivaTtsSummary ? 'RIVA' : 'Default') : '—');
    const asr = isTextOnly ? 'n/a' : ('ASR: ' + asrModel);
    const mid = hasCamera ? ('VLM: ' + llmModel) : ('LLM: ' + llmModel);
    const tts = hasSpeaker ? ('TTS: ' + ttsLabel) : 'n/a';
    return escapeHtml(asr + ' | ' + mid + ' | ' + tts);
}

/** Update pipeline segment SVG and --seg-slant so slants stay 45°/135° at any size (inset = height/2). Polygon inset by 1 so 1px stroke is not clipped by viewBox. */
function updatePipelineSegShapes(container) {
    if (!container) return;
    var segs = container.querySelectorAll('.pipeline-seg');
    var inset = 1;
    segs.forEach(function (seg) {
        var w = seg.offsetWidth;
        var h = seg.offsetHeight;
        if (w <= 0 || h <= 0) return;
        var s = Math.max(1, Math.floor(h / 2));
        seg.style.setProperty('--seg-slant', s + 'px');
        var svg = seg.querySelector('.pipeline-seg-shape');
        var poly = seg.querySelector('.pipeline-seg-poly');
        if (svg && poly) {
            svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
            var x0 = inset;
            var y0 = inset;
            var x1 = w - inset;
            var y1 = h - inset;
            var midY = h / 2;
            poly.setAttribute('points', x0 + ',' + y0 + ' ' + (x1 - s) + ',' + y0 + ' ' + x1 + ',' + midY + ' ' + (x1 - s) + ',' + y1 + ' ' + x0 + ',' + y1 + ' ' + (x0 + s) + ',' + midY);
        }
    });
}

/** Approximate chars that fit in a pixel width (monospace-ish). */
var PX_PER_CHAR_DEVICE = 7;
var PX_PER_CHAR_SEG = 6;
var DEVICE_SLOT_NON_LABEL_PX = 52;

/** Re-trim device badge and pipeline-seg labels based on current container width (call on resize). */
function updatePipelineLabelTrimming(container) {
    if (!container) return;
    var grid = container.querySelector('.pipeline-grid');
    if (!grid) return;

    // Device slots: each slot gets ~1/3 of grid; reserve space for icon + padding
    var slots = container.querySelectorAll('.pipeline-device-slot');
    slots.forEach(function (slot) {
        var nameEl = slot.querySelector('.pipeline-device-slot-name');
        var full = slot.getAttribute('data-full-label');
        if (!nameEl || full == null) return;
        var w = slot.offsetWidth;
        var labelPx = Math.max(0, w - DEVICE_SLOT_NON_LABEL_PX);
        var maxLen = Math.max(2, Math.floor(labelPx / PX_PER_CHAR_DEVICE));
        var t = truncateWithTitle(full, maxLen);
        nameEl.textContent = t.short;
        slot.setAttribute('title', t.title || '');
    });

    // Pipeline segments: each seg has flex:1; trim label to fit
    var segs = container.querySelectorAll('.pipeline-seg');
    segs.forEach(function (seg) {
        var labelEl = seg.querySelector('.pipeline-seg-label');
        var full = labelEl && labelEl.getAttribute('data-full-label');
        if (!labelEl || full == null) return;
        var w = seg.offsetWidth;
        var labelPx = Math.max(0, w - 24);
        var maxLen = Math.max(2, Math.floor(labelPx / PX_PER_CHAR_SEG));
        var display = full.length <= maxLen ? full : full.slice(0, maxLen - 3) + '\u2026';
        labelEl.textContent = display;
    });
}

/** Refresh pipeline display when config or devices change (live session uses currentConfig and real device labels). */
function refreshPipelineDisplay() {
    const el = document.getElementById('pipeline-config');
    if (!el) return;
    if (state.isLiveSession) {
        var deviceLabels = {
            mic: getDeviceDisplayLabel('mic'),
            camera: getDeviceDisplayLabel('camera'),
            speaker: getDeviceDisplayLabel('speaker')
        };
        var deviceTypes = { mic: getDeviceDisplayType('mic'), camera: getDeviceDisplayType('camera'), speaker: getDeviceDisplayType('speaker') };
        el.innerHTML = getPipelineTableHtml(currentConfig, { condensed: false, deviceLabels: deviceLabels, deviceTypes: deviceTypes });
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        requestAnimationFrame(function () { updatePipelineSegShapes(el); updatePipelineLabelTrimming(el); });
        updateImagePlaceholderContent();
    } else if (state.selectedSession && state.selectedSession.config) {
        var c = state.selectedSession.config;
        var opts = { condensed: false };
        if (c.device_labels && (c.device_labels.mic != null || c.device_labels.camera != null || c.device_labels.speaker != null)) opts.deviceLabels = c.device_labels;
        var types = c.device_types ? { ...c.device_types } : {};
        var d = c.devices || {};
        if ((d.speaker && (String(d.speaker).indexOf('alsa:') === 0 || String(d.speaker).indexOf('pyaudio:') === 0)) && types.speaker !== 'usb') types.speaker = 'usb';
        if ((d.microphone && (String(d.microphone).indexOf('alsa:') === 0 || String(d.microphone).indexOf('pyaudio:') === 0)) && types.mic !== 'usb') types.mic = 'usb';
        if (d.camera && String(d.camera).indexOf('/dev/') === 0 && types.camera !== 'usb') types.camera = 'usb';
        opts.deviceTypes = types;
        el.innerHTML = getPipelineTableHtml(c, opts);
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
        requestAnimationFrame(function () { updatePipelineSegShapes(el); updatePipelineLabelTrimming(el); });
    }
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded fired!');

    console.log('Loading UI settings...');
    try {
        loadUISettings();
        console.log('UI settings loaded:', uiSettings);
        applyUISettingsToLayout();
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

    var pipelineEl = document.getElementById('pipeline-config');
    if (pipelineEl && typeof ResizeObserver !== 'undefined') {
        var ro = new ResizeObserver(function () {
            updatePipelineSegShapes(pipelineEl);
            updatePipelineLabelTrimming(pipelineEl);
        });
        ro.observe(pipelineEl);
    }

    console.log('Setting up modal handlers...');
    try {
        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', openSettingsModal);
        document.getElementById('settings-modal-close').addEventListener('click', closeSettingsModal);
        document.getElementById('settings-save-btn').addEventListener('click', saveSettingsFromModal);
        document.getElementById('settings-reset-btn').addEventListener('click', resetUISettings);

        // Session filename copy button (event delegation)
        var sessionMetaEl = document.getElementById('session-meta');
        if (sessionMetaEl) {
            sessionMetaEl.addEventListener('click', function (e) {
                var btn = e.target.closest('.session-filename-copy');
                if (!btn || !btn.dataset.filename) return;
                var filename = btn.dataset.filename;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(filename).then(function () {
                        btn.setAttribute('aria-label', 'Copied');
                        setTimeout(function () { btn.setAttribute('aria-label', 'Copy filename'); }, 1500);
                    }).catch(function () {});
                }
            });
        }

        // Close modal on backdrop click
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target.id === 'settings-modal') {
                closeSettingsModal();
            }
        });

        // Settings modal: pane navigation
        document.querySelectorAll('.settings-nav-item').forEach(btn => {
            btn.addEventListener('click', () => {
                const pane = btn.getAttribute('data-pane');
                document.querySelectorAll('.settings-nav-item').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.settings-pane').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                const content = document.getElementById('settings-pane-' + pane);
                if (content) content.classList.add('active');
            });
        });

        // Timeline height: Auto checkbox enables/disables slider and updates value display
        const timelineHeightAutoEl = document.getElementById('ui-timeline-height-auto');
        const timelineHeightPxEl = document.getElementById('ui-timeline-height-px');
        const timelineHeightPxValueEl = document.getElementById('ui-timeline-height-px-value');
        if (timelineHeightAutoEl && timelineHeightPxEl && timelineHeightPxValueEl) {
            timelineHeightAutoEl.addEventListener('change', () => {
                const auto = timelineHeightAutoEl.checked;
                timelineHeightPxEl.disabled = auto;
                timelineHeightPxValueEl.textContent = timelineHeightPxEl.value;
            });
            timelineHeightPxEl.addEventListener('input', () => {
                timelineHeightPxValueEl.textContent = timelineHeightPxEl.value;
            });
        }

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
                closeSessionMenus();
            }
        });

        // Session list: menu button and dropdown actions (delegation)
        const sessionItems = document.getElementById('session-items');
        if (sessionItems) {
            sessionItems.addEventListener('click', (e) => {
                const menuBtn = e.target.closest('.session-item-menu-btn');
                const menuitem = e.target.closest('.session-item-dropdown [data-action]');
                if (menuBtn) {
                    e.preventDefault();
                    e.stopPropagation();
                    const sid = menuBtn.getAttribute('data-session-id');
                    const idx = menuBtn.getAttribute('data-session-index');
                    if (sid != null) toggleSessionMenu(e, sid, idx != null ? parseInt(idx, 10) : 0);
                } else if (menuitem) {
                    e.preventDefault();
                    const dropdown = e.target.closest('.session-item-dropdown');
                    const sid = dropdown && dropdown.getAttribute('data-session-id');
                    if (sid && menuitem.getAttribute('data-action') === 'rename') renameSession(sid);
                    if (sid && menuitem.getAttribute('data-action') === 'delete') deleteSession(sid);
                }
            });
        }
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.session-item-menu-btn') && !e.target.closest('.session-item-dropdown')) closeSessionMenus();
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

    // Load current settings into controls
    document.getElementById('ui-combine-speech-lanes').checked = uiSettings.combineSpeechLanes;
    document.getElementById('ui-show-session-thumbnails').checked = uiSettings.showSessionThumbnails;
    document.getElementById('ui-show-pipeline-in-session-list').checked = uiSettings.showPipelineInSessionList;
    document.getElementById('ui-show-new-chat-with-default-config').checked = uiSettings.showNewChatWithDefaultConfig;
    document.getElementById('ui-record-preview-in-session-history').checked = uiSettings.recordPreviewInSessionHistory;
    document.getElementById('ui-auto-scroll-chat').checked = uiSettings.autoScrollChat;
    document.getElementById('ui-show-timestamps').checked = uiSettings.showTimestamps;
    document.getElementById('ui-show-debug-info').checked = uiSettings.showDebugInfo;

    const timelineAuto = document.getElementById('ui-timeline-height-auto');
    const timelinePx = document.getElementById('ui-timeline-height-px');
    const timelinePxValue = document.getElementById('ui-timeline-height-px-value');
    if (timelineAuto) timelineAuto.checked = uiSettings.timelineHeightAuto;
    if (timelinePx) {
        timelinePx.value = String(uiSettings.timelineHeightPx);
        timelinePx.disabled = !!uiSettings.timelineHeightAuto;
    }
    if (timelinePxValue) timelinePxValue.textContent = String(uiSettings.timelineHeightPx);

    const micPreviewGainEl = document.getElementById('ui-mic-preview-gain');
    if (micPreviewGainEl) {
        var g = (typeof uiSettings.micPreviewGain === 'number' && !isNaN(uiSettings.micPreviewGain)) ? uiSettings.micPreviewGain : 2;
        g = Math.max(1, Math.min(4, Math.round(g)));
        micPreviewGainEl.value = String(g);
    }

    modal.classList.add('show');
}

function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    modal.classList.remove('show');
}

/** Apply UI settings that affect layout (New Chat button label, timeline panel height). Call after loadUISettings or after saving. */
function applyUISettingsToLayout() {
    const newBtn = document.getElementById('new-session-btn');
    if (newBtn) {
        newBtn.textContent = uiSettings.showNewChatWithDefaultConfig ? '+ New Chat with Default Config' : '+ New Voice Chat';
    }
    const panel = document.getElementById('timeline-panel');
    if (panel) {
        if (uiSettings.timelineHeightAuto) {
            panel.style.height = '';
            panel.classList.add('timeline-panel--height-auto');
        } else {
            panel.classList.remove('timeline-panel--height-auto');
            const px = Math.max(200, Math.min(600, uiSettings.timelineHeightPx || 400));
            panel.style.height = px + 'px';
        }
    }
    if (state.selectedSession) {
        renderTimeline();
    }
}

function saveSettingsFromModal() {
    // Read values from controls
    uiSettings.combineSpeechLanes = document.getElementById('ui-combine-speech-lanes').checked;
    uiSettings.showSessionThumbnails = document.getElementById('ui-show-session-thumbnails').checked;
    uiSettings.showPipelineInSessionList = document.getElementById('ui-show-pipeline-in-session-list').checked;
    uiSettings.showNewChatWithDefaultConfig = document.getElementById('ui-show-new-chat-with-default-config').checked;
    uiSettings.recordPreviewInSessionHistory = document.getElementById('ui-record-preview-in-session-history').checked;
    uiSettings.autoScrollChat = document.getElementById('ui-auto-scroll-chat').checked;
    uiSettings.showTimestamps = document.getElementById('ui-show-timestamps').checked;
    uiSettings.showDebugInfo = document.getElementById('ui-show-debug-info').checked;

    const timelineAuto = document.getElementById('ui-timeline-height-auto');
    const timelinePx = document.getElementById('ui-timeline-height-px');
    uiSettings.timelineHeightAuto = timelineAuto ? timelineAuto.checked : true;
    uiSettings.timelineHeightPx = timelinePx ? Math.max(200, Math.min(600, parseInt(timelinePx.value, 10) || 400)) : 400;

    const micPreviewGainEl = document.getElementById('ui-mic-preview-gain');
    if (micPreviewGainEl) {
        var g = parseInt(micPreviewGainEl.value, 10);
        uiSettings.micPreviewGain = (!isNaN(g) && g >= 1 && g <= 4) ? g : 2;
    }

    saveUISettings();
    applyUISettingsToLayout();

    renderSessionList();
    if (state.selectedSession) {
        renderTimeline();
    }
    updateVoiceDebugPanel();

    closeSettingsModal();
}

function resetUISettings() {
    if (confirm('Reset all UI settings to defaults?')) {
        uiSettings.combineSpeechLanes = false;
        uiSettings.showSessionThumbnails = true;
        uiSettings.autoScrollChat = true;
        uiSettings.showTimestamps = false;
        uiSettings.showDebugInfo = false;
        uiSettings.showPipelineInSessionList = false;
        uiSettings.showNewChatWithDefaultConfig = false;
        uiSettings.recordPreviewInSessionHistory = true;
        uiSettings.timelineHeightAuto = true;
        uiSettings.timelineHeightPx = 400;
        uiSettings.micPreviewGain = 2;
        uiSettings.userAudioGain = 2;
        uiSettings.aiAudioGain = 2;
        uiSettings.userVoiceThreshold = 5;

        saveUISettings();

        document.getElementById('ui-combine-speech-lanes').checked = uiSettings.combineSpeechLanes;
        document.getElementById('ui-show-session-thumbnails').checked = uiSettings.showSessionThumbnails;
        document.getElementById('ui-show-pipeline-in-session-list').checked = uiSettings.showPipelineInSessionList;
        document.getElementById('ui-show-new-chat-with-default-config').checked = uiSettings.showNewChatWithDefaultConfig;
        document.getElementById('ui-record-preview-in-session-history').checked = uiSettings.recordPreviewInSessionHistory;
        document.getElementById('ui-auto-scroll-chat').checked = uiSettings.autoScrollChat;
        document.getElementById('ui-show-timestamps').checked = uiSettings.showTimestamps;
        document.getElementById('ui-show-debug-info').checked = uiSettings.showDebugInfo;
        const timelineAuto = document.getElementById('ui-timeline-height-auto');
        const timelinePx = document.getElementById('ui-timeline-height-px');
        const timelinePxValue = document.getElementById('ui-timeline-height-px-value');
        if (timelineAuto) timelineAuto.checked = uiSettings.timelineHeightAuto;
        if (timelinePx) {
            timelinePx.value = String(uiSettings.timelineHeightPx);
            timelinePx.disabled = !!uiSettings.timelineHeightAuto;
        }
        if (timelinePxValue) timelinePxValue.textContent = String(uiSettings.timelineHeightPx);
        var micPreviewGainEl = document.getElementById('ui-mic-preview-gain');
        if (micPreviewGainEl) micPreviewGainEl.value = String(uiSettings.micPreviewGain);
        const ug = document.getElementById('timeline-user-audio-gain');
        const ag = document.getElementById('timeline-ai-audio-gain');
        if (ug) ug.value = String(uiSettings.userAudioGain);
        if (ag) ag.value = String(uiSettings.aiAudioGain);
        const uvt = document.getElementById('timeline-user-voice-threshold');
        const uvtVal = document.getElementById('timeline-user-voice-threshold-value');
        if (uvt) {
            var uv = (typeof uiSettings.userVoiceThreshold === 'number' && !isNaN(uiSettings.userVoiceThreshold)) ? uiSettings.userVoiceThreshold : 5;
            uv = Math.max(0, Math.min(30, uv));
            uvt.value = String(uv);
            if (uvtVal) uvtVal.textContent = String(uv);
        }
        applyUISettingsToLayout();
        renderSessionList();
        if (state.selectedSession || (state.isLiveSession && state.sessionState === 'stopped')) {
            renderTimeline();
        }
    }
}
