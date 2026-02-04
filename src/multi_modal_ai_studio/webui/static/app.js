// Multi-modal AI Studio - WebUI App
// Handles session loading, display, and timeline visualization

// ===== State Management =====
const state = {
    sessions: [],
    selectedSession: null,
    activeConfigTab: 'asr',
    timelineZoom: 1.0,
    timelineOffset: 0,
    isLiveSession: false,
    sessionState: 'setup', // 'setup', 'live', 'stopped'
    
    // UI state (for future persistence)
    ui: {
        configPanelCollapsed: false,
        timelinePanelCollapsed: false,
        sessionListVisible: true, // For mobile
    }
};

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

function renderConfig() {
    const session = state.selectedSession;
    const config = session.config || {};
    const tab = state.activeConfigTab;
    const contentEl = document.getElementById('config-tab-content');
    
    let content = '';
    
    switch (tab) {
        case 'asr':
            content = renderConfigSection('ASR Configuration', config.asr || {});
            break;
        case 'llm':
            content = renderConfigSection('LLM Configuration', config.llm || {});
            break;
        case 'tts':
            content = renderConfigSection('TTS Configuration', config.tts || {});
            break;
        case 'device':
            content = renderConfigSection('Device Configuration', config.devices || {});
            break;
        case 'app':
            content = renderConfigSection('App Configuration', config.app || {});
            break;
    }
    
    contentEl.innerHTML = content;
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

function renderChatHistory() {
    const session = state.selectedSession;
    const turns = session.turns || [];
    const chatEl = document.getElementById('chat-history');
    
    if (turns.length === 0) {
        chatEl.innerHTML = `
            <div class="empty-state">
                <p>No conversation yet</p>
            </div>
        `;
        return;
    }
    
    chatEl.innerHTML = turns.map(turn => {
        const userConfidence = turn.user_confidence ? 
            `<span class="chat-meta">Confidence: ${(turn.user_confidence * 100).toFixed(0)}%</span>` : '';
        const turnMetrics = turn.latencies ? 
            `<span class="chat-meta">TTL: ${formatLatency(turn.latencies.ttl)}</span>` : '';
        
        return `
            <div class="chat-message user">
                <div class="chat-avatar">👤</div>
                <div class="chat-bubble">
                    <div class="chat-text">${escapeHtml(turn.user_transcript || '...')}</div>
                    ${userConfidence}
                </div>
            </div>
            <div class="chat-message ai">
                <div class="chat-avatar">🤖</div>
                <div class="chat-bubble">
                    <div class="chat-text">${escapeHtml(turn.ai_response || '...')}</div>
                    ${turnMetrics}
                </div>
            </div>
        `;
    }).join('');
    
    // Auto-scroll to bottom
    chatEl.scrollTop = chatEl.scrollHeight;
}

function renderTimelineMetrics() {
    const session = state.selectedSession;
    const metrics = session.metrics || {};
    const metricsEl = document.getElementById('timeline-metrics');
    
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

// ===== Timeline Canvas Rendering =====
function renderTimeline() {
    if (!state.selectedSession) return;
    
    const canvas = document.getElementById('timeline-canvas');
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
    
    const timeline = state.selectedSession.timeline || [];
    if (timeline.length === 0) {
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No timeline data', width / 2, height / 2);
        return;
    }
    
    // Timeline rendering constants
    const LANE_HEIGHT = 40;
    const LANE_GAP = 10;
    const PADDING_TOP = 20;
    const PADDING_LEFT = 100;
    const PADDING_RIGHT = 20;
    
    const lanes = ['system', 'audio', 'speech', 'llm', 'tts'];
    const laneColors = {
        system: getComputedStyle(document.documentElement).getPropertyValue('--timeline-system'),
        audio: getComputedStyle(document.documentElement).getPropertyValue('--timeline-audio'),
        speech: getComputedStyle(document.documentElement).getPropertyValue('--timeline-speech'),
        llm: getComputedStyle(document.documentElement).getPropertyValue('--timeline-llm'),
        tts: getComputedStyle(document.documentElement).getPropertyValue('--timeline-tts'),
    };
    
    // Calculate time range
    const maxTime = Math.max(...timeline.map(e => e.timestamp));
    const timeScale = (width - PADDING_LEFT - PADDING_RIGHT) / maxTime;
    
    // Draw lane labels
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    lanes.forEach((lane, i) => {
        const y = PADDING_TOP + i * (LANE_HEIGHT + LANE_GAP) + LANE_HEIGHT / 2;
        ctx.fillText(lane.toUpperCase(), PADDING_LEFT - 10, y + 4);
    });
    
    // Draw lane backgrounds
    lanes.forEach((lane, i) => {
        const y = PADDING_TOP + i * (LANE_HEIGHT + LANE_GAP);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-tertiary');
        ctx.fillRect(PADDING_LEFT, y, width - PADDING_LEFT - PADDING_RIGHT, LANE_HEIGHT);
    });
    
    // Draw events
    timeline.forEach(event => {
        const laneIndex = lanes.indexOf(event.lane);
        if (laneIndex === -1) return;
        
        const x = PADDING_LEFT + event.timestamp * timeScale;
        const y = PADDING_TOP + laneIndex * (LANE_HEIGHT + LANE_GAP);
        
        // Draw event marker
        ctx.fillStyle = laneColors[event.lane] || '#888';
        ctx.beginPath();
        ctx.arc(x, y + LANE_HEIGHT / 2, 4, 0, Math.PI * 2);
        ctx.fill();
        
        // Highlight TTL-critical events
        if (event.event_type === 'user_speech_end' || event.event_type === 'tts_first_audio') {
            ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--ttl-highlight');
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y + LANE_HEIGHT / 2, 7, 0, Math.PI * 2);
            ctx.stroke();
        }
    });
    
    // Draw time axis
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING_LEFT, height - 30);
    ctx.lineTo(width - PADDING_RIGHT, height - 30);
    ctx.stroke();
    
    // Draw time labels
    const timeSteps = 5;
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i <= timeSteps; i++) {
        const t = (maxTime / timeSteps) * i;
        const x = PADDING_LEFT + t * timeScale;
        ctx.fillText(`${t.toFixed(1)}s`, x, height - 10);
    }
}

// ===== Live Session Management =====
function updateLiveSessionUI() {
    const deviceControls = document.getElementById('device-controls-container');
    const videoFeed = document.getElementById('video-feed');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const previewImage = document.getElementById('preview-image');
    const sessionTitle = document.getElementById('session-title');
    const sessionStats = document.getElementById('session-stats');
    const startBtn = document.getElementById('start-session-btn');
    const stopBtn = document.getElementById('stop-session-btn');
    
    if (state.isLiveSession) {
        // Show device controls
        deviceControls.style.display = 'flex';
        
        // Show video feed (and hide preview image)
        videoFeed.style.display = 'block';
        previewImage.style.display = 'none';
        imagePlaceholder.style.display = 'none';
        
        // Update session meta
        const now = new Date();
        sessionTitle.textContent = `Live Session - ${now.toLocaleTimeString()}`;
        
        // Update stats to show PREVIEW/LIVE status
        if (state.sessionState === 'setup') {
            sessionStats.innerHTML = '<span class="stat-value" style="color: var(--accent-secondary);">PREVIEW MODE</span>';
            startBtn.style.display = 'flex';
            stopBtn.style.display = 'none';
        } else if (state.sessionState === 'live') {
            sessionStats.innerHTML = '<span class="stat-value" style="color: #ef4444;">🔴 RECORDING</span>';
            startBtn.style.display = 'none';
            stopBtn.style.display = 'flex';
        }
    } else {
        // Hide device controls and video feed for historical sessions
        deviceControls.style.display = 'none';
        videoFeed.style.display = 'none';
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
        
        // Update session stats (horizontal: Date | Turns | TTL)
        const date = new Date(session.created_at).toLocaleDateString();
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
        
        // Show preview image (hide video feed)
        videoFeed.style.display = 'none';
        
        // Try to load thumbnail if exists
        if (session.thumbnail) {
            previewImage.src = session.thumbnail;
            previewImage.style.display = 'block';
            imagePlaceholder.style.display = 'none';
        } else {
            // Show placeholder
            previewImage.style.display = 'none';
            imagePlaceholder.style.display = 'flex';
        }
        
        // Show the preview container
        sessionMeta.style.display = 'flex';
    } else if (!state.selectedSession) {
        // No session selected - show empty state
        sessionTitle.textContent = 'Select a session';
        sessionStats.innerHTML = '';
        previewImage.style.display = 'none';
        videoFeed.style.display = 'none';
        imagePlaceholder.style.display = 'flex';
    }
}

function startNewSession() {
    // Create new live session
    state.isLiveSession = true;
    state.sessionState = 'setup';
    state.selectedSession = null;
    
    // Clear chat history
    document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p>Configure devices and click START to begin</p>
        </div>
    `;
    
    // Update UI
    renderSessionList();
    updateLiveSessionUI();
    
    console.log('New session created - Setup mode');
}

function startSessionRecording() {
    if (state.sessionState !== 'setup') return;
    
    state.sessionState = 'live';
    updateLiveSessionUI();
    
    console.log('Session recording started!');
    // TODO: Initialize WebRTC, start ASR/LLM/TTS pipeline, begin timeline recording
    
    // Clear chat and show live indicator
    document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p>🔴 Session is LIVE - Start talking!</p>
        </div>
    `;
}

function stopSessionRecording() {
    if (state.sessionState !== 'live') return;
    
    state.sessionState = 'stopped';
    state.isLiveSession = false;
    
    console.log('Session recording stopped!');
    // TODO: Stop WebRTC, close connections, finalize session, save to sessions list
    
    // Show completion message
    document.getElementById('chat-history').innerHTML = `
        <div class="empty-state">
            <p>✅ Session saved! Check the session list.</p>
        </div>
    `;
    
    updateLiveSessionUI();
}

// ===== Event Handlers =====
function setupEventHandlers() {
    console.log('setupEventHandlers() called');
    
    // Theme toggle
    try {
        document.getElementById('theme-toggle').addEventListener('click', () => {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            document.getElementById('theme-toggle').textContent = newTheme === 'dark' ? '🌙' : '☀️';
            // Re-render timeline with new colors
            if (state.selectedSession) {
                renderTimeline();
            }
        });
        console.log('✓ Theme toggle handler attached');
    } catch (e) {
        console.error('✗ Error attaching theme toggle:', e);
    }
    
    // Config tabs
    document.querySelectorAll('.config-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.config-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.activeConfigTab = tab.dataset.tab;
            renderConfig();
        });
    });
    
    // Timeline zoom controls
    document.getElementById('timeline-zoom-in').addEventListener('click', () => {
        state.timelineZoom *= 1.2;
        renderTimeline();
    });
    
    document.getElementById('timeline-zoom-out').addEventListener('click', () => {
        state.timelineZoom /= 1.2;
        renderTimeline();
    });
    
    document.getElementById('timeline-reset').addEventListener('click', () => {
        state.timelineZoom = 1.0;
        state.timelineOffset = 0;
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
    
    // Device selection handlers (stubs for now)
    document.getElementById('camera-select').addEventListener('change', (e) => {
        console.log('Camera changed:', e.target.value);
        // TODO: Initialize camera stream
    });
    
    document.getElementById('mic-select').addEventListener('change', (e) => {
        console.log('Microphone changed:', e.target.value);
        // TODO: Initialize microphone stream
    });
    
    document.getElementById('speaker-select').addEventListener('change', (e) => {
        console.log('Speaker changed:', e.target.value);
        // TODO: Set audio output device
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

function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded fired!');
    console.log('Setting up event handlers...');
    try {
        setupEventHandlers();
        console.log('Event handlers set up successfully');
    } catch (error) {
        console.error('Error setting up event handlers:', error);
    }
    
    console.log('Loading sessions...');
    try {
        loadSessions();
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
    
    // Future: Load saved UI state
    // loadUIState();
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
