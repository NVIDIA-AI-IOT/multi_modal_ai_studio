# Future UI Enhancements - Collapsible Panels

## Overview

The current design uses a 2-column grid (session list | main panel) with config panel collapse. Future enhancements will add collapsible timeline and responsive mobile design.

## Current Structure

```
.container (2 columns: session | main)
├── .session-list
└── .main-panel (2 rows: content | timeline)
    ├── .content-area (2 columns: config | chat)
    │   ├── .config-panel-bar (collapse indicator)
    │   ├── .config-panel ✅ COLLAPSIBLE
    │   └── .chat-panel
    └── .timeline-panel 🔜 WILL BE COLLAPSIBLE
```

---

## 1. Timeline Panel Collapse 🔜

### Goal
Collapse timeline to a thin bar at the bottom, showing only header with expand button.

### UI Design

**Expanded (Current)**:
```
┌─────────────────────────────┐
│ Timeline        [ - ]  ⟲    │ ← Header with collapse button
├─────────────────────────────┤
│ █████████████████████████   │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━   │ ← Canvas visualization
│                             │
│ Metrics: TTL 285ms ...      │
└─────────────────────────────┘
Height: 300px
```

**Collapsed**:
```
┌─────────────────────────────┐
│ Timeline ▲  TTL:285ms  12 turns │ ← Mini bar with key metrics
└─────────────────────────────┘
Height: 40px
```

### Implementation Plan

**HTML Structure** (add class):
```html
<section class="timeline-panel" id="timeline-panel">
    <!-- Add collapse button in header -->
    <div class="timeline-header">
        <h3>Timeline</h3>
        <div class="timeline-controls">
            <button id="timeline-collapse-btn" class="icon-btn" title="Collapse timeline">▼</button>
            <!-- existing zoom buttons -->
        </div>
    </div>
    <!-- rest of timeline -->
</section>
```

**CSS** (add to styles.css):
```css
/* Timeline collapsed state */
.timeline-panel.collapsed {
    height: 40px;
    overflow: hidden;
}

.timeline-panel.collapsed .timeline-content,
.timeline-panel.collapsed .timeline-metrics {
    display: none;
}

.timeline-panel.collapsed .timeline-header {
    /* Show mini metrics inline */
}

/* Mini metrics bar when collapsed */
.timeline-mini-metrics {
    display: none;
    margin-left: auto;
    gap: 1rem;
    font-size: 0.85rem;
}

.timeline-panel.collapsed .timeline-mini-metrics {
    display: flex;
}
```

**JavaScript** (add to app.js):
```javascript
// Timeline collapse toggle
document.getElementById('timeline-collapse-btn').addEventListener('click', () => {
    const timelinePanel = document.getElementById('timeline-panel');
    const btn = document.getElementById('timeline-collapse-btn');
    
    timelinePanel.classList.toggle('collapsed');
    btn.textContent = timelinePanel.classList.contains('collapsed') ? '▲' : '▼';
    btn.title = timelinePanel.classList.contains('collapsed') ? 
        'Expand timeline' : 'Collapse timeline';
});
```

**Benefits**:
- More vertical space for chat when timeline not needed
- Key metrics still visible in collapsed state
- Smooth animation (height transition)
- No layout shifts for chat panel

---

## 2. Session List Mobile Collapse 🔜

### Goal
Hide session list on mobile, show via hamburger menu overlay.

### UI Design

**Desktop (Current)**:
```
┌──────────┬─────────────────┐
│ Sessions │ Main Content    │
│  List    │                 │
│  280px   │                 │
└──────────┴─────────────────┘
```

**Mobile (<768px)**:
```
┌─────────────────────────────┐
│ ☰  Multi-modal AI Studio    │ ← Hamburger menu
├─────────────────────────────┤
│ Main Content (Full Width)   │
│                             │
│                             │
└─────────────────────────────┘

When hamburger clicked:
┌─────────────────────────────┐
│ Sessions │ Main (dimmed)    │ ← Overlay
│  List    │                  │
│ [X]      │                  │
└─────────────────────────────┘
```

### Implementation Plan

**HTML** (add hamburger button to header):
```html
<header class="header">
    <div class="header-left">
        <button id="session-menu-toggle" class="icon-btn hamburger-btn">☰</button>
        <h1>🎙️ Multi-modal AI Studio</h1>
    </div>
    <!-- rest of header -->
</header>
```

**CSS** (add responsive behavior):
```css
/* Hamburger menu button - hidden on desktop */
.hamburger-btn {
    display: none;
}

/* Mobile layout */
@media (max-width: 768px) {
    .hamburger-btn {
        display: block;
    }
    
    .session-list {
        position: fixed;
        left: -280px; /* Hidden off-screen */
        top: 64px;
        bottom: 0;
        width: 280px;
        z-index: 1000;
        transition: left 0.3s ease;
        box-shadow: 2px 0 8px rgba(0,0,0,0.3);
    }
    
    .session-list.open {
        left: 0; /* Slide in */
    }
    
    /* Backdrop overlay */
    .session-list-backdrop {
        display: none;
        position: fixed;
        top: 64px;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        z-index: 999;
    }
    
    .session-list-backdrop.visible {
        display: block;
    }
    
    .container {
        grid-template-columns: 1fr; /* Full width main panel */
    }
}
```

**JavaScript** (add mobile menu toggle):
```javascript
// Mobile session list toggle
const sessionMenuToggle = document.getElementById('session-menu-toggle');
const sessionList = document.querySelector('.session-list');

// Create backdrop
const backdrop = document.createElement('div');
backdrop.className = 'session-list-backdrop';
document.body.appendChild(backdrop);

function toggleSessionList() {
    sessionList.classList.toggle('open');
    backdrop.classList.toggle('visible');
}

sessionMenuToggle.addEventListener('click', toggleSessionList);
backdrop.addEventListener('click', toggleSessionList);
```

**Benefits**:
- Standard mobile UX pattern (hamburger menu)
- Session list slides in as overlay
- Main content remains full width
- Backdrop dims main content, clear interaction

---

## 3. Combined Collapsed State (Future)

### Desktop - All Collapsed
```
┌──────────┬──┬──────────────────┐
│ Sessions │║ │ Chat (Full)      │ ← Max space
│  List    │║ │                  │
│          │║ │                  │
├──────────┴─┴──────────────────┤
│ Timeline ▲  TTL:285ms ...     │ ← Minimized
└─────────────────────────────────┘
```

### Mobile - Session Hidden, Timeline Collapsed
```
┌─────────────────────────────┐
│ ☰  Multi-modal AI Studio    │
├─────────────────────────────┤
│ Chat (Full Width & Height)  │
│                             │
│                             │
├─────────────────────────────┤
│ Timeline ▲  TTL:285ms       │
└─────────────────────────────┘
```

---

## 4. State Management

### JavaScript State Object (extend current):
```javascript
const state = {
    // Existing
    sessions: [],
    selectedSession: null,
    isLiveSession: false,
    sessionState: 'setup',
    
    // Add UI state
    ui: {
        configPanelCollapsed: false,
        timelinePanelCollapsed: false,
        sessionListVisible: true, // For mobile
    }
};
```

### Persist UI State (localStorage):
```javascript
// Save user's panel preferences
function saveUIState() {
    localStorage.setItem('ui-state', JSON.stringify(state.ui));
}

function loadUIState() {
    const saved = localStorage.getItem('ui-state');
    if (saved) {
        state.ui = { ...state.ui, ...JSON.parse(saved) };
        applyUIState();
    }
}

function applyUIState() {
    if (state.ui.configPanelCollapsed) {
        toggleConfigPanel(true);
    }
    if (state.ui.timelinePanelCollapsed) {
        toggleTimelinePanel(true);
    }
}
```

---

## 5. Responsive Breakpoints

### Defined Breakpoints
```css
/* Desktop: Full layout */
@media (min-width: 1025px) {
    /* Session list: 280px */
    /* Config: 50% */
    /* Chat: 50% */
}

/* Tablet: Hide config by default */
@media (max-width: 1024px) and (min-width: 769px) {
    /* Session list: 200px */
    /* Config: collapsible */
    /* Chat: more space */
}

/* Mobile: Hamburger menu */
@media (max-width: 768px) {
    /* Session list: overlay */
    /* Chat: full width */
    /* Timeline: collapsed by default */
}

/* Small mobile: Minimal */
@media (max-width: 480px) {
    /* Even more aggressive space saving */
}
```

---

## 6. Keyboard Shortcuts (Future)

Allow power users to toggle panels quickly:

```javascript
// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+B - Toggle session list (like VS Code sidebar)
    if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        toggleSessionList();
    }
    
    // Ctrl+\ - Toggle config panel
    if (e.ctrlKey && e.key === '\\') {
        e.preventDefault();
        document.querySelector('.config-header').click();
    }
    
    // Ctrl+J - Toggle timeline panel
    if (e.ctrlKey && e.key === 'j') {
        e.preventDefault();
        document.getElementById('timeline-collapse-btn').click();
    }
});
```

---

## 7. Animation Considerations

### Smooth Transitions
All panel collapses should animate smoothly:

```css
/* Config panel - already implemented */
.config-panel {
    transition: opacity 0.3s ease, transform 0.3s ease;
}

/* Timeline panel - to implement */
.timeline-panel {
    transition: height 0.3s ease;
}

/* Session list mobile - to implement */
.session-list {
    transition: left 0.3s ease;
}
```

### Performance
- Use `transform` instead of `left/right` when possible (GPU accelerated)
- Use `will-change` hint for frequently toggled elements
- Avoid layout thrashing (batch DOM reads/writes)

---

## 8. Accessibility

### ARIA Labels
```html
<button id="timeline-collapse-btn" 
        aria-label="Collapse timeline panel"
        aria-expanded="true">
    ▼
</button>

<aside class="session-list" 
       aria-label="Session history"
       role="navigation">
    <!-- sessions -->
</aside>
```

### Focus Management
When panels open/close, manage focus appropriately:
- Opening session list: focus first session
- Collapsing timeline: focus collapse button
- Closing mobile menu: return focus to hamburger

---

## 9. Implementation Priority

### Phase 1 (Current) ✅
- Config panel collapse with bar indicator
- Historical session preview
- Basic responsive grid

### Phase 2 (Next) 🔜
- Timeline collapse functionality
- Mobile hamburger menu for sessions
- UI state persistence

### Phase 3 (Polish) 💎
- Keyboard shortcuts
- Advanced animations
- Touch gestures for mobile
- Accessibility audit

---

## 10. Testing Checklist

Before implementing each collapsible feature:

- [ ] Panel collapse/expand smoothly
- [ ] No content jumps or reflows
- [ ] Other panels maintain position/size
- [ ] Keyboard navigation works
- [ ] Touch interactions work (mobile)
- [ ] State persists across page reloads
- [ ] Works in all supported browsers
- [ ] Meets WCAG accessibility standards
- [ ] Performance is smooth (60fps animations)

---

## Current Flexibility Assessment

### Already Flexible ✅
1. **Grid-based layout** - Easy to add/remove columns
2. **Main-panel wrapper** - Clean separation of concerns
3. **Class-based toggling** - No inline styles
4. **Transition-ready** - CSS animations prepared

### Needs Enhancement 🔧
1. **Timeline structure** - Add collapse button and mini metrics
2. **Mobile breakpoints** - Refine for phone sizes
3. **State management** - Track all panel states
4. **Backdrop element** - For mobile overlays

### Ready for Implementation 🚀
The current structure is well-designed for these future enhancements:
- Clear component boundaries
- Flexible grid system
- Animation-ready CSS
- Modular JavaScript

---

**Recommendation**: The current design is solid! When ready to implement timeline collapse or mobile menu, the changes will be localized and non-breaking. No major refactoring needed.
