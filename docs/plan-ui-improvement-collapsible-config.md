# UI improvement: collapsible config and session layout

**Branch:** `ui-improvement-collapsible-config` (shorter alternative: `ui-collapsible-config`).  
**Merge path:** After PR from `merge/dev-vlm` is merged to main, rebase or merge main into this branch, then open a new PR to main.

**Reference:** Live VLM WebUI layout/overlays (full-screen overlay, transparent balloons, mirror, etc.).

---

## Phase 1: Config lock and auto-hide (priority)

### 1.1 Config pane inactive once session starts
- **Goal:** Once the session starts, the Configuration pane is read-only and visually inactive (grayed out); no config changes during or after the session.
- **Files:** `app.js` (set disabled/readonly on all config inputs when session goes live), `styles.css` (`.config-panel--locked` or reuse readonly styling).

### 1.2 Replace "Save as default config" with "Auto hide once session start <<"
- **Goal:** Add a button "Auto hide once session start <<" that toggles a preference. When on, the Configuration pane collapses once the session starts.
- **Files:** `index.html`, `app.js` (toggle state, localStorage; on session start, collapse if enabled), `styles.css`.

### 1.3 Collapsible Configuration pane when in session
- **Goal:** When in session, the left Configuration pane is collapsible. When collapsed:
  - **chat-panel** takes the full width of the content area (excluding session-list).
  - **session-meta-container** full width at top.
  - **pipeline-config-container** and **session-image-container** on the left; chat/conversation area on the right.
- **Layout (collapsed):**  
  `[session-list] | [session-meta full width] [pipeline-config + session-image (left)] [chat balloons (right)]`
- **Files:** `index.html`, `styles.css`, `app.js`.

---

## Phase 2: Session controls and video overlay (can be split)

### 2.1 Stop button; Start morphs to Stop
- **Goal:** When session starts, Start button is replaced by a **Stop** button. One clear Stop control during the session. Move the STOP control onto the video container (remove the big red STOP button from its current spot).

### 2.2 Full-screen / shrink overlay and Mirror
- **Goal:** Full-screen overlay toggle; Mirror button for horizontal flip of the video feed.

### 2.3 Full-screen overlay content (align with Live VLM WebUI)
- **Goal:** In full-screen mode: AI balloon at top, user balloon at bottom, transparent; overlay focused on video.

---

## Implementation order (suggested)

1. **1.1** Config pane inactive/grayed once session starts.  
2. **1.2** "Auto hide once session start <<" button and preference; on start, collapse config if enabled.  
3. **1.3** Collapsible config pane + chat-panel layout when collapsed.  
4. **2.1** Stop button; Start → Stop morph.  
5. **2.2** Full-screen/shrink and Mirror.  
6. **2.3** Full-screen overlay content.

---

## Key DOM / state references

- **Config panel:** `#config-panel`, `.config-panel--editable`, `.config-header`, `#config-content`, `#config-tab-content`.
- **Session state:** `state.sessionState` (`'setup'` | `'live'` | `'stopped'`), `state.isLiveSession`.
- **Layout:** `.content-area` → `.config-panel` + `.chat-panel`. Inside `.chat-panel`: `.session-meta-container`, `.pipeline-config-container`, `.session-image-container`, chat content.
- **Start/Stop:** `#start-session-btn`, `#start-session-overlay`; stop flow exists.

---

## PR strategy

- **Now:** Branch is from `merge/dev-vlm`. Do not merge until the current merge/dev-vlm → main PR is merged.
- **After that merge:** Update this branch from main, then open a new PR: `ui-improvement-collapsible-config` → `main`.
