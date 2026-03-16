# TODO: Frontend Config Defaults Refactor

**Status**: Planned  
**Priority**: Medium  
**Relates to**: Preset management, `schema.py`, `app.js`

## Background

Today the frontend (`app.js`) contains a hardcoded `defaultConfig` object (~70 lines)
that duplicates the defaults defined in `schema.py`.  When a new config field is added
to the schema, the JS object must also be updated — a maintenance hazard.

As of v0.1.0, a new **`GET /api/config/defaults`** endpoint was added to `server.py`
(Piece 1 below).  It returns a full `SessionConfig()` with all schema defaults as JSON,
establishing `schema.py` as the single source of truth.

The remaining work is to make the frontend actually use that endpoint and to produce
cleaner preset YAML files when saving from the UI.

## Pieces

### Piece 1 — `/api/config/defaults` endpoint (DONE)

Added in `server.py`.  Returns `dataclasses.asdict(SessionConfig())`.

### Piece 2 — Frontend fetches defaults from server

Replace the hardcoded `defaultConfig` in `app.js` with a dynamic fetch:

1. Change `var defaultConfig = { ... }` to `var defaultConfig = {}`.
2. On page load, `fetch('/api/config/defaults')`, normalize the response through
   `_normalizePresetToFrontend()`, and assign into `defaultConfig`.
3. Gate `renderConfig()` and the rest of init on this fetch completing.

**Key concerns:**
- **Key-name mismatch**: The backend schema uses `scheme`/`server` while the frontend
  uses `backend`/`riva_server`.  The existing `_normalizePresetToFrontend()` handles
  most of this, but needs an audit to confirm every UI-rendered field survives the
  normalization.
- **Frontend-only keys**: Fields like `camera` (alias of `video_source`) exist only
  on the frontend.  The normalizer must continue synthesising these.
- **Fallback**: If the fetch fails or is slow, the UI currently has no defaults.
  Consider keeping a minimal inline fallback or showing a loading state.

### Piece 3 — Sparse preset save

When saving via the `[+]` button, diff `currentConfig` against `defaultConfig` and
only write changed fields:

```javascript
function _sparseDiff(current, defaults) {
    var out = {};
    for (var k of Object.keys(current)) {
        if (typeof current[k] === 'object' && !Array.isArray(current[k]) && defaults[k]) {
            var sub = _sparseDiff(current[k], defaults[k]);
            if (Object.keys(sub).length) out[k] = sub;
        } else if (JSON.stringify(current[k]) !== JSON.stringify(defaults[k])) {
            out[k] = current[k];
        }
    }
    return out;
}
```

This produces compact, readable YAML that only contains intentional overrides.
New fields added in future schema versions automatically pick up their defaults.

**Note:** The existing hand-written presets (00–03, demo-local, demo-live, etc.) are
intentionally dense/explicit.  This is fine — having all values visible is easier for
developers to understand.  Sparse save is mainly for user-created presets via the UI.

## Files involved

| File | Change |
|---|---|
| `src/.../webui/server.py` | Piece 1 (done) |
| `src/.../webui/static/app.js` | Pieces 2 + 3 |
| `src/.../config/schema.py` | No change (already the source of truth) |
