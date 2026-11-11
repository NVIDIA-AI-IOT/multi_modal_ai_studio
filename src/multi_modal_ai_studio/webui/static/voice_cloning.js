// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Reference audio stays in the server-local library, never in session config.
let referenceVoices = [];
let designedVoicePreview = null;
let ttsPreviewText = '';

function voiceDesignFingerprint(config) {
    return JSON.stringify([config.request_profile, config.api_base, config.model,
        config.language, config.instructions, config.sample_rate, config.speed]);
}

function rememberDesignedVoicePreview(config, blob, encodedText) {
    if (config.request_profile !== 'qwen3-voicedesign' || !encodedText) return;
    designedVoicePreview = {blob, text: decodeURIComponent(encodedText),
        fingerprint: voiceDesignFingerprint(config)};
    const button = document.getElementById('save-designed-voice');
    if (button) button.disabled = false;
}

function renderReferenceVoiceControls(config, readonly) {
    const clone = config.request_profile === 'qwen3-base';
    const design = config.request_profile === 'qwen3-voicedesign';
    if (readonly) return clone ? `<div class="form-group compact-config-field--wide"><label>Reference voice</label>
        <span class="input-hint">${escapeHtml(config.reference_voice_id || 'Not selected')} · ${escapeHtml(config.reference_voice_mode || 'icl')}. Audio is stored separately on this server.</span></div>` : '';
    const preview = `<div class="form-group compact-config-field--wide"><label for="tts-preview-text">Preview text (optional)</label>
        <input type="text" id="tts-preview-text" maxlength="500" value="${escapeHtml(ttsPreviewText)}" placeholder="Leave blank for a short sample in the selected language"
            oninput="ttsPreviewText = this.value">
        <span class="input-hint">Use the play button beside Voice. This text is only for the preview, not the LLM prompt.</span></div>`;
    if (!clone && !design) return preview;
    const validPreview = designedVoicePreview && designedVoicePreview.fingerprint === voiceDesignFingerprint(config);
    return preview + (clone ? `<div class="form-group compact-config-field--wide">
        <label for="reference-voice-mode">Cloning mode</label>
        <select id="reference-voice-mode" onchange="stopTTSVoicePreview(); updateConfig('tts', 'reference_voice_mode', this.value)">
            <option value="icl" ${config.reference_voice_mode !== 'x-vector' ? 'selected' : ''}>Audio + exact transcript (ICL)</option>
            <option value="x-vector" ${config.reference_voice_mode === 'x-vector' ? 'selected' : ''}>Voice features only (no transcript)</option>
        </select>
        <span class="input-hint">ICL also uses the reference delivery; voice-features-only may sound less expressive. Base does not use Voice instructions.</span>
        <button type="button" class="btn-secondary" onclick="loadReferenceVoices()">Reload saved voices</button>
        <button type="button" id="delete-reference-voice" class="btn-secondary" onclick="deleteReferenceVoice()" ${config.reference_voice_id ? '' : 'disabled'}>Delete selected reference…</button>
        <audio id="reference-voice-audio" controls preload="none" style="display:none; max-width:100%; margin-top:8px" aria-label="Saved reference recording"></audio>
        </div>` : '') + `<div class="form-group compact-config-field--wide">
        <details><summary>${design ? 'Save this VoiceDesign preview for cloning' : 'Add a reference voice'}</summary>
        <label for="reference-voice-name">Reference name</label>
        <input type="text" id="reference-voice-name" maxlength="80" placeholder="My robot voice">
        ${clone ? `<label for="reference-voice-file">Reference WAV</label>
        <input id="reference-voice-file" type="file" accept=".wav,audio/wav">
        <span class="input-hint">1–30 seconds, PCM16 mono, 8–48 kHz, maximum 3 MiB. Use a clean single-speaker recording.</span>
        <label for="reference-voice-transcript">Exact reference transcript (required for ICL)</label>
        <textarea id="reference-voice-transcript" rows="2" maxlength="2000" placeholder="The words spoken in the reference, not new text to synthesize"></textarea>` :
        '<span class="input-hint">Generate a preview first. Its audio and spoken text will be saved, then select the Voice Clone preset and this reference. No training is needed.</span>'}
        <label><input type="checkbox" id="reference-voice-consent"> I have permission to use this voice for synthesis.</label>
        <button type="button" class="btn-secondary" ${design ? 'id="save-designed-voice"' : ''}
            ${design && !validPreview ? 'disabled' : ''} onclick="saveReferenceVoice(${design}, this)">${design ? 'Save generated preview' : 'Upload and select reference'}</button>
        <span class="input-hint">Saved on this VAW server, accessible to its users. Sessions and presets contain only a reference ID; they do not embed the recording. Delete removes the saved reference, not past synthesized output.</span>
        </details><div id="reference-voice-status" class="input-hint" role="status" aria-live="polite"></div></div>`;
}

async function loadReferenceVoices() {
    const select = document.getElementById('tts-rest-voice-select');
    const config = currentConfig.tts;
    if (!select || state.sessionState !== 'setup' || config.request_profile !== 'qwen3-base') return;
    try {
        const response = await fetch(getApiBase() + '/api/tts/reference-voices');
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not load references');
        if (!select.isConnected || currentConfig.tts !== config || state.sessionState !== 'setup') return;
        referenceVoices = data.voices || [];
        const selected = config.reference_voice_id || '';
        select.innerHTML = '<option value="">Select a saved reference…</option>' + referenceVoices.map(v =>
            `<option value="${escapeHtml(v.id)}" ${v.id === selected ? 'selected' : ''}>${escapeHtml(v.name)} · ${Number(v.duration).toFixed(1)} s${v.transcript ? '' : ' · no transcript'}</option>`).join('');
        if (selected && !referenceVoices.some(v => v.id === selected)) {
            select.innerHTML += `<option value="${escapeHtml(selected)}" selected>Reference unavailable on this server</option>`;
        }
        select.disabled = false;
        const hint = document.getElementById('tts-rest-metadata-hint');
        if (hint) hint.textContent = 'Server-local reference recordings for Qwen3-TTS Base. No named CustomVoice speaker is sent.';
        updateReferenceAudio(selected);
    } catch (error) {
        if (select.isConnected && currentConfig.tts === config && state.sessionState === 'setup') referenceVoiceStatus(error.message);
    }
}

function updateReferenceAudio(id) {
    const audio = document.getElementById('reference-voice-audio');
    if (!audio) return;
    audio.pause();
    if (referenceVoices.some(v => v.id === id)) {
        audio.src = getApiBase() + '/api/tts/reference-voices/' + encodeURIComponent(id) + '/audio';
        audio.style.display = 'block';
    } else { audio.removeAttribute('src'); audio.style.display = 'none'; }
}

function selectReferenceVoice(id) {
    if (state.sessionState !== 'setup') return;
    stopTTSVoicePreview();
    const voice = referenceVoices.find(v => v.id === id);
    currentConfig.tts.voice = voice ? voice.name : '';
    updateConfig('tts', 'reference_voice_id', id);
    updateReferenceAudio(id);
}

function referenceVoiceStatus(message) {
    const status = document.getElementById('reference-voice-status');
    if (status) status.textContent = message;
}

async function saveReferenceVoice(fromDesign, button) {
    if (state.sessionState !== 'setup') return;
    const config = currentConfig.tts;
    const originSelect = document.getElementById('tts-rest-voice-select');
    try {
        const name = document.getElementById('reference-voice-name').value.trim();
        const consent = document.getElementById('reference-voice-consent').checked;
        if (!name || !consent) throw new Error('Enter a name and confirm permission to use this voice.');
        const design = fromDesign ? designedVoicePreview : null;
        if (fromDesign && (!design || design.fingerprint !== voiceDesignFingerprint(config))) {
            throw new Error('Voice settings changed. Generate a new VoiceDesign preview before saving.');
        }
        const audio = fromDesign ? design.blob : document.getElementById('reference-voice-file').files[0];
        if (!audio) throw new Error('Select a WAV recording.');
        if (audio.size > 3 * 1024 * 1024) throw new Error('Reference WAV must be at most 3 MiB.');
        const transcript = fromDesign ? design.text : document.getElementById('reference-voice-transcript').value;
        const form = new FormData();
        form.append('audio', audio, 'reference.wav');
        form.append('name', name);
        form.append('transcript', transcript);
        form.append('consent', 'true');
        form.append('source', fromDesign ? 'voicedesign-preview' : 'upload');
        button.disabled = true;
        referenceVoiceStatus('Saving reference…');
        const response = await fetch(getApiBase() + '/api/tts/reference-voices', {method: 'POST', body: form});
        const entry = await response.json();
        if (!response.ok) throw new Error(entry.error || 'Could not save reference');
        if (!originSelect.isConnected || currentConfig.tts !== config || state.sessionState !== 'setup') return;
        if (!fromDesign) {
            config.voice = entry.name;
            if (!entry.transcript) config.reference_voice_mode = 'x-vector';
            updateConfig('tts', 'reference_voice_id', entry.id);
            await loadReferenceVoices();
        }
        referenceVoiceStatus('Saved “' + entry.name + '”.' + (fromDesign ? ' Select Qwen3-TTS — Voice Clone, then choose this reference.' : ' Use the play button beside Voice to test new text.'));
    } catch (error) {
        if (originSelect?.isConnected && currentConfig.tts === config && state.sessionState === 'setup') referenceVoiceStatus(error.message);
    }
    finally { if (button.isConnected) button.disabled = false; }
}

async function deleteReferenceVoice() {
    if (state.sessionState !== 'setup') return;
    const config = currentConfig.tts;
    const id = config.reference_voice_id;
    if (!id || !window.confirm('Delete this saved reference recording? Presets using it will need another reference.')) return;
    try {
        stopTTSVoicePreview();
        const response = await fetch(getApiBase() + '/api/tts/reference-voices/' + encodeURIComponent(id), {method: 'DELETE'});
        if (!response.ok) throw new Error((await response.json()).error || 'Delete failed');
        if (currentConfig.tts !== config || state.sessionState !== 'setup') return;
        config.voice = '';
        updateConfig('tts', 'reference_voice_id', '');
        await loadReferenceVoices();
        referenceVoiceStatus('Reference deleted.');
    } catch (error) {
        if (currentConfig.tts === config && state.sessionState === 'setup') referenceVoiceStatus(error.message);
    }
}
