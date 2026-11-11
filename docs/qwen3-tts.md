# Qwen3-TTS voice controls (0.3.0 development)

This feature is under development on `codex/qwen3-tts-voice-controls`. It is not part of the published Voice AI Workbench 0.2.0 release. The existing Speaches and NVIDIA Speech quick starts are unchanged.

## Choose a workflow

| Workflow | Model | Voice input | Status in the workbench |
| --- | --- | --- | --- |
| CustomVoice | Qwen3-TTS 1.7B CustomVoice | A built-in speaker plus optional delivery instructions | Preview, conversation, presets, and recorded configuration |
| VoiceDesign | Qwen3-TTS 1.7B VoiceDesign | A required description of the voice and delivery | Preview, conversation, presets, and recorded configuration; experimental |
| Voice Cloning | Qwen3-TTS 1.7B Base | Saved reference audio, optionally its exact transcript | WAV upload, preview, conversation, deletion, and reference IDs in presets/sessions; experimental |

CustomVoice instructions direct the delivery of a selected speaker; they do not replace the speaker's identity. VoiceDesign creates a voice from a description on each request, so identity may vary between requests or text chunks. Save a VoiceDesign preview as a reference and switch to Base cloning to reuse that recording. The workbench stores reference recordings, not trained models or cached voice embeddings. Reusing a reference does not guarantee identical delivery across generations.

Use 1.7B CustomVoice for instruction control. The upstream 0.6B CustomVoice model supports built-in speakers but not instructions; the workbench rejects a nonempty instruction for that model. Descriptions such as “warm, lightly textured, playful, and clearly articulated” are directions, not guarantees of a specific timbre.

## Server setup

The Python package is an API client: it does not install a Qwen model server or model weights. These profiles target **vLLM-Omni 0.26.0** and its `/v1/audio/speech` extensions. They do not require an additional workbench proxy.

The following commands reflect the Thor configuration used for development. They are not an Orin Nano memory-fit claim. Check available GPU memory and existing containers/ports before starting another model. Start one service if you only need one workflow; each model consumes GPU memory when run together. ASR and LLM remain independent services.

CustomVoice:

```bash
docker run -d --name vaw-qwen3-customvoice --restart unless-stopped \
  --runtime=nvidia --network host --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-omni@sha256:5cba1538c6f8ee81e8bea6708c24e68d7b2640f466a9fbf2ef15e68f2168b48b \
  vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --revision 0c0e3051f131929182e2c023b9537f8b1c68adfe \
  --omni --host 127.0.0.1 --port 18083 \
  --stage-overrides '{"0":{"gpu_memory_utilization":0.10,"max_num_seqs":1},"1":{"gpu_memory_utilization":0.10,"max_num_seqs":1}}'
```

VoiceDesign:

```bash
docker run -d --name vaw-qwen3-voicedesign --restart unless-stopped \
  --runtime=nvidia --network host --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-omni@sha256:5cba1538c6f8ee81e8bea6708c24e68d7b2640f466a9fbf2ef15e68f2168b48b \
  vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --revision 5ecdb67327fd37bb2e042aab12ff7391903235d3 \
  --omni --host 127.0.0.1 --port 18085 \
  --stage-overrides '{"0":{"gpu_memory_utilization":0.10,"max_num_seqs":1},"1":{"gpu_memory_utilization":0.10,"max_num_seqs":1}}'
```

Wait for model loading, then check the selected endpoint:

```bash
curl --fail http://127.0.0.1:18083/v1/models
# Or, for VoiceDesign:
curl --fail http://127.0.0.1:18085/v1/models
```

Voice Clone (a separate **Base** model service, not CustomVoice):

```bash
docker run -d --name vaw-qwen3-base --restart unless-stopped \
  --runtime=nvidia --network host --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-omni@sha256:5cba1538c6f8ee81e8bea6708c24e68d7b2640f466a9fbf2ef15e68f2168b48b \
  vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --revision fd4b254389122332181a7c3db7f27e918eec64e3 \
  --omni --host 127.0.0.1 --port 18086 \
  --stage-overrides '{"0":{"gpu_memory_utilization":0.10,"max_num_seqs":1},"1":{"gpu_memory_utilization":0.10,"max_num_seqs":1}}'
curl --fail http://127.0.0.1:18086/v1/models
```

## Thor runtime qualification

The Base and VoiceDesign paths were also exercised on a shared Thor using
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`, `enforce_eager: true`, and
`kv_cache_memory_bytes: 536870912` in each stage override. The explicit KV limit
avoids the runtime's much larger profiling fallback; eager mode avoids the
CUDA Graph initialization failure encountered with this allocator on that host.
VoiceDesign additionally used `max_num_batched_tokens: 2048` for both stages:
its otherwise 65,536-token decoder warmup synthesized 4,096 codec frames,
while the smaller batch warmed up 128 frames. These are qualified development
settings, not a memory-fit guarantee for another device or concurrent workload.

When using that profile, add `-e PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`
to the relevant Docker command and use this stage override:

```bash
--stage-overrides '{"0":{"gpu_memory_utilization":0.10,"kv_cache_memory_bytes":536870912,"max_num_batched_tokens":2048,"max_num_seqs":1,"enforce_eager":true},"1":{"gpu_memory_utilization":0.10,"kv_cache_memory_bytes":536870912,"max_num_batched_tokens":2048,"max_num_seqs":1,"enforce_eager":true}}'
```

The smaller batch was exercised with VoiceDesign; Base was exercised with the
explicit KV limit and eager mode at its default batch size. CustomVoice was
exercised with the original command above. Keep the host-specific settings
separate from model capabilities.

Confirm that `/v1/models` reports the **selected Qwen model**, not merely that a
service responds. The ports above are examples: update the corresponding
preset when using another free port, and verify its independent ASR and LLM
endpoints too. After editing a preset on disk, select it again in a new session;
an existing browser configuration does not update automatically.

## Workbench controls

1. Use this feature branch's source installation (not the published 0.2.0 wheel). Preset selection and saving are available only during new-session setup; recorded sessions are read-only.
2. Select **Qwen3-TTS — CustomVoice (Experimental)** or **Qwen3-TTS — VoiceDesign (Experimental)** from Preset. These presets expect Speaches ASR on port 18080 and Gemma's OpenAI-compatible LLM on port 8080. Adjust those independent endpoints for your system.
3. In **TTS → OpenAI REST API**, check the endpoint, model, and **Request profile**. Changing the profile does not change the endpoint or load a different server model.
4. For CustomVoice, choose a built-in speaker. For VoiceDesign, the Voice field is disabled because the description replaces the speaker selection.
5. Enter **Voice instructions** and press the voice preview button. Example CustomVoice direction: `Speak warmly and clearly, with a playful cartoon-like delivery. Keep the words easy to understand.` Example VoiceDesign description: `A friendly adult robot with a warm, lightly textured mid-range voice, clear English articulation, and a playful but calm delivery.`
6. Start a conversation. The same instruction is sent for each synthesized text chunk. Save a preset to reuse it; saved sessions retain the instruction and profile for read-only review. Instructions are plain text in those files, so do not include secrets.

The instruction is separate from spoken text and the LLM system prompt. You do **not** need the LLM to emit stage directions or emotion tags. Current workbench controls apply one instruction to the whole session; per-turn or per-sentence instructions are not implemented.

CustomVoice speakers: Aiden, Dylan, Eric, Ono_Anna, Ryan, Serena, Sohee, Uncle_Fu, and Vivian. The profiles offer English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, and Auto. Speaker/native-accent metadata does not limit the language selector. When a provider does not report metadata, these choices come from model capabilities rather than live discovery.

## Reference voice workflow

1. Select **Qwen3-TTS — Voice Clone (Experimental)**. Check the Base model endpoint on port 18086.
2. In the TTS tab, expand **Add a reference voice**. Give it a name, choose a clean single-speaker WAV, and confirm permission to use it. Uploads must be **PCM16 mono, 8–48 kHz, 1–30 seconds, at most 3 MiB**. MP3 and browser microphone recording are not supported in this increment.
3. For **Audio + exact transcript (ICL)**, enter the words spoken in that recording. They are not the new words to synthesize. Without a transcript, uploads select **Voice features only (no transcript)** automatically. This uses `x_vector_only_mode`; it may preserve less expressive delivery.
4. **Upload and select reference** saves it locally. The embedded audio player plays the original reference. The play button beside **Voice** synthesizes new words; use **Preview text** to specify them.
5. Start a conversation or save a preset. The same reference is sent for each TTS request. Base cloning does not accept the CustomVoice/VoiceDesign instructions field.

To reuse a designed voice, select **VoiceDesign**, generate a preview, then expand **Save this VoiceDesign preview for cloning**. Enter a name and confirm permission, then save. The generated WAV and the actual preview text are stored together. Switch to **Voice Clone** and select that name. Changing the design settings requires generating another preview before saving it. This is reference-conditioned synthesis, not fine-tuning.

References are stored under `$XDG_DATA_HOME/voice-ai-workbench/voices` (default `~/.local/share/voice-ai-workbench/voices`). Set `VAW_VOICE_LIBRARY_DIR` before starting the app to choose a different directory; use the same setting for CLI smoke tests. Files are private to the OS user (0600), with a maximum of 100 references. The UI lists names and durations; only opaque reference IDs, names, and mode are saved in presets and sessions, not the recording or its transcript. A copied preset needs the matching reference library on the destination server, or a newly selected reference.

This is a **trusted-network, shared-server library**, not a multi-user private vault: anyone with access to this VAW server can list, play, and delete references. Do not expose it publicly. Reference audio is forwarded to the configured TTS endpoint, so use a trusted endpoint. The API accepts bounded file uploads, never arbitrary file paths or download URLs. Browser cross-origin writes are rejected. **Delete selected reference** removes the saved reference, not previous generated output or backups; presets that used it require another reference. Use only recordings you have permission to use and label generated/cloned audio appropriately when sharing it.

## Streaming and limitations

- These profiles use **HTTP audio streaming**, not the WebSocket Realtime API. Requests include `stream: true`, `stream_format: "audio"`, and `response_format: "pcm"`. vLLM-Omni otherwise defaults its streaming response to SSE, which must not be played as PCM.
- The tested vLLM-Omni streaming path uses mono PCM16 at 24 kHz and `speed: 1.0`. The UI locks those fields for Qwen profiles. For CustomVoice and VoiceDesign, describe speaking pace in the instruction; this is not an exact numeric speed control. Base cloning has no instruction field.
- **Start speaking before LLM finishes** is a separate application-level text-chunking choice. The example presets disable it initially for prosody assessment. Audio can still stream from an individual TTS request before that request finishes.
- Long text is still split at the workbench's existing text boundaries. Short requests can finish before playback begins. VoiceDesign can change identity across chunks; it is not a stored speaker asset.
- Local cancellation closes the pending HTTP request and discards old-turn audio. It does not prove that the model server immediately stops all GPU work. Verify audible barge-in and recovery on the actual device.
- Generic OpenAI-compatible profiles send `instructions` only when nonempty. Leave it blank for providers that do not support that field. Riva and WebSocket TTS instruction support is not added by this change.

## Repeatable smoke test

Run from this checkout using its development environment:

```bash
python scripts/test_qwen3_tts.py \
  --preset presets/qwen3-tts-customvoice.yaml \
  --text "That sounds egg-citing! Shall we explore the park this weekend?" \
  --output-dir /tmp/vaw-qwen3-customvoice-smoke

python scripts/test_qwen3_tts.py \
  --preset presets/qwen3-tts-voicedesign.yaml \
  --text "Hello! I am ready to help you plan a little adventure." \
  --output-dir /tmp/vaw-qwen3-voicedesign-smoke

# Use the reference ID recorded in a saved Voice Clone preset (or GET
# /api/tts/reference-voices); the CLI must use the same voice library as the UI.
python scripts/test_qwen3_tts.py \
  --preset presets/qwen3-tts-voice-clone.yaml \
  --reference-voice-id YOUR_REFERENCE_ID --reference-voice-mode icl \
  --text "Hello! This is new speech using my saved reference." \
  --output-dir /tmp/vaw-qwen3-clone-smoke
```

The script uses the same backend as conversation and preview. It writes `speech.wav` and `metrics.json` and reports time to the first PCM chunk delivered by the client adapter, total request time, audio duration, and RTF (`generation seconds / audio seconds`; lower is faster). This is **not** acoustic time-to-first-audio at a physical speaker. Model caching, contention, and cold starts affect the result.

Before merging, test preview and a short conversation for each profile, interrupt a long reply, check the following reply is audible, save/reopen the session, and verify the recorded profile/instructions. Repeat a generic Speaches or NVIDIA Speech test with empty instructions to check the existing path.

## Development validation — November 7, 2026

Validated on Jetson AGX Thor with the pinned vLLM-Omni image above. The reference was a VoiceDesign-generated friendly robot voice, not a third-party recording.

- Automated tests: 291 Python unit tests passed, one skipped; 61 JavaScript tests passed.
- Browser/API checks: VoiceDesign preview and save, Base ICL and x-vector previews, WAV upload without a transcript, automatic x-vector selection, reference playback controls, deletion, and persistence after reload. No browser errors were observed.
- Real model cancellation: cancel an active Base synthesis request after audio arrives, then synthesize another reply using the same backend. The cancelled client request completed promptly and the following reply returned audio. This does not measure server-side GPU cancellation or confirm physical speaker playback.
- Generic Speaches/Kokoro synthesis still returned valid PCM with no Qwen-specific fields in its request.
- A local development wheel built successfully and included the voice library, UI script, and Voice Clone preset. It was not published.

One request per mode, using the same saved reference and text (`That sounds egg-citing! What would you like to build this weekend?`):

| Cloning mode | First client PCM | Total request time | Generated audio | RTF |
| --- | ---: | ---: | ---: | ---: |
| Audio + exact transcript (ICL) | 1.308 s | 2.100 s | 4.00 s | 0.525 |
| Voice features only | 1.171 s | 2.088 s | 4.32 s | 0.483 |

These single-run smoke measurements are not a comparative benchmark. Physical listening, conversational barge-in, and saved-session review with the new reference voice remain manual checks before merging.

## Follow-up for 0.3.0

- Reference microphone recording, more upload formats, and explicit library export/import.
- Optional server-side embedding reuse, subject to provider support and privacy controls.
- Explicit provider capabilities instead of expanding implicit model-name heuristics.
- Optional per-turn directions, with strict separation between direction and spoken text.
- Quality/latency evaluation on target hardware; do not infer Orin support from Thor tests.

References: [Qwen3-TTS upstream](https://github.com/QwenLM/Qwen3-TTS), [vLLM-Omni](https://github.com/vllm-project/vllm-omni).
