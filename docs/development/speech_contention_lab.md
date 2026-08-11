# Speech Contention Lab

`benchmarks/speech_contention_lab.py` is a controlled, headless benchmark for
distinguishing TTS/LLM resource contention from browser and conversational
pipeline behavior.

It runs the same inputs in three scenarios:

1. `tts-only`
2. `llm-only`
3. `overlap` — LLM starts first; fixed TTS begins when the same MMAS
   `TTSChunkBuffer` used by the application emits its first text chunk

The TTS synthesis text remains fixed. Its start is triggered from the streamed
LLM output after the configured number of speech units (10 by default, with an
earlier natural phrase break allowed in the same way as MMAS). This preserves a
repeatable synthesis load while reproducing MMAS's LLM-prefill → token-buffer →
TTS ordering. Use `--overlap-start-mode simultaneous` only for a deliberate
worst-case stress test.

## OpenAI-compatible Magpie and vLLM

```bash
PYTHONPATH=src python3 benchmarks/speech_contention_lab.py run \
  --label no-mig \
  --warmup \
  --repeats 3 \
  --llm-api-base http://localhost:8000/v1 \
  --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --tts-backend openai-rest \
  --tts-api-base http://localhost:8082/v1 \
  --tts-model nvidia/magpie_tts_multilingual_357m \
  --tts-voice Sofia
```

The MMAS-like trigger can be adjusted explicitly:

```bash
  --overlap-start-mode mmas-units \
  --overlap-trigger-units 10
```

For controlled start-threshold experiments, `strict-units` disables MMAS's
early phrase-boundary flush and waits until the requested number of units has
actually arrived. `after-llm` provides the non-overlap endpoint:

```bash
  --overlap-start-mode strict-units --overlap-trigger-units 30
  --overlap-start-mode after-llm
```

For a comparison set, use labels with the same ordered axes and assign an
explicit display order, for example:

```text
riva-grpc-magpie-thor-r38.4-short-1x
openai-rest-magpie-thor-r38.4-short-1x
riva-grpc-magpie-thor-r38.4-standard-1x
```

Use `--display-order` and `--comparison-role` to make the baseline and each
single-axis variation explicit in the browser matrix.

## Riva Magpie and an OpenAI-compatible LLM

Use the exact voice name reported by the Riva deployment:

```bash
PYTHONPATH=src python3 benchmarks/speech_contention_lab.py run \
  --label riva-no-mig \
  --warmup \
  --repeats 3 \
  --llm-api-base http://localhost:8000/v1 \
  --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --tts-backend riva \
  --tts-server localhost:50051 \
  --tts-voice Magpie-Multilingual.EN-US.Sofia \
  --tts-sample-rate 22050
```

## Browser report

Start the report server after one or more benchmark runs:

```bash
PYTHONPATH=src python3 benchmarks/speech_contention_lab.py serve \
  --results-dir ./benchmark-results/speech-contention \
  --host 0.0.0.0 \
  --port 8097
```

Open `http://JETSON-IP:8097`. The report displays:

- a linked experiment matrix that highlights controlled and changed conditions
- an MMAS-style overlap timeline for LLM, TTS, and simulated AI playout
- LLM TTFT and tokens per second
- TTS TTFA, RTF, chunk count, and delivery timing
- sampled GPU utilization
- the clean generated waveform and audio
- a simulated live-playout waveform/audio with measured delivery gaps inserted
- estimated playback buffer underruns

The generated audio is always preserved separately from the simulated playout.
If clean audio is continuous but simulated playout is broken, the likely
problem is delivery cadence or playback buffering. If clean audio is also
broken, inspect TTS inference and text/audio chunk joining.

## MIG comparison

MIG configuration and process placement are intentionally kept outside this
tool. Configure the LLM and TTS services on their target MIG device UUIDs, then
run the same command again with a different label:

```bash
PYTHONPATH=src python3 benchmarks/speech_contention_lab.py run \
  --label mig-llm-2g-tts-1g \
  --warmup \
  --repeats 3
```

Serve the common result parent directory. Both runs appear in the same report.
Keep prompts, text, models, clocks/power mode, repetitions, and the simulated
playout prebuffer unchanged.

This controlled benchmark tests resource contention. A final MMAS browser
conversation remains necessary to validate WebSocket scheduling, AudioContext
behavior, sentence segmentation, and barge-in.

Run the commands from an installed MMAS development environment. On a machine
where the MMAS dependencies were installed into a target directory rather than
the system Python, include both paths, for example:

```bash
PYTHONPATH=src:/home/jetson/.local/share/mmas-runtime \
  python3 benchmarks/speech_contention_lab.py --help
```
