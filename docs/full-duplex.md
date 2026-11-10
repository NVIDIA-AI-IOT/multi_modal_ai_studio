# Full-duplex speech sessions

Full speech-to-speech WebSocket sessions have their own recording view. A
continuous model stream does not create the ASR → LLM → TTS turns used by a
cascade pipeline.

## Nemotron VoiceChat

Run a NeMo-Speech.cpp VoiceChat server separately, then load
`presets/nemotron-voicechat-thor.yaml`. The preset expects the server at
`ws://localhost:18088/v1/realtime`; edit this URL for your deployment. Select
**Nemotron VoiceChat (full duplex)** as the API wire format. Browser microphone
and speaker are the preset defaults. VoiceChat handles interruption through
incoming speech; keep the microphone unmuted for natural interruption.

For an isolated local trial, run `scripts/run_voicechat_trial.sh` with the
project dependencies installed. `VAW_PYTHON` selects the Python executable;
`VAW_TRIAL_PORT` defaults to 8110. The launcher listens on loopback by default.

The native adapter sends paced 80 ms PCM frames, including silence while capture
is muted or idle. It bounds queued capture to 400 ms and drains the provider's
final transcripts on stop. Native response IDs identify a stream, so separate
utterance IDs track individual assistant statements.

## Conversation and timeline

Recordings reconstruct user messages from final input transcripts and assistant
messages from output transcripts. Greetings, short acknowledgements and repeated
answers remain separate utterances. Older recordings are adapted when read;
their original files are not rewritten.

The AI speech lane shows one latest transcript marker per utterance, replacing
partial snapshots with the final transcript. Labels are shortened to available
space and move to the left at the viewport's right edge; hover reveals the complete text.
Vertical Cyan → Purple input bars and Purple → Magenta output bars identify the
end-to-end path. The LLM lane identifies the integrated
model and has no separate LLM timing. CPU/GPU utilization is sampled during
Realtime sessions and saved for replay. These are host utilization measurements,
including other workloads. Older recordings without samples display a missing
data message.

## Independent metrics

- **User / AI utterances:** transcript counts, independent of cascade turns.
- **FD-TTL estimate:** first associated assistant audio onset minus input speech
  end. Negative values indicate overlap.
- **First reply estimate:** that onset minus input speech start.
- **FD-TTL samples:** aligned replies / user utterances, exposing coverage.
- **Duration:** session stop time, or last observed event for older recordings.

For the native adapter, input boundaries use an energy observer on the PCM sent
to the model, at 20 ms resolution. Silence confirmation time is excluded from
the boundary timestamp. Multiple regions do not automatically suppress a TTL.
One completed region of at least 400 ms can supply a dominant-region endpoint
estimate if it is at least three times the next longest region. Relative
dominance avoids a hard cutoff on secondary-region duration. This is an explicit energy heuristic, not
proof that shorter regions are noise. Use that region's end instead of moving
the endpoint to a later short burst, and retain later ends as a latency range.
The timeline marks these estimates with ≈; hover shows the endpoint, scheduled
audio onset, method and range. Estimates contribute to the FD-TTL average.
Completed input regions are assigned by their end timestamp, not their start.
A region continuing beyond a final transcript stays available for the next
transcript instead of losing that next question's endpoint. If only an ongoing
region is available, its endpoint remains missing. Comparable completed regions
remain ambiguous and are excluded. Energy alone cannot distinguish within-phrase
pauses, background sounds or microphone echo. The UI shows an uncertain-endpoint count.
This observer is for measurement, not model turn control. The
browser reports the scheduled time of the first active output sample only when
its audio context is running and output is unmuted. This is a browser scheduling
estimate, not a measurement at the physical speaker.

Reply association is temporal, not semantic ground truth. The first transcript
candidate after an input starts is retained, including acknowledgements such as
“Sure.” An utterance is associated with a new acoustic onset only when there is
one pending candidate. Continuous overlapping output without an identifiable
boundary remains unaligned. An unaligned acknowledgement is never replaced by a
later substantive answer to obtain a latency value.

Acoustic regions have IDs independent of transcript utterance IDs. The browser
records an unmuted region's scheduled onset immediately, even if its transcript
has not arrived. An explicit one-to-one link joins that region to its text
only when a single pending pair exists. Multiple pending regions or unresolved
text inside an already-owned continuous region remain unaligned; later text is
not used to hide that ambiguity. Old recordings lacking region onset events
cannot recover this association from subtitles or amplitude envelopes alone.

Missing input or output alignment displays **Not aligned** and contributes no
latency sample. Historical text events and amplitude envelopes alone cannot
establish exact word-to-audio timing. Full speech-to-speech providers without
the native measurement events still show transcripts and utilization, but
latency remains unavailable. The cascade metrics retain their existing path.
