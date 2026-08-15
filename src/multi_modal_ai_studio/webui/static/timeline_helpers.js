// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(function (root, factory) {
    const helpers = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = helpers;
    }
    root.MMASTimelineHelpers = helpers;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    // Build coarse TTS playback segments for sessions recorded before dense
    // 25 ms TTS RMS samples were persisted. Playback starts at first audio,
    // not at synthesis start.
    function buildTtsSegmentsFromTimeline(timeline) {
        if (!timeline || !timeline.length) return [];
        const starts = timeline.filter(function (e) { return e.event_type === 'tts_start'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        const firstAudios = timeline.filter(function (e) { return e.event_type === 'tts_first_audio'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        const completes = timeline.filter(function (e) { return e.event_type === 'tts_complete'; }).sort(function (a, b) { return (a.timestamp || 0) - (b.timestamp || 0); });
        if (!starts.length) return [];
        return starts.map(function (s, i) {
            const synthesisStart = s.timestamp != null ? Number(s.timestamp) : 0;
            const nextSynthesisStart = starts[i + 1] && starts[i + 1].timestamp != null ? Number(starts[i + 1].timestamp) : Infinity;
            const firstAudio = firstAudios.find(function (e) {
                const t = e.timestamp != null ? Number(e.timestamp) : NaN;
                return !isNaN(t) && t >= synthesisStart && t < nextSynthesisStart;
            });
            const endEvent = completes.find(function (e) {
                const t = e.timestamp != null ? Number(e.timestamp) : NaN;
                return !isNaN(t) && t >= synthesisStart && t < nextSynthesisStart;
            });
            const startTime = firstAudio && firstAudio.timestamp != null ? Number(firstAudio.timestamp) : synthesisStart;
            const endTime = endEvent && endEvent.timestamp != null ? Math.max(startTime + 0.025, Number(endEvent.timestamp)) : startTime + 0.1;
            return { startTime: startTime, endTime: endTime, amplitude: 50 };
        });
    }

    // A legacy OpenAI-REST recording stored one RMS value per multi-second
    // HTTP chunk. Treat that as sparse; otherwise interpolation draws only
    // isolated purple pixels.
    function hasDenseTtsAmplitudeTimeline(timeline) {
        if (!timeline || !timeline.length) return false;
        const samples = timeline.filter(function (e) {
            return e.event_type === 'audio_amplitude' && (e.source === 'tts' || e.source === 'ai');
        }).map(function (e) {
            return e.timestamp != null ? Number(e.timestamp) : NaN;
        }).filter(function (t) {
            return !isNaN(t);
        }).sort(function (a, b) {
            return a - b;
        });
        for (let i = 1; i < samples.length; i++) {
            if (samples[i] - samples[i - 1] <= 0.1) return true;
        }
        return false;
    }

    // Reconstruct browser-style continuous playout for recordings created
    // before client-timed TTS segments were persisted. Server timeline
    // amplitudes carry generation/send timestamps, which can overlap or have
    // large gaps even though WebAudio queued the PCM chunks continuously.
    function rebuildTtsPlaybackSegments(timeline, ttlBands) {
        if (!Array.isArray(timeline)) return [];
        const groups = [];
        let current = null;
        timeline.forEach(function (event) {
            if (!event) return;
            if (event.event_type === 'tts_start') {
                current = {
                    start: Number(event.timestamp || 0),
                    firstAudio: null,
                    playbackStop: null,
                    amplitudes: [],
                };
                groups.push(current);
                return;
            }
            if (!current) return;
            if (event.event_type === 'tts_first_audio' && current.firstAudio == null) {
                current.firstAudio = Number(event.timestamp);
                return;
            }
            if (
                event.event_type === 'tts_playback_stopped'
                && current.playbackStop == null
            ) {
                current.playbackStop = Number(event.timestamp);
                return;
            }
            if (
                event.event_type === 'audio_amplitude'
                && (event.source === 'tts' || event.source === 'ai')
            ) {
                current.amplitudes.push(Number(event.amplitude || 0));
            }
        });

        const bands = Array.isArray(ttlBands) ? ttlBands : [];
        const windowSeconds = 0.025;
        const segments = [];
        groups.forEach(function (group, index) {
            if (!group.amplitudes.length) return;
            const nextStart = groups[index + 1] ? groups[index + 1].start : Infinity;
            const matchingBand = bands.find(function (band) {
                const end = Number(band && band.end);
                return Number.isFinite(end) && end >= group.start && end < nextStart;
            });
            const bandStart = matchingBand ? Number(matchingBand.end) : NaN;
            const playbackStart = Number.isFinite(bandStart)
                ? bandStart
                : group.firstAudio;
            if (!Number.isFinite(playbackStart)) return;
            group.amplitudes.forEach(function (amplitude, sampleIndex) {
                const startTime = playbackStart + sampleIndex * windowSeconds;
                if (
                    Number.isFinite(group.playbackStop)
                    && startTime >= group.playbackStop
                ) {
                    return;
                }
                segments.push({
                    startTime: startTime,
                    endTime: Number.isFinite(group.playbackStop)
                        ? Math.min(startTime + windowSeconds, group.playbackStop)
                        : startTime + windowSeconds,
                    amplitude: amplitude,
                });
            });
        });
        return segments;
    }

    // Convert the server-relative session_start timestamp into the browser
    // wall-clock origin used by live rendering and browser-generated events.
    //
    // The START button is clicked before the WebSocket connects and the
    // server creates its Timeline. Keeping that click time as the origin
    // makes server-timestamped microphone samples appear seconds in the past.
    // Anchoring when session_start arrives keeps both clocks in one timebase.
    function syncLiveSessionClock(receivedAt, eventTimestamp) {
        const received = Number(receivedAt);
        if (!Number.isFinite(received)) return 0;
        const serverTime = Number(eventTimestamp);
        return received - (
            Number.isFinite(serverTime) && serverTime >= 0 ? serverTime : 0
        );
    }

    // Collapse multiple telemetry samples that land on the same screen pixel,
    // retaining the largest value so short GPU bursts remain visible.
    function buildPeakPreservingPoints(samples, getX, getValue) {
        if (!Array.isArray(samples)) return [];
        const buckets = new Map();
        samples.forEach(function (sample) {
            const x = Number(getX(sample));
            const rawValue = getValue(sample);
            if (rawValue == null) return;
            const value = Number(rawValue);
            if (!Number.isFinite(x) || !Number.isFinite(value)) return;
            const pixel = Math.floor(x);
            const current = buckets.get(pixel);
            if (!current || value > current.value) {
                buckets.set(pixel, { x: x, value: value, sample: sample });
            }
        });
        return Array.from(buckets.values()).sort(function (a, b) {
            return a.x - b.x;
        });
    }

    // VAD and speech events can be emitted as two aliases at the same
    // timestamp. Treat them as one boundary so ASR turn indices do not shift.
    function dedupeTimelineEventsByTimestamp(events, toleranceSeconds) {
        const tolerance = Number.isFinite(Number(toleranceSeconds))
            ? Math.max(0, Number(toleranceSeconds))
            : 0.005;
        const ordered = (Array.isArray(events) ? events : []).slice().sort(function (a, b) {
            return Number(a.timestamp || 0) - Number(b.timestamp || 0);
        });
        return ordered.filter(function (event, index, all) {
            if (index === 0) return true;
            return Math.abs(
                Number(event.timestamp || 0)
                - Number(all[index - 1].timestamp || 0)
            ) > tolerance;
        });
    }

    // Pair point events into non-overlapping intervals.  Each end belongs to
    // the closest preceding start and may not cross the following start.
    function pairTimelineEvents(timeline, startType, endType) {
        if (!Array.isArray(timeline)) return [];
        const starts = timeline.filter(function (event) {
            return event && event.event_type === startType
                && Number.isFinite(Number(event.timestamp));
        }).slice().sort(function (a, b) {
            return Number(a.timestamp) - Number(b.timestamp);
        });
        const ends = timeline.filter(function (event) {
            return event && event.event_type === endType
                && Number.isFinite(Number(event.timestamp));
        }).slice().sort(function (a, b) {
            return Number(a.timestamp) - Number(b.timestamp);
        });
        const usedEnds = new Set();
        const intervals = [];
        starts.forEach(function (startEvent, index) {
            const start = Number(startEvent.timestamp);
            const nextStart = index + 1 < starts.length
                ? Number(starts[index + 1].timestamp)
                : Infinity;
            const endIndex = ends.findIndex(function (endEvent, candidateIndex) {
                if (usedEnds.has(candidateIndex)) return false;
                const end = Number(endEvent.timestamp);
                return end >= start && end < nextStart;
            });
            if (endIndex < 0) return;
            usedEnds.add(endIndex);
            const endEvent = ends[endIndex];
            intervals.push({
                start: start,
                end: Number(endEvent.timestamp),
                startEvent: startEvent,
                endEvent: endEvent,
            });
        });
        return intervals;
    }

    // Prefer turn-scoped playback events over gap-based waveform grouping.
    // Barge-in can leave only a short gap before the next reply, making two
    // independent responses look like one continuous waveform group.
    function selectFirstPlaybackTimes(
        turnCount,
        firstAudioByTurn,
        firstAmplitudeByTurn,
        responseGroups
    ) {
        const count = Math.max(0, Number(turnCount) || 0);
        return Array.from({ length: count }, function (_, index) {
            const firstAudio = Number(firstAudioByTurn && firstAudioByTurn[index]);
            if (Number.isFinite(firstAudio) && firstAudio > 0) return firstAudio;
            const firstAmplitude = Number(
                firstAmplitudeByTurn && firstAmplitudeByTurn[index]
            );
            if (Number.isFinite(firstAmplitude) && firstAmplitude > 0) {
                return firstAmplitude;
            }
            const group = (
                responseGroups
                && Array.isArray(responseGroups[index])
            ) ? responseGroups[index] : [];
            const starts = group.filter(function (segment) {
                return segment && Number.isFinite(Number(segment.startTime));
            }).map(function (segment) {
                return Number(segment.startTime);
            });
            return starts.length ? Math.min.apply(null, starts) : null;
        });
    }

    // A recorded session's browser-observed TTL bands are the source of
    // truth.  Reconstructing them from down-sampled waveform data can move
    // the detected end-of-speech boundary and make replay disagree with the
    // value shown live.  Only use reconstruction for legacy recordings that
    // do not contain persisted bands.
    function selectReplayTtlBands(persistedBands, fallbackFactory) {
        if (Array.isArray(persistedBands) && persistedBands.length > 0) {
            return persistedBands;
        }
        const fallback = typeof fallbackFactory === 'function'
            ? fallbackFactory()
            : [];
        return Array.isArray(fallback) ? fallback : [];
    }

    // REST ASR can submit an utterance that later returns an empty transcript.
    // Such an orphan request must not be drawn as though it belonged to the
    // following successful speech turn. Match each final only to the closest
    // completed request since the preceding final.
    function matchAsrRequestIntervalsToFinals(intervals, finals) {
        if (!Array.isArray(intervals) || !Array.isArray(finals)) return [];
        const orderedIntervals = intervals.slice().sort(function (a, b) {
            return Number(a.start) - Number(b.start);
        });
        const orderedFinals = finals.slice().sort(function (a, b) {
            return Number(a.timestamp) - Number(b.timestamp);
        });
        const used = new Set();
        return orderedFinals.map(function (finalEvent, finalIndex) {
            const finalTime = Number(finalEvent && finalEvent.timestamp);
            const previousFinalTime = finalIndex > 0
                ? Number(orderedFinals[finalIndex - 1].timestamp)
                : -Infinity;
            if (!Number.isFinite(finalTime)) return null;
            let matchIndex = -1;
            orderedIntervals.forEach(function (interval, intervalIndex) {
                if (used.has(intervalIndex)) return;
                const start = Number(interval && interval.start);
                const end = Number(interval && interval.end);
                if (
                    Number.isFinite(start)
                    && Number.isFinite(end)
                    && start > previousFinalTime
                    && end <= finalTime + 0.01
                ) {
                    matchIndex = intervalIndex;
                }
            });
            if (matchIndex < 0) return null;
            used.add(matchIndex);
            return orderedIntervals[matchIndex];
        }).filter(Boolean);
    }

    // Do not interpolate utilization across missing telemetry. A gap is
    // unknown data, not a gradual decay from the previous value.
    function splitPointsAtTimeGaps(points, maxGapSeconds) {
        if (!Array.isArray(points) || points.length === 0) return [];
        const maximumGap = Number.isFinite(Number(maxGapSeconds))
            ? Math.max(0, Number(maxGapSeconds))
            : 0.2;
        const runs = [];
        let current = [];
        points.forEach(function (point) {
            const timestamp = Number(point && point.t);
            const previous = current.length
                ? Number(current[current.length - 1].t)
                : null;
            if (
                current.length
                && Number.isFinite(timestamp)
                && Number.isFinite(previous)
                && timestamp - previous > maximumGap
            ) {
                runs.push(current);
                current = [];
            }
            current.push(point);
        });
        if (current.length) runs.push(current);
        return runs;
    }

    // The browser-observed playback segment is the authoritative TTFA point.
    // Server first-byte timestamps can differ slightly because the browser and
    // server session clocks are synchronized after the WebSocket is created.
    function resolveTtsFirstAudioTimes(starts, firstAudios, playbackSegments) {
        const orderedStarts = (Array.isArray(starts) ? starts : []).slice().sort(function (a, b) {
            return Number(a.timestamp) - Number(b.timestamp);
        });
        const orderedFirstAudios = (Array.isArray(firstAudios) ? firstAudios : []).slice().sort(function (a, b) {
            return Number(a.timestamp) - Number(b.timestamp);
        });
        const segments = (Array.isArray(playbackSegments) ? playbackSegments : []).slice().sort(function (a, b) {
            return Number(a.startTime) - Number(b.startTime);
        });
        if (!orderedStarts.length) {
            return orderedFirstAudios.map(function (event) {
                return Number(event.timestamp);
            }).filter(Number.isFinite);
        }
        return orderedStarts.map(function (startEvent, index) {
            const start = Number(startEvent.timestamp);
            const nextStart = index + 1 < orderedStarts.length
                ? Number(orderedStarts[index + 1].timestamp)
                : Infinity;
            const observed = segments.find(function (segment) {
                const timestamp = Number(segment && segment.startTime);
                return Number.isFinite(timestamp)
                    && timestamp >= start
                    && timestamp < nextStart;
            });
            if (observed) return Number(observed.startTime);
            const serverEvent = orderedFirstAudios.find(function (event) {
                const timestamp = Number(event && event.timestamp);
                return Number.isFinite(timestamp)
                    && timestamp >= start
                    && timestamp < nextStart;
            });
            return serverEvent ? Number(serverEvent.timestamp) : null;
        }).filter(Number.isFinite);
    }

    // Find the strongest observed GPU sample inside each explicit interval.
    // The renderer gives these point-in-time peaks a minimum visual width;
    // this helper intentionally does not widen the measured time window.
    function buildIntervalPeakMarkers(samples, intervals, minimumValue) {
        const threshold = Number.isFinite(Number(minimumValue))
            ? Number(minimumValue)
            : 5;
        if (!Array.isArray(samples) || !Array.isArray(intervals)) return [];
        return intervals.map(function (interval) {
            const start = Number(interval.start);
            const end = Number(interval.end);
            if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
                return null;
            }
            let peak = null;
            samples.forEach(function (sample) {
                const timestamp = Number(sample && sample.t);
                const value = Number(sample && sample.gpu);
                if (
                    !Number.isFinite(timestamp)
                    || !Number.isFinite(value)
                    || timestamp < start
                    || timestamp > end
                ) return;
                if (!peak || value > peak.value) {
                    peak = {
                        t: timestamp,
                        value: value,
                        sample: sample,
                        start: start,
                        end: end,
                    };
                }
            });
            return peak && peak.value >= threshold ? peak : null;
        }).filter(Boolean);
    }

    // Keep the ASR legend truthful across fundamentally different transports.
    // REST has one observable request per utterance; Realtime and Riva keep a
    // stream open and expose transcript updates instead.
    function getAsrLegendSpec(config) {
        const asr = config && config.asr ? config.asr : (config || {});
        const scheme = String(asr.scheme || asr.backend || '').toLowerCase();
        let mode = 'generic';
        if (scheme) {
            if (scheme.indexOf('realtime') >= 0) {
                mode = 'realtime';
            } else if (scheme === 'riva') {
                mode = 'riva';
            } else if (
                scheme === 'openai'
                || scheme.indexOf('rest') >= 0
            ) {
                mode = 'rest';
            }
        } else if (asr.realtime_url) {
            mode = 'realtime';
        } else if (
            asr.riva_server
            || String(asr.server || '').indexOf(':50051') >= 0
        ) {
            mode = 'riva';
        } else if (asr.api_base) {
            mode = 'rest';
        }

        const common = {
            mode: mode,
            speechLabel: 'Speech / VAD',
        };
        if (mode === 'rest') {
            return Object.assign(common, {
                speechTitle: 'Local energy VAD detected user speech',
                endpointLabel: 'Endpoint Wait',
                endpointTitle: 'Local silence hold before submitting the utterance',
                activeLabel: 'ASR Request',
                activeTitle: 'REST transcription request: audio upload through response receipt',
                activeSwatch: 'request',
            });
        }
        if (mode === 'realtime') {
            return Object.assign(common, {
                speechTitle: 'Speech activity from provider events when available; otherwise inferred from transcript timing',
                endpointLabel: 'Finalization',
                endpointTitle: 'Last partial to completed transcript; may include endpointing, commit, and final decoding',
                activeLabel: 'Streaming ASR',
                activeTitle: 'Partial transcript updates observed on the persistent Realtime stream; not exact GPU time',
                activeSwatch: 'streaming',
            });
        }
        if (mode === 'riva') {
            return Object.assign(common, {
                speechTitle: 'Local energy VAD observed the same PCM stream sent to Riva',
                endpointLabel: 'Endpoint / Finalize',
                endpointTitle: 'Local energy speech end to Riva final transcript; includes endpointing and final decoding',
                activeLabel: 'Streaming ASR',
                activeTitle: 'Riva interim transcript activity on the persistent gRPC stream; not exact GPU time',
                activeSwatch: 'streaming',
            });
        }
        return Object.assign(common, {
            speechTitle: 'Detected or inferred user speech activity',
            endpointLabel: 'Endpoint / Finalize',
            endpointTitle: 'Speech endpointing and final transcript completion',
            activeLabel: 'ASR Active',
            activeTitle: 'Observed recognition activity; exact meaning depends on the ASR backend',
            activeSwatch: 'streaming',
        });
    }

    // Apply the speech timing state shared by the REST and Realtime paths.
    // Returns true when the caller should not process the event further.
    function applySpeechTimingEvent(state, evt) {
        if (!state || !evt) return false;
        if (evt.event_type === 'vad_start' || evt.event_type === 'user_speech_start') {
            state.voiceTurnActive = true;
            // Any TTS response already in progress belongs to the previous
            // turn and must not close the new turn's TTL band.
            state.ttsEligibleForCurrentTtl = false;
            state.voiceSilenceCandidate = null;
            state.voiceSilenceConsecutiveCount = 0;
            return true;
        }
        if (evt.event_type === 'vad_end' || evt.event_type === 'user_speech_end') {
            const vadEndTime = evt.timestamp != null ? Number(evt.timestamp) : NaN;
            if (!isNaN(vadEndTime) && state.liveTtlBandStartTime == null) {
                state.liveTtlBandStartTime = vadEndTime;
                state.voiceTurnActive = true;
            }
            return true;
        }
        if (evt.event_type === 'asr_final' && state.liveTtlBandStartTime == null) {
            state.ttsEligibleForCurrentTtl = false;
            const finalTime = evt.timestamp != null ? Number(evt.timestamp) : NaN;
            if (!isNaN(finalTime)) {
                state.liveTtlBandStartTime = state.lastAsrPartialTime != null
                    ? state.lastAsrPartialTime
                    : finalTime - 0.2;
                state.voiceTurnActive = true;
            }
        }
        return false;
    }

    // Close one browser-observed TTL band only when the audio belongs to the
    // response for the current user turn. This prevents an interrupted TTS
    // response from producing a negative band or lending its audio to the
    // following turn.
    function closeTtlBandAt(state, bandEnd) {
        if (!state || state.liveTtlBandStartTime == null) return false;
        if (state.ttsEligibleForCurrentTtl === false) return false;
        const start = Number(state.liveTtlBandStartTime);
        const end = Number(bandEnd);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
            return false;
        }
        if (!Array.isArray(state.liveTtlBands)) state.liveTtlBands = [];
        state.liveTtlBands.push({
            start: start,
            end: end,
            ttlMs: Math.round((end - start) * 1000),
        });
        state.liveTtlBandStartTime = null;
        state.voiceTurnActive = false;
        state.lastAsrPartialTime = null;
        state.firstTtsPlayTimeThisResponse = null;
        state.earliestTtsPlayTimeAboveThreshold = null;
        return true;
    }

    // Remove generated/scheduled AI audio after browser playback is stopped.
    // Barge-in may occur after future WebAudio buffers have already been
    // scheduled, so simply stopping BufferSources is not enough to keep the
    // visible purple waveform faithful to what was actually played.
    function truncateAudioSegmentsAt(segments, cutoff) {
        if (!Array.isArray(segments)) return [];
        const end = Number(cutoff);
        if (!Number.isFinite(end)) return segments.slice();
        return segments.reduce(function (result, segment) {
            if (!segment) return result;
            const start = Number(
                segment.startTime != null ? segment.startTime : segment.timestamp
            );
            if (!Number.isFinite(start) || start >= end) return result;
            const clipped = Object.assign({}, segment);
            if (clipped.endTime != null) {
                const segmentEnd = Number(clipped.endTime);
                if (Number.isFinite(segmentEnd)) {
                    clipped.endTime = Math.min(segmentEnd, end);
                }
            }
            result.push(clipped);
            return result;
        }, []);
    }

    // Split scheduled browser audio at the exact WebAudio stop point.  The
    // played side remains the authoritative purple waveform; the discarded
    // side is summarized as a hatched diagnostic region.
    function splitAudioSegmentsAt(segments, cutoff) {
        const stop = Number(cutoff);
        const played = [];
        const discarded = [];
        if (!Array.isArray(segments) || !Number.isFinite(stop)) {
            return {
                played: Array.isArray(segments) ? segments.slice() : [],
                discarded: [],
                discardedEnd: null,
            };
        }
        segments.forEach(function (segment) {
            if (!segment) return;
            const start = Number(
                segment.startTime != null ? segment.startTime : segment.timestamp
            );
            const rawEnd = Number(
                segment.endTime != null ? segment.endTime : start
            );
            if (!Number.isFinite(start) || !Number.isFinite(rawEnd)) return;
            const end = Math.max(start, rawEnd);
            if (start < stop) {
                const before = Object.assign({}, segment, {
                    startTime: start,
                    endTime: Math.min(end, stop),
                });
                if (before.endTime > before.startTime) played.push(before);
            }
            if (end > stop) {
                const after = Object.assign({}, segment, {
                    startTime: Math.max(start, stop),
                    endTime: end,
                });
                if (after.endTime > after.startTime) discarded.push(after);
            }
        });
        const discardedEnd = discarded.length
            ? Math.max.apply(null, discarded.map(function (segment) {
                return Number(segment.endTime);
            }))
            : null;
        return {
            played: played,
            discarded: discarded,
            discardedEnd: discardedEnd,
        };
    }

    // Pair each browser playback stop with the cancellation completion for the
    // same response.  The next tts_start is a hard boundary so a missing
    // completion cannot lend its wait interval to the following turn.
    function buildBargeInWindows(timeline) {
        if (!Array.isArray(timeline)) return [];
        const ordered = timeline.slice().sort(function (a, b) {
            return Number(a.timestamp || 0) - Number(b.timestamp || 0);
        });
        const stops = ordered.filter(function (event) {
            return event && event.event_type === 'tts_playback_stopped'
                && (!event.data || !event.data.reason || event.data.reason === 'barge_in');
        });
        return stops.map(function (stopEvent) {
            const start = Number(stopEvent.timestamp);
            const nextStart = ordered.find(function (event) {
                return event.event_type === 'tts_start'
                    && Number(event.timestamp) > start;
            });
            const limit = nextStart ? Number(nextStart.timestamp) : Infinity;
            const cancelEvent = ordered.find(function (event) {
                const timestamp = Number(event.timestamp);
                if (!(timestamp >= start && timestamp < limit)) return false;
                if (event.event_type === 'tts_cancelled') return true;
                return event.event_type === 'tts_complete'
                    && event.data && event.data.cancelled === true;
            }) || null;
            const data = stopEvent.data || {};
            const discardedEnd = Number(data.discarded_audio_end);
            return {
                stopEvent: stopEvent,
                start: start,
                cancelEnd: cancelEvent ? Number(cancelEvent.timestamp) : null,
                cancelEvent: cancelEvent,
                nextTtsStart: Number.isFinite(limit) ? limit : null,
                discardedEnd: Number.isFinite(discardedEnd) && discardedEnd > start
                    ? Math.min(discardedEnd, limit)
                    : null,
            };
        }).filter(function (window) {
            return Number.isFinite(window.start);
        });
    }

    return {
        applySpeechTimingEvent: applySpeechTimingEvent,
        buildBargeInWindows: buildBargeInWindows,
        buildIntervalPeakMarkers: buildIntervalPeakMarkers,
        buildPeakPreservingPoints: buildPeakPreservingPoints,
        buildTtsSegmentsFromTimeline: buildTtsSegmentsFromTimeline,
        closeTtlBandAt: closeTtlBandAt,
        dedupeTimelineEventsByTimestamp: dedupeTimelineEventsByTimestamp,
        getAsrLegendSpec: getAsrLegendSpec,
        hasDenseTtsAmplitudeTimeline: hasDenseTtsAmplitudeTimeline,
        matchAsrRequestIntervalsToFinals: matchAsrRequestIntervalsToFinals,
        pairTimelineEvents: pairTimelineEvents,
        rebuildTtsPlaybackSegments: rebuildTtsPlaybackSegments,
        selectFirstPlaybackTimes: selectFirstPlaybackTimes,
        selectReplayTtlBands: selectReplayTtlBands,
        resolveTtsFirstAudioTimes: resolveTtsFirstAudioTimes,
        splitPointsAtTimeGaps: splitPointsAtTimeGaps,
        syncLiveSessionClock: syncLiveSessionClock,
        splitAudioSegmentsAt: splitAudioSegmentsAt,
        truncateAudioSegmentsAt: truncateAudioSegmentsAt,
    };
}));
