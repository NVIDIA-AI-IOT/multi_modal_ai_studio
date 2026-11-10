"""PCM pacing and acoustic turn boundaries for continuous speech streams."""

import numpy as np


class ContinuousPCMInput:
    """Bound capture latency and keep the model advancing when capture is idle."""

    frame_bytes = 3840  # 80 ms, 24 kHz mono PCM16

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, pcm: bytes) -> None:
        self.buffer.extend(pcm[:len(pcm) // 2 * 2])
        # Keep at most 400 ms, discarding stale input after capture stalls.
        if len(self.buffer) > self.frame_bytes * 5:
            del self.buffer[:-self.frame_bytes * 5]

    def clear(self) -> None:
        self.buffer.clear()

    def take(self) -> bytes:
        frame = bytes(self.buffer[:self.frame_bytes])
        del self.buffer[:self.frame_bytes]
        return frame.ljust(self.frame_bytes, b"\0")


class AcousticTurn:
    """Track audible regions independently of the provider's stream response ID."""

    def __init__(self):
        self.last_active = None

    def observe(self, pcm: bytes, now: float) -> tuple[bool, bool]:
        samples = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2")
        active = bool(len(samples) and np.max(np.abs(samples.astype(np.int32))) > 128)
        if active:
            self.last_active = now
        ended = self.last_active is not None and now - self.last_active >= 0.4
        if ended:
            self.last_active = None
        return active, ended


class AudioTextMatcher:
    """Pair unique pending acoustic regions/text in either arrival order.

    A new text inside already-owned continuous audio has no distinct onset.
    Multiple pending candidates remain unpaired instead of guessing by order.
    """

    def __init__(self):
        self.texts = []
        self.regions = []
        self.ambiguous = False

    def _match(self):
        if not self.ambiguous and len(self.texts) == len(self.regions) == 1:
            return self.texts.pop(), self.regions.pop()
        return None

    def text_started(self, utterance_id, *, audio_active):
        if self.regions or not audio_active:
            self.texts.append(utterance_id)
        else:
            # An unresolved text inside owned audio must not steal a later onset.
            self.ambiguous = True
        return self._match()

    def audio_started(self, region_id):
        self.regions.append(region_id)
        return self._match()
