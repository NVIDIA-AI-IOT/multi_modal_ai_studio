"""Acoustic onset bookkeeping must work in either transport arrival order."""
from multi_modal_ai_studio.backends.realtime.duplex import AudioTextMatcher


def test_text_before_audio_links_once():
    matcher = AudioTextMatcher()
    assert matcher.text_started(1, audio_active=False) is None
    assert matcher.audio_started(1) == (1, 1)
    assert matcher.texts == matcher.regions == []


def test_audio_before_sure_text_keeps_onset_available():
    matcher = AudioTextMatcher()
    assert matcher.text_started(1, audio_active=False) is None
    assert matcher.audio_started(1) == (1, 1)
    assert matcher.audio_started(2) is None  # New speech before earlier text final.
    assert matcher.text_started(2, audio_active=True) == (2, 2)


def test_new_text_inside_owned_continuous_audio_has_no_distinct_onset():
    matcher = AudioTextMatcher()
    matcher.text_started(1, audio_active=False)
    matcher.audio_started(1)
    assert matcher.text_started(2, audio_active=True) is None
    assert matcher.audio_started(2) is None  # Do not assign a later answer to it.
    assert matcher.text_started(3, audio_active=True) is None


def test_multiple_pending_regions_are_not_paired_by_order():
    matcher = AudioTextMatcher()
    matcher.audio_started(1)
    matcher.audio_started(2)
    assert matcher.text_started(1, audio_active=True) is None
    assert matcher.text_started(2, audio_active=True) is None
