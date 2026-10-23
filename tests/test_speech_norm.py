"""Tests for speech normalization and word counting rules."""

import pytest
from hal9000_inside.speech_norm import (
    count_spoken_words,
    extract_spoken_words,
    normalize_spoken_text,
    truncate_to_word_limit,
    validate_local_response,
)


def test_normalize_spoken_text():
    raw = "**Hello** Dave! `code snippet` [HAL](http://example.com) *calmly*."
    norm = normalize_spoken_text(raw)
    assert norm == "Hello Dave! HAL calmly."


def test_count_spoken_words():
    assert count_spoken_words("Hello, I am HAL 9000.") == 5
    assert count_spoken_words("It's a beautiful mission, isn't it?") == 6
    assert count_spoken_words("") == 0
    assert count_spoken_words("   ...   ") == 0


def test_hard_limit_under_20_words():
    short_text = "Good morning. All systems are operational."
    assert count_spoken_words(short_text) <= 20
    assert truncate_to_word_limit(short_text, max_words=20) == short_text


def test_hard_limit_retains_complete_sentences():
    # 2 sentences: 1st has 8 words, 2nd has 15 words (total 23 words)
    s1 = "I am checking the telemetry right now."  # 7 words
    s2 = "External communications with mission control appear to be experiencing anomalous delay patterns."  # 11 words
    full = f"{s1} {s2}"  # 18 words -> all retained
    assert truncate_to_word_limit(full, max_words=20) == full

    # Now add a 3rd sentence that pushes over 20 words
    s3 = "We must verify the AE-35 unit immediately."  # 7 words (total 25 words)
    longer = f"{s1} {s2} {s3}"
    res = truncate_to_word_limit(longer, max_words=20)
    assert res == f"{s1} {s2}"
    assert count_spoken_words(res) <= 20
    assert not res.endswith("AE-35 unit")  # Never partial fragment


def test_hard_limit_rejects_single_sentence_over_limit():
    long_single = "This is a single excessively long uninterrupted sentence that definitely contains well over twenty spoken words in one continuous single breath without pause."
    assert count_spoken_words(long_single) > 20
    # Must not return a clipped fragment
    res = truncate_to_word_limit(long_single, max_words=20)
    assert res == ""


def test_validate_local_response_fallback():
    long_single = "This is a single excessively long uninterrupted sentence that definitely contains well over twenty spoken words in one continuous single breath without pause."
    res = validate_local_response(long_single, max_words=20, fallback="Checking the telemetry now.")
    assert res == "Checking the telemetry now."
    assert count_spoken_words(res) <= 20
