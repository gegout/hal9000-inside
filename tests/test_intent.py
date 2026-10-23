"""Tests for turn intent classification."""

import pytest
from hal9000_inside.llm.intent import classify_turn_intent


def test_classify_social():
    assert classify_turn_intent("Hello HAL") == "social"
    assert classify_turn_intent("Good morning") == "social"
    assert classify_turn_intent("Thank you HAL.") == "social"
    assert classify_turn_intent("Goodbye") == "social"


def test_classify_action():
    assert classify_turn_intent("stop") == "action"
    assert classify_turn_intent("cancel") == "action"
    assert classify_turn_intent("stop talking") == "action"


def test_classify_substantive_overrides():
    assert classify_turn_intent("Thanks, but explain why the sky is blue") == "substantive"
    assert classify_turn_intent("Why did OpenSSL change license?") == "substantive"
    assert classify_turn_intent("What is the capital of France?") == "substantive"
    assert classify_turn_intent("Compare Debian and Ubuntu") == "substantive"
    assert classify_turn_intent("Tell me more about robotics") == "substantive"
