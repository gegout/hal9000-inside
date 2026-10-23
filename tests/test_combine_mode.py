import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from hal9000_inside.tts.speech_queue import OrderedSpeechQueue
from hal9000_inside.speech_norm import count_spoken_words, validate_local_response, truncate_to_word_limit
from hal9000_inside.llm.intent import classify_turn_intent


@pytest.mark.asyncio
async def test_ordered_speech_queue_local_then_remote():
    events = []

    def mock_play(pcm):
        events.append(f"play_{pcm.decode('utf-8')}")

    tts = MagicMock(spec=["synthesize_raw_pcm"])
    tts.synthesize_raw_pcm.side_effect = lambda t: t.encode('utf-8')

    started = []
    stopped = []

    queue = OrderedSpeechQueue(
        tts=tts,
        play_pcm_fn=mock_play,
        notify_start_fn=lambda: started.append(True),
        notify_stop_fn=lambda: stopped.append(True),
    )

    # Local plays
    res_local = await queue.speak_local_ack("Looking into that now.")
    assert res_local is True

    # Remote substantive plays
    res_remote = await queue.speak_substantive_remote("Ubuntu 24.04 LTS includes Linux kernel 6.8.")
    assert res_remote is True

    # Verify order: local first, then remote
    assert events == ["play_Looking into that now.", "play_Ubuntu 24.04 LTS includes Linux kernel 6.8."]
    assert len(started) == 2
    assert len(stopped) == 2


@pytest.mark.asyncio
async def test_ordered_speech_queue_suppress_local_when_remote_ready():
    events = []

    def mock_play(pcm):
        events.append(f"play_{pcm.decode('utf-8')}")

    tts = MagicMock(spec=["synthesize_raw_pcm"])
    tts.synthesize_raw_pcm.side_effect = lambda t: t.encode('utf-8')

    queue = OrderedSpeechQueue(
        tts=tts,
        play_pcm_fn=mock_play,
        notify_start_fn=lambda: None,
        notify_stop_fn=lambda: None,
    )

    # Remote finishes immediately before local starts playing
    queue.mark_remote_ready()
    res_local = await queue.speak_local_ack("Looking into that now.")
    assert res_local is False

    res_remote = await queue.speak_substantive_remote("The capital of France is Paris.")
    assert res_remote is True

    # Local was suppressed because remote was already ready
    assert events == ["play_The capital of France is Paris."]


@pytest.mark.asyncio
async def test_ordered_speech_queue_cancellation():
    events = []

    def mock_play(pcm):
        events.append(f"play_{pcm.decode('utf-8')}")

    tts = MagicMock(spec=["synthesize_raw_pcm"])
    tts.synthesize_raw_pcm.side_effect = lambda t: t.encode('utf-8')

    queue = OrderedSpeechQueue(
        tts=tts,
        play_pcm_fn=mock_play,
        notify_start_fn=lambda: None,
        notify_stop_fn=lambda: None,
    )

    queue.cancel()
    assert queue._cancelled is True
    res = await queue.speak_local_ack("This should not play.")
    assert res is False
    assert len(events) == 0


def test_hard_local_word_limit():
    assert count_spoken_words("Good morning, Dave. I am completely operational.") == 7
    assert count_spoken_words("Word " * 25) == 25

    # Truncation preserving complete sentences
    truncated = truncate_to_word_limit("First sentence is short. Second sentence is also short. Third sentence pushes it past twenty words limit definitely.", max_words=20)
    assert count_spoken_words(truncated) <= 20
    assert truncated.endswith(".")

    # Validation
    res = validate_local_response("Affirmative, Dave. System is nominal.", max_words=20)
    assert res == "Affirmative, Dave. System is nominal."
    assert count_spoken_words(res) <= 20


def test_intent_classification_combine():
    assert classify_turn_intent("hello HAL") == "social"
    assert classify_turn_intent("stop speaking") == "action"
    assert classify_turn_intent("cancel current speech") == "action"
    assert classify_turn_intent("What is the release cadence of Ubuntu?") == "substantive"
    assert classify_turn_intent("Thanks, but explain why") == "substantive"
