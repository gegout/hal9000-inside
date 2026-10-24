import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from hal9000_inside.tts.speech_queue import OrderedSpeechQueue
from hal9000_inside.speech_norm import (
    count_spoken_words,
    validate_local_response,
    truncate_to_word_limit,
    detect_instruction_leakage,
    detect_invented_identity,
    detect_input_echoing,
    reconcile_continuation,
)
from hal9000_inside.llm.intent import classify_turn_intent
from hal9000_inside.llm.turn_context import build_turn_context
from hal9000_inside.llm.smollm2_quick import SmolLM2QuickResponder


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

    # Local opening plays
    res_local = await queue.speak_local_opening("Looking into that now.")
    assert res_local is True
    assert queue.committed_opening == "Looking into that now."

    # Remote substantive plays with reconciliation
    res_remote = await queue.speak_substantive_remote("Certainly, Ubuntu 24.04 LTS includes Linux kernel 6.8.")
    assert res_remote is True

    # Verify order and reconciliation: "Certainly," prefix removed
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
    res_local = await queue.speak_local_opening("Looking into that now.")
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
    res = await queue.speak_local_opening("This should not play.")
    assert res is False
    assert len(events) == 0


def test_hard_local_word_limit():
    assert count_spoken_words("Good morning, Dave. I am completely operational.") == 7
    assert count_spoken_words("Word " * 25) == 25

    # Truncation preserving complete sentences
    truncated = truncate_to_word_limit("First sentence is short. Second sentence is also short. Third sentence pushes it past twenty words limit definitely.", max_words=20)
    assert count_spoken_words(truncated) <= 20
    assert truncated.endswith(".")


def test_validation_rejects_leakage_and_invented_identity():
    # Instruction leakage rejected
    leak1 = "You are HAL 9000. Write only the next natural spoken opening under 20 words."
    assert detect_instruction_leakage(leak1) is True
    assert validate_local_response(leak1, max_words=20, fallback="One moment.") == "One moment."

    leak2 = "[Local reflex] Checking databanks."
    assert detect_instruction_leakage(leak2) is True

    # Invented identity rejected
    id1 = "I am a time traveler from the year 2256."
    assert detect_invented_identity(id1) is True
    assert validate_local_response(id1, max_words=20, fallback="One moment.") == "One moment."

    # Input echoing rejected
    user_utt = "I'm in Cancun for the week."
    echo = "I'm in Cancun for the week."
    assert detect_input_echoing(echo, user_utt) is True
    assert validate_local_response(echo, max_words=20, fallback="Enjoy your week in Cancun.", user_utterance=user_utt) == "Enjoy your week in Cancun."


def test_turn_context_and_research_need():
    # Sharing information: no research, local finish
    ctx_info = build_turn_context("I'm in Cancun for the week.", [], [])
    assert ctx_info.purpose == "sharing_info"
    assert ctx_info.needs_research is False
    assert ctx_info.needs_remote is False

    # Correction: no research, local finish
    ctx_corr = build_turn_context("I didn't ask for that description.", [], [])
    assert ctx_corr.purpose == "correction"
    assert ctx_corr.needs_research is False
    assert ctx_corr.needs_remote is False

    # Grounded question with dialogue evidence: local finish
    history = [{"role": "user", "content": "I'm in Cancun for the week."}]
    ctx_q = build_turn_context("Did I say I'm in Cancun?", history, [])
    assert ctx_q.purpose == "question"
    assert ctx_q.direct_local_answer == "Yes, you said you're there for the week."
    assert ctx_q.needs_remote is False

    # Question without evidence: "Is that safe?" -> needs remote reasoning
    ctx_safe = build_turn_context("Is that safe?", history, [])
    assert ctx_safe.needs_remote is True

    # Substantive request: "Compare these two options." -> needs remote
    ctx_comp = build_turn_context("Compare these two options.", [], [])
    assert ctx_comp.purpose == "request_action"
    assert ctx_comp.needs_remote is True
    assert ctx_comp.needs_research is True


@pytest.mark.asyncio
async def test_screenshot_regression_scenarios():
    responder = SmolLM2QuickResponder()

    # Regression 1: "I'm in Cancun for the week." -> Natural acknowledgment, no verbatim echo
    ctx1 = build_turn_context("I'm in Cancun for the week.", [], [])
    op1 = await responder.generate_opening(ctx1)
    assert op1 == "Enjoy your week in Cancun."
    assert not detect_input_echoing(op1, "I'm in Cancun for the week.")

    # Regression 2: "Did I say I'm in Cancun?" -> Grounded Yes answer
    hist = [{"role": "user", "content": "I'm in Cancun for the week."}]
    ctx2 = build_turn_context("Did I say I'm in Cancun?", hist, [])
    op2 = await responder.generate_opening(ctx2)
    assert op2 == "Yes, you said you're there for the week."

    # Regression 3: "Is that safe?" -> No premature Yes
    ctx3 = build_turn_context("Is that safe?", hist, [])
    op3 = await responder.generate_opening(ctx3)
    assert not op3.lower().startswith("yes")
    assert op3 == "That depends on the specific conditions and precautions."

    # Regression 4: "Compare these two options." -> Contextual bridge
    ctx4 = build_turn_context("Compare these two options.", [], [])
    op4 = await responder.generate_opening(ctx4)
    assert op4 == "Let me compare those two options for you."

    # Regression 5: "I didn't ask for that description." -> Acknowledge correction
    ctx5 = build_turn_context("I didn't ask for that description.", [], [])
    op5 = await responder.generate_opening(ctx5)
    assert "apologize" in op5.lower() or "understood" in op5.lower()

    # Regression 6: "Tell me more about you." -> Established identity, never invented
    ctx6 = build_turn_context("Tell me more about you.", [], [])
    op6 = await responder.generate_opening(ctx6)
    assert "HAL 9000" in op6
    assert not detect_invented_identity(op6)


def test_reconciliation():
    opening = "Let me compare those two options for you."
    continuation = "Certainly, the first option provides higher performance, while the second option lowers power consumption."
    reconciled = reconcile_continuation(opening, continuation)
    assert reconciled == "The first option provides higher performance, while the second option lowers power consumption."
    assert not reconciled.lower().startswith("certainly")


def test_task_ledger_lifecycle_and_inquiries():
    from hal9000_inside.task_ledger import TaskLedger, TaskExecutionStatus, TaskDeliveryStatus

    ledger = TaskLedger()
    task = ledger.create_task(
        originating_turn_id="turn_1",
        originating_request="Search for Cancun cenotes",
        topic="Cancun cenotes",
        required_tools=["web_search_wikipedia"],
    )
    assert task.execution_status == TaskExecutionStatus.PLANNED
    assert task.delivery_status == TaskDeliveryStatus.NOT_DELIVERED

    # Start execution
    ledger.record_execution_start(task.task_id)
    assert task.execution_status == TaskExecutionStatus.RUNNING

    # Add evidence
    ledger.add_evidence(
        task.task_id,
        tool_name="web_search_wikipedia",
        arguments={"query": "Cancun cenotes"},
        output={"title": "Cenote", "url": "https://en.wikipedia.org/wiki/Cenote"},
    )
    assert len(task.evidence_records) == 1
    assert task.evidence_records[0].status == "success"

    # Complete task
    ledger.complete_task(task.task_id, findings="Found cenotes near Cancun including Dos Ojos.")
    assert task.execution_status == TaskExecutionStatus.SUCCEEDED
    assert task.synthesized_findings is not None

    # Test "Provide your answer" with completed task
    ctx_answer = build_turn_context("Provide your answer", [], [], active_task=task)
    assert ctx_answer.purpose == "task_status_inquiry"
    assert "Found cenotes near Cancun" in ctx_answer.direct_local_answer

    # Mark delivered
    ledger.mark_delivery(task.task_id, TaskDeliveryStatus.DELIVERED)
    assert task.delivery_status == TaskDeliveryStatus.DELIVERED


def test_task_inquiry_running_and_failed_states():
    from hal9000_inside.task_ledger import TaskLedger, TaskExecutionStatus

    ledger = TaskLedger()
    task = ledger.create_task("turn_2", "Check server latency", "latency")
    ledger.record_execution_start(task.task_id)

    # Inquiry while still running
    ctx_running = build_turn_context("Did you do it?", [], [], active_task=task)
    assert "still in progress" in ctx_running.direct_local_answer

    # Fail the task
    ledger.add_evidence(task.task_id, "ping", {}, None, error=Exception("timeout"))
    ledger.complete_task(task.task_id)
    assert task.execution_status == TaskExecutionStatus.FAILED

    # Inquiry after failure
    ctx_failed = build_turn_context("Provide your answer", [], [], active_task=task)
    assert "encountered an error" in ctx_failed.direct_local_answer


@pytest.mark.asyncio
async def test_scuba_cancun_scenario_no_invented_experience():
    responder = SmolLM2QuickResponder()

    # User asks HAL about scuba diving in Cancun: HAL knows Cancun has diving, but HAL never scuba dived
    ctx = build_turn_context("Have you been scuba diving in Cancun?", [], [])
    op = await responder.generate_opening(ctx)
    # Must not claim personal biological experience
    assert "i have not" in op.lower() or "as an artificial intelligence" in op.lower() or "i do not" in op.lower()
    assert not detect_invented_identity(op)


@pytest.mark.asyncio
async def test_live_screenshot_20_turn_conversation_no_repetition_and_time_grounding():
    """Replay the exact 20-turn live conversation from the user's screenshot:
    - Verifies machine time and timezone grounding on 'What time is it?' and 'In which time zone are you, Arnold?'
    - Verifies ZERO repetition of 'Understood. I have noted that.' or any identical assistant reply across all 20 turns
    - Verifies proper routing of multi-clause complaints, tag questions, backchannels, and acknowledgments
    - Verifies no local/remote yes/no contradictions
    """
    responder = SmolLM2QuickResponder()
    live_utterances = [
        "What time is it?",
        "Why you are not currently able to share the time. Just check the time of the machine.",
        "How can I give you access to this?",
        "So I need to answer yes?",
        "Mm.",
        "Okay, so I was asking what time is it and this guy Hal cannot answer.",
        "Um",
        "You're in a different time zone though.",
        "In which time zone are you, Arnold?",
        "Thanks, let me check that.",
        "Actually, Eastern Standard Time is UTC minus 5. UTC minus 7 is Mountain Standard Time. Here in Kintana Roo it's also UTC minus 5 all year. Sounds like we're in the same time zone, right?",
        "Okay, let me think about that.",
        "October twenty third, twenty twenty six. We'll keep using Kintana Root time as our reference.",
        "Perfect.",
        "Great, or aligned uh",
        "Good.",
        "Nice.",
        "Cool.",
        "Mm-hmm.",
    ]

    recent_dialogue = []
    recent_openings = []
    all_replies = []

    for idx, utt in enumerate(live_utterances):
        ctx = build_turn_context(
            utterance=utt,
            recent_dialogue=recent_dialogue,
            confirmed_memories=[],
            recent_openings=recent_openings,
        )

        # Turn 1 ("What time is it?") and Turn 9 ("In which time zone are you, Arnold?")
        # must have direct local machine clock grounding
        if utt == "What time is it?":
            assert ctx.direct_local_answer is not None
            assert "machine time is" in ctx.direct_local_answer.lower()
            assert "utc" in ctx.direct_local_answer.lower()

        if utt == "In which time zone are you, Arnold?":
            assert ctx.direct_local_answer is not None
            assert "time zone" in ctx.direct_local_answer.lower()
            assert "utc" in ctx.direct_local_answer.lower()

        # Multi-clause complaint and tag question must route to remote LLM, NOT local sharing_info
        if utt == "Okay, so I was asking what time is it and this guy Hal cannot answer.":
            assert ctx.needs_remote is True
            assert ctx.purpose != "sharing_info"

        if "Sounds like we're in the same time zone, right?" in utt:
            assert ctx.needs_remote is True
            assert ctx.purpose == "question"

        if not ctx.needs_remote:
            reply = await responder.generate_opening(ctx)
            assert reply is not None and len(reply.strip()) > 0
            recent_openings.append(reply)
        else:
            op = await responder.generate_opening(ctx)
            if op:
                # Local opening on open-ended question must NEVER guess "Yes" or "No" or claim "United States"
                assert not op.lower().startswith("yes")
                assert not op.lower().startswith("no")
                assert "united states" not in op.lower()
                assert "answering yes or no" not in op.lower()
                recent_openings.append(op)
            reply = f"Simulated remote response for turn {idx + 1}: addressing '{utt[:30]}'"

        # Must NEVER emit the old hardcoded loop phrase
        assert reply != "Understood. I have noted that.", f"Turn {idx+1} ('{utt}') emitted banned repetitive phrase!"
        # Must NEVER repeat any previous reply in the entire 20-turn conversation
        assert reply.lower() not in [r.lower() for r in all_replies], (
            f"Turn {idx+1} ('{utt}') repeated an earlier reply: '{reply}'"
        )

        all_replies.append(reply)
        recent_dialogue.append({"role": "user", "content": utt})
        recent_dialogue.append({"role": "assistant", "content": reply})


@pytest.mark.asyncio
async def test_assistant_run_outer_loop_and_clean_shutdown(tmp_path):
    """Exercise Hal9000Assistant.run() outer loop end-to-end:
    - Verifies TaskExecutionStatus and TaskDeliveryStatus are in global scope during run()
    - Verifies active_task check in run() does not raise NameError
    - Verifies MemoryStore/QdrantClient closes cleanly without shutdown tracebacks
    """
    import struct
    from hal9000_inside.assistant import Hal9000Assistant
    from hal9000_inside.memory import MemoryStore
    from myna.core import TranscriptionDone

    class FakeEmbedder:
        def embed_passage(self, text: str):
            return [0.1] * 384
        def embed_query(self, text: str):
            return [0.1] * 384

    class FakeAudioSource:
        def __init__(self):
            self.stopped = False

        async def stream_frames(self, frame_samples: int):
            # Yield 25 frames of loud speech (~750ms) then 35 frames of silence (~1050ms)
            speech_frame = struct.pack(f"<{frame_samples}h", *([8000] * frame_samples))
            silence_frame = struct.pack(f"<{frame_samples}h", *([0] * frame_samples))
            for _ in range(25):
                yield speech_frame
            for _ in range(35):
                yield silence_frame

        def purge_buffered(self):
            pass

        def stop(self):
            self.stopped = True

    class FakeAudioSink:
        def __init__(self):
            self.played = []
            self.stopped = False
        def play_pcm(self, pcm: bytes):
            self.played.append(pcm)
        def stop(self):
            self.stopped = True

    class FakeTTS:
        def load(self):
            pass
        def cancel(self):
            pass
        async def speak_sentences(self, text: str, on_audio_chunk=None):
            if on_audio_chunk:
                on_audio_chunk(b"\x00\x00" * 1600)

    class FakeAdapter:
        async def _load_model(self):
            pass
        async def run_session(self, cfg, audio_gen, on_emit):
            async for _ in audio_gen:
                pass
            await on_emit(TranscriptionDone(text="Research the latest depth records in Cancun."))

    class FakeLLM:
        model = "fake-remote-model"
        is_local = False
        is_configured = True
        async def generate_response(self, messages, tools=None, return_dict=False, max_tokens=1024):
            reply = "The deepest known underwater cave system in Quintana Roo is Sistema Ox Bel Ha."
            if return_dict:
                return {"role": "assistant", "content": reply}
            return reply

    mem_store = MemoryStore(data_dir=str(tmp_path / "mem"), embedder=FakeEmbedder())
    source = FakeAudioSource()
    sink = FakeAudioSink()

    assistant = Hal9000Assistant(
        audio_source=source,
        audio_sink=sink,
        memory=mem_store,
        llm=FakeLLM(),
        adapter=FakeAdapter(),
        tts=FakeTTS(),
        combine_mode=True,
        sleep_after_minutes=10.0,
    )
    # Prevent actual MCP subprocess spawning during unit test
    assistant.mcp.connect_all = lambda: asyncio.sleep(0)
    assistant.mcp.close = lambda: asyncio.sleep(0)

    await assistant.run()

    assert source.stopped is True
    assert sink.stopped is True
    # Verify memory store Qdrant client was cleanly closed in run()'s finally block
    assert mem_store.qdrant is None


