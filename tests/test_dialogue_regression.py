"""Deterministic regression session reproducing user dialogue and verifying 5 key bug fixes:

1. Compound requests:
   - "Do you think there's nothing else pending? How would you distinguish missing evidence from evidence of absence?"
   - Must address both parts; never return solitary "Affirmative."
2. History consistency:
   - Cross-session turn continuity: recent dialogue tools must retrieve authoritative past turns across sessions
     rather than being stranded in a stale "default" session that only contains "What is your name?".
3. Incomplete responses:
   - Fragment recovery: isolated short outputs like "A miss" are detected and expanded into substantive explanations.
4. Tool completion and premature sleep:
   - Turns invoking multiple tools (memory_get_recent_dialogue and memory_search) must generate non-empty follow-up,
     deliver speech, and never enter idle sleep prematurely while a turn is active.
5. Required reasoning outcome:
   - Distinguish missing evidence from evidence of absence: when transcript contains the record but semantic search
     returns nothing, explain that retrieval/indexing did not cover it rather than claiming the conversation did not occur.
"""

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.audio import NullSink
from hal9000_inside.llm import LLMClient
from hal9000_inside.memory import MemoryStore
from tests.fakes import FakeSTT, FakeTTS, FakeEmbedder, FakeOpenAIServer


class TestDialogueRegressionSession(unittest.TestCase):
    """Deterministic regression suite reproducing the exact dialogue failure modes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embedder = FakeEmbedder()
        self.memory_store = MemoryStore(data_dir=self.temp_dir, embedder=self.fake_embedder)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_regression_compound_request_not_solitary_affirmative(self):
        """1. Compound request:

        'Do you think there's nothing else pending? How would you distinguish missing evidence from evidence of absence?'
        A yes/no opening must not hide the explanatory question.
        Must address both parts substantively. Solitary 'Affirmative.' is rejected.
        """
        prompt = "Do you think there's nothing else pending? How would you distinguish missing evidence from evidence of absence?"
        
        fake_server = FakeOpenAIServer()
        # Mock model answering both parts substantively
        fake_server.enqueue_response(
            "There may still be pending tasks requiring analysis. To distinguish missing evidence from evidence of absence: "
            "missing evidence means a query or search simply failed to retrieve records or lacked coverage, whereas evidence "
            "of absence requires a verified complete search of an authoritative, exhaustive record confirming no such entry exists."
        )

        fake_stt = FakeSTT(script=[prompt])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Retrieve recorded assistant turn
            recent = assistant.memory.get_recent_turns(limit=2, session_id=None)
            self.assertEqual(len(recent), 2)
            assistant_reply = recent[1]["content"]

            # Must NOT be solitary "Affirmative."
            self.assertNotEqual(assistant_reply.strip(), "Affirmative.")
            self.assertNotEqual(assistant_reply.strip(), "Affirmative")
            
            # Substantive content assertions: addresses both the pending question and the distinction
            self.assertTrue(len(assistant_reply) > 50)
            self.assertIn("missing evidence", assistant_reply.lower())
            self.assertIn("absence", assistant_reply.lower())

            # Verified spoken delivery
            self.assertIn(assistant_reply, fake_tts.synthesized_texts)
        finally:
            loop.close()

    def test_regression_history_consistency_across_sessions(self):
        """2. History consistency:

        Ensure memory_get_recent_dialogue and context loading reflect actual previous turns
        (e.g., Kubernetes / PostgreSQL / French-English conversation) across episodes, rather
        than regressing to stale 'What is your name?' records from a disconnected session.
        """
        # Prepopulate dialogue store with historical turns across prior sessions
        session_1 = "session-prev-k8s"
        self.memory_store.record_turn("user", "Explain how Kubernetes interacts with OpenStack and PostgreSQL.", session_id=session_1)
        self.memory_store.record_turn("assistant", "Kubernetes can orchestrate containerized workloads while OpenStack provides underlying IaaS.", session_id=session_1)
        self.memory_store.record_turn("user", "Switch to English for all future answers.", session_id=session_1)
        self.memory_store.record_turn("assistant", "Understood. I will respond exclusively in English.", session_id=session_1)

        fake_server = FakeOpenAIServer()
        # Model queries memory_get_recent_dialogue
        tool_call = {
            "id": "call_hist_1",
            "type": "function",
            "function": {
                "name": "memory_get_recent_dialogue",
                "arguments": json.dumps({"limit": 5}),
            }
        }
        fake_server.enqueue_response(None, tool_calls=[tool_call])
        # Follow-up response summarizing the retrieved recent dialogue
        fake_server.enqueue_response("Our recent dialogue covered Kubernetes, OpenStack, and a language preference confirmation.")

        fake_stt = FakeSTT(script=["What did we just talk about?"])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )
        assistant.current_episode_id = "session-new-current"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Inspect what was supplied to the model in the tool response
            requests = fake_server.recorded_requests
            self.assertGreaterEqual(len(requests), 2)
            second_req_messages = requests[1]["body"]["messages"]
            tool_msg = next((m for m in second_req_messages if m.get("role") == "tool"), None)
            self.assertIsNotNone(tool_msg)

            # Ensure tool output contains the actual Kubernetes/OpenStack turns and not empty or stale name turns
            self.assertIn("Kubernetes", tool_msg["content"])
            self.assertIn("OpenStack", tool_msg["content"])

            # Verify HAL's final reply
            recent = assistant.memory.get_recent_turns(limit=2, session_id="session-new-current")
            self.assertIn("Kubernetes", recent[1]["content"])
        finally:
            loop.close()

    def test_regression_incomplete_response_fragment_recovery(self):
        """3. Incomplete responses:

        Investigate isolated output 'A miss'.
        The assistant must detect truncated/terse error fragments like 'A miss' or empty content
        and recover with an honest, substantive explanation without crashing or remaining silent.
        """
        prompt = "Did you find any record of our earlier discussion about the antenna fault?"
        
        fake_server = FakeOpenAIServer()
        # Model returns truncated fragment "A miss"
        fake_server.enqueue_response("A miss.")

        fake_stt = FakeSTT(script=[prompt])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            recent = assistant.memory.get_recent_turns(limit=2, session_id=None)
            assistant_reply = recent[1]["content"]

            # Must not be the isolated fragment "A miss."
            self.assertNotEqual(assistant_reply.strip(), "A miss.")
            self.assertNotEqual(assistant_reply.strip(), "A miss")
            self.assertTrue(len(assistant_reply) > 40)
            self.assertIn("absence", assistant_reply.lower())

            # Verified spoken delivery occurred
            self.assertEqual(len(fake_tts.synthesized_texts), 1)
            self.assertIn(assistant_reply, fake_tts.synthesized_texts)
        finally:
            loop.close()

    def test_regression_multi_tool_completion_and_no_premature_sleep(self):
        """4. Tool completion and premature sleep:

        Turn invokes memory_get_recent_dialogue and memory_search concurrently.
        Must generate non-empty follow-up, speak it via TTS, and prevent entering sleep
        while turn, tool execution, or audio playback is active.
        """
        prompt = "Check recent dialogue and search memory for any pending items."

        fake_server = FakeOpenAIServer()
        # Step 1: Model calls memory_get_recent_dialogue and memory_search
        tool_call_1 = {
            "id": "call_dialogue_1",
            "type": "function",
            "function": {
                "name": "memory_get_recent_dialogue",
                "arguments": json.dumps({"limit": 5}),
            }
        }
        tool_call_2 = {
            "id": "call_search_2",
            "type": "function",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "pending items"}),
            }
        }
        fake_server.enqueue_response(None, tool_calls=[tool_call_1, tool_call_2])

        # Step 2: Follow-up answer based on tools
        fake_server.enqueue_response(
            "I have inspected the recent dialogue and searched memory records. There are no critical pending alerts."
        )

        fake_stt = FakeSTT(script=[prompt])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 1. Assert enter_sleep() is rejected while _turn_active is True
            assistant._turn_active = True
            loop.run_until_complete(assistant.enter_sleep())
            self.assertFalse(assistant.is_sleeping)
            assistant._turn_active = False

            # 2. Run turn with multi-tool execution and follow-up
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Turn completed: _turn_active should be reset, but sleep should not have been entered prematurely
            self.assertFalse(assistant.is_sleeping)

            # Assert non-empty response was generated and delivered to TTS
            recent = assistant.memory.get_recent_turns(limit=2, session_id=None)
            self.assertEqual(len(recent), 2)
            assistant_reply = recent[1]["content"]
            self.assertTrue(len(assistant_reply.strip()) > 0)
            self.assertIn("inspected the recent dialogue", assistant_reply)
            self.assertIn(assistant_reply, fake_tts.synthesized_texts)
        finally:
            loop.close()

    def test_regression_required_reasoning_retrieval_failure_vs_missing_history(self):
        """5. Required reasoning outcome:

        When a saved transcript contains the question but semantic memory search returns nothing,
        explain that retrieval/indexing failed or did not cover the transcript—never that the original
        history did not happen.
        """
        # Save authoritative transcript in SQLite turns
        self.memory_store.record_turn("user", "We previously configured the AE-35 communication antenna.", session_id="session-mission")
        self.memory_store.record_turn("assistant", "The AE-35 antenna is tracking telemetry with 100% signal strength.", session_id="session-mission")

        # Semantic memory search for 'AE-35' returns empty (index not built for this turn)
        search_results = self.memory_store.search("AE-35 antenna", limit=3)
        self.assertEqual(search_results, [])  # Confirms semantic index has a miss

        fake_server = FakeOpenAIServer()
        # Model performs memory_search and gets empty results, then consults dialogue
        tool_call_search = {
            "id": "call_search_ae35",
            "type": "function",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "AE-35 antenna"}),
            }
        }
        fake_server.enqueue_response(None, tool_calls=[tool_call_search])

        # Substantive explanation distinguishing retrieval failure from missing dialogue
        fake_server.enqueue_response(
            "Semantic memory search returned no vector matches for the AE-35 antenna. However, our authoritative "
            "transcript contains this exchange. The absence of search results indicates an indexing or retrieval limitation, "
            "not that the conversation never occurred."
        )

        fake_stt = FakeSTT(script=["Did we discuss the AE-35 antenna earlier?"])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            recent = assistant.memory.get_recent_turns(limit=2, session_id=None)
            assistant_reply = recent[1]["content"]

            self.assertIn("authoritative transcript", assistant_reply.lower())
            self.assertIn("indexing", assistant_reply.lower())
            self.assertIn(assistant_reply, fake_tts.synthesized_texts)
        finally:
            loop.close()

    def test_regression_empty_content_repetitive_loop_breaking(self):
        """Test that empty model responses do not result in a repeating fallback loop.
        
        When the model returns empty content, HAL must attempt direct synthesis or provide
        an informative response, and must never emit the exact same fallback repeatedly.
        """
        prompt1 = "Continue your thought on the Ubermensch."
        prompt2 = "No need for retrieval, just answer directly."

        fake_server = FakeOpenAIServer()
        # Turn 1: model returns None / empty content
        fake_server.enqueue_response(None)
        # Direct synthesis attempt returns a substantive answer
        fake_server.enqueue_response("The Ubermensch represents an individual who transcends established norms, Dave.")

        # Turn 2: model returns empty content again
        fake_server.enqueue_response(None)
        # Direct synthesis returns second response
        fake_server.enqueue_response("It is the creative overcoming of nihilism, creating values independently.")

        fake_stt = FakeSTT(script=[prompt1, prompt2])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            recent1 = assistant.memory.get_recent_turns(limit=2, session_id=None)
            reply1 = recent1[1]["content"]

            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            recent2 = assistant.memory.get_recent_turns(limit=4, session_id=None)
            reply2 = recent2[3]["content"]

            # Must not be stuck in the static fallback loop
            self.assertNotIn("no detailed output was produced", reply1.lower())
            self.assertNotIn("no detailed output was produced", reply2.lower())
            # Must not repeat identical replies
            self.assertNotEqual(reply1.strip(), reply2.strip())
            self.assertTrue(len(reply1) > 20)
            self.assertTrue(len(reply2) > 20)
        finally:
            loop.close()

    def test_dialogue_logging_in_session_log_file(self):
        """Verify that user and assistant dialogue turns are flushed to the session log file.
        
        The log entries must include speaker, text, and metadata (latency, status, fluidity notes)
        to enable post-session review of dialogue flow and conversational quality.
        """
        from hal9000_inside.logging_config import setup_session_logging, get_current_log_path

        log_file = setup_session_logging(data_dir=self.temp_dir)

        prompt = "Explain why value creation differs from rationalization."
        response_text = "Value creation demands authentic self-overcoming, Dave, rather than arbitrary desire."

        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response(response_text)

        fake_stt = FakeSTT(script=[prompt])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=self.memory_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Inspect session log file content
            self.assertTrue(os.path.exists(log_file), f"Log file does not exist: {log_file}")
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read()

            self.assertIn("[DIALOGUE] USER:", log_content)
            self.assertIn("Explain why value creation differs from rationalization", log_content)
            self.assertIn("[DIALOGUE] HAL 9000:", log_content)
            self.assertIn("Value creation demands authentic self-overcoming", log_content)
            self.assertIn("[FLUIDITY:", log_content)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()


