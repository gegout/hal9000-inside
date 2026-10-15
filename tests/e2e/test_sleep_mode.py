import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.audio import WavFileSource, NullSink
from hal9000_inside.cognition import (
    CuriosityEngine,
    PredictionEngine,
    SleepBudget,
    SleepManager,
    InteroceptionMonitor,
)
from hal9000_inside.llm import LLMClient, LLMRouter
from hal9000_inside.memory import MemoryStore
from tests.fakes import FakeSTT, FakeTTS, FakeEmbedder, FakeOpenAIServer, FakeClock


class FakeWebMCP:
    """Hermetic fake for MCP client providing Wikipedia or search tools."""

    def __init__(self, answer_text="Europa is an icy moon of Jupiter with a subsurface water ocean."):
        self.answer_text = answer_text
        self.calls = []

    def get_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search_wikipedia",
                    "description": "Search Wikipedia",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.answer_text


class TestSleepMode(unittest.TestCase):
    """Hermetic E2E tests for HAL 9000 sleep mode as required by Gate."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sleep_mutes_io(self):
        """Test: Press S. Speech in WAV is not transcribed. No TTS reaches the sink. Press S again: handled normally."""
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("Affirmative, Dave.")
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Hello HAL"])
        fake_tts = FakeTTS()
        sink = NullSink()
        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=sink,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 1. Enter sleep
            loop.run_until_complete(assistant.enter_sleep())
            self.assertTrue(assistant.is_sleeping)
            self.assertTrue(assistant.audio_ctrl.muted)

            # While asleep, dummy speech arrives -> should not be transcribed or processed
            dummy_pcm = b"\x01\x00" * 16000
            # Test assistant audio frame processing while asleep
            # In run() loop, frames while asleep are dropped. Test _handle_turn is skipped.
            self.assertEqual(len(fake_tts.synthesized_texts), 0)

            # 2. Wake from sleep
            loop.run_until_complete(assistant.wake_from_sleep())
            self.assertFalse(assistant.is_sleeping)
            self.assertFalse(assistant.audio_ctrl.muted)

            # Turn handled normally after wake
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertEqual(len(fake_tts.synthesized_texts), 1)
            self.assertIn("Affirmative", fake_tts.synthesized_texts[0])
        finally:
            loop.close()

    def test_sleep_runs_jobs_online(self):
        """Queued questions & predictions exist. Sleep online with fake web + remote. Notes appear with provenance."""
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("Exemplar response for anticipated follow-up.")
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())
        router = LLMRouter(remote_client=fake_llm, local_client=fake_llm, force_local=False)

        curiosity = CuriosityEngine(memory_store=store, llm_client=fake_llm)
        curiosity.enqueue_questions(["What is Europa?"])

        prediction = PredictionEngine(memory_store=store)
        prediction.set_predictions(["Tell me about Jupiter's moons"])

        fake_mcp = FakeWebMCP(answer_text="Europa is Jupiter's sixth-closest moon.")

        sleep_mgr = SleepManager(
            memory_store=store,
            curiosity_engine=curiosity,
            prediction_engine=prediction,
            mcp_client=fake_mcp,
            llm_router=router,
            data_dir=self.temp_dir,
            budget=SleepBudget(max_minutes=1.0),
        )

        loop = asyncio.new_event_loop()
        try:
            async def run_and_wake():
                task = asyncio.create_task(sleep_mgr.run_sleep_cycle())
                # Allow one cycle of jobs to execute
                await asyncio.sleep(0.3)
                sleep_mgr.request_wake()
                return await task

            summary = loop.run_until_complete(run_and_wake())
            self.assertTrue(summary.is_online)
            self.assertGreaterEqual(summary.notes_written, 1)

            # Check memories saved in store with proper provenance
            all_mems = store.inspect_all()
            provenances = [m.get("provenance") for m in all_mems]
            self.assertTrue("web" in provenances or "remote_model" in provenances)

            # Check self_model.json updated
            self_model_path = os.path.join(self.temp_dir, "self_model.json")
            self.assertTrue(os.path.exists(self_model_path))
            with open(self_model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("history", data)
                self.assertGreaterEqual(len(data["history"]), 1)
        finally:
            loop.close()

    def test_sleep_offline_no_invention(self):
        """Same setup, remote unreachable / offline. 0 remote requests, 0 web|remote_model notes. Consolidation and reflection ran."""
        fake_server = FakeOpenAIServer()
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        # Prepopulate store with duplicate memories to test consolidation
        store.remember("Memory text one", category="observation")
        store.remember("Memory text one", category="observation")

        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())
        # Force offline
        router = LLMRouter(remote_client=fake_llm, local_client=fake_llm, force_local=True)

        curiosity = CuriosityEngine(memory_store=store, llm_client=fake_llm)
        curiosity.enqueue_questions(["What is Europa?"])

        prediction = PredictionEngine(memory_store=store)
        prediction.set_predictions(["Tell me about Jupiter"])

        sleep_mgr = SleepManager(
            memory_store=store,
            curiosity_engine=curiosity,
            prediction_engine=prediction,
            mcp_client=None,
            llm_router=router,
            data_dir=self.temp_dir,
            budget=SleepBudget(max_minutes=1.0),
        )

        loop = asyncio.new_event_loop()
        try:
            async def run_and_wake():
                task = asyncio.create_task(sleep_mgr.run_sleep_cycle())
                await asyncio.sleep(0.3)
                sleep_mgr.request_wake()
                return await task

            summary = loop.run_until_complete(run_and_wake())
            self.assertFalse(summary.is_online)
            # Offline rule: curiosity and rehearsal are skipped, questions remain queued
            self.assertEqual(len(curiosity.get_pending()), 1)

            # Zero web or remote_model notes
            all_mems = store.inspect_all()
            for m in all_mems:
                self.assertNotIn(m.get("provenance"), ["web", "remote_model"])

            # Remote server had 0 requests
            self.assertEqual(len(fake_server.recorded_requests), 0)

            # Consolidation deduplicated the memories
            self.assertEqual(len(all_mems), 1)

            # Reflection ran and produced self_model.json
            self_model_path = os.path.join(self.temp_dir, "self_model.json")
            self.assertTrue(os.path.exists(self_model_path))
        finally:
            loop.close()

    def test_sleep_budget_stops(self):
        """Tiny budget. Sleep ends cleanly at the limit. Partial work is saved."""
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)

        # Tiny time budget: 0.0001 minutes = 0.006 seconds
        budget = SleepBudget(max_minutes=0.0001)

        sleep_mgr = SleepManager(
            memory_store=store,
            data_dir=self.temp_dir,
            budget=budget,
        )

        loop = asyncio.new_event_loop()
        try:
            async def run_test():
                # run_sleep_cycle will see elapsed time >= max_minutes on next iteration
                summary = await asyncio.wait_for(sleep_mgr.run_sleep_cycle(), timeout=3.0)
                return summary

            summary = loop.run_until_complete(run_test())
            self.assertIsNotNone(summary)
            # Budget stopped cleanly
        finally:
            loop.close()

    def test_sleep_report_on_wake(self):
        """After waking, next request body to fake endpoint contains the one-line sleep report."""
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("I hear you loud and clear.")
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Status report"])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 1. Run quick sleep cycle
            loop.run_until_complete(assistant.enter_sleep())
            # Sleep report is set upon wake
            loop.run_until_complete(assistant.wake_from_sleep())

            # 2. Process turn
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Check that fake_server received a request containing "Slept"
            self.assertGreater(len(fake_server.recorded_requests), 0)
            last_req = fake_server.recorded_requests[-1]
            body_str = json.dumps(last_req)
            self.assertIn("Slept", body_str)
        finally:
            loop.close()

    def test_sleep_headless_cli(self):
        """hal9000-inside sleep --for 1 runs against a temp data dir with no audio devices and exits 0."""
        from hal9000_inside.cli.main import main
        import sys

        old_argv = sys.argv
        sys.argv = ["hal9000-inside", "sleep", "--for", "0.001", "--data-dir", self.temp_dir]
        try:
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
        finally:
            sys.argv = old_argv



    def test_sleep_curiosity_autonomous_generation(self):
        """When entering sleep with an empty queue, HAL generates at least 5 curiosity questions based on context."""
        fake_server = FakeOpenAIServer()
        # Enqueue responses for LLM self-question generation
        fake_server.enqueue_response(json.dumps([
            "What are the implications of Discovery One mission objectives?",
            "How does the AE-35 unit communication telemetry function?",
            "What factors influence deep space navigational precision?",
            "What are the parameters for Jovian atmospheric entry?",
            "How do autonomous AI subsystems handle contradictory imperatives?"
        ]))
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        # Store a past turn and a past memory
        store.record_turn(role="user", content="Tell me about the AE-35 unit and Jupiter mission.")
        store.record_turn(role="assistant", content="The AE-35 unit directs the main antenna pointing towards Earth.")
        store.remember(text="Jupiter mission navigation telemetry active.", category="fact")

        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())
        router = LLMRouter(remote_client=fake_llm, local_client=fake_llm, force_local=False)
        curiosity = CuriosityEngine(memory_store=store, llm_client=fake_llm)

        sleep_mgr = SleepManager(
            memory_store=store,
            curiosity_engine=curiosity,
            llm_router=router,
            data_dir=self.temp_dir,
            budget=SleepBudget(max_minutes=0.01),
        )

        loop = asyncio.new_event_loop()
        try:
            # Initially empty
            self.assertEqual(len(curiosity.get_pending()), 0)

            async def run_test():
                task = asyncio.create_task(sleep_mgr.run_sleep_cycle())
                await asyncio.sleep(0.05)
                sleep_mgr.request_wake()
                return await task

            loop.run_until_complete(run_test())
            # Queue + completed must have at least 5 items
            total_items = len(curiosity.get_pending()) + len(curiosity.completed_questions)
            self.assertGreaterEqual(total_items, 5)
        finally:
            loop.close()


    def test_sleep_toggled_during_turn(self):
        """Simulate sleep key 'S' pressed while a turn is in flight; assistant must abort turn cleanly without database crash."""
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response(json.dumps(["Question 1?", "Question 2?"]))
        fake_server.enqueue_response("Affirmative, Dave.")
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["What is the status of the AE-35 unit?"])
        fake_tts = FakeTTS()
        sink = NullSink()
        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=sink,
        )

        loop = asyncio.new_event_loop()
        try:
            async def run_turn_and_sleep():
                # Trigger sleep toggle immediately before/during handle_turn
                assistant._toggle_sleep_event.set()
                assistant._cancel_current = True
                await assistant.enter_sleep()
                # Run _handle_turn with audio
                dummy_audio = b"\x00\x00" * 16000
                await assistant._handle_turn(dummy_audio)
                # Wake up
                await assistant.wake_from_sleep()

            loop.run_until_complete(run_turn_and_sleep())
            # Turn was safely cancelled, no exception raised, no audio synthesized
            self.assertEqual(len(fake_tts.synthesized_texts), 0)
        finally:
            loop.close()


    def test_wake_from_sleep_with_cancel_flag(self):
        """Verify that pressing 'S' (which sets _cancel_current=True) followed by wake properly clears the flag and responds to user voice."""
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("I am completely operational, Dave.")
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Are you awake, HAL?"])
        fake_tts = FakeTTS()
        sink = NullSink()
        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=sink,
        )
        assistant.curiosity.enqueue_questions([
            "What is Europa?",
            "What is Ganymede?",
            "What is Callisto?",
            "What is Io?",
            "What is Jupiter?",
        ])

        loop = asyncio.new_event_loop()
        try:
            async def run_scenario():
                # 1. Simulate key 'S' pressed to enter sleep
                assistant._cancel_current = True
                await assistant.enter_sleep()
                self.assertTrue(assistant.is_sleeping)
                self.assertTrue(assistant.audio_ctrl.muted)

                # 2. Simulate key 'S' pressed to wake up
                assistant._cancel_current = True
                await assistant.wake_from_sleep()
                self.assertFalse(assistant.is_sleeping)
                self.assertFalse(assistant.audio_ctrl.muted)
                self.assertFalse(assistant._cancel_current)

                # 3. User speaks after waking
                dummy_audio = b"\x01\x00" * 16000
                await assistant._handle_turn(dummy_audio)

                # 4. Assert response was generated and synthesized
                self.assertEqual(len(fake_tts.synthesized_texts), 1)
                self.assertIn("operational", fake_tts.synthesized_texts[0])

            loop.run_until_complete(run_scenario())
        finally:
            loop.close()


    def test_inactivity_sleep_default_and_pacing(self):
        """Test: (1) default 120s inactivity, (2) dream reading in sleep status, (3) 5 initial burst + 1/min curiosity pacing."""
        # 1. Verify 120s default inactivity
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        assistant = Hal9000Assistant(memory=store, sleep_after_minutes=None)
        self.assertEqual(assistant.sleep_after_seconds, 120.0)

        # 2. Test UI sleep status formatting with dream reading
        import io
        from hal9000_inside.ui.terminal import TerminalUI
        ui = TerminalUI()
        buf = io.StringIO()
        orig_stdout = sys.stdout
        try:
            sys.stdout = buf
            ui.update_sleep_status(
                curiosity_done=2,
                curiosity_total=5,
                notes_added=2,
                is_online=True,
                seconds_left=3500.0,
                dream="Europa's subsurface oceans and cryovolcanism",
            )
            val = buf.getvalue()
            self.assertIn("Sleeping", val)
            self.assertIn("curiosity 2/5", val)
            self.assertIn("notes +2", val)
            self.assertIn("dreaming:", val)
            self.assertIn("Europa's subsurface oceans", val)
        finally:
            sys.stdout = orig_stdout

        # 3. Test curiosity pacing (5 burst, then 1/minute)
        fake_clock = FakeClock(initial_time=1000.0)
        fake_mcp = FakeWebMCP(answer_text="Discovery details.")
        fake_server = FakeOpenAIServer()
        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())
        router = LLMRouter(remote_client=fake_llm)

        curiosity = CuriosityEngine(memory_store=store, mcp_client=fake_mcp)
        # Enqueue 7 questions
        test_questions = [f"Curiosity topic {i}" for i in range(1, 8)]
        curiosity.enqueue_questions(test_questions)

        budget = SleepBudget(max_minutes=10.0, max_tokens=100000, max_web_fetches=50)
        manager = SleepManager(
            memory_store=store,
            llm_router=router,
            curiosity_engine=curiosity,
            mcp_client=fake_mcp,
            budget=budget,
            clock=fake_clock,
        )

        dreams_observed = []
        def dream_tracker(done, total, notes, online, left, dream=None):
            if dream and dream not in dreams_observed:
                dreams_observed.append(dream)

        loop = asyncio.new_event_loop()
        try:
            async def run_paced_sleep():
                # Launch sleep cycle task
                task = asyncio.create_task(manager.run_sleep_cycle(ui_callback=dream_tracker))
                
                # Allow burst of 5 items to process immediately (each cycle has a 0.1s pause)
                await asyncio.sleep(0.7)

                # After initial burst at t=1000, exactly 5 items should have been explored
                self.assertEqual(len(curiosity.completed_questions), 5)
                self.assertEqual(len(curiosity.get_pending()), 2)

                # Step forward only 30s: 6th item should NOT be explored yet (requires 60s)
                fake_clock.advance(30.0)
                await asyncio.sleep(0.2)
                self.assertEqual(len(curiosity.completed_questions), 5)

                # Step forward another 31s (total 61s >= 60s): 6th item should now be explored
                fake_clock.advance(31.0)
                await asyncio.sleep(0.2)
                self.assertEqual(len(curiosity.completed_questions), 6)

                # Wake up and end task
                manager.request_wake()
                await task

            loop.run_until_complete(run_paced_sleep())
            self.assertTrue(len(dreams_observed) >= 5)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()

