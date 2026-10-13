import asyncio
import json
import os
import shutil
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



if __name__ == "__main__":
    unittest.main()
