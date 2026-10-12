import asyncio
import json
import shutil
import tempfile
import unittest

from hal9000_inside.cognition import (
    AttentionTracker,
    BackgroundJob,
    CognitiveScheduler,
    CuriosityEngine,
    PredictionEngine,
    TurnDigestAnalyzer,
)
from hal9000_inside.llm import LLMClient
from hal9000_inside.memory import MemoryStore
from tests.fakes import FakeClock, FakeEmbedder, FakeOpenAIServer


class TestCognitionPhase3(unittest.TestCase):
    """Hermetic unit tests for Phase 3: Scheduler, Digest, Curiosity, Attention, Prediction."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scheduler_priority_and_preemption(self):
        fake_clock = FakeClock(initial_time=100.0)
        scheduler = CognitiveScheduler(max_concurrent=2, clock=fake_clock)

        executed_jobs = []

        async def fast_job():
            executed_jobs.append("fast")

        long_job_started = asyncio.Event()

        async def long_job():
            executed_jobs.append("long_started")
            long_job_started.set()
            await asyncio.sleep(10.0)
            executed_jobs.append("long_finished")

        # Enqueue jobs
        scheduler.enqueue(BackgroundJob("j1", "test", long_job, priority=5))
        scheduler.enqueue(BackgroundJob("j2", "test", fast_job, priority=10))

        loop = asyncio.new_event_loop()
        try:
            async def run_scenario():
                # j2 has higher priority, so it runs first along with j1
                started = await scheduler.run_once()
                self.assertEqual(started, 2)
                await long_job_started.wait()
                # Simulate user speech start (VAD trigger)
                scheduler.on_speech_start()
                # Yield control to allow cancellation to propagate
                await asyncio.sleep(0.01)

            loop.run_until_complete(run_scenario())
            self.assertIn("fast", executed_jobs)
            self.assertIn("long_started", executed_jobs)
            self.assertNotIn("long_finished", executed_jobs)
            self.assertEqual(len(scheduler.active_tasks), 0)
        finally:
            loop.close()

    def test_turn_digest_extraction(self):
        fake_server = FakeOpenAIServer()
        digest_payload = {
            "facts_to_remember": [
                {"text": "The user's favorite celestial body is Europa.", "confidence": 0.95, "salience": 0.8, "category": "fact"}
            ],
            "open_questions": ["What is the composition of Europa's subsurface ocean?"],
            "focus_topics": ["Europa", "Jupiter", "Astrobiology"],
            "predicted_followups": ["Tell me more about Europa's ice shell."],
            "user_corrections": [],
            "answer_confidence": 0.98,
        }
        fake_server.enqueue_response(json.dumps(digest_payload))

        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())
        analyzer = TurnDigestAnalyzer(llm_client=fake_llm)

        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(
                analyzer.analyze_turn("I am fascinated by Europa.", "Europa has a subsurface ocean.")
            )
            self.assertIsNotNone(res)
            self.assertEqual(len(res.facts_to_remember), 1)
            self.assertEqual(res.facts_to_remember[0].text, "The user's favorite celestial body is Europa.")
            self.assertEqual(res.open_questions[0], "What is the composition of Europa's subsurface ocean?")
            self.assertEqual(res.focus_topics, ["Europa", "Jupiter", "Astrobiology"])
            self.assertEqual(res.predicted_followups, ["Tell me more about Europa's ice shell."])
        finally:
            loop.close()

    def test_attention_and_prediction(self):
        tracker = AttentionTracker(decay_rate=0.5, max_focus_topics=3)
        tracker.update_from_turn(["Jupiter", "Europa"])
        self.assertIn("jupiter", tracker.get_active_focus())
        self.assertIn("europa", tracker.get_active_focus())

        boost = tracker.get_salience_boost("Tell me about Jupiter and the moons")
        self.assertGreater(boost, 0.0)

        # Update next turn with different topic
        tracker.update_from_turn(["Mars"])
        self.assertIn("mars", tracker.get_active_focus())

        # Prediction engine
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        store.remember("Europa is covered with water ice.", category="astronomy")

        prediction = PredictionEngine(memory_store=store)
        prediction.set_predictions(["What is Europa's crust made of?"])

        loop = asyncio.new_event_loop()
        try:
            matches = loop.run_until_complete(prediction.prefetch_candidates())
            self.assertGreaterEqual(len(matches), 1)
        finally:
            loop.close()

    def test_curiosity_exploration_and_memory(self):
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("Europa is thought to have a liquid water ocean under a thick ice shell.")

        fake_llm = LLMClient(api_key="sk-test", transport=fake_server.get_transport())
        fake_embedder = FakeEmbedder()
        store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)

        curiosity = CuriosityEngine(memory_store=store, llm_client=fake_llm)
        curiosity.enqueue_questions(["Does Europa have liquid water?"])
        self.assertEqual(len(curiosity.get_pending()), 1)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(curiosity.explore_next())
            self.assertIsNotNone(result)
            self.assertEqual(result["question"], "Does Europa have liquid water?")
            self.assertEqual(len(curiosity.get_pending()), 0)

            # Check memory entry with provenance='curiosity'
            memories = store.search("liquid water", limit=1)
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0]["provenance"], "curiosity")
            self.assertIn("Europa", memories[0]["text"])
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
