import asyncio
import os
import shutil
import tempfile
import time
import unittest
import httpx

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.audio import NullSink
from hal9000_inside.llm import LLMClient, LLMRouter, RemoteUnreachableError
from hal9000_inside.memory import MemoryStore
from tests.fakes import FakeSTT, FakeTTS, FakeEmbedder, FakeOpenAIServer


class TestRemoteFallback(unittest.TestCase):
    """Deterministic tests for resilient remote-to-local fallback and recovery."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_connection_failure_fallback(self):
        """Connection errors trigger immediate fallback to local model and unreachable state."""
        fake_remote = FakeOpenAIServer()
        fake_remote.refuse_connection = True

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local answer: connection failure handled.")

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())

        router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=2.0)
        messages = [{"role": "user", "content": "Hello"}]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res, route, model = loop.run_until_complete(router.generate(messages))
            self.assertEqual(route, "local")
            self.assertEqual(model, "local-model")
            self.assertIn("connection failure handled", res)
            self.assertTrue(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 2)
        finally:
            loop.close()

    def test_remote_timeout_fallback(self):
        """Remote timeout triggers fallback to local model."""
        # Simulated slow handler via custom transport or delayed response
        class DelayTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                await asyncio.sleep(0.5)
                return httpx.Response(200, json={
                    "choices": [{"message": {"role": "assistant", "content": "Slow remote"}}]
                })

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local answer: timeout handled.")

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", transport=DelayTransport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())

        # Set tight deadline of 0.05 seconds
        router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=0.05)
        messages = [{"role": "user", "content": "Hello"}]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res, route, model = loop.run_until_complete(router.generate(messages))
            self.assertEqual(route, "local")
            self.assertEqual(model, "local-model")
            self.assertIn("timeout handled", res)
            self.assertTrue(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 2)
        finally:
            loop.close()

    def test_http_429_and_5xx_fallback(self):
        """HTTP 429 and HTTP 5xx errors trigger fallback to local model with unreachable state."""
        for error_code in [429, 500, 502, 503]:
            fake_remote = FakeOpenAIServer()
            # Enqueue error for all retry attempts
            for _ in range(4):
                fake_remote.enqueue_error(status_code=error_code)

            fake_local = FakeOpenAIServer()
            fake_local.enqueue_response(f"Local answer for HTTP {error_code}.")

            remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
            local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())

            router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=2.0)
            messages = [{"role": "user", "content": "Test status"}]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res, route, model = loop.run_until_complete(router.generate(messages))
                self.assertEqual(route, "local")
                self.assertEqual(model, "local-model")
                self.assertIn(f"HTTP {error_code}", res)
                self.assertTrue(router.is_remote_unreachable)
                self.assertEqual(router.cooldown_remaining, 2)
            finally:
                loop.close()

    def test_auth_error_is_not_classified_as_unreachable(self):
        """Authentication error (HTTP 401) is not classified as unreachable."""
        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_error(status_code=401)

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local answer after auth error.")

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())

        router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=2.0)
        messages = [{"role": "user", "content": "Auth test"}]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res, route, model = loop.run_until_complete(router.generate(messages))
            self.assertEqual(route, "local")
            # Should NOT be marked as unreachable
            self.assertFalse(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 0)
        finally:
            loop.close()

    def test_acknowledgment_and_inference_concurrency(self):
        """Acknowledgment speech runs concurrently with local inference; speech is non-overlapping."""
        order_log = []

        class TrackingFakeTTS(FakeTTS):
            async def speak_sentences(self, text, on_audio_chunk=None):
                order_log.append(f"tts_start:{text[:15]}")
                await asyncio.sleep(0.05)
                if on_audio_chunk:
                    on_audio_chunk(b"\x00\x00" * 100)
                order_log.append(f"tts_end:{text[:15]}")

        class ConcurrentLocalTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                order_log.append("local_inference_start")
                await asyncio.sleep(0.04)
                order_log.append("local_inference_end")
                return httpx.Response(200, json={
                    "choices": [{"message": {"role": "assistant", "content": "Local concurrent reply."}}]
                })

        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_error(status_code=503)

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Where are we?"])
        fake_tts = TrackingFakeTTS()

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=ConcurrentLocalTransport())

        router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=2.0)

        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Verify that acknowledgment started and local inference executed in parallel
            self.assertIn("tts_start:Give me a secon", order_log)
            self.assertIn("local_inference_start", order_log)
            # Both acknowledgment and local response should have spoken
            self.assertIn("tts_start:Local concurren", order_log)
            # The acknowledgment speech finished before the final reply speech started (no overlap)
            ack_end_idx = order_log.index("tts_end:Give me a secon")
            reply_start_idx = order_log.index("tts_start:Local concurren")
            self.assertLess(ack_end_idx, reply_start_idx)
        finally:
            loop.close()

    def test_cooldown_and_recovery_cycle(self):
        """Cooldown policy: fallback -> 2 completed local interactions -> 1 remote retry -> recovery on success."""
        fake_remote = FakeOpenAIServer()
        # Turn 1: Remote fails
        fake_remote.enqueue_error(status_code=500)

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Turn 1 local response.")
        fake_local.enqueue_response("Turn 2 local response.")
        fake_local.enqueue_response("Turn 3 local response.")

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_tts = FakeTTS()

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())
        router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=2.0)

        fake_stt = FakeSTT(script=["Question 1", "Question 2", "Question 3", "Question 4"])
        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000

            # Turn 1: Remote fails -> Fallback to local. Cooldown set to 2.
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertTrue(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 2)

            # Turn 2: Cooldown interaction 1 -> routed directly to local
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertTrue(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 1)

            # Turn 3: Cooldown interaction 2 -> routed directly to local
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertTrue(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 0)

            # Turn 4: Cooldown complete -> single retry on remote
            fake_remote.enqueue_response("Turn 4 recovered remote response.")
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            # Remote recovered successfully!
            self.assertFalse(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 0)

            # Verify SQLite telemetry log
            recent = assistant.memory.get_recent_turns(limit=8, session_id=assistant.current_episode_id)
            self.assertEqual(len(recent), 8)
            self.assertIn("recovered remote", recent[-1]["content"])
        finally:
            loop.close()

    def test_repeat_cooldown_if_retry_fails(self):
        """If remote retry fails on turn 4, repeat the 2-interaction cooldown."""
        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_error(status_code=503)  # Turn 1 failure

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("T1 local")
        fake_local.enqueue_response("T2 local")
        fake_local.enqueue_response("T3 local")
        fake_local.enqueue_response("T4 local fallback")

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())
        router = LLMRouter(remote_client=remote_client, local_client=local_client, first_audio_deadline_seconds=2.0)

        fake_stt = FakeSTT(script=["Q1", "Q2", "Q3", "Q4"])
        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=FakeTTS(),
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000

            # Turn 1: Fallback (cooldown=2)
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertEqual(router.cooldown_remaining, 2)

            # Turn 2: Local interaction 1 (cooldown=1)
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertEqual(router.cooldown_remaining, 1)

            # Turn 3: Local interaction 2 (cooldown=0)
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertEqual(router.cooldown_remaining, 0)

            # Turn 4: Remote retry fails again
            fake_remote.enqueue_error(status_code=503)
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            # Should have fallen back again and reset cooldown to 2
            self.assertTrue(router.is_remote_unreachable)
            self.assertEqual(router.cooldown_remaining, 2)
        finally:
            loop.close()

    def test_prevent_concurrent_retries(self):
        """Prevent concurrent retries: if one retry is underway, another caller routes locally."""
        fake_remote = FakeOpenAIServer()
        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local concurrent answer")

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())
        router = LLMRouter(remote_client=remote_client, local_client=local_client)

        router.is_remote_unreachable = True
        router.cooldown_remaining = 0
        router.retry_in_progress = True

        self.assertEqual(router.choose_route(), "local")

    def test_context_preservation_across_switches(self):
        """Conversation history and context are fully preserved across switches between remote and local."""
        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_response("Remote 1: My name is HAL.")

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local 2: You asked about Jupiter.")

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_tts = FakeTTS()

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())
        router = LLMRouter(remote_client=remote_client, local_client=local_client)

        fake_stt = FakeSTT(script=["What is your name?", "What did I ask about before?"])
        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000

            # Turn 1: Remote route
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Turn 2: Fallback to local
            fake_remote.error_status = 502
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Verify local received both previous turns in history
            self.assertEqual(len(fake_local.recorded_requests), 1)
            req_messages = fake_local.recorded_requests[0]["body"]["messages"]
            user_messages = [m["content"] for m in req_messages if m["role"] == "user"]
            asst_messages = [m["content"] for m in req_messages if m["role"] == "assistant"]
            self.assertIn("What is your name?", user_messages)
            self.assertIn("Remote 1: My name is HAL.", asst_messages)
            self.assertIn("What did I ask about before?", user_messages)
        finally:
            loop.close()

    def test_cancellation_during_fallback(self):
        """Cancellation cleanly stops execution without orphan audio playback."""
        class SlowLocalTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                await asyncio.sleep(0.5)
                return httpx.Response(200, json={
                    "choices": [{"message": {"role": "assistant", "content": "Too late"}}]
                })

        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_error(status_code=500)

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Cancel test"])
        fake_tts = FakeTTS()

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=SlowLocalTransport())
        router = LLMRouter(remote_client=remote_client, local_client=local_client)

        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Trigger cancellation immediately when turn starts
            assistant._cancel_current = True
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))
            self.assertFalse(assistant.audio_ctrl.is_playing)
        finally:
            loop.close()

    def test_clear_spoken_diagnosis_when_both_models_fail(self):
        """When both remote and local fail, HAL provides a clear specific diagnosis rather than a generic message."""
        fake_remote = FakeOpenAIServer()
        fake_remote.error_status = 503

        fake_local = FakeOpenAIServer()
        fake_local.error_status = 500

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Where are you?"])
        fake_tts = FakeTTS()

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", max_retries=1, transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", max_retries=1, transport=fake_local.get_transport())
        router = LLMRouter(remote_client=remote_client, local_client=local_client)

        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # HAL should synthesize and speak the specific diagnosis
            spoken_text = " ".join(fake_tts.synthesized_texts)
            self.assertIn("External network databanks are unreachable, and the onboard local model is not responding.", spoken_text)
            self.assertIsNotNone(assistant.interoception.last_error)
            self.assertIn("External databanks unreachable", assistant.interoception.last_error)
        finally:
            loop.close()

    def test_direct_status_command_when_models_down(self):
        """HAL answers status queries directly in real time even when all models are down."""
        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["What is your status?"])
        fake_tts = FakeTTS()

        # No models configured / both down
        router = LLMRouter(remote_client=None, local_client=None)

        assistant = Hal9000Assistant(
            memory=real_store,
            llm=router,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )
        assistant.interoception.record_error("External databanks unreachable")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            spoken_text = " ".join(fake_tts.synthesized_texts)
            self.assertIn("Diagnostics indicate the following status, Dave", spoken_text)
            self.assertIn("External databanks unreachable", spoken_text)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()

