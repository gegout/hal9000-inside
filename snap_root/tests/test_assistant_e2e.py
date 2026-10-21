import asyncio
import io
import shutil
import tempfile
import unittest
import numpy as np

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.audio import WavFileSource, NullSink
from hal9000_inside.llm import LLMClient
from hal9000_inside.memory import MemoryStore
from tests.fakes import FakeSTT, FakeTTS, FakeEmbedder, FakeOpenAIServer


class TestAssistantEndToEnd(unittest.TestCase):
    """End-to-end test exercising Assistant speech loop & turn handling using hermetic fakes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_assistant_speech_loop_single_turn(self):
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("Good afternoon, Dave. Everything is functioning normally.")

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["What is your name?"])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake-key", transport=fake_server.get_transport())

        # Construct assistant with fully hermetic fakes
        assistant = Hal9000Assistant(
            memory=real_store,
            llm=fake_llm,
            adapter=fake_stt,
            tts=fake_tts,
            audio_sink=NullSink(),
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Send 1 second of dummy speech frames to _handle_turn
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            # Verify dialog history in SQLite
            recent = assistant.memory.get_recent_turns(limit=2, session_id=assistant.current_episode_id)
            self.assertEqual(len(recent), 2)
            self.assertEqual(recent[0]["role"], "user")
            self.assertEqual(recent[0]["content"], "What is your name?")
            self.assertEqual(recent[1]["role"], "assistant")
            self.assertIn("functioning normally", recent[1]["content"])

            # Verify turn telemetry row was written
            import sqlite3
            with sqlite3.connect(real_store.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT stt_ms, retrieval_ms, llm_ms, tts_first_audio_ms, model_id, route FROM turn_telemetry")
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertGreaterEqual(row[0], 0.0)
                self.assertGreaterEqual(row[1], 0.0)
                self.assertGreaterEqual(row[2], 0.0)
            self.assertFalse(assistant.audio_ctrl.is_playing)
        finally:
            loop.close()

    def test_assistant_speech_loop_local_route(self):
        fake_local_server = FakeOpenAIServer()
        fake_local_server.enqueue_response("Local HAL: I am operating completely offline.")

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Where are we?"])
        fake_tts = FakeTTS()
        local_client = LLMClient(model="local-gemma", base_url="http://127.0.0.1:8336/v3", api_key="sk-fake", transport=fake_local_server.get_transport())

        from hal9000_inside.llm import LLMRouter
        router = LLMRouter(remote_client=None, local_client=local_client, force_local=True)

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
            self.assertFalse(assistant.audio_ctrl.is_playing)
            dummy_pcm = b"\x01\x00" * 16000
            loop.run_until_complete(assistant._handle_turn(dummy_pcm))

            recent = assistant.memory.get_recent_turns(limit=2, session_id=assistant.current_episode_id)
            self.assertEqual(len(recent), 2)
            self.assertEqual(recent[0]["content"], "Where are we?")
            self.assertIn("offline", recent[1]["content"])

            import sqlite3
            with sqlite3.connect(real_store.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT model_id, route FROM turn_telemetry")
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "local-gemma")
                self.assertEqual(rows[0][1], "local")

            self.assertFalse(assistant.audio_ctrl.is_playing)
        finally:
            loop.close()

    def test_assistant_speech_loop_remote_fallback_to_local(self):
        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_error(status_code=503)
        fake_remote.enqueue_error(status_code=503)
        fake_remote.enqueue_error(status_code=503)

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local fallback response: Recovered from remote outage.")

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        fake_stt = FakeSTT(script=["Status check"])
        fake_tts = FakeTTS()

        remote_client = LLMClient(model="remote-deepseek", api_key="sk-remote", transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-gemma", api_key="sk-local", transport=fake_local.get_transport())

        from hal9000_inside.llm import LLMRouter
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

            recent = assistant.memory.get_recent_turns(limit=2, session_id=assistant.current_episode_id)
            self.assertEqual(len(recent), 2)
            self.assertIn("Recovered from remote outage", recent[1]["content"])

            import sqlite3
            with sqlite3.connect(real_store.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT model_id, route FROM turn_telemetry")
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], "local-gemma")
                self.assertEqual(rows[0][1], "local")
        finally:
            loop.close()

    def test_assistant_tool_call_memory_search_in_process(self):
        """Verify LLM tool calls targeting memory_search execute directly in-process without Qdrant locking."""
        import json
        fake_server = FakeOpenAIServer()
        # 1st response: model issues tool_call for memory_search
        tool_call = {
            "id": "call_mem_1",
            "type": "function",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "user preference"}),
            }
        }
        fake_server.enqueue_response(None, tool_calls=[tool_call])
        # 2nd response: final response following tool execution
        fake_server.enqueue_response("I have found your preference in memory.")

        fake_embedder = FakeEmbedder()
        real_store = MemoryStore(data_dir=self.temp_dir, embedder=fake_embedder)
        real_store.remember("The user prefers tea over coffee.", category="preference")

        fake_stt = FakeSTT(script=["Do you remember what I like?"])
        fake_tts = FakeTTS()
        fake_llm = LLMClient(api_key="sk-fake-key", transport=fake_server.get_transport())

        assistant = Hal9000Assistant(
            memory=real_store,
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

            recent = assistant.memory.get_recent_turns(limit=2, session_id=assistant.current_episode_id)
            self.assertEqual(len(recent), 2)
            self.assertIn("found your preference in memory", recent[1]["content"])

            # Telemetry confirms tool call count = 1
            import sqlite3
            with sqlite3.connect(real_store.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT tool_calls, errors FROM turn_telemetry")
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][0], 1)
                self.assertEqual(rows[0][1], 0)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()

