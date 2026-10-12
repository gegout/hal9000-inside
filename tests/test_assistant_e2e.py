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
            recent = assistant.memory.get_recent_turns(limit=2)
            self.assertEqual(len(recent), 2)
            self.assertEqual(recent[0]["role"], "user")
            self.assertEqual(recent[0]["content"], "What is your name?")
            self.assertEqual(recent[1]["role"], "assistant")
            self.assertIn("functioning normally", recent[1]["content"])
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
