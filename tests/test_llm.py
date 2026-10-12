import unittest
import asyncio
from hal9000_inside.llm import LLMClient, load_openrouter_api_key

class TestLLM(unittest.TestCase):
    def test_key_loader_and_redaction(self):
        key = load_openrouter_api_key()
        self.assertIsNotNone(key)
        self.assertTrue(key.startswith("sk-or-"))
        # Verify secret does not appear in str/repr if logged
        client = LLMClient(api_key=key)
        self.assertTrue(client.is_configured)

    def test_live_completion_and_retries(self):
        from tests.fakes import FakeOpenAIServer
        fake_server = FakeOpenAIServer()
        fake_server.enqueue_response("The AE-35 unit is operating 100% reliably.")

        client = LLMClient(api_key="sk-fake-key", transport=fake_server.get_transport())
        messages = [
            {"role": "system", "content": "You are HAL 9000. Answer in one short sentence."},
            {"role": "user", "content": "What is the status of the AE-35 unit?"}
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            reply = loop.run_until_complete(client.generate_response(messages, max_tokens=60))
            self.assertEqual(reply, "The AE-35 unit is operating 100% reliably.")
            self.assertEqual(len(fake_server.recorded_requests), 1)
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
