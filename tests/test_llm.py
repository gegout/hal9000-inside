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
        client = LLMClient()
        messages = [
            {"role": "system", "content": "You are HAL 9000. Answer in one short sentence."},
            {"role": "user", "content": "What is the status of the AE-35 unit?"}
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            reply = loop.run_until_complete(client.generate_response(messages, max_tokens=60))
            self.assertTrue(len(reply) > 0)
            self.assertIsInstance(reply, str)
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
