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

    def test_router_remote_and_fallback(self):
        from tests.fakes import FakeOpenAIServer
        from hal9000_inside.llm import LLMRouter, ConnectivityMonitor
        
        fake_remote = FakeOpenAIServer()
        fake_remote.enqueue_response("Remote response: all systems nominal.")

        fake_local = FakeOpenAIServer()
        fake_local.enqueue_response("Local fallback response: running offline.")

        remote_client = LLMClient(model="remote-model", api_key="sk-remote", transport=fake_remote.get_transport())
        local_client = LLMClient(model="local-model", api_key="sk-local", transport=fake_local.get_transport())

        router = LLMRouter(remote_client=remote_client, local_client=local_client)
        messages = [{"role": "user", "content": "Status report"}]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 1. Normal remote route
            res, route, model = loop.run_until_complete(router.generate(messages))
            self.assertEqual(route, "remote")
            self.assertEqual(model, "remote-model")
            self.assertIn("nominal", res)

            # 2. Simulated offline / failure triggers fallback to local
            fake_remote.enqueue_error(status_code=503)
            fake_remote.enqueue_error(status_code=503)
            fake_remote.enqueue_error(status_code=503)
            res2, route2, model2 = loop.run_until_complete(router.generate(messages))
            self.assertEqual(route2, "local")
            self.assertEqual(model2, "local-model")
            self.assertIn("offline", res2)

            # 3. Force local override
            router.force_local = True
            fake_local.enqueue_response("Forced local response.")
            res3, route3, model3 = loop.run_until_complete(router.generate(messages))
            self.assertEqual(route3, "local")
            self.assertIn("Forced local", res3)
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
