import unittest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from hal9000_inside.stt import MynaSTTClient

class TestSTTContract(unittest.TestCase):
    def test_client_handshake_and_events(self):
        client = MynaSTTClient(socket_path="/tmp/mock_myna.sock")
        
        # Test event dispatching
        provisional_events = []
        committed_events = []

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            json.dumps({"type": "session.created", "session_id": "test-session-123"}),
            json.dumps({"type": "transcription.provisional", "text": "hello"}),
            json.dumps({"type": "transcription.committed", "text": "hello world"}),
            json.dumps({"type": "session.finished"}),
        ])
        
        client._ws = mock_ws
        client._active_utterance = True

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(client.commit_utterance())
            self.assertEqual(res, "hello world")
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
