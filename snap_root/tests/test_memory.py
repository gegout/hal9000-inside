import unittest
import os
import shutil
import tempfile
from hal9000_inside.memory import MemoryStore, MultilingualE5Embedder

class TestMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = MemoryStore(data_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_e5_embeddings_format(self):
        embedder = self.store.embedder
        # Verify prefixing and 384-d normalized vectors
        q_vec = embedder.encode_query("Where is the communication dish?")
        self.assertEqual(len(q_vec), 384)
        norm = sum(x**2 for x in q_vec)**0.5
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_store_recall_correct_forget_flow(self):
        # 1. Store a fact
        mid = self.store.remember("Dave prefers tea over coffee in the morning.", category="preference")
        self.assertTrue(len(mid) > 0)

        # 2. Recall through paraphrase
        results = self.store.search("What does Bowman like to drink when he wakes up?", limit=3)
        self.assertTrue(len(results) > 0)
        self.assertIn("tea over coffee", results[0]["text"])

        # 3. Cross-lingual recall (French query)
        fr_results = self.store.search("Que boit Dave le matin ?", limit=3)
        self.assertTrue(len(fr_results) > 0)
        self.assertIn("tea over coffee", fr_results[0]["text"])

        # 4. Correct the fact
        self.store.correct(mid, "Dave prefers black coffee in the morning.")
        updated_results = self.store.search("What does Bowman drink?", limit=3)
        self.assertTrue(len(updated_results) > 0)
        self.assertIn("black coffee", updated_results[0]["text"])

        # 5. Forget the fact
        self.store.forget(mid)
        empty_results = self.store.search("What does Bowman drink?", limit=3)
        self.assertEqual(len(empty_results), 0)

    def test_dialogue_history_sqlite(self):
        t1 = self.store.record_turn("user", "Hello HAL.", session_id="test-session")
        t2 = self.store.record_turn("assistant", "Good afternoon, Dave.", session_id="test-session")
        
        turns = self.store.get_recent_turns(limit=5, session_id="test-session")
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[0]["content"], "Hello HAL.")
        self.assertEqual(turns[1]["role"], "assistant")
        self.assertEqual(turns[1]["content"], "Good afternoon, Dave.")

if __name__ == "__main__":
    unittest.main()
