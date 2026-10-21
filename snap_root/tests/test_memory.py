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

    def test_schema_migration_v1_to_v2(self):
        # Create a legacy v1 sqlite db directly
        v1_dir = tempfile.mkdtemp()
        try:
            sqlite_path = os.path.join(v1_dir, "dialogue_history.db")
            import sqlite3
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute("""
                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        text TEXT NOT NULL,
                        category TEXT,
                        source_turn_id TEXT,
                        is_active INTEGER DEFAULT 1
                    )
                """)
                conn.execute("INSERT INTO memories VALUES ('mem-1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'Test fact', 'fact', NULL, 1)")
                conn.commit()

            # Now open MemoryStore pointing to v1_dir; it should auto-migrate to v2
            v2_store = MemoryStore(data_dir=v1_dir, embedder=self.store.embedder)
            with sqlite3.connect(sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(memories)")
                col_names = [r[1] for r in cursor.fetchall()]
                self.assertIn("provenance", col_names)
                self.assertIn("salience", col_names)
                self.assertIn("confidence", col_names)
                self.assertIn("access_count", col_names)
                
                cursor.execute("SELECT version FROM schema_version")
                ver = cursor.fetchone()[0]
                self.assertEqual(ver, 2)
        finally:
            shutil.rmtree(v1_dir, ignore_errors=True)

    def test_v2_provenance_and_access_tracking(self):
        mid = self.store.remember(
            "Europa mission destination",
            category="fact",
            provenance="web",
            salience=0.9,
            confidence=0.95
        )
        results = self.store.search("Europa", limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["provenance"], "web")
        self.assertEqual(results[0]["salience"], 0.9)
        
        # Verify access count incremented in SQLite
        import sqlite3
        with sqlite3.connect(self.store.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT access_count, last_accessed_at FROM memories WHERE id = ?", (mid,))
            row = cursor.fetchone()
            self.assertEqual(row[0], 1)
            self.assertIsNotNone(row[1])

if __name__ == "__main__":
    unittest.main()
