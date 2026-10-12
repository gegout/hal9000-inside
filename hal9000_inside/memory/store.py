"""Conversational memory store integrating SQLite for dialogue history and Qdrant for semantic search.

Supports:
- timestamped dialogue in SQLite
- long-term facts, preferences, relationships in embedded Qdrant
- inspect, remember, correct, forget, and clear commands
- French/English cross-language recall
"""

import datetime
import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embeddings import MultilingualE5Embedder

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.environ.get("SNAP_USER_COMMON") or os.path.expanduser("~/.local/share/hal9000-inside")
COLLECTION_NAME = "hal9000_memories"
VECTOR_DIM = 384


class MemoryStore:
    """Embedded conversational memory using local SQLite and local Qdrant."""

    def __init__(self, data_dir: Optional[str] = None, embedder: Optional[MultilingualE5Embedder] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True, mode=0o700)
        
        self.sqlite_path = os.path.join(self.data_dir, "dialogue_history.db")
        self.qdrant_path = os.path.join(self.data_dir, "qdrant_storage")
        
        self.embedder = embedder or MultilingualE5Embedder()
        self._init_sqlite()
        self._init_qdrant()

    def _init_sqlite(self):
        """Initialise SQLite tables with restrictive permissions."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    text TEXT NOT NULL,
                    category TEXT,
                    source_turn_id TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)
            conn.commit()

    def _init_qdrant(self):
        """Initialise embedded Qdrant client."""
        self.qdrant = QdrantClient(path=self.qdrant_path)
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection {COLLECTION_NAME} (size={VECTOR_DIM}, cosine)")

    def record_turn(self, role: str, content: str, session_id: str = "default") -> str:
        """Record a single conversational turn in SQLite."""
        turn_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO turns (id, timestamp, role, content, session_id) VALUES (?, ?, ?, ?, ?)",
                (turn_id, ts, role, content, session_id),
            )
            conn.commit()
        return turn_id

    def get_recent_turns(self, limit: int = 6, session_id: str = "default") -> List[Dict[str, str]]:
        """Retrieve recent dialogue turns for follow-up and pronoun resolution."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cursor.fetchall()
            # Return in chronological order
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def remember(self, text: str, category: str = "fact", source_turn_id: Optional[str] = None) -> str:
        """Store a long-term fact or preference with passage embedding."""
        mem_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # 1. Compute 384-d normalized vector with 'passage: ' prefix
        vec = self.embedder.encode_passage(text)

        # 2. Store in Qdrant
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=mem_id,
                    vector=vec,
                    payload={"text": text, "category": category, "created_at": ts, "is_active": True},
                )
            ],
        )

        # 3. Store in SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO memories (id, created_at, updated_at, text, category, source_turn_id, is_active) VALUES (?, ?, ?, ?, ?, ?, 1)",
                (mem_id, ts, ts, text, category, source_turn_id),
            )
            conn.commit()
        return mem_id

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Semantic retrieval using query embedding."""
        query_vec = self.embedder.encode_query(query)
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=limit,
            score_threshold=score_threshold,
        ).points
        
        memories = []
        for p in results:
            if p.payload and p.payload.get("is_active", True):
                memories.append({
                    "id": str(p.id),
                    "text": p.payload.get("text", ""),
                    "category": p.payload.get("category", ""),
                    "score": p.score,
                })
        return memories

    def correct(self, mem_id: str, new_text: str) -> bool:
        """Update/supersede an existing memory."""
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        vec = self.embedder.encode_passage(new_text)

        # Update Qdrant
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=mem_id,
                    vector=vec,
                    payload={"text": new_text, "updated_at": ts, "is_active": True},
                )
            ],
        )

        # Update SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "UPDATE memories SET text = ?, updated_at = ? WHERE id = ?",
                (new_text, ts, mem_id),
            )
            conn.commit()
        return True

    def forget(self, mem_id: str) -> bool:
        """Completely delete/purge a memory from Qdrant and SQLite so it never returns."""
        # 1. Delete from Qdrant
        self.qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[mem_id],
        )

        # 2. Mark inactive or delete from SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            conn.commit()
        return True

    def clear(self):
        """Purge all stored memories and dialogue history."""
        self.qdrant.delete_collection(collection_name=COLLECTION_NAME)
        self.qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM turns")
            conn.commit()

    def inspect_all(self) -> List[Dict[str, Any]]:
        """List all active memories."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, category, created_at, updated_at FROM memories WHERE is_active = 1")
            rows = cursor.fetchall()
            return [
                {"id": r[0], "text": r[1], "category": r[2], "created_at": r[3], "updated_at": r[4]}
                for r in rows
            ]
