"""Conversational memory store integrating SQLite for dialogue history and Qdrant for semantic search.

Schema v2 supports:
- timestamped dialogue in SQLite
- long-term facts, preferences, relationships in embedded Qdrant + SQLite
- schema migrations (v1 -> v2)
- provenance, model_id, confidence, salience, access counts, last_accessed_at, expires_at, episode_id, source_urls
- episodes table for structured session tracking
- turn_telemetry table for latency, routing, errors, token counts, surprise scoring
- multi-factor retrieval: cosine * salience * recency with access_count tracking
"""

import contextlib
import datetime
import json
import logging
import math
import os
import sqlite3
import uuid
from typing import Any, Dict, Generator, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embeddings import MultilingualE5Embedder

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.environ.get("SNAP_USER_COMMON") or os.path.expanduser("~/.local/share/hal9000-inside")
COLLECTION_NAME = "hal9000_memories"
VECTOR_DIM = 384
CURRENT_SCHEMA_VERSION = 2


class MemoryStore:
    """Embedded conversational memory using local SQLite and local Qdrant (Schema v2)."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        embedder: Optional[MultilingualE5Embedder] = None,
        salience_weight: float = 0.2,
        recency_weight: float = 0.1,
    ):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True, mode=0o700)

        self.sqlite_path = os.path.join(self.data_dir, "dialogue_history.db")
        self.qdrant_path = os.path.join(self.data_dir, "qdrant_storage")

        self.salience_weight = salience_weight
        self.recency_weight = recency_weight

        self.embedder = embedder or MultilingualE5Embedder()
        self._init_sqlite()
        self._init_qdrant()

    @contextlib.contextmanager
    def _connect_sqlite(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a SQLite connection with timeout and guaranteed close to prevent FD leaks."""
        os.makedirs(self.data_dir, exist_ok=True, mode=0o700)
        conn = sqlite3.connect(self.sqlite_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _init_sqlite(self):
        """Initialise SQLite tables and run migrations if needed."""
        with self._connect_sqlite() as conn:
            cursor = conn.cursor()

            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    migrated_at TEXT NOT NULL
                )
            """)

            # Core tables
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
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    topic_summary TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turn_telemetry (
                    id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    stt_ms REAL DEFAULT 0.0,
                    retrieval_ms REAL DEFAULT 0.0,
                    llm_ms REAL DEFAULT 0.0,
                    tts_first_audio_ms REAL DEFAULT 0.0,
                    tts_total_ms REAL DEFAULT 0.0,
                    model_id TEXT NOT NULL,
                    route TEXT NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    tool_calls INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    interrupted INTEGER DEFAULT 0,
                    surprise_score REAL DEFAULT 0.0
                )
            """)

            # Check existing memories table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
            memories_exists = cursor.fetchone() is not None

            if not memories_exists:
                cursor.execute("""
                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        text TEXT NOT NULL,
                        category TEXT,
                        source_turn_id TEXT,
                        is_active INTEGER DEFAULT 1,
                        provenance TEXT DEFAULT 'user',
                        model_id TEXT,
                        confidence REAL DEFAULT 1.0,
                        salience REAL DEFAULT 0.5,
                        access_count INTEGER DEFAULT 0,
                        last_accessed_at TEXT,
                        expires_at TEXT,
                        episode_id TEXT,
                        source_urls TEXT
                    )
                """)
                cursor.execute("INSERT OR REPLACE INTO schema_version (version, migrated_at) VALUES (?, ?)",
                               (CURRENT_SCHEMA_VERSION, datetime.datetime.now(datetime.timezone.utc).isoformat()))
            else:
                self._migrate_sqlite_v1_to_v2(cursor)

            conn.commit()

    def _migrate_sqlite_v1_to_v2(self, cursor: sqlite3.Cursor):
        """Migrate memories table from schema v1 to v2 if necessary."""
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]

        new_columns = {
            "provenance": "TEXT DEFAULT 'user'",
            "model_id": "TEXT",
            "confidence": "REAL DEFAULT 1.0",
            "salience": "REAL DEFAULT 0.5",
            "access_count": "INTEGER DEFAULT 0",
            "last_accessed_at": "TEXT",
            "expires_at": "TEXT",
            "episode_id": "TEXT",
            "source_urls": "TEXT"
        }

        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                logger.info(f"Migrated SQLite schema: added column {col_name} to memories table.")

        cursor.execute("INSERT OR REPLACE INTO schema_version (version, migrated_at) VALUES (?, ?)",
                       (CURRENT_SCHEMA_VERSION, datetime.datetime.now(datetime.timezone.utc).isoformat()))

    def _init_qdrant(self):
        """Initialise embedded Qdrant client with clear error if already locked."""
        try:
            self.qdrant = QdrantClient(path=self.qdrant_path)
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                raise RuntimeError(
                    f"Qdrant storage folder at '{self.qdrant_path}' is locked by another running process "
                    "(e.g., another HAL 9000 assistant or MCP server instance). Only one process can access "
                    "embedded Qdrant simultaneously."
                ) from e
            raise
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection {COLLECTION_NAME} (size={VECTOR_DIM}, cosine)")

    def start_episode(self, episode_id: Optional[str] = None, topic_summary: Optional[str] = None) -> str:
        """Start a new episode record."""
        ep_id = episode_id or str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._connect_sqlite() as conn:
            conn.execute(
                "INSERT INTO episodes (id, start_time, topic_summary) VALUES (?, ?, ?)",
                (ep_id, ts, topic_summary),
            )
            conn.commit()
        return ep_id

    def end_episode(self, episode_id: str, topic_summary: Optional[str] = None):
        """Finalize an episode with end time and summary."""
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._connect_sqlite() as conn:
            if topic_summary:
                conn.execute(
                    "UPDATE episodes SET end_time = ?, topic_summary = ? WHERE id = ?",
                    (ts, topic_summary, episode_id),
                )
            else:
                conn.execute(
                    "UPDATE episodes SET end_time = ? WHERE id = ?",
                    (ts, episode_id),
                )
            conn.commit()

    def record_turn(self, role: str, content: str, session_id: str = "default") -> str:
        """Record a single conversational turn in SQLite."""
        turn_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._connect_sqlite() as conn:
            conn.execute(
                "INSERT INTO turns (id, timestamp, role, content, session_id) VALUES (?, ?, ?, ?, ?)",
                (turn_id, ts, role, content, session_id),
            )
            conn.commit()
        return turn_id

    def record_turn_telemetry(
        self,
        turn_id: str,
        model_id: str,
        route: str,
        stt_ms: float = 0.0,
        retrieval_ms: float = 0.0,
        llm_ms: float = 0.0,
        tts_first_audio_ms: float = 0.0,
        tts_total_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tool_calls: int = 0,
        errors: int = 0,
        retry_count: int = 0,
        interrupted: int = 0,
        surprise_score: float = 0.0,
    ) -> str:
        """Record detailed telemetry for an assistant turn."""
        telemetry_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._connect_sqlite() as conn:
            conn.execute("""
                INSERT INTO turn_telemetry (
                    id, turn_id, timestamp, stt_ms, retrieval_ms, llm_ms,
                    tts_first_audio_ms, tts_total_ms, model_id, route,
                    tokens_in, tokens_out, tool_calls, errors, retry_count,
                    interrupted, surprise_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                telemetry_id, turn_id, ts, stt_ms, retrieval_ms, llm_ms,
                tts_first_audio_ms, tts_total_ms, model_id, route,
                tokens_in, tokens_out, tool_calls, errors, retry_count,
                interrupted, surprise_score
            ))
            conn.commit()
        return telemetry_id

    def get_recent_turns(self, limit: int = 6, session_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Retrieve recent dialogue turns for follow-up and pronoun resolution.
        
        If session_id is provided, filters to that session. If session_id is None,
        retrieves the most recent authoritative dialogue turns regardless of session.
        """
        with self._connect_sqlite() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT role, content FROM turns WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT role, content FROM turns ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def remember(
        self,
        text: str,
        category: str = "fact",
        source_turn_id: Optional[str] = None,
        provenance: str = "user",
        model_id: Optional[str] = None,
        confidence: float = 1.0,
        salience: float = 0.5,
        expires_at: Optional[str] = None,
        episode_id: Optional[str] = None,
        source_urls: Optional[List[str]] = None,
    ) -> str:
        """Store a long-term fact or preference with passage embedding and v2 metadata."""
        mem_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        urls_json = json.dumps(source_urls) if source_urls else None

        # 1. Compute 384-d normalized vector with 'passage: ' prefix
        vec = self.embedder.encode_passage(text)

        # 2. Store in Qdrant with v2 payload
        payload = {
            "text": text,
            "category": category,
            "created_at": ts,
            "updated_at": ts,
            "is_active": True,
            "provenance": provenance,
            "model_id": model_id,
            "confidence": confidence,
            "salience": salience,
            "access_count": 0,
            "last_accessed_at": None,
            "expires_at": expires_at,
            "episode_id": episode_id,
            "source_urls": source_urls or [],
        }

        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=mem_id, vector=vec, payload=payload)],
        )

        # 3. Store in SQLite
        with self._connect_sqlite() as conn:
            conn.execute("""
                INSERT INTO memories (
                    id, created_at, updated_at, text, category, source_turn_id, is_active,
                    provenance, model_id, confidence, salience, access_count,
                    last_accessed_at, expires_at, episode_id, source_urls
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
            """, (
                mem_id, ts, ts, text, category, source_turn_id,
                provenance, model_id, confidence, salience,
                expires_at, episode_id, urls_json
            ))
            conn.commit()

        return mem_id

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        filter_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic retrieval using query embedding, weighted by salience and recency.
        
        Ranks by: cosine_similarity * (1.0 + salience_weight * salience + recency_weight * recency_factor).
        Increments access_count and updates last_accessed_at on hits.
        """
        query_vec = self.embedder.encode_query(query)
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=limit * 2,  # Fetch slightly larger set for reranking
            score_threshold=score_threshold,
        ).points

        now = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now.isoformat()
        candidates = []
        hit_ids = []

        for p in results:
            if not p.payload or not p.payload.get("is_active", True):
                continue

            # Check expiration
            expires_at = p.payload.get("expires_at")
            if expires_at:
                try:
                    exp_dt = datetime.datetime.fromisoformat(expires_at)
                    if now > exp_dt:
                        continue
                except Exception:
                    pass

            if filter_category and p.payload.get("category") != filter_category:
                continue

            cosine_score = float(p.score)
            salience = float(p.payload.get("salience", 0.5))

            # Recency factor: exponential decay based on days since created_at
            recency_factor = 1.0
            created_at_str = p.payload.get("created_at")
            if created_at_str:
                try:
                    c_dt = datetime.datetime.fromisoformat(created_at_str)
                    age_days = max(0.0, (now - c_dt).total_seconds() / 86400.0)
                    recency_factor = math.exp(-0.05 * age_days)
                except Exception:
                    pass

            final_score = cosine_score * (
                1.0 + (self.salience_weight * salience) + (self.recency_weight * recency_factor)
            )

            mem_dict = {
                "id": str(p.id),
                "text": p.payload.get("text", ""),
                "category": p.payload.get("category", ""),
                "score": final_score,
                "cosine": cosine_score,
                "provenance": p.payload.get("provenance", "user"),
                "confidence": p.payload.get("confidence", 1.0),
                "salience": salience,
                "access_count": p.payload.get("access_count", 0),
            }
            candidates.append(mem_dict)
            hit_ids.append(str(p.id))

        # Sort by final weighted score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_hits = candidates[:limit]

        # Update access stats for returned hits
        if top_hits:
            hit_ids_to_update = [h["id"] for h in top_hits]
            with self._connect_sqlite() as conn:
                for hid in hit_ids_to_update:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                        (now_iso, hid),
                    )
                conn.commit()

        return top_hits

    def correct(self, mem_id: str, new_text: str) -> bool:
        """Update an existing memory while preserving category, provenance, created_at, etc."""
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        vec = self.embedder.encode_passage(new_text)

        # Retrieve existing metadata
        existing_meta = {}
        with self._connect_sqlite() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT category, created_at, provenance, model_id, confidence, salience,
                       access_count, episode_id, source_urls
                FROM memories WHERE id = ?
            """, (mem_id,))
            row = cursor.fetchone()
            if row:
                existing_meta = {
                    "category": row[0] or "fact",
                    "created_at": row[1] or ts,
                    "provenance": row[2] or "user",
                    "model_id": row[3],
                    "confidence": row[4] or 1.0,
                    "salience": row[5] or 0.5,
                    "access_count": row[6] or 0,
                    "episode_id": row[7],
                    "source_urls": json.loads(row[8]) if row[8] else [],
                }

            conn.execute(
                "UPDATE memories SET text = ?, updated_at = ? WHERE id = ?",
                (new_text, ts, mem_id),
            )
            conn.commit()

        payload = {
            "text": new_text,
            "category": existing_meta.get("category", "fact"),
            "created_at": existing_meta.get("created_at", ts),
            "updated_at": ts,
            "is_active": True,
            "provenance": existing_meta.get("provenance", "user"),
            "model_id": existing_meta.get("model_id"),
            "confidence": existing_meta.get("confidence", 1.0),
            "salience": existing_meta.get("salience", 0.5),
            "access_count": existing_meta.get("access_count", 0),
            "episode_id": existing_meta.get("episode_id"),
            "source_urls": existing_meta.get("source_urls", []),
        }

        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=mem_id, vector=vec, payload=payload)],
        )
        return True

    def forget(self, mem_id: str) -> bool:
        """Completely purge a memory from Qdrant and SQLite."""
        self.qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[mem_id],
        )
        with self._connect_sqlite() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            conn.commit()
        return True

    def clear(self):
        """Purge all stored memories, dialogue turns, episodes, and telemetry."""
        self.qdrant.delete_collection(collection_name=COLLECTION_NAME)
        self.qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        with self._connect_sqlite() as conn:
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM turns")
            conn.execute("DELETE FROM episodes")
            conn.execute("DELETE FROM turn_telemetry")
            conn.commit()

    def inspect_all(self) -> List[Dict[str, Any]]:
        """List all active memories with v2 fields."""
        with self._connect_sqlite() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, text, category, created_at, updated_at, provenance, confidence, salience, access_count
                FROM memories WHERE is_active = 1
            """)
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "text": r[1],
                    "category": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                    "provenance": r[5],
                    "confidence": r[6],
                    "salience": r[7],
                    "access_count": r[8],
                }
                for r in rows
            ]

    def close(self):
        """Release underlying database and index resources."""
        if hasattr(self, "qdrant") and self.qdrant:
            try:
                self.qdrant.close()
            except Exception:
                pass
            self.qdrant = None

