"""Deterministic fake multilingual E5 embedder for test isolation."""

import hashlib
import numpy as np
from typing import List, Union


class FakeEmbedder:
    """Produces deterministic normalized 384-d vectors based on text hash."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> List[float]:
        # Clean text
        clean = text.lower()
        if clean.startswith("query:"):
            clean = clean[6:].strip()
        elif clean.startswith("passage:"):
            clean = clean[8:].strip()

        words = [w.strip(".,!?:;\"'()[]{}") for w in clean.split() if len(w) > 2]
        if not words:
            words = ["empty"]

        # Aggregate normalized token vectors
        combined = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            h = hashlib.sha256(w.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed)
            token_vec = rng.normal(0, 1.0, size=self.dim).astype(np.float32)
            token_norm = np.linalg.norm(token_vec)
            if token_norm > 0:
                token_vec /= token_norm
            combined += token_vec

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined.tolist()

    def encode_query(self, query: str) -> List[float]:
        return self._hash_to_vector(f"query: {query.strip()}")

    def encode_passage(self, passage: str) -> List[float]:
        return self._hash_to_vector(f"passage: {passage.strip()}")

    def encode(self, texts: Union[str, List[str]], is_query: bool = False) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            return self.encode_query(texts) if is_query else self.encode_passage(texts)
        return [self.encode_query(t) if is_query else self.encode_passage(t) for t in texts]
