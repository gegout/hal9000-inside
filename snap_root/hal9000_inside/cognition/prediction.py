"""Prediction and anticipation engine for HAL 9000 cognition.

Generates anticipated user follow-ups, queries, or topics based on recent turn context,
enabling pre-fetching of candidate memories and pre-computation of embeddings.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Manages predicted follow-ups and candidate prefetching."""

    def __init__(self, memory_store: Optional[Any] = None):
        self.memory = memory_store
        self.last_predictions: List[str] = []

    def set_predictions(self, followups: List[str]):
        """Store the latest predictions from turn digest."""
        self.last_predictions = [f.strip() for f in followups if f.strip()]

    def get_predictions(self) -> List[str]:
        return list(self.last_predictions)

    async def prefetch_candidates(self) -> List[dict]:
        """Pre-fetch memories matching predicted followups to warm cache."""
        if not self.memory or not self.last_predictions:
            return []

        all_matches = []
        for pred in self.last_predictions[:2]:
            try:
                results = self.memory.search(pred, limit=2, score_threshold=0.0)
                all_matches.extend(results)
            except Exception as e:
                logger.debug(f"Prediction prefetch error for '{pred}': {e}")
        return all_matches
