"""Attention tracker and focus manager for HAL 9000 cognition.

Tracks active topics across recent turns, boosts salience of focused concepts,
and coordinates background memory enrichment without interfering with active turns.
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AttentionTracker:
    """Maintains active attention state across conversation turns."""

    def __init__(self, decay_rate: float = 0.8, max_focus_topics: int = 5):
        self.decay_rate = decay_rate
        self.max_focus_topics = max_focus_topics
        # topic -> attention weight (0.0 to 1.0)
        self.focus_weights: Dict[str, float] = {}
        self.last_updated: float = time.time()

    def update_from_turn(self, topics: List[str]):
        """Decay older focus weights and reinforce newly mentioned topics."""
        # Decay current topics
        for t in list(self.focus_weights.keys()):
            self.focus_weights[t] *= self.decay_rate
            if self.focus_weights[t] < 0.1:
                del self.focus_weights[t]

        # Reinforce new topics
        for t in topics:
            clean_t = t.strip().lower()
            if not clean_t:
                continue
            current = self.focus_weights.get(clean_t, 0.0)
            self.focus_weights[clean_t] = min(1.0, current + 0.5)

        # Retain top topics
        sorted_topics = sorted(self.focus_weights.items(), key=lambda x: x[1], reverse=True)
        self.focus_weights = dict(sorted_topics[: self.max_focus_topics])
        self.last_updated = time.time()

    def get_active_focus(self) -> List[str]:
        """Returns the currently active focus topics ordered by salience."""
        return list(self.focus_weights.keys())

    def get_salience_boost(self, text: str) -> float:
        """Calculates attention boost factor for a given memory text."""
        boost = 0.0
        lower = text.lower()
        for topic, weight in self.focus_weights.items():
            if topic in lower:
                boost += weight * 0.3
        return min(0.5, boost)
