"""Deterministic routing and intent classification for HAL 9000 Inside."""

import re
from typing import Optional, Tuple

# Pure social greetings and polite courtesies suitable for immediate local-only handling
SOCIAL_PATTERNS = [
    r"^(hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings)(\s+hal)?[\.!?]*$",
    r"^(how\s+are\s+you|how\s+do\s+you\s+feel|how\s+are\s+things)(\s+hal)?[\.!?]*$",
    r"^(thank\s+you|thanks|thank\s+you\s+very\s+much|appreciate\s+it)(\s+hal)?[\.!?]*$",
    r"^(goodbye|bye|bye\s+bye|see\s+you|farewell)(\s+hal)?[\.!?]*$",
]

# Patterns that explicitly require reasoning or substantive explanation even if containing "thanks"
EXPLANATION_OVERRIDE_PATTERNS = [
    r"explain",
    r"why",
    r"how\s+does",
    r"what\s+is",
    r"tell\s+me\s+more",
    r"but",
    r"however",
    r"compare",
    r"search",
    r"find",
    r"who\s+was",
]


def classify_turn_intent(user_text: str) -> str:
    """Classify the user turn intent:
    - 'social': Pure greetings or courteous pleasantries. Handled locally.
    - 'action': Direct assistant control commands handled deterministically.
    - 'substantive': Requires full remote reasoning, external knowledge, or deep memory recall.
    """
    clean = user_text.strip().lower()
    if not clean:
        return "social"

    # Stop / cancel commands
    if any(clean == s or clean.startswith(s + " ") for s in (
        "stop", "cancel", "halt", "be quiet", "stop talking", "stop speaking",
        "cancel speech", "cancel current speech", "shut up"
    )):
        return "action"

    # Check explanation overrides: "Thanks, but explain why" -> substantive
    for pat in EXPLANATION_OVERRIDE_PATTERNS:
        if re.search(r"\b" + pat + r"\b", clean):
            return "substantive"

    # Social patterns
    for pat in SOCIAL_PATTERNS:
        if re.match(pat, clean):
            return "social"

    return "substantive"
