"""Structured Turn Digest for HAL 9000 cognition.

Extracts:
- facts_to_remember[] (text, confidence, salience)
- open_questions[]
- focus_topics[]
- predicted_followups[]
- user_corrections[]
- answer_confidence
Strict validation using Pydantic. Runs asynchronously after a turn.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FactToRemember(BaseModel):
    text: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    category: str = "fact"


class UserCorrection(BaseModel):
    target_id: Optional[str] = None
    corrected_text: str


class TurnDigestResult(BaseModel):
    facts_to_remember: List[FactToRemember] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    focus_topics: List[str] = Field(default_factory=list)
    predicted_followups: List[str] = Field(default_factory=list)
    user_corrections: List[UserCorrection] = Field(default_factory=list)
    answer_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


DIGEST_PROMPT = """Analyze the following conversational turn between the User and HAL 9000.
Produce a strictly valid JSON object matching this schema:
{
  "facts_to_remember": [{"text": "...", "confidence": 0.9, "salience": 0.6, "category": "fact"}],
  "open_questions": ["..."],
  "focus_topics": ["..."],
  "predicted_followups": ["..."],
  "user_corrections": [],
  "answer_confidence": 0.95
}
Do not include any Markdown fencing, explanations, or prose outside the JSON object.
"""


class TurnDigestAnalyzer:
    """Performs structured extraction on a completed turn."""

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    async def analyze_turn(self, user_text: str, hal_reply: str) -> Optional[TurnDigestResult]:
        """Call LLM with turn transcript and parse Pydantic result."""
        content = f"User: {user_text}\nHAL 9000: {hal_reply}"
        messages = [
            {"role": "system", "content": DIGEST_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            raw = await self.llm.generate_response(messages, max_tokens=300)
            text = str(raw).strip()
            # Strip markdown json code block if present
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            data = json.loads(text)
            return TurnDigestResult.model_validate(data)
        except Exception as e:
            logger.debug(f"Turn digest extraction failed or invalid json: {e}")
            return None
