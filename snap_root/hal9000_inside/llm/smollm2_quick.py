"""SmolLM2 fast quick-response client for local concurrent generation in combine mode."""

import asyncio
import logging
from typing import Dict, List, Optional
import httpx

from hal9000_inside.speech_norm import validate_local_response
from hal9000_inside.smollm2 import DEFAULT_SMOLLM2_ENDPOINT, DEFAULT_SMOLLM2_MODEL

logger = logging.getLogger(__name__)

SMOLLM2_QUICK_ACK_PROMPT = """You are HAL 9000's local cognitive reflex.
Your role is ONLY to emit a brief, polite, helpful spoken acknowledgement or quick reaction.
CRITICAL RULES:
1. STRICT HARD LIMIT: At most 12 spoken words. Never exceed 15 words.
2. Address the user's intent politely (e.g. "I am checking the databanks on that now.", "Understood. Accessing records.", "Checking telemetry on that right away.")
3. DO NOT invent facts, numbers, or full answers.
4. DO NOT use markdown or bullet points. Speak as HAL 9000.
5. In nominal mood, NEVER use the name "Dave".
"""


class SmolLM2QuickResponder:
    """Async client generating quick spoken acknowledgements concurrently with remote reasoning."""

    def __init__(
        self,
        endpoint: str = DEFAULT_SMOLLM2_ENDPOINT,
        model: str = DEFAULT_SMOLLM2_MODEL,
        timeout_seconds: float = 2.5,
        max_words: int = 20,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_words = max_words

    async def generate_social_reply(self, user_text: str) -> Optional[str]:
        """Generate a brief social reply using local SmolLM2."""
        messages = [
            {"role": "system", "content": "You are HAL 9000. Reply politely and calmly in strictly under 15 words. Never use bullet points or markdown."},
            {"role": "user", "content": user_text},
        ]
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 25,
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        raw = choices[0].get("message", {}).get("content", "").strip()
                        valid = validate_local_response(raw, max_words=self.max_words)
                        if valid:
                            return valid
        except Exception as e:
            logger.debug(f"SmolLM2 social reply failed: {e}")
        return None

    async def generate_quick_ack(
        self,
        user_text: str,
        recent_turns: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Generate a concise spoken acknowledgement within word limits.
        Returns validated text or None if generation failed, timed out, or exceeded limits.
        """
        # If user is asking for research, comparison, facts, return an appropriate prompt
        messages = [
            {"role": "system", "content": "You are HAL 9000. Output only a brief spoken acknowledgment in under 10 words indicating you are processing or checking databanks. Do not answer the question itself."},
            {"role": "user", "content": f"Acknowledge this request concisely: {user_text}"},
        ]

        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 25,
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        raw_content = choices[0].get("message", {}).get("content", "").strip()
                        valid_text = validate_local_response(raw_content, max_words=self.max_words)
                        if valid_text:
                            return valid_text
                        logger.debug(f"SmolLM2 response discarded: {raw_content!r}")
        except Exception as e:
            logger.debug(f"SmolLM2 quick ack generation skipped: {e}")

        return None
