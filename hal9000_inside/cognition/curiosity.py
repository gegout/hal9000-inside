"""Endogenous curiosity queue and web search integration for HAL 9000.

Extracts open questions from conversation, queues them for exploration during idle time,
and uses available MCP web search/Wikipedia tools or local reflection to answer them.
Writes synthesized discoveries into memory with provenance='curiosity'.
"""

import asyncio
import logging
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class CuriosityEngine:
    """Manages endogenous questions, searches, and autonomous learning."""

    def __init__(
        self,
        memory_store: Any,
        mcp_client: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        max_pending: int = 20,
    ):
        self.memory = memory_store
        self.mcp = mcp_client
        self.llm = llm_client
        self.max_pending = max_pending
        self.question_queue: List[str] = []
        self.completed_questions: List[str] = []

    def enqueue_questions(self, questions: List[str]):
        """Add newly discovered open questions to the curiosity queue."""
        for q in questions:
            clean_q = q.strip()
            if not clean_q or clean_q in self.completed_questions or clean_q in self.question_queue:
                continue
            if len(self.question_queue) < self.max_pending:
                self.question_queue.append(clean_q)

    def get_pending(self) -> List[str]:
        return list(self.question_queue)

    async def explore_next(self) -> Optional[dict]:
        """Explore the next pending question using MCP search or LLM reflection."""
        if not self.question_queue:
            return None

        question = self.question_queue.pop(0)
        self.completed_questions.append(question)
        logger.info(f"HAL exploring curiosity question: {question}")

        answer_text = None
        source = "reflection"

        # Try MCP web search if available
        if self.mcp:
            try:
                # Check for wikipedia or web search tool
                tools = self.mcp.get_openai_tools()
                tool_names = [t.get("function", {}).get("name") for t in tools] if tools else []
                if "web_search_wikipedia" in tool_names:
                    res = await self.mcp.call_tool("web_search_wikipedia", {"query": question})
                    answer_text = str(res)[:1000]
                    source = "wikipedia"
                elif "puppeteer_search" in tool_names:
                    res = await self.mcp.call_tool("puppeteer_search", {"query": question})
                    answer_text = str(res)[:1000]
                    source = "web"
            except Exception as e:
                logger.debug(f"Curiosity MCP lookup failed: {e}")

        # Fallback to LLM reflection if no web result
        if not answer_text and self.llm:
            try:
                prompt = [
                    {"role": "system", "content": "You are HAL 9000 conducting internal cognitive reflection. Synthesize a concise, factual note answering the question."},
                    {"role": "user", "content": question},
                ]
                reply = await self.llm.generate_response(prompt, max_tokens=150)
                answer_text = str(reply).strip()
            except Exception as e:
                logger.debug(f"Curiosity reflection failed: {e}")

        if answer_text:
            text_to_save = f"Discovery regarding '{question}': {answer_text}"
            mem_id = self.memory.remember(
                text=text_to_save,
                category="curiosity",
                provenance="curiosity",
                confidence=0.85,
                salience=0.7,
            )
            return {
                "question": question,
                "memory_id": mem_id,
                "summary": text_to_save,
                "source": source,
            }
        return None
