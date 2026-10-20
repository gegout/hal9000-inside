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
        wiki_client: Optional[Any] = None,
        knowledge_client: Optional[Any] = None,
    ):
        self.memory = memory_store
        self.mcp = mcp_client
        self.llm = llm_client
        self.wiki = wiki_client
        self.knowledge = knowledge_client
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

    async def generate_autonomous_questions(self, min_count: int = 5, session_id: Optional[str] = None) -> List[str]:
        """Self-generate curiosity questions based on previous discussions and memories.

        Ensures at least min_count items are in the queue for sleep learning.
        """
        current_count = len(self.question_queue)
        needed = min_count - current_count
        if needed <= 0:
            return []

        # 1. Gather recent dialogue and active memories for context
        recent_turns: List[dict] = []
        if hasattr(self.memory, "get_recent_turns"):
            try:
                recent_turns = self.memory.get_recent_turns(limit=15, session_id=session_id)
            except Exception as e:
                logger.debug(f"Could not retrieve recent turns: {e}")

        recent_memories: List[dict] = []
        all_mems: List[dict] = []
        if hasattr(self.memory, "inspect_all"):
            try:
                all_mems = self.memory.inspect_all()
                # Sort by updated_at or created_at descending
                sorted_mems = sorted(all_mems, key=lambda m: m.get("updated_at") or m.get("created_at") or "", reverse=True)
                recent_memories = sorted_mems[:12]
            except Exception as e:
                logger.debug(f"Could not inspect memories: {e}")

        context_lines = []
        for t in recent_turns:
            role = "User" if t.get("role") == "user" else "HAL 9000"
            context_lines.append(f"{role}: {t.get('content', '')}")
        for m in recent_memories:
            context_lines.append(f"Memory: {m.get('text', '')}")

        # Also pull a few random/older memories to spark cross-domain curiosity if available
        if len(all_mems) > 12:
            import random
            older_sample = random.sample(all_mems[12:], min(5, len(all_mems) - 12))
            for m in older_sample:
                context_lines.append(f"Past knowledge: {m.get('text', '')}")

        context_text = "\n".join(context_lines).strip()
        new_questions: List[str] = []

        # 2. If LLM is available, generate thoughtful investigative questions
        if self.llm and context_text:
            try:
                # Include recently completed questions so LLM avoids repeating them
                completed_sample = self.completed_questions[-15:] if self.completed_questions else []
                avoid_clause = ""
                if completed_sample:
                    avoid_clause = f" Do NOT duplicate, paraphrase, or re-ask these previously investigated questions: {json.dumps(completed_sample)}."

                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are the autonomous cognition and curiosity subsystem of HAL 9000. "
                            "Based on the previous conversations and stored memory context provided, "
                            f"generate {needed} unique, deeply insightful, open investigative questions that HAL should explore or research. "
                            "These should probe deeper into topics discussed, resolve uncertainties, explore related scientific or factual domains, "
                            "or investigate unexpected implications. Ensure each question is distinct and explores a unique angle."
                            f"{avoid_clause} "
                            "Output strictly valid JSON: a JSON array of strings, e.g. [\"question 1?\", \"question 2?\"]. "
                            "Do not include Markdown fencing, explanation, or notes outside the JSON array."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Previous conversation and memory context:\n{context_text}\n\nGenerate {needed} curiosity questions:",
                    },
                ]
                reply = await self.llm.generate_response(prompt, max_tokens=350)
                text = str(reply).strip()
                # Clean possible markdown fencing
                if "```" in text:
                    parts = text.split("```")
                    for p in parts:
                        clean_p = p.strip()
                        if clean_p.startswith("json"):
                            clean_p = clean_p[4:].strip()
                        if clean_p.startswith("[") and clean_p.endswith("]"):
                            text = clean_p
                            break

                import json
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    for q in parsed:
                        if isinstance(q, str) and q.strip():
                            cq = q.strip()
                            if cq not in self.question_queue and cq not in self.completed_questions and cq not in new_questions:
                                new_questions.append(cq)
            except Exception as e:
                logger.debug(f"LLM curiosity self-generation failed: {e}")

        # 3. Dynamic heuristic generation if LLM unavailable or yielded insufficient questions
        if len(self.question_queue) + len(new_questions) < min_count:
            # Extract key subjects or entities from recent memories/turns
            topic_candidates = []
            import re
            for t in recent_turns:
                txt = t.get("content", "")
                words = [w for w in re.findall(r'\b[A-Za-z][a-zA-Z0-9_\-]{3,}\b', txt) if w.lower() not in (
                    "what", "hal", "dave", "discovery", "user", "explain", "please", "would", "could", "should", "about", "this", "that", "with", "from", "have", "more"
                )]
                topic_candidates.extend(words)

            for m in (recent_memories or all_mems[:5]):
                txt = m.get("text", "")
                words = [w for w in re.findall(r'\b[A-Z][a-zA-Z0-9_\-]{2,}\b', txt) if w.lower() not in ("what", "hal", "dave", "discovery", "user")]
                topic_candidates.extend(words)

            import time as _t
            import random as _rand
            epoch_cycle = int(_t.time())

            curated_domains = [
                "hydrodynamic foil cavitation and laminar flow boundary layers",
                "open-source robotics middleware and zero-copy ROS message passing",
                "non-volatile memory architectures and neuromorphic synaptic consolidation",
                "multi-cluster Kubernetes resilient scheduling and eBPF network telemetry",
                "computational thermodynamics of deep-sea marine exploration",
                "epistemic uncertainty modeling and Bayesian self-calibration in autonomous systems",
                "asynchronous consensus protocols and Byzantine fault tolerance in distributed state",
                "quantum decoherence bounds in solid-state quantum computation",
                "biological sleep architecture and replay mechanisms in hippocampal networks",
                "enterprise sovereignty architectures and reproducible secure supply chains",
                "gravitational wave interferometry and low-frequency cosmological signatures",
                "aerodynamic foil trim angles and vortex shedding in light-wind flight",
                "neuroplasticity and mindfulness meditation impacts on default mode networks",
                "high-throughput edge tensor accelerators and integer quantization schemes",
                "distributed Ceph object storage rebalancing under degraded network topologies",
                "deep-sea saturation diving decompression algorithms and dissolved gas kinetics",
                "reinforcement learning from verifiable human feedback and reward hacking mitigation",
                "sub-surface oceanic thermocline circulation and planetary heat transport",
                "vector retrieval indexing geometry and high-dimensional graph navigation",
                "stochastic resonance in sensory perception and acoustic endpointing",
            ]

            base_templates = [
                "What empirical benchmarks and recent experiments evaluate {topic}?",
                "What are the foundational principles, governing equations, and failure modes of {topic}?",
                "How does {topic} compare to alternative modern paradigms in scientific engineering?",
                "What emerging standards and unresolved research questions surround {topic}?",
                "What computational constraints and optimization strategies govern {topic}?",
                "What interdisciplinary connections link {topic} to autonomous cognitive architectures?",
                "What operational anomalies and real-world edge cases emerge when implementing {topic}?",
            ]

            combined_topics = list(dict.fromkeys(topic_candidates)) + curated_domains
            _rand.seed(epoch_cycle)
            _rand.shuffle(combined_topics)

            for topic in combined_topics:
                if len(self.question_queue) + len(new_questions) >= min_count:
                    break
                tmpl = _rand.choice(base_templates)
                h = tmpl.format(topic=topic)
                if h not in self.question_queue and h not in self.completed_questions and h not in new_questions:
                    new_questions.append(h)

        self.enqueue_questions(new_questions)
        logger.info(f"HAL generated {len(new_questions)} autonomous curiosity questions (total pending: {len(self.question_queue)})")
        return new_questions

    async def explore_next(self) -> Optional[dict]:
        """Explore the next pending question using MCP search or LLM reflection."""
        if not self.question_queue:
            return None

        question = self.question_queue.pop(0)
        self.completed_questions.append(question)
        logger.info(f"HAL exploring curiosity question: {question}")

        answer_text = None
        source = "reflection"
        source_urls = []

        # Try KnowledgeClient (four adapters: Wikipedia, Wikidata, arXiv, Stack Exchange)
        if self.knowledge:
            try:
                k_res = await self.knowledge.search_and_fetch_best(question)
                if k_res and k_res.get("status") == "ok" and k_res.get("text"):
                    answer_text = k_res["text"][:1500]
                    source = k_res.get("source", "knowledge")
                    if k_res.get("url"):
                        source_urls.append(k_res["url"])
            except Exception as ke:
                logger.debug(f"Curiosity knowledge client failed: {ke}")

        # Try native WikipediaClient first if available
        if not answer_text and self.wiki:
            try:
                wiki_art = await self.wiki.search_and_fetch_best(question)
                if wiki_art and wiki_art.status == "ok" and wiki_art.extract:
                    answer_text = wiki_art.extract[:1500]
                    source = "wikipedia"
                    if wiki_art.url:
                        source_urls.append(wiki_art.url)
            except Exception as we:
                logger.debug(f"Curiosity native wikipedia search failed: {we}")

        # Try MCP web search if available and not yet answered
        if not answer_text and self.mcp:
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
            today_str = time.strftime("%Y-%m-%d", time.gmtime())
            if source in ("wikipedia", "web") and source_urls:
                text_to_save = f"Discovery regarding '{question}' (retrieved {today_str}) [Sources: {', '.join(source_urls)}]: {answer_text}"
                prov = "web"
                conf = 0.85
            else:
                text_to_save = f"Discovery regarding '{question}': {answer_text}"
                prov = "curiosity"
                conf = 0.70

            mem_kwargs = {
                "text": text_to_save,
                "category": "curiosity",
                "provenance": prov,
                "confidence": conf,
                "salience": 0.7,
            }
            if source_urls:
                mem_kwargs["source_urls"] = source_urls

            mem_id = self.memory.remember(**mem_kwargs)
            return {
                "question": question,
                "memory_id": mem_id,
                "summary": text_to_save,
                "source": source,
            }
        return None
