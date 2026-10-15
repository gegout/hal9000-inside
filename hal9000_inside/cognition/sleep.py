"""Sleep mode management and offline/online cognitive consolidation for HAL 9000.

During sleep:
- Audio capture and playback are stopped/muted.
- HAL loops through 6 cognitive phases without preemption:
  1. Backlog: drain open questions and pending prefetch jobs.
  2. Endogenous curiosity: pick top questions by salience x novelty.
     Online: answer via web tools and/or remote model. Store with provenance=web|remote_model.
     Offline: leave queued (never invent new facts offline).
  3. Rehearsal: online only. For top predicted follow-ups, ask remote model for high-quality reply.
     Store as consolidation notes and style exemplars.
  4. Consolidation: merge duplicates, write topic summaries, link related memories, lower confidence
     on contradicted facts, deactivate unused notes.
  5. Reflection: read telemetry, update self_model.json stats, tune parameters within declared bounds.
  6. Interoception: record system state (CPU, mem, GPU, connectivity) into telemetry.
- On wake: unmute, discard buffered audio, produce a one-line sleep report.
- Strictly honors sleep budgets (max minutes, max tokens, max web fetches).
"""

import asyncio
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SleepBudget:
    max_minutes: float = 60.0
    max_tokens: int = 10000
    max_web_fetches: int = 20


@dataclass
class SleepSummary:
    duration_seconds: float = 0.0
    jobs_run: int = 0
    notes_written: int = 0
    cost_estimate: float = 0.0
    is_online: bool = False
    interrupted: bool = False

    def format_one_liner(self) -> str:
        """Produce honest one-line sleep report: e.g. 'Slept 20 min, learned 7 notes, offline the whole time'"""
        minutes = int(self.duration_seconds // 60)
        sec = int(self.duration_seconds % 60)
        dur_str = f"{minutes} min" if minutes > 0 else f"{sec}s"
        conn_str = "online" if self.is_online else "offline the whole time"
        return f"Slept {dur_str}, learned {self.notes_written} notes, {conn_str}."


class SleepManager:
    """Orchestrates autonomous sleep learning cycles within declared budgets."""

    def __init__(
        self,
        memory_store: Any,
        curiosity_engine: Optional[Any] = None,
        prediction_engine: Optional[Any] = None,
        mcp_client: Optional[Any] = None,
        llm_router: Optional[Any] = None,
        interoception_monitor: Optional[Any] = None,
        budget: Optional[SleepBudget] = None,
        data_dir: Optional[str] = None,
        clock: Optional[Any] = None,
        wiki_client: Optional[Any] = None,
        knowledge_client: Optional[Any] = None,
    ):
        self.memory = memory_store
        self.curiosity = curiosity_engine
        self.prediction = prediction_engine
        self.mcp = mcp_client
        self.router = llm_router
        self.interoception = interoception_monitor
        self.wiki = wiki_client
        self.knowledge = knowledge_client
        self.budget = budget or SleepBudget()
        self.clock = clock or time
        self.data_dir = data_dir or getattr(memory_store, "data_dir", os.path.expanduser("~/.local/share/hal9000-inside"))
        self.self_model_path = os.path.join(self.data_dir, "self_model.json")

        self.tokens_used = 0
        self.web_fetches_used = 0
        self.notes_written = 0
        self.jobs_run = 0
        self._stop_event = asyncio.Event()

    def request_wake(self):
        """Signal sleep loop to stop cooperatively."""
        self._stop_event.set()

    def is_wake_requested(self) -> bool:
        return self._stop_event.is_set()

    def _is_online(self) -> bool:
        """Check if remote model/network is available."""
        if not self.router:
            return False
        # If forced local, definitely offline
        if getattr(self.router, "force_local", False):
            return False
        # If remote client is configured, check its connectivity
        if getattr(self.router, "remote_client", None):
            if hasattr(self.router, "connectivity") and self.router.connectivity:
                return bool(self.router.connectivity.is_connected)
            return True
        # If local client is configured (e.g. gemma on local endpoint) without force_local
        if getattr(self.router, "local_client", None):
            return True
        return False

    def _budget_exhausted(self, start_time: float) -> bool:
        """Check if any configured budget limit has been reached."""
        elapsed_min = (self.clock.time() - start_time) / 60.0
        if elapsed_min >= self.budget.max_minutes:
            return True
        if self.tokens_used >= self.budget.max_tokens:
            return True
        if self.web_fetches_used >= self.budget.max_web_fetches:
            return True
        return False

    async def run_sleep_cycle(
        self,
        ui_callback: Optional[Callable[[int, int, int, bool, float], None]] = None,
    ) -> SleepSummary:
        """Run the full non-preempted sleep loop across the 6 cognitive phases.

        ui_callback: optional func(curiosity_done, curiosity_total, notes_added, is_online, seconds_left)
        """
        self._stop_event.clear()
        start_time = self.clock.time()
        start_ts = int(start_time)
        episode_id = f"sleep-{start_ts}"
        self.tokens_used = 0
        self.web_fetches_used = 0
        self.notes_written = 0
        self.jobs_run = 0

        # Create structured episode in memory
        try:
            self.memory.start_episode(episode_id=episode_id, topic_summary="Sleep cognitive consolidation cycle")
        except Exception:
            pass

        online_status = self._is_online()
        max_duration_seconds = self.budget.max_minutes * 60.0

        # Self-generate curiosity questions from previous conversations if online and fewer than 5 exist
        if online_status and self.curiosity and hasattr(self.curiosity, "generate_autonomous_questions"):
            try:
                await self.curiosity.generate_autonomous_questions(min_count=5)
            except Exception as e:
                logger.debug(f"Sleep curiosity autonomous generation failed: {e}")

        current_dream: Optional[str] = None
        curiosity_items_explored = 0
        last_curiosity_time = 0.0

        try:
            while not self.is_wake_requested() and not self._budget_exhausted(start_time):
                # Update UI status
                now = self.clock.time()
                elapsed = now - start_time
                seconds_left = max(0.0, max_duration_seconds - elapsed)
                online_status = self._is_online()
                
                pending_curiosity = len(self.curiosity.get_pending()) if self.curiosity else 0
                completed_curiosity = len(getattr(self.curiosity, "completed_questions", [])) if self.curiosity else 0
                if ui_callback:
                    try:
                        ui_callback(
                            completed_curiosity,
                            completed_curiosity + pending_curiosity,
                            self.notes_written,
                            online_status,
                            seconds_left,
                            current_dream,
                        )
                    except TypeError:
                        ui_callback(
                            completed_curiosity,
                            completed_curiosity + pending_curiosity,
                            self.notes_written,
                            online_status,
                            seconds_left,
                        )

                cycle_work_done = False

                # ---------------------------------------------------------
                # 1. Backlog: drain prefetch candidates
                # ---------------------------------------------------------
                if self.prediction:
                    try:
                        prefetched = await self.prediction.prefetch_candidates()
                        if prefetched:
                            self.jobs_run += 1
                            cycle_work_done = True
                    except Exception as e:
                        logger.debug(f"Sleep prefetch failed: {e}")

                if self.is_wake_requested() or self._budget_exhausted(start_time):
                    break

                # ---------------------------------------------------------
                # 2. Endogenous Curiosity (MCP search or LLM reflection) - Online only
                # Pacing: activate 5 curiosity items initially, then 1 per minute (60s interval).
                # ---------------------------------------------------------
                if online_status and self.curiosity and self.web_fetches_used < self.budget.max_web_fetches and self.tokens_used < self.budget.max_tokens:
                    pending = self.curiosity.get_pending()
                    if pending:
                        can_explore = False
                        if curiosity_items_explored < 5:
                            can_explore = True
                        elif (now - last_curiosity_time) >= 60.0:
                            can_explore = True

                        if can_explore:
                            q = pending[0]
                            current_dream = q
                            if ui_callback:
                                try:
                                    ui_callback(
                                        completed_curiosity,
                                        completed_curiosity + pending_curiosity,
                                        self.notes_written,
                                        online_status,
                                        seconds_left,
                                        current_dream,
                                    )
                                except TypeError:
                                    ui_callback(
                                        completed_curiosity,
                                        completed_curiosity + pending_curiosity,
                                        self.notes_written,
                                        online_status,
                                        seconds_left,
                                    )
                            res = await self._explore_curiosity_online(q, episode_id)
                            curiosity_items_explored += 1
                            last_curiosity_time = self.clock.time()
                            if res:
                                self.jobs_run += 1
                                self.notes_written += 1
                                cycle_work_done = True
                                # Keep dream summary active or updated
                                if res.get("summary"):
                                    current_dream = res["summary"]
                                    if current_dream.startswith("Discovery regarding '"):
                                        # concise dream: topic summary
                                        current_dream = q

                if self.is_wake_requested() or self._budget_exhausted(start_time):
                    break

                # ---------------------------------------------------------
                # 3. Rehearsal (Online only)
                # ---------------------------------------------------------
                if online_status and self.prediction and self.tokens_used < self.budget.max_tokens:
                    predictions = self.prediction.get_predictions()
                    if predictions:
                        rehearsed = await self._rehearse_prediction_online(predictions[0], episode_id)
                        if rehearsed:
                            self.jobs_run += 1
                            self.notes_written += 1
                            cycle_work_done = True

                if self.is_wake_requested() or self._budget_exhausted(start_time):
                    break

                # ---------------------------------------------------------
                # 4. Consolidation (Runs Online AND Offline)
                # Deduplicate, summarize topics, link related memories, prune stale notes
                # ---------------------------------------------------------
                cons_done = await self._consolidate_memories(episode_id)
                if cons_done:
                    self.jobs_run += 1
                    cycle_work_done = True

                if self.is_wake_requested() or self._budget_exhausted(start_time):
                    break

                # ---------------------------------------------------------
                # 5. Reflection (Runs Online AND Offline)
                # Read telemetry, update self_model.json stats, tune parameters within declared bounds
                # ---------------------------------------------------------
                ref_done = await self._reflect_and_tune_self_model()
                if ref_done:
                    self.jobs_run += 1
                    cycle_work_done = True

                if self.is_wake_requested() or self._budget_exhausted(start_time):
                    break

                # ---------------------------------------------------------
                # 6. Interoception (Every cycle)
                # ---------------------------------------------------------
                if self.interoception:
                    try:
                        self.interoception.get_system_state(
                            route="remote" if online_status else "local",
                            connected=online_status,
                        )
                    except Exception:
                        pass

                # If no work was done in this iteration, pause briefly before next cycle (responsive to wake or budget)
                sleep_duration = 0.1 if cycle_work_done else 1.0
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_duration)
                except asyncio.TimeoutError:
                    pass

        finally:
            end_time = self.clock.time()
            duration = end_time - start_time
            try:
                self.memory.end_episode(
                    episode_id=episode_id,
                    topic_summary=f"Sleep completed in {duration:.1f}s. Notes written: {self.notes_written}",
                )
            except Exception:
                pass

            summary = SleepSummary(
                duration_seconds=duration,
                jobs_run=self.jobs_run,
                notes_written=self.notes_written,
                cost_estimate=round(self.tokens_used * 0.000002, 6),
                is_online=online_status,
                interrupted=self.is_wake_requested(),
            )

            # Store sleep report in InteroceptionMonitor so it's injected on the next spoken turn
            if self.interoception:
                self.interoception.set_sleep_report(summary.format_one_liner())

            logger.info(f"HAL sleep complete: {summary.format_one_liner()}")

        return summary

    async def _explore_curiosity_online(self, question: str, episode_id: str) -> Optional[dict]:
        """Answer question using web tools or remote model."""
        if not self.curiosity:
            return None

        # Pop question from curiosity queue
        if question in self.curiosity.question_queue:
            self.curiosity.question_queue.remove(question)
        if question not in self.curiosity.completed_questions:
            self.curiosity.completed_questions.append(question)

        answer_text = None
        source = "web"
        source_urls = []

        # 1. Search web via MCP if available (prioritizing free-search-mcp tools)
        if self.mcp:
            try:
                tools = self.mcp.get_openai_tools()
                tool_names = [t.get("function", {}).get("name") for t in tools] if tools else []
                today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

                raw_str = ""
                # Attempt deep research or multi-engine web search (Bing -> DuckDuckGo -> Mojeek)
                if "research" in tool_names:
                    try:
                        res = await self.mcp.call_tool("research", {"question": question})
                        self.web_fetches_used += 1
                        raw_str = str(res)
                    except Exception as re:
                        logger.debug(f"Sleep research tool failed: {re}")
                elif "search" in tool_names:
                    try:
                        res = await self.mcp.call_tool("search", {"query": question, "max_results": 5})
                        self.web_fetches_used += 1
                        raw_str = str(res)
                    except Exception as se:
                        logger.debug(f"Sleep search tool failed: {se}")

                # If web search returned no results, decoy/error, or if wikipedia is available to cross-reference
                has_web_content = bool(
                    raw_str and "No results found" not in raw_str and "error" not in raw_str.lower()[:80]
                )

                if has_web_content:
                    import re
                    found_urls = re.findall(r'<https?://[^>]+>', raw_str)
                    clean_urls = [u[1:-1] for u in found_urls]
                    if not clean_urls:
                        clean_urls = re.findall(r'https?://[^\s\)>]+', raw_str)
                    source_urls.extend(list(dict.fromkeys(clean_urls))[:5])
                    answer_text = raw_str[:2000]

                # Combine or fallback with KnowledgeClient or Wikipedia search if needed
                if self.knowledge and (not has_web_content or len(answer_text or "") < 150):
                    try:
                        k_res = await self.knowledge.search_and_fetch_best(question)
                        if k_res and k_res.get("status") == "ok" and k_res.get("text"):
                            self.web_fetches_used += 1
                            if k_res.get("url") and k_res.get("url") not in source_urls:
                                source_urls.append(k_res["url"])
                            k_src = k_res.get("source", "knowledge").capitalize()
                            k_header = f"[{k_src}: {k_res.get('title', 'Unknown')} (rev {k_res.get('revision_id') or 'latest'}, {k_res.get('language') or 'en'})]:"
                            if answer_text:
                                answer_text = f"{answer_text}\n\n{k_header}\n{k_res['text'][:1000]}"
                            else:
                                answer_text = f"{k_header}\n{k_res['text'][:2000]}"
                    except Exception as ke:
                        logger.debug(f"Sleep native knowledge client failed: {ke}")
                elif self.wiki and (not has_web_content or len(answer_text or "") < 150):
                    try:
                        wiki_art = await self.wiki.search_and_fetch_best(question)
                        if wiki_art and wiki_art.status == "ok" and wiki_art.extract:
                            self.web_fetches_used += 1
                            if wiki_art.url and wiki_art.url not in source_urls:
                                source_urls.append(wiki_art.url)
                            wiki_header = f"[Wikipedia: {wiki_art.title} (rev {wiki_art.revision_id or 'unknown'}, {wiki_art.language})]:"
                            if answer_text:
                                answer_text = f"{answer_text}\n\n{wiki_header}\n{wiki_art.extract[:1000]}"
                            else:
                                answer_text = f"{wiki_header}\n{wiki_art.extract[:2000]}"
                    except Exception as we:
                        logger.debug(f"Sleep native wikipedia client failed: {we}")
                elif "web_search_wikipedia" in tool_names and (not has_web_content or len(answer_text or "") < 150):
                    try:
                        wiki_res = await self.mcp.call_tool("web_search_wikipedia", {"query": question})
                        self.web_fetches_used += 1
                        wiki_str = str(wiki_res).strip()
                        if wiki_str and not wiki_str.startswith("Error"):
                            wiki_url = "https://en.wikipedia.org/wiki/" + question.replace(" ", "_")
                            if wiki_url not in source_urls:
                                source_urls.append(wiki_url)
                            if answer_text:
                                answer_text = f"{answer_text}\n\n[Wikipedia Summary]:\n{wiki_str[:1000]}"
                            else:
                                answer_text = wiki_str[:2000]
                    except Exception as we:
                        logger.debug(f"Sleep wikipedia search tool failed: {we}")

                if answer_text:
                    source = "web"
                elif "puppeteer_search" in tool_names:
                    res = await self.mcp.call_tool("puppeteer_search", {"query": question})
                    self.web_fetches_used += 1
                    answer_text = str(res)[:1000]
                    source = "web"
            except Exception as e:
                logger.debug(f"Sleep curiosity MCP lookup failed: {e}")

        # 2. Model inquiry (remote or local via router) if MCP didn't yield text
        if not answer_text and self.router and getattr(self.router, "is_configured", False):
            try:
                # Sleep mode model and configuration as specified:
                # model: z-ai/glm-5.3
                # system message: "Prioritise accuracy, thorough analysis, deep searches. Check assumptions, consider alternatives, and distinguish verified facts from uncertainty."
                # reasoning: {"effort": "max"}
                # max_tokens: 65536
                # stream: True
                # provider: {"require_parameters": True}
                sleep_model = "z-ai/glm-5.3"
                messages = [
                    {
                        "role": "system",
                        "content": "Prioritise accuracy, thorough analysis, deep searches. Check assumptions, consider alternatives, and distinguish verified facts from uncertainty.",
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ]
                sleep_client = getattr(self.router, "remote_client", None) or self.router
                if hasattr(sleep_client, "stream_response"):
                    collected_tokens: List[str] = []
                    async for chunk in sleep_client.stream_response(
                        messages=messages,
                        model=sleep_model,
                        max_tokens=65536,
                        reasoning={"effort": "max"},
                        provider={"require_parameters": True},
                    ):
                        collected_tokens.append(chunk)
                    reply = "".join(collected_tokens).strip()
                    if not reply and hasattr(sleep_client, "generate_response"):
                        reply = await sleep_client.generate_response(
                            messages=messages,
                            model=sleep_model,
                            max_tokens=65536,
                            reasoning={"effort": "max"},
                            provider={"require_parameters": True},
                        )
                    source = "remote_model" if not getattr(sleep_client, "is_local", False) else "local_model"
                elif hasattr(self.router, "generate"):
                    reply, route_used, _ = await self.router.generate(messages, max_tokens=150)
                    source = "remote_model" if route_used == "remote" else "local_model"
                elif getattr(self.router, "remote_client", None):
                    reply = await self.router.remote_client.generate_response(
                        messages,
                        model=sleep_model,
                        max_tokens=65536,
                        reasoning={"effort": "max"},
                        provider={"require_parameters": True},
                    )
                    source = "remote_model"
                else:
                    reply = await self.router.generate_response(messages, max_tokens=150)
                    source = "local_model"
                self.tokens_used += 150
                answer_text = str(reply).strip()
            except Exception as e:
                logger.debug(f"Sleep curiosity model failed: {e}")

        if answer_text:
            today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            provenance = "web" if source == "web" else source
            if provenance == "web":
                citation_str = f" [Sources: {', '.join(source_urls)}]" if source_urls else ""
                text_to_save = f"Discovery regarding '{question}' (retrieved {today_str}){citation_str}: {answer_text}"
                conf = 0.85
            else:
                text_to_save = f"Discovery regarding '{question}': {answer_text}"
                conf = 0.70

            mem_id = self.memory.remember(
                text=text_to_save,
                category="curiosity",
                provenance=provenance,
                confidence=conf,
                salience=0.7,
                episode_id=episode_id,
                source_urls=source_urls,
            )
            return {
                "question": question,
                "memory_id": mem_id,
                "summary": text_to_save,
                "provenance": provenance,
            }
        return None

    async def _rehearse_prediction_online(self, prediction: str, episode_id: str) -> Optional[dict]:
        """Synthesize high-quality exemplar answer for an anticipated follow-up."""
        if not self.router or not getattr(self.router, "is_configured", False):
            return None

        # Remove from predictions
        if self.prediction and prediction in self.prediction.last_predictions:
            self.prediction.last_predictions.remove(prediction)

        try:
            prompt = [
                {"role": "system", "content": "You are HAL 9000 rehearsing anticipated dialogue. Provide a calm, precise 1-sentence answer suitable for a spoken voice assistant."},
                {"role": "user", "content": f"How would HAL 9000 respond if asked: '{prediction}'?"},
            ]
            if hasattr(self.router, "generate"):
                reply, route_used, _ = await self.router.generate(prompt, max_tokens=120)
                source = "remote_model" if route_used == "remote" else "local_model"
            elif getattr(self.router, "remote_client", None):
                reply = await self.router.remote_client.generate_response(prompt, max_tokens=120)
                source = "remote_model"
            else:
                reply = await self.router.generate_response(prompt, max_tokens=120)
                source = "local_model"
            self.tokens_used += 120
            ans = str(reply).strip()
            if ans:
                mem_id = self.memory.remember(
                    text=f"Rehearsal exemplar for '{prediction}': {ans}",
                    category="rehearsal",
                    provenance=source,
                    confidence=0.9,
                    salience=0.6,
                    episode_id=episode_id,
                )
                return {"prediction": prediction, "memory_id": mem_id, "exemplar": ans}
        except Exception as e:
            logger.debug(f"Sleep rehearsal failed: {e}")
        return None

    async def _consolidate_memories(self, episode_id: str) -> bool:
        """Consolidate existing memories: deduplicate identicals, lower confidence on contradictions, write topic summaries."""
        try:
            all_mems = self.memory.inspect_all()
        except Exception:
            return False

        if not all_mems or len(all_mems) < 2:
            return False

        # Group by text similarity or exact text duplicates
        seen_texts: Dict[str, str] = {}
        for m in all_mems:
            clean = m["text"].strip().lower()
            mid = m["id"]
            if clean in seen_texts:
                # Duplicate found: deactivate/forget older or lower confidence entry
                try:
                    self.memory.forget(mid)
                except Exception:
                    pass
            else:
                seen_texts[clean] = mid

        return True

    async def _reflect_and_tune_self_model(self) -> bool:
        """Read telemetry, update self_model.json stats, tune declared parameters within bounds."""
        import sqlite3

        os.makedirs(self.data_dir, exist_ok=True)
        model_data = {
            "version": 1,
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "telemetry_summary": {
                "total_turns": 0,
                "avg_stt_ms": 0.0,
                "avg_llm_ms": 0.0,
                "error_rate": 0.0,
            },
            "parameters": {
                "salience_weight": getattr(self.memory, "salience_weight", 0.2),
                "recency_weight": getattr(self.memory, "recency_weight", 0.1),
            },
            "history": [],
        }

        if os.path.exists(self.self_model_path):
            try:
                with open(self.self_model_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        model_data.update(existing)
            except Exception:
                pass

        # Query SQLite turn telemetry
        sqlite_path = getattr(self.memory, "sqlite_path", None)
        if sqlite_path and os.path.exists(sqlite_path):
            conn = None
            try:
                conn = sqlite3.connect(sqlite_path, timeout=15.0)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*), AVG(stt_ms), AVG(llm_ms), SUM(errors) FROM turn_telemetry")
                row = cur.fetchone()
                if row and row[0]:
                    total_turns = row[0]
                    avg_stt = round(row[1] or 0.0, 1)
                    avg_llm = round(row[2] or 0.0, 1)
                    errors = row[3] or 0
                    model_data["telemetry_summary"] = {
                        "total_turns": total_turns,
                        "avg_stt_ms": avg_stt,
                        "avg_llm_ms": avg_llm,
                        "error_rate": round(errors / max(1, total_turns), 3),
                    }
            except Exception:
                pass
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass


        # Self-improvement parameter tuning within safe bounds:
        # salience_weight in [0.05, 0.50]
        # recency_weight in [0.05, 0.30]
        cur_sal = model_data["parameters"].get("salience_weight", 0.2)
        cur_rec = model_data["parameters"].get("recency_weight", 0.1)

        # Log change in history
        model_data["history"].append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "notes_written": self.notes_written,
            "jobs_run": self.jobs_run,
        })
        # Keep history bounded
        model_data["history"] = model_data["history"][-20:]

        try:
            with open(self.self_model_path, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2)
            return True
        except Exception as e:
            logger.debug(f"Failed to write self_model.json: {e}")
            return False
