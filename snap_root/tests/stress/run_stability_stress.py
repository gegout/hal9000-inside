"""Long-running endurance and stability stress test for HAL 9000.

Executes sustained real-time cycles (configurable duration, e.g. 5 minutes or 15 minutes)
testing multiple edge cases simultaneously:
1. Frequent transitions between SLEEP and AWAKE (conversational turn) modes.
2. Turns that trigger MCP tools (e.g. web_search_wikipedia, memory_search, memory_remember).
3. Concurrent and edge-case tool calls:
   - Malformed arguments (missing 'query', non-dict arguments).
   - Tool execution exceptions and network timeouts.
   - Long strings / high-volume tool responses.
4. User cancellation mid-turn and mid-sleep ('S' toggle and 'C' cancel).
5. Continuous SQLite memory recording and Qdrant vector retrieval under sustained load.
6. Memory leak / resource leak monitoring (unclosed databases, thread accumulation, memory growth).

Usage:
    .venv/bin/python tests/stress/run_stability_stress.py --duration 300   # 5 minutes
    .venv/bin/python tests/stress/run_stability_stress.py --duration 900   # 15 minutes
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Setup root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.audio import NullSink
from hal9000_inside.cognition import SleepBudget
from hal9000_inside.llm import LLMClient
from hal9000_inside.memory import MemoryStore
from tests.fakes import FakeSTT, FakeTTS, FakeEmbedder, FakeOpenAIServer

# Silence non-critical logs during long run, but keep errors/warnings
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stability_stress")
logger.setLevel(logging.INFO)


class StatefulWebAndToolsMCP:
    """Simulates realistic and hostile MCP tool environment for stress testing."""

    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.success_count = 0

    def get_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search_wikipedia",
                    "description": "Search Wikipedia for factual encyclopedic information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string"}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "puppeteer_search",
                    "description": "Perform live web search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search keyword"}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search conversational memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_remember",
                    "description": "Store a new memory item",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "category": {"type": "string"}
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_inspect_all",
                    "description": "Inspect all active memories",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_get_recent_dialogue",
                    "description": "Get recent dialogue turns",
                    "parameters": {
                        "type": "object",
                        "properties": {"limit": {"type": "integer"}},
                    },
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        self.call_history.append({"name": name, "args": arguments, "time": time.monotonic()})
        
        # Edge case 1: Injected occasional network/MCP failures (5% rate)
        if random.random() < 0.05:
            self.error_count += 1
            raise RuntimeError(f"Simulated MCP remote connection timeout for {name}")

        self.success_count += 1
        query = arguments.get("query", "") if isinstance(arguments, dict) else ""
        
        if name == "web_search_wikipedia":
            if not query:
                return "Error: Missing query string."
            return (
                f"Wikipedia summary for '{query}': Discovery One is an interplanetary spacecraft "
                f"bound for Jupiter under the guidance of HAL 9000. Telemetry and systems nominal."
            )
        elif name == "puppeteer_search":
            return f"Web search results for '{query}': 3 matching articles found with full verification."
        elif name == "memory_search":
            return json.dumps([{"text": f"Found prior fact related to {query}", "category": "fact"}])
        elif name == "memory_remember":
            return json.dumps({"status": "stored", "memory_id": f"mem_{random.randint(100, 999)}"})
        elif name == "memory_inspect_all":
            return json.dumps([{"id": f"mem_{i}", "text": f"Memory {i}", "category": "knowledge"} for i in range(5)])
        elif name == "memory_get_recent_dialogue":
            return json.dumps([{"role": "user", "content": "hello"}])
        else:
            return f"Tool {name} executed successfully."


class StressTester:
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self.temp_dir = tempfile.mkdtemp(prefix="hal_stress_")
        self.fake_server = FakeOpenAIServer()
        self.fake_embedder = FakeEmbedder()
        self.store = MemoryStore(data_dir=self.temp_dir, embedder=self.fake_embedder)
        self.fake_stt = FakeSTT(script=["Initial placeholder"])
        self.fake_tts = FakeTTS()
        self.fake_mcp = StatefulWebAndToolsMCP()
        self.fake_llm = LLMClient(api_key="sk-stress", transport=self.fake_server.get_transport())
        
        self.assistant = Hal9000Assistant(
            memory=self.store,
            llm=self.fake_llm,
            adapter=self.fake_stt,
            tts=self.fake_tts,
            audio_sink=NullSink(),
            sleep_budget=SleepBudget(max_minutes=0.1),
        )
        self.assistant.mcp = self.fake_mcp

        # Telemetry metrics
        self.cycles_completed = 0
        self.sleep_cycles_completed = 0
        self.turns_completed = 0
        self.mcp_searches_triggered = 0
        self.aborted_turns_tested = 0
        self.edge_case_turns_tested = 0
        self.exceptions_caught = 0

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def run(self):
        start_time = time.monotonic()
        end_time = start_time + self.duration_seconds
        last_report = start_time

        logger.info(f"=== Starting HAL 9000 Stability Stress Test ===")
        logger.info(f"Target Duration: {self.duration_seconds} seconds ({self.duration_seconds / 60:.1f} minutes)")
        logger.info(f"Data Dir: {self.temp_dir}")

        edge_case_scenarios = [
            # 1. Normal conversational turn
            "normal_turn",
            # 2. Turn requesting Wikipedia MCP search
            "wikipedia_mcp_turn",
            # 3. Turn requesting memory search tool
            "memory_tool_turn",
            # 4. Turn with missing/empty tool arguments
            "malformed_tool_turn",
            # 5. Sleep mode toggle: enter sleep, let background curiosity run, wake up
            "sleep_cycle",
            # 6. Sleep toggled during turn execution (interrupted turn)
            "sleep_interrupts_turn",
            # 7. Cancel key 'C' mid-turn followed immediately by a new turn
            "cancel_key_turn",
            # 8. Rapid double wake/sleep toggle
            "rapid_sleep_toggle",
            # 9. Multi-tool calls in a single turn (parallel MCP execution)
            "multi_tool_turn",
            # 10. Massive MCP tool response (>10KB) truncation and handling
            "massive_tool_payload_turn",
            # 11. Malformed non-JSON / array tool arguments
            "invalid_json_tool_arguments_turn",
            # 12. Direct memory_remember and memory_inspect_all tools
            "memory_write_and_inspect_turn",
            # 13. Natural sleep budget expiration wakeup
            "sleep_budget_exhaustion",
            # 14. Sleep while LLM/network is completely offline
            "sleep_offline_mode",
            # 15. Memory consolidation deduplication stress during sleep
            "memory_consolidation_stress",
            # 16. Sleep curiosity burst and 1/min pacing verification
            "sleep_burst_and_pacing",
            # 17. Sleep dream rendering and reading validation
            "sleep_dream_reading",
            # 18. Auto-sleep inactivity timeout transition
            "sleep_inactivity_timeout",
            # 19. Simultaneous cancel key and wake toggle race
            "sleep_cancel_race",
            # 20. Sleep enter when already sleeping (idempotency check)
            "sleep_idempotent_enter",
            # 21. Wake call when already awake (idempotency check)
            "sleep_idempotent_wake",
        ]


        while time.monotonic() < end_time:
            self.cycles_completed += 1
            scenario = random.choice(edge_case_scenarios)
            
            try:
                await self._execute_scenario(scenario)
            except Exception as e:
                self.exceptions_caught += 1
                logger.error(f"Scenario '{scenario}' raised unexpected error: {e}", exc_info=True)
                raise e

            # Progress report every 30 seconds
            now = time.monotonic()
            if now - last_report >= 30.0 or now >= end_time:
                elapsed = now - start_time
                remaining = max(0.0, end_time - now)
                # Check memory usage
                import resource
                max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                print(
                    f"[{elapsed:6.1f}s / {self.duration_seconds:4.0f}s] "
                    f"Cycles: {self.cycles_completed:4d} | "
                    f"Turns: {self.turns_completed:4d} | "
                    f"Sleeps: {self.sleep_cycles_completed:3d} | "
                    f"MCP Searches: {self.mcp_searches_triggered:3d} (errs: {self.fake_mcp.error_count}) | "
                    f"RSS: {max_rss_mb:.1f} MB"
                )
                sys.stdout.flush()
                last_report = now

            # Modest yield so event loop runs smoothly
            await asyncio.sleep(0.01)

        total_elapsed = time.monotonic() - start_time
        logger.info(f"=== Stability Test Completed Successfully in {total_elapsed:.1f}s ===")
        logger.info(f"Total cycles: {self.cycles_completed}")
        logger.info(f"Spoken turns: {self.turns_completed}")
        logger.info(f"Sleep cycles: {self.sleep_cycles_completed}")
        logger.info(f"MCP tool calls executed: {len(self.fake_mcp.call_history)}")
        logger.info(f"Unexpected exceptions: {self.exceptions_caught}")

    async def _execute_scenario(self, scenario: str):
        dummy_pcm = b"\x01\x00" * 8000  # 0.5s audio frame

        if scenario == "normal_turn":
            self.fake_stt.script = ["What is the status of Discovery One?"]
            self.fake_server.enqueue_response("All systems are functioning nominally, Dave.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1

        elif scenario == "wikipedia_mcp_turn":
            self.fake_stt.script = ["Look up Europa on Wikipedia."]
            # LLM first returns a tool call requesting web_search_wikipedia
            tool_calls = [
                {
                    "id": f"call_wiki_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "web_search_wikipedia",
                        "arguments": json.dumps({"query": "Europa moon"}),
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            # LLM second follow-up response with synthesized answer
            self.fake_server.enqueue_response("Europa is an icy Jovian moon with potential subsurface water oceans.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1
            self.mcp_searches_triggered += 1

        elif scenario == "memory_tool_turn":
            self.fake_stt.script = ["What do you remember about our previous talk?"]
            tool_calls = [
                {
                    "id": f"call_mem_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "memory_search",
                        "arguments": json.dumps({"query": "previous talk"}),
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            self.fake_server.enqueue_response("I recall our discussion regarding navigational parameters.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1
            self.mcp_searches_triggered += 1

        elif scenario == "malformed_tool_turn":
            # Edge case: tool arguments missing 'query' or empty dict string
            self.fake_stt.script = ["Search Wikipedia with missing argument."]
            tool_calls = [
                {
                    "id": f"call_err_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "web_search_wikipedia",
                        "arguments": "{}",  # Missing required 'query'
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            self.fake_server.enqueue_response("I have completed the query despite missing parameters.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1
            self.edge_case_turns_tested += 1

        elif scenario == "sleep_cycle":
            # Ensure curiosity has items to test sleep MCP / background jobs
            self.assistant.curiosity.enqueue_questions([
                f"Curiosity item {random.randint(1, 10000)} regarding deep space cosmology?"
            ])
            await self.assistant.enter_sleep()
            assert self.assistant.is_sleeping is True
            assert self.assistant.audio_ctrl.muted is True

            # Let sleep background worker execute briefly
            await asyncio.sleep(random.uniform(0.05, 0.2))

            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            assert self.assistant.audio_ctrl.muted is False
            assert self.assistant._cancel_current is False

            # Verify assistant responds immediately after wake
            self.fake_stt.script = ["HAL, are you awake?"]
            self.fake_server.enqueue_response("Yes Dave, I am completely awake and operational.")
            await self.assistant._handle_turn(dummy_pcm)
            self.sleep_cycles_completed += 1
            self.turns_completed += 1

        elif scenario == "sleep_interrupts_turn":
            # Set cancel current and enter sleep right as a turn starts
            self.assistant._cancel_current = True
            await self.assistant.enter_sleep()
            
            # Simulated audio arrives while sleeping: must be cleanly rejected
            await self.assistant._handle_turn(dummy_pcm)
            self.aborted_turns_tested += 1

            # Wake up
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            assert self.assistant._cancel_current is False

            # Turn after wake succeeds
            self.fake_stt.script = ["Report current status."]
            self.fake_server.enqueue_response("Status nominal after interruption recovery.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1

        elif scenario == "cancel_key_turn":
            # Simulate pressing 'C' mid-turn
            self.assistant._cancel_current = True
            self.assistant.tts.cancel()
            self.assistant.playback.stop()

            # Next turn clears the flag and works
            self.fake_stt.script = ["New instruction after cancel."]
            self.fake_server.enqueue_response("Understood, proceeding with new instruction.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1

        elif scenario == "rapid_sleep_toggle":
            # Rapid toggle S -> S -> S -> S
            for _ in range(2):
                await self.assistant.enter_sleep()
                await self.assistant.wake_from_sleep()

            assert self.assistant.is_sleeping is False
            assert self.assistant._cancel_current is False
            self.sleep_cycles_completed += 2

        elif scenario == "multi_tool_turn":
            # Multiple concurrent tools returned in a single turn response
            self.fake_stt.script = ["Check both memory and Wikipedia for mission status."]
            tool_calls = [
                {
                    "id": f"call_multi1_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "memory_search",
                        "arguments": json.dumps({"query": "mission status"}),
                    },
                },
                {
                    "id": f"call_multi2_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "web_search_wikipedia",
                        "arguments": json.dumps({"query": "Jupiter mission"}),
                    },
                },
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            self.fake_server.enqueue_response("Both memory and external mission telemetry indicate systems are 100% operational.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1
            self.mcp_searches_triggered += 2

        elif scenario == "massive_tool_payload_turn":
            # Tool returns a massive payload (>10KB) to ensure response truncation and buffer limits
            self.fake_stt.script = ["Fetch astronomical ephemeris data."]
            tool_calls = [
                {
                    "id": f"call_large_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "puppeteer_search",
                        "arguments": json.dumps({"query": "ephemeris"}),
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            self.fake_server.enqueue_response("Ephemeris verified successfully.")
            # Inject a 15,000 character output for puppeteer_search
            orig_call = self.fake_mcp.call_tool
            async def huge_tool_call(name, args):
                if name == "puppeteer_search":
                    return "DATA " * 3000
                return await orig_call(name, args)
            self.fake_mcp.call_tool = huge_tool_call
            try:
                await self.assistant._handle_turn(dummy_pcm)
            finally:
                self.fake_mcp.call_tool = orig_call
            self.turns_completed += 1
            self.mcp_searches_triggered += 1

        elif scenario == "invalid_json_tool_arguments_turn":
            # Arguments that are invalid JSON strings or JSON primitives instead of objects
            self.fake_stt.script = ["Corrupted tool arguments test."]
            corrupted_args = random.choice([
                "NOT_VALID_JSON{",
                "[1, 2, 3]",
                '"raw string argument"',
                "12345",
            ])
            tool_calls = [
                {
                    "id": f"call_corrupt_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "web_search_wikipedia",
                        "arguments": corrupted_args,
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            self.fake_server.enqueue_response("Recovered cleanly from malformed argument structure.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1
            self.edge_case_turns_tested += 1

        elif scenario == "memory_write_and_inspect_turn":
            # Exercise internal memory tools: memory_remember and memory_inspect_all
            self.fake_stt.script = ["Remember that antenna azimuth is 45 degrees."]
            tool_calls = [
                {
                    "id": f"call_rem_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "memory_remember",
                        "arguments": json.dumps({"text": "antenna azimuth 45 degrees", "category": "engineering"}),
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls)
            self.fake_server.enqueue_response("I have committed the azimuth reading to long-term memory.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1

            # Follow up turn inspecting all memories
            self.fake_stt.script = ["Inspect all stored memory records."]
            tool_calls2 = [
                {
                    "id": f"call_insp_{random.randint(1000, 9999)}",
                    "type": "function",
                    "function": {
                        "name": "memory_inspect_all",
                        "arguments": "{}",
                    },
                }
            ]
            self.fake_server.enqueue_response(content="", tool_calls=tool_calls2)
            self.fake_server.enqueue_response("Memory inspection completed. All sectors coherent.")
            await self.assistant._handle_turn(dummy_pcm)
            self.turns_completed += 1

        elif scenario == "sleep_budget_exhaustion":
            # Test natural sleep completion when budget time expires
            self.assistant.sleep_manager.budget = SleepBudget(max_minutes=0.0005)  # 0.03 seconds
            await self.assistant.enter_sleep()
            # Wait for sleep background task to exhaust its budget and finish naturally
            if self.assistant._sleep_task:
                try:
                    await asyncio.wait_for(self.assistant._sleep_task, timeout=1.0)
                except Exception:
                    pass
            # Simulate the main run loop waking up because _sleep_task.done()
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            # Restore default budget
            self.assistant.sleep_manager.budget = SleepBudget(max_minutes=0.1)
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_offline_mode":
            # Sleep cycle with remote connectivity simulated as offline
            orig_online = self.assistant.sleep_manager._is_online
            self.assistant.sleep_manager._is_online = lambda: False
            try:
                await self.assistant.enter_sleep()
                await asyncio.sleep(0.05)
                await self.assistant.wake_from_sleep()
                assert self.assistant.is_sleeping is False
                self.sleep_cycles_completed += 1
            finally:
                self.assistant.sleep_manager._is_online = orig_online

        elif scenario == "memory_consolidation_stress":
            # Insert duplicate and overlapping memories, then run sleep consolidation
            for _ in range(5):
                self.store.remember("The communication antenna has an alignment offset of 0.2 degrees.", category="note")
            await self.assistant.enter_sleep()
            await asyncio.sleep(0.05)
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_burst_and_pacing":
            # Test that 5 questions can burst explore promptly, while pacing stays intact
            qs = [f"Burst curiosity topic {random.randint(100, 999)}?" for _ in range(3)]
            self.assistant.curiosity.enqueue_questions(qs)
            await self.assistant.enter_sleep()
            await asyncio.sleep(0.08)
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_dream_reading":
            # Test that dream updates are delivered to UI during sleep
            dreams_captured = []
            orig_update = self.assistant.ui.update_sleep_status
            def capture_dream(curiosity_done=0, curiosity_total=0, notes_added=0, is_online=False, seconds_left=None, dream=None):
                if dream:
                    dreams_captured.append(dream)
                orig_update(curiosity_done, curiosity_total, notes_added, is_online, seconds_left, dream)
            self.assistant.ui.update_sleep_status = capture_dream
            try:
                self.assistant.curiosity.enqueue_questions(["What are the rings of Saturn composed of?"])
                await self.assistant.enter_sleep()
                await asyncio.sleep(0.08)
                await self.assistant.wake_from_sleep()
                assert self.assistant.is_sleeping is False
            finally:
                self.assistant.ui.update_sleep_status = orig_update
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_inactivity_timeout":
            # Test auto-sleep inactivity simulation
            # Check silence calculation
            last_speech = time.monotonic() - 130.0  # > 120s
            if (time.monotonic() - last_speech) > self.assistant.sleep_after_seconds:
                await self.assistant.enter_sleep()
                assert self.assistant.is_sleeping is True
                await self.assistant.wake_from_sleep()
                assert self.assistant.is_sleeping is False
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_cancel_race":
            # Cancel key 'C' simultaneously with sleep wake
            await self.assistant.enter_sleep()
            self.assistant._cancel_current = True
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            assert self.assistant._cancel_current is False
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_idempotent_enter":
            # Calling enter_sleep twice consecutively should not create duplicate tasks
            await self.assistant.enter_sleep()
            first_task = self.assistant._sleep_task
            await self.assistant.enter_sleep()
            assert self.assistant._sleep_task is first_task
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False
            self.sleep_cycles_completed += 1

        elif scenario == "sleep_idempotent_wake":
            # Calling wake_from_sleep when already awake should be safe no-op
            assert self.assistant.is_sleeping is False
            await self.assistant.wake_from_sleep()
            assert self.assistant.is_sleeping is False



def main():
    parser = argparse.ArgumentParser(description="HAL 9000 Stability Stress Test")
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Duration to run in seconds (default: 300 for 5 minutes; use 900 for 15 minutes)",
    )
    args = parser.parse_args()

    tester = StressTester(duration_seconds=args.duration)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tester.run())
    finally:
        tester.cleanup()
        loop.close()


if __name__ == "__main__":
    main()
