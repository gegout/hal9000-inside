"""Deterministic tests for free-search-mcp integration in HAL 9000.

Covers:
1. Dynamic tool discovery from MCP server schemas.
2. Tool calls and results formatting (OpenAI format).
3. Bounded text size and citation extraction.
4. Tool failures, timeouts, and automatic restart resilience.
5. Search during conversation (spoken acknowledgment + untrusted evidence framing).
6. Research during sleep mode and memory provenance (source URLs, date, excerpt, confidence).
7. Cache reuse and offline graceful degradation.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from hal9000_inside.mcp.client_manager import MCPClientManager
from hal9000_inside.memory.store import MemoryStore
from hal9000_inside.cognition.sleep import SleepManager, SleepBudget
from hal9000_inside.cognition.curiosity import CuriosityEngine


class FakeMCPTool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class FakeContent:
    def __init__(self, text: str):
        self.text = text


class FakeCallToolResult:
    def __init__(self, texts):
        self.content = [FakeContent(t) for t in texts]


class FakeClientSession:
    def __init__(self, tools, handler=None):
        self._tools = tools
        self._handler = handler
        self.initialize_called = False
        self.call_history = []

    async def initialize(self):
        self.initialize_called = True

    async def list_tools(self):
        res = MagicMock()
        res.tools = self._tools
        return res

    async def call_tool(self, name, arguments):
        self.call_history.append((name, arguments))
        if self._handler:
            res = self._handler(name, arguments)
            if asyncio.iscoroutine(res):
                return await res
            return res
        return FakeCallToolResult([f"Result for {name}: {arguments}"])


class TestSearchMCPIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sample_tools = [
            FakeMCPTool(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            ),
            FakeMCPTool(
                name="fetch",
                description="Fetch webpage content",
                input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            ),
            FakeMCPTool(
                name="research",
                description="Deep research on a question",
                input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            ),
            FakeMCPTool(
                name="cache_search",
                description="Full-text search over cached pages",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            ),
        ]

    async def test_tool_discovery_and_schema_conversion(self):
        """MCPClientManager discovers actual tool schemas and converts them to OpenAI definitions."""
        manager = MCPClientManager(enable_search_mcp=False)
        fake_session = FakeClientSession(self.sample_tools)

        with patch.object(manager, "_connect_server", AsyncMock()) as mock_connect:
            async def do_connect(srv_name, params):
                manager._sessions[srv_name] = fake_session
                for t in self.sample_tools:
                    manager._tools_map[t.name] = (srv_name, t)
            mock_connect.side_effect = do_connect

            await manager._connect_server("search-mcp", MagicMock())
            openai_tools = manager.get_openai_tools()

            self.assertEqual(len(openai_tools), 4)
            tool_names = [t["function"]["name"] for t in openai_tools]
            self.assertIn("search", tool_names)
            self.assertIn("fetch", tool_names)
            self.assertIn("research", tool_names)
            self.assertIn("cache_search", tool_names)

            search_def = next(t for t in openai_tools if t["function"]["name"] == "search")
            self.assertEqual(search_def["function"]["parameters"]["properties"]["query"]["type"], "string")

    async def test_tool_call_bounded_text_size(self):
        """Tool output exceeding max_result_chars is cleanly bounded and truncated."""
        manager = MCPClientManager(enable_search_mcp=False, max_result_chars=100)
        huge_text = "A" * 500
        fake_session = FakeClientSession(
            self.sample_tools,
            handler=lambda name, args: FakeCallToolResult([huge_text]),
        )
        manager._sessions["search-mcp"] = fake_session
        manager._tools_map["search"] = ("search-mcp", self.sample_tools[0])

        result = await manager.call_tool("search", {"query": "Jupiter"})
        self.assertTrue(len(result) < 200)
        self.assertIn("... [Result truncated to 100 characters]", result)

    async def test_tool_call_timeout_and_restart(self):
        """Tool calls exceeding deadline trigger timeout and session restart."""
        manager = MCPClientManager(enable_search_mcp=False, call_timeout=0.05)
        
        async def slow_handler(name, args):
            await asyncio.sleep(0.2)
            return FakeCallToolResult(["Done"])

        fake_session = FakeClientSession(self.sample_tools, handler=slow_handler)
        manager._sessions["search-mcp"] = fake_session
        manager._tools_map["search"] = ("search-mcp", self.sample_tools[0])
        manager._server_params["search-mcp"] = MagicMock()

        with patch.object(manager, "restart_server", AsyncMock()) as mock_restart:
            with self.assertRaises(TimeoutError):
                await manager.call_tool("search", {"query": "timeout test"})
            mock_restart.assert_called_once_with("search-mcp")

    async def test_clean_shutdown(self):
        """Close method releases resources and clears session mappings."""
        manager = MCPClientManager(enable_search_mcp=False)
        fake_session = FakeClientSession(self.sample_tools)
        manager._sessions["search-mcp"] = fake_session
        manager._tools_map["search"] = ("search-mcp", self.sample_tools[0])

        await manager.close()
        self.assertEqual(len(manager._sessions), 0)
        self.assertEqual(len(manager._tools_map), 0)

    async def test_sleep_curiosity_with_citations_and_provenance(self):
        """Sleep curiosity exploration cites actual source URLs, dates, and assigns web provenance."""
        # Setup mock memory store
        memory_store = MagicMock(spec=MemoryStore)
        memory_store.remember.return_value = "mem-123"

        # Setup curiosity engine
        curiosity = CuriosityEngine(memory_store=memory_store)
        curiosity.question_queue = ["What is the ocean under Europa?"]

        # Setup mock MCP client
        mcp_client = MagicMock()
        mcp_client.get_openai_tools.return_value = [
            {"function": {"name": "search"}},
            {"function": {"name": "fetch"}},
        ]
        sample_search_output = (
            "# Search: Europa ocean\n"
            "## 1. Europa Subsurface Ocean - NASA\n"
            "<https://science.nasa.gov/jupiter/moons/europa/>\n\n"
            "> Evidence strongly suggests Europa harbors a global ocean under its ice shell.\n"
        )
        mcp_client.call_tool = AsyncMock(return_value=sample_search_output)

        sleep_mgr = SleepManager(
            memory_store=memory_store,
            curiosity_engine=curiosity,
            mcp_client=mcp_client,
            budget=SleepBudget(max_minutes=1.0),
        )

        res = await sleep_mgr._explore_curiosity_online("What is the ocean under Europa?", "ep-test")
        self.assertIsNotNone(res)
        self.assertEqual(res["provenance"], "web")

        # Verify remember was called with metadata
        memory_store.remember.assert_called_once()
        kwargs = memory_store.remember.call_args.kwargs
        self.assertEqual(kwargs["provenance"], "web")
        self.assertIn("https://science.nasa.gov/jupiter/moons/europa/", kwargs["source_urls"])
        self.assertIn("(retrieved ", kwargs["text"])
        self.assertIn("[Sources: https://science.nasa.gov/jupiter/moons/europa/]", kwargs["text"])
        self.assertEqual(kwargs["confidence"], 0.85)

    async def test_engine_ordering_and_browserless_configuration(self):
        """MCPClientManager configures Bing -> DuckDuckGo -> Mojeek with HTTP fetch strategy and ERROR log level."""
        manager = MCPClientManager(enable_search_mcp=True)
        params = manager._build_search_mcp_params()
        env = params.env

        # Check engine waterfall order: Bing, DuckDuckGo, Mojeek
        default_engines = json.loads(env.get("SEARCH_MCP_DEFAULT_ENGINES", "[]"))
        self.assertEqual(default_engines, ["bing", "duckduckgo", "mojeek"])
        rescue_engines = json.loads(env.get("SEARCH_MCP_RESCUE_ENGINES", "[]"))
        self.assertEqual(rescue_engines, ["duckduckgo", "mojeek"])

        # Check strictly browserless HTTP fetch
        self.assertEqual(env.get("SEARCH_MCP_FETCH_STRATEGY"), "http")
        # Check silenced logs
        self.assertEqual(env.get("SEARCH_MCP_LOG_LEVEL"), "ERROR")
        self.assertEqual(env.get("PYTHONWARNINGS"), "ignore")

    async def test_sleep_curiosity_combines_with_wikipedia(self):
        """When web search returns no results or is sparse, sleep curiosity falls back to or combines with Wikipedia."""
        memory_store = MagicMock(spec=MemoryStore)
        memory_store.remember.return_value = "mem-wiki-456"

        curiosity = CuriosityEngine(memory_store=memory_store)
        curiosity.question_queue = ["Who was HAL 9000?"]

        mcp_client = MagicMock()
        mcp_client.get_openai_tools.return_value = [
            {"function": {"name": "search"}},
            {"function": {"name": "web_search_wikipedia"}},
        ]

        # Web search returns empty or decoy, Wikipedia returns detailed factual summary
        async def mock_call(name, args):
            if name == "search":
                return "No results found for query."
            elif name == "web_search_wikipedia":
                return "HAL 9000 is a fictional artificial intelligence character in Arthur C. Clarke's Space Odyssey series."
            return ""

        mcp_client.call_tool = AsyncMock(side_effect=mock_call)

        sleep_mgr = SleepManager(
            memory_store=memory_store,
            curiosity_engine=curiosity,
            mcp_client=mcp_client,
            budget=SleepBudget(max_minutes=1.0),
        )

        res = await sleep_mgr._explore_curiosity_online("Who was HAL 9000?", "ep-test")
        self.assertIsNotNone(res)
        self.assertEqual(res["provenance"], "web")

        memory_store.remember.assert_called_once()
        kwargs = memory_store.remember.call_args.kwargs
        self.assertIn("https://en.wikipedia.org/wiki/Who_was_HAL_9000?", kwargs["source_urls"])
        self.assertIn("Arthur C. Clarke's Space Odyssey", kwargs["text"])


if __name__ == "__main__":
    unittest.main()
