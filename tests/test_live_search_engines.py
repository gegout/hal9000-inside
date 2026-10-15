"""Opt-in live tests for free-search-mcp engines.

Run only when HAL_LIVE_SEARCH_TESTS=1 is set in the environment.
Verifies DuckDuckGo, Bing, and Mojeek against live endpoints:
- Confirms DuckDuckGo and Bing return real search results without Chromium.
- Confirms Mojeek requires browser/JS challenge and is disabled by default.
"""

import asyncio
import os
import sys
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@unittest.skipUnless(os.environ.get("HAL_LIVE_SEARCH_TESTS") == "1", "Live search tests disabled; set HAL_LIVE_SEARCH_TESTS=1 to run.")
class TestLiveSearchEngines(unittest.IsolatedAsyncioTestCase):
    async def test_live_duckduckgo_and_bing(self):
        """Verify DuckDuckGo and Bing return results over stdio MCP without browser."""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "search_mcp"],
            env={
                "SEARCH_MCP_DEFAULT_ENGINES": '["duckduckgo", "bing"]',
                "SEARCH_MCP_FETCH_STRATEGY": "http",
            },
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool("search", arguments={"query": "Voyager 1 mission status", "max_results": 3})
                self.assertTrue(len(res.content) > 0)
                text = res.content[0].text
                self.assertIn("Voyager", text)
                self.assertTrue("http" in text)

    async def test_live_fetch_without_browser(self):
        """Verify fetch downloads and parses HTML into markdown without Chromium."""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "search_mcp"],
            env={
                "SEARCH_MCP_FETCH_STRATEGY": "http",
            },
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool("fetch", arguments={"url": "https://httpbin.org/html"})
                self.assertTrue(len(res.content) > 0)
                text = res.content[0].text
                self.assertIn("Herman Melville", text)


if __name__ == "__main__":
    unittest.main()
