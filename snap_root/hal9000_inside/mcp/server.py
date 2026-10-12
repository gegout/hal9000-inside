"""HAL 9000 Model Context Protocol (MCP) Server.

Provides tools for:
- Reading and searching HAL 9000 embedded conversational memory (facts, turns, history)
- Adding/remembering new learned knowledge into memory
- Searching the web and Wikipedia via Puppeteer headless browser
- Fetching and reading full web page content via Puppeteer
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from hal9000_inside.memory import MemoryStore
from hal9000_inside.mcp.puppeteer_browser import PuppeteerBrowser

logger = logging.getLogger(__name__)

# Initialize MCP Server instance with mcp 2.x MCPServer
mcp_server = MCPServer("hal9000-mcp")

# Shared singletons
_memory_store: Optional[MemoryStore] = None
_browser: Optional[PuppeteerBrowser] = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def get_browser() -> PuppeteerBrowser:
    global _browser
    if _browser is None:
        _browser = PuppeteerBrowser()
    return _browser


@mcp_server.tool(name="memory_search", description="Search HAL 9000's embedded memory for user preferences, facts, past statements, and context using semantic retrieval.")
def memory_search(query: str, limit: int = 5) -> str:
    """Semantic retrieval from HAL 9000 memory."""
    mem = get_memory_store()
    results = mem.search(query, limit=limit)
    return json.dumps(results, indent=2)


@mcp_server.tool(name="memory_inspect_all", description="Inspect all stored long-term facts, relationships, and user preferences stored in HAL 9000 memory.")
def memory_inspect_all() -> str:
    """List all stored active memories."""
    mem = get_memory_store()
    results = mem.inspect_all()
    return json.dumps(results, indent=2)


@mcp_server.tool(name="memory_get_recent_dialogue", description="Retrieve recent dialogue turns between user and HAL 9000 for conversation continuity.")
def memory_get_recent_dialogue(limit: int = 10) -> str:
    """Retrieve recent conversation turns."""
    mem = get_memory_store()
    turns = mem.get_recent_turns(limit=limit)
    return json.dumps(turns, indent=2)


@mcp_server.tool(name="memory_remember", description="Store a new fact, preference, or learned knowledge into HAL 9000's persistent memory.")
def memory_remember(text: str, category: str = "knowledge") -> str:
    """Add new fact/knowledge to memory."""
    mem = get_memory_store()
    mem_id = mem.remember(text, category=category)
    return json.dumps({"status": "stored", "memory_id": mem_id})


@mcp_server.tool(name="web_search_wikipedia", description="Search Wikipedia using Puppeteer headless browser and extract readable text knowledge.")
async def web_search_wikipedia(query: str, language: str = "en") -> str:
    """Search Wikipedia using headless browser navigation."""
    browser = get_browser()
    result = await browser.search_wikipedia(query, lang=language)
    return json.dumps(result, indent=2)


@mcp_server.tool(name="web_fetch_page", description="Fetch and extract readable text from any web URL using Puppeteer headless Chrome (e.g. documentation, articles, https://mcpservers.org/servers/progamesigner/wikipedia-mcp).")
async def web_fetch_page(url: str) -> str:
    """Fetch text content from a web page using headless Chrome."""
    browser = get_browser()
    result = await browser.fetch_url(url)
    return json.dumps(result, indent=2)


def run_stdio():
    """Run standard stdio MCP server loop."""
    asyncio.run(mcp_server.run_stdio_async())


if __name__ == "__main__":
    run_stdio()
