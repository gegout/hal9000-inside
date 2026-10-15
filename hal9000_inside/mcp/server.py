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
    try:
        mem = get_memory_store()
        results = mem.search(query, limit=limit)
        return json.dumps(results, indent=2)
    except RuntimeError as e:
        if "locked by another running process" in str(e) or "already accessed" in str(e):
            return json.dumps({"error": "Memory store is currently locked by active assistant process."})
        raise


@mcp_server.tool(name="memory_inspect_all", description="Inspect all stored long-term facts, relationships, and user preferences stored in HAL 9000 memory.")
def memory_inspect_all() -> str:
    """List all stored active memories."""
    try:
        mem = get_memory_store()
        results = mem.inspect_all()
        return json.dumps(results, indent=2)
    except RuntimeError as e:
        if "locked by another running process" in str(e) or "already accessed" in str(e):
            return json.dumps({"error": "Memory store is currently locked by active assistant process."})
        raise


@mcp_server.tool(name="memory_get_recent_dialogue", description="Retrieve recent dialogue turns between user and HAL 9000 for conversation continuity.")
def memory_get_recent_dialogue(limit: int = 10) -> str:
    """Retrieve recent conversation turns."""
    try:
        mem = get_memory_store()
        turns = mem.get_recent_turns(limit=limit)
        return json.dumps(turns, indent=2)
    except RuntimeError as e:
        if "locked by another running process" in str(e) or "already accessed" in str(e):
            return json.dumps({"error": "Memory store is currently locked by active assistant process."})
        raise


@mcp_server.tool(name="memory_remember", description="Store a new fact, preference, or learned knowledge into HAL 9000's persistent memory.")
def memory_remember(text: str, category: str = "knowledge") -> str:
    """Add new fact/knowledge to memory."""
    try:
        mem = get_memory_store()
        mem_id = mem.remember(text, category=category)
        return json.dumps({"status": "stored", "memory_id": mem_id})
    except RuntimeError as e:
        if "locked by another running process" in str(e) or "already accessed" in str(e):
            return json.dumps({"error": "Memory store is currently locked by active assistant process."})
        raise


from hal9000_inside.wikipedia import WikipediaClient, KnowledgeClient

_wiki_client: Optional[WikipediaClient] = None
_knowledge_client: Optional[KnowledgeClient] = None


def get_wiki_client() -> WikipediaClient:
    global _wiki_client
    if _wiki_client is None:
        _wiki_client = WikipediaClient()
    return _wiki_client


def get_knowledge_client() -> KnowledgeClient:
    global _knowledge_client
    if _knowledge_client is None:
        _knowledge_client = KnowledgeClient()
    return _knowledge_client


@mcp_server.tool(name="web_search_wikipedia", description="Search Wikipedia using native MediaWiki API and return candidate articles with page IDs and snippets.")
async def web_search_wikipedia(query: str, language: str = "en") -> str:
    """Search Wikipedia using MediaWiki API."""
    client = get_wiki_client()
    try:
        results = await client.search(query, limit=5, language=language)
        if not results:
            return json.dumps({"status": "empty", "message": f"No Wikipedia results found for '{query}'.", "results": []})
        return json.dumps({"status": "ok", "query": query, "language": language, "results": [r.to_dict() for r in results]}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Wikipedia search failed: {e}"})


@mcp_server.tool(name="web_fetch_wikipedia_page", description="Fetch structured Wikipedia article content by numeric page ID using direct MediaWiki API.")
async def web_fetch_wikipedia_page(page_id: int, language: str = "en") -> str:
    """Fetch structured article content from Wikipedia by page ID."""
    client = get_wiki_client()
    try:
        article = await client.fetch(page_id, language=language)
        return json.dumps(article.to_dict(), indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Wikipedia fetch failed: {e}"})


@mcp_server.tool(
    name="knowledge_search",
    description="Search across Wikipedia, Wikidata, arXiv, or Stack Exchange using native asynchronous APIs without external subprocesses.",
)
async def knowledge_search(
    query: str,
    source: str = "auto",
    limit: int = 5,
    language: str = "en",
    site: str = "stackoverflow",
) -> str:
    """Multi-source knowledge search (wikipedia, wikidata, arxiv, stackexchange)."""
    client = get_knowledge_client()
    try:
        res = await client.search(
            query=query,
            source=source,
            limit=limit,
            language=language,
            site=site,
        )
        return json.dumps(res, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Knowledge search failed: {e}"})


@mcp_server.tool(
    name="knowledge_fetch",
    description="Fetch structured article, Wikidata claims, arXiv abstract, or Stack Exchange Q&A with accepted answers by identifier.",
)
async def knowledge_fetch(
    source: str,
    identifier: str,
    max_chars: int = 16000,
    language: str = "en",
    site: str = "stackoverflow",
) -> str:
    """Multi-source knowledge fetch for specific items."""
    client = get_knowledge_client()
    try:
        res = await client.fetch(
            source=source,
            identifier=identifier,
            max_chars=max_chars,
            language=language,
            site=site,
        )
        return json.dumps(res, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Knowledge fetch failed: {e}"})


@mcp_server.tool(name="web_fetch_page", description="Fetch and extract readable text from any web URL using Puppeteer headless Chrome (e.g. documentation, articles, external websites).")
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
