"""MCP Client Manager for HAL 9000.

Connects to configured MCP servers (from ~/.gemini/config/mcp_config.json,
environment variables, or custom paths), or defaults to an embedded
free-search-mcp stdio subprocess server.

Discovers actual tool schemas dynamically, formats tools for OpenAI function calling,
and provides robust tool execution with timeouts, output bounding, restart, and clean shutdown.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_PATHS = [
    os.path.expanduser("~/.gemini/config/mcp_config.json"),
    os.path.expanduser("~/.config/hal9000/mcp_config.json"),
]

DEFAULT_TOOL_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RESULT_CHARS = 4000


class MCPClientManager:
    """Manages connections to MCP servers (including managed free-search-mcp) and exposes tools."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_search_mcp: bool = True,
        call_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    ):
        self.config_path = config_path or self._find_config_path()
        self.enable_search_mcp = enable_search_mcp
        self.call_timeout = call_timeout
        self.max_result_chars = max_result_chars

        self._exit_stack = AsyncExitStack()
        self._server_params: Dict[str, StdioServerParameters] = {}
        self._sessions: Dict[str, ClientSession] = {}
        # tool_name -> (server_name, tool_def)
        self._tools_map: Dict[str, Tuple[str, Any]] = {}
        self._initialized = False

    def _find_config_path(self) -> Optional[str]:
        for p in DEFAULT_MCP_CONFIG_PATHS:
            if os.path.isfile(p):
                return p
        return None

    def _build_search_mcp_params(self) -> StdioServerParameters:
        """Construct StdioServerParameters for local free-search-mcp subprocess.
        
        Configured to run without Chromium:
        - Default engines: DuckDuckGo and Bing.
        - Mojeek disabled (requires JS browser challenge).
        - Fetch strategy: http only (no Playwright browser).
        - Cache directory configured to user cache or snap user data.
        """
        env = dict(os.environ)
        # Search engine waterfall: Bing -> DuckDuckGo -> Mojeek (all browserless HTTP)
        env["SEARCH_MCP_DEFAULT_ENGINES"] = json.dumps(["bing", "duckduckgo", "mojeek"])
        env["SEARCH_MCP_RESCUE_ENGINES"] = json.dumps(["duckduckgo", "mojeek"])
        env["SEARCH_MCP_RESERVE_ENGINES"] = json.dumps([])
        env["SEARCH_MCP_FRESH_ENGINES"] = json.dumps([])
        env["SEARCH_MCP_FETCH_STRATEGY"] = "http"
        # Silence third-party search and parser loggers to prevent terminal clutter
        env["SEARCH_MCP_LOG_LEVEL"] = "ERROR"
        env["PYTHONWARNINGS"] = "ignore"

        cache_base = os.environ.get("SNAP_USER_DATA", os.path.expanduser("~"))
        env["SEARCH_MCP_CACHE_DIR"] = os.path.join(cache_base, ".cache", "search-mcp")

        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "search_mcp"],
            env=env,
        )

    def load_server_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load configured MCP servers from JSON file."""
        if not self.config_path or not os.path.isfile(self.config_path):
            logger.debug(f"No MCP config file found at {self.config_path}")
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("mcpServers", {})
        except Exception as e:
            logger.warning(f"Error parsing MCP config file {self.config_path}: {e}")
            return {}

    async def _connect_server(self, srv_name: str, params: StdioServerParameters):
        """Connect to a single MCP server given its StdioServerParameters."""
        self._server_params[srv_name] = params
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._sessions[srv_name] = session

        # Discover tools
        tool_list = await session.list_tools()
        for tool in tool_list.tools:
            tool_key = tool.name
            # If name exists from another server, namespace with prefix
            if tool_key in self._tools_map and self._tools_map[tool_key][0] != srv_name:
                tool_key = f"{srv_name}_{tool.name}"
            self._tools_map[tool_key] = (srv_name, tool)
            logger.info(f"Discovered MCP tool: {tool_key} from [{srv_name}]")

    async def connect_all(self):
        """Connect to all defined MCP servers and discover their tools."""
        server_configs = self.load_server_configs()

        # Connect user-configured servers from JSON
        for srv_name, srv_conf in server_configs.items():
            command = srv_conf.get("command")
            if not command:
                continue

            args = srv_conf.get("args", [])
            env = dict(os.environ)
            if "env" in srv_conf and isinstance(srv_conf["env"], dict):
                env.update(srv_conf["env"])
            cwd = srv_conf.get("cwd")

            params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
            try:
                await self._connect_server(srv_name, params)
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server [{srv_name}]: {e}")

        # If search-mcp is enabled and not already provided by config, start managed search-mcp
        has_search_tool = any(
            t in self._tools_map for t in ("search", "fetch", "research", "cache_search")
        )
        if self.enable_search_mcp and not has_search_tool:
            try:
                search_params = self._build_search_mcp_params()
                await self._connect_server("search-mcp", search_params)
                logger.info("Started managed free-search-mcp stdio subprocess.")
            except Exception as e:
                logger.warning(f"Failed to start managed free-search-mcp: {e}")

        self._initialized = True

    async def restart_server(self, srv_name: str):
        """Restart a failed or timed-out MCP server session."""
        if srv_name not in self._server_params:
            raise ValueError(f"No stored parameters for MCP server '{srv_name}'")

        params = self._server_params[srv_name]
        logger.info(f"Restarting MCP server [{srv_name}]...")

        # Remove old session from active map
        self._sessions.pop(srv_name, None)
        # Re-connect
        await self._connect_server(srv_name, params)
        logger.info(f"MCP server [{srv_name}] successfully restarted.")

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Format discovered MCP tools as OpenAI-compatible function definitions."""
        openai_tools = []
        for name, (_, tool) in self._tools_map.items():
            parameters = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description or f"Tool {name}",
                    "parameters": parameters,
                }
            })
        return openai_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any], timeout: Optional[float] = None) -> str:
        """Call an MCP tool with timeout, bounded response size, and automatic restart on crash."""
        if name not in self._tools_map:
            raise KeyError(f"MCP tool '{name}' not found.")

        srv_name, original_tool = self._tools_map[name]
        session = self._sessions.get(srv_name)
        if not session:
            # Try restarting server
            await self.restart_server(srv_name)
            session = self._sessions.get(srv_name)
            if not session:
                raise RuntimeError(f"Session for MCP server '{srv_name}' is not active.")

        real_tool_name = original_tool.name
        call_timeout = timeout or self.call_timeout

        try:
            result = await asyncio.wait_for(
                session.call_tool(real_tool_name, arguments=arguments),
                timeout=call_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"MCP tool '{name}' timed out after {call_timeout}s on [{srv_name}]. Restarting server...")
            try:
                await self.restart_server(srv_name)
            except Exception as re:
                logger.debug(f"Restart after timeout failed: {re}")
            raise TimeoutError(f"Tool '{name}' execution timed out after {call_timeout}s.")
        except Exception as e:
            logger.warning(f"MCP tool '{name}' failed with error: {e}. Attempting server restart...")
            try:
                await self.restart_server(srv_name)
            except Exception as re:
                logger.debug(f"Restart after error failed: {re}")
            raise

        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
            else:
                texts.append(str(content))
        combined = "\n".join(texts)

        # Enforce bounded result size
        if len(combined) > self.max_result_chars:
            combined = combined[:self.max_result_chars] + f"\n... [Result truncated to {self.max_result_chars} characters]"
        return combined

    async def close(self):
        """Close all MCP sessions and shut down subprocesses cleanly."""
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.debug(f"Error closing MCP sessions: {e}")
        self._sessions.clear()
        self._tools_map.clear()
        self._server_params.clear()
        self._initialized = False
