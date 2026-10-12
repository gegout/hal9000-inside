"""MCP Client Manager for HAL 9000.

Connects to configured MCP servers (from ~/.gemini/config/mcp_config.json,
environment variables, or custom paths), discovers tools, and provides a unified
tool-call interface formatted for OpenAI-compatible function calling.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_PATHS = [
    os.path.expanduser("~/.gemini/config/mcp_config.json"),
    os.path.expanduser("~/.config/hal9000/mcp_config.json"),
]


class MCPClientManager:
    """Manages connections to multiple MCP servers and exposes unified tools."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_path()
        self._exit_stack = AsyncExitStack()
        self._sessions: Dict[str, ClientSession] = {}
        # tool_name -> (server_name, tool_def)
        self._tools_map: Dict[str, Tuple[str, Any]] = {}
        self._initialized = False

    def _find_config_path(self) -> Optional[str]:
        for p in DEFAULT_MCP_CONFIG_PATHS:
            if os.path.isfile(p):
                return p
        return None

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

    async def connect_all(self):
        """Connect to all defined MCP servers and discover their tools."""
        server_configs = self.load_server_configs()
        if not server_configs:
            logger.debug("No MCP servers configured to connect.")
            self._initialized = True
            return

        for srv_name, srv_conf in server_configs.items():
            command = srv_conf.get("command")
            if not command:
                continue

            args = srv_conf.get("args", [])
            env = dict(os.environ)
            if "env" in srv_conf and isinstance(srv_conf["env"], dict):
                env.update(srv_conf["env"])

            cwd = srv_conf.get("cwd")

            try:
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env,
                    cwd=cwd,
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
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
                    # Prefix if conflict exists across servers
                    if tool_key in self._tools_map:
                        tool_key = f"{srv_name}_{tool.name}"
                    self._tools_map[tool_key] = (srv_name, tool)
                    logger.info(f"Discovered MCP tool: {tool_key} from [{srv_name}]")

            except Exception as e:
                logger.warning(f"Failed to connect to MCP server [{srv_name}]: {e}")

        self._initialized = True

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Format discovered MCP tools as OpenAI-compatible function definitions."""
        openai_tools = []
        for name, (_, tool) in self._tools_map.items():
            parameters = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
            # Clean pydantic schema titles if necessary
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description or f"Tool {name}",
                    "parameters": parameters,
                }
            })
        return openai_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool on the appropriate server and return text result."""
        if name not in self._tools_map:
            raise KeyError(f"MCP tool '{name}' not found.")

        srv_name, original_tool = self._tools_map[name]
        session = self._sessions.get(srv_name)
        if not session:
            raise RuntimeError(f"Session for MCP server '{srv_name}' is not active.")

        real_tool_name = original_tool.name
        result = await session.call_tool(real_tool_name, arguments=arguments)
        
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
            else:
                texts.append(str(content))
        return "\n".join(texts)

    async def close(self):
        """Close all MCP sessions."""
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.debug(f"Error closing MCP sessions: {e}")
        self._sessions.clear()
        self._tools_map.clear()
        self._initialized = False
