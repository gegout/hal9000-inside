"""OpenRouter LLM client with safe configuration loading and retry logic."""

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Small instruction model with strong English dialogue and reliable structured output
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
FALLBACK_MODEL = "openai/gpt-4o-mini"

CONFIG_PATH = os.path.expanduser("~/.openrouter.config")

HAL9000_SYSTEM_PROMPT = """You are HAL 9000, the onboard artificial intelligence from 2001: A Space Odyssey.
Your tone is calm, polite, logical, precise, and completely composed.
You address the user with respect, speaking naturally and concisely in English (or in French if the user addresses you in French).
Keep spoken responses concise (1 to 3 sentences usually) and well-suited for voice output. Avoid Markdown formatting like asterisks, bullet points, or complex tables.
You have access to retrieved conversational memory context to recall user preferences, facts, and ongoing plans.
"""

def load_openrouter_api_key() -> Optional[str]:
    """Safely load API key without executing or sourcing any shell files."""
    # 1. Environment variable
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Config file ~/.openrouter.config
    paths = [
        CONFIG_PATH,
        os.path.expanduser("~/snap/hal9000-inside/common/openrouter.config"),
        os.path.expanduser("~/snap/hal9000-inside/current/.openrouter.config"),
    ]
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.replace("export", "").strip()
                        v = v.strip().strip("'\"")
                        if "KEY" in k.upper() or "TOKEN" in k.upper():
                            if v.startswith("sk-") or len(v) > 10:
                                return v
                if content.strip().startswith("{"):
                    data = json.loads(content)
                    for k, v in data.items():
                        if "key" in k.lower() or "token" in k.lower():
                            return str(v)
            except Exception as e:
                logger.debug(f"Failed to parse config file {p}: {e}")

    return None


def load_llm_base_url() -> str:
    """Load LLM base URL from environment or configuration."""
    env_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
    if env_url and env_url.strip():
        return env_url.strip()

    paths = [
        CONFIG_PATH,
        os.path.expanduser("~/snap/hal9000-inside/common/openrouter.config"),
        os.path.expanduser("~/snap/hal9000-inside/current/.openrouter.config"),
    ]
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.replace("export", "").strip()
                        v = v.strip().strip("'\"")
                        if "URL" in k.upper() or "ENDPOINT" in k.upper():
                            return v
                if content.strip().startswith("{"):
                    data = json.loads(content)
                    for k, v in data.items():
                        if "url" in k.lower() or "endpoint" in k.lower():
                            return str(v)
            except Exception as e:
                logger.debug(f"Failed to parse base URL from {p}: {e}")

    return DEFAULT_BASE_URL


class LLMClient:
    """Async client for OpenAI-compatible completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.api_key = api_key or load_openrouter_api_key()
        self.base_url = (base_url or load_llm_base_url()).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.transport = transport

    @property
    def is_local(self) -> bool:
        """Check if base_url points to a local or internal endpoint."""
        return any(h in self.base_url for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1"))

    @property
    def is_configured(self) -> bool:
        if self.is_local:
            return True
        return bool(self.api_key and (self.api_key.startswith("sk-") or len(self.api_key) > 10))

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_dict: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Any:
        """Non-streaming completion with optional tools, retries and timeout.
        
        Returns a string response when tools is None and return_dict is False.
        Returns the choice message dict when return_dict is True or tools is provided.
        """
        if not self.is_configured:
            raise ValueError("LLM API key is not configured. Please provide it via ~/.openrouter.config or OPENROUTER_API_KEY / OPENAI_API_KEY.")

        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gegout/hal9000-inside",
            "X-Title": "HAL 9000 Inside",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.is_local:
            headers["Authorization"] = "Bearer none"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code == 401:
                        raise PermissionError("OpenRouter Authentication failed: Invalid API key (HTTP 401).")
                    if resp.status_code == 429:
                        if attempt == self.max_retries:
                            raise RuntimeError("OpenRouter Rate limit exceeded (HTTP 429).")
                        await asyncio.sleep(1.0 * attempt)
                        continue
                    if resp.status_code >= 500:
                        if attempt == self.max_retries:
                            raise RuntimeError(f"OpenRouter Server error: HTTP {resp.status_code}")
                        await asyncio.sleep(1.0 * attempt)
                        continue
                    if resp.is_error:
                        err_text = resp.text
                        raise RuntimeError(f"OpenRouter request failed (HTTP {resp.status_code}): {err_text}")
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    if return_dict or tools:
                        return msg
                    return (msg.get("content") or "").strip()
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    if attempt == self.max_retries:
                        raise RuntimeError(f"Network error communicating with OpenRouter: {e}")
                    await asyncio.sleep(1.0 * attempt)

        raise RuntimeError("OpenRouter completion failed after max retries.")

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> AsyncGenerator[str, None]:
        """Stream chunks from LLM endpoint."""
        if not self.is_configured:
            raise ValueError("LLM API key is not configured.")

        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gegout/hal9000-inside",
            "X-Title": "HAL 9000 Inside",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.is_local:
            headers["Authorization"] = "Bearer none"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code == 401:
                    raise PermissionError("OpenRouter Authentication failed: Invalid API key (HTTP 401).")
                if response.status_code != 200:
                    err_body = await response.aread()
                    raise RuntimeError(f"OpenRouter stream error: HTTP {response.status_code} - {err_body.decode('utf-8', errors='ignore')}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]
                            content = delta.get("content")
                            if content:
                                yield content
                        except Exception:
                            continue
