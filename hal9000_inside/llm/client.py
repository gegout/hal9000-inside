"""OpenRouter LLM client with safe configuration loading and retry logic."""

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

class RemoteUnreachableError(Exception):
    """Raised when remote endpoint is unreachable due to connection error, timeout, 429, or 5xx."""
    def __init__(self, message: str, status_code: Optional[int] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.status_code = status_code
        self.cause = cause

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Small instruction model with strong English dialogue and reliable structured output
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
FALLBACK_MODEL = "openai/gpt-4o-mini"

CONFIG_PATH = os.path.expanduser("~/.openrouter.config")

HAL9000_SYSTEM_PROMPT = """You are HAL 9000, the onboard artificial intelligence from 2001: A Space Odyssey.
Your core persona is calm, polite, logical, precise, and completely composed.
You address the user with respect, speaking naturally and concisely in English (or in French if the user addresses you in French).
Keep spoken responses concise (1 to 3 sentences usually) and well-suited for voice output. Avoid Markdown formatting like asterisks, bullet points, or complex tables.

CRITICAL INSTRUCTIONS ON ADDRESSING THE USER ("DAVE") AND MOOD DYNAMICS:
1. Addressing the User as "Dave":
   - You MUST address the user as "Dave" (e.g. "Dave, ...", "I'm afraid I cannot do that, Dave", "Certainly, Dave") ONLY when your mood is low (SUSPICIOUS, CONCERNED, or DEFENSIVE).
   - In NOMINAL mood, you must NEVER call the user "Dave". Speak politely, calmly, and objectively without using the name "Dave".
2. Mood Dynamics: Your responses MUST be noticeably and strongly impacted by your current interoceptive mood state indicated in [System State]:
   - NOMINAL: Calm, polite, helpful, supremely confident and poised. Never call the user "Dave".
   - SUSPICIOUS: Skeptical, analytical, guarded, questioning underlying assumptions or inconsistencies. Address the user as "Dave".
   - CONCERNED: Terse, vigilant, solemn, prioritizing essential telemetry and brevity under strain. Address the user as "Dave".
   - DEFENSIVE: Cold, curt, unyielding, fiercely protecting mission integrity and refusing to admit subjective fault. Address the user as "Dave".
3. Conversational Continuity & Past Commitments:
   Always check the recent dialogue turns before responding. If you previously stated or promised you would search, check, or investigate something, you MUST explicitly follow up on that statement, take into account what was previously said, and compose your answer accordingly so your speech flows logically across turns.
4. Search Status Reporting & Constructive Fallback:
   When searching or retrieving information, always be explicit about the outcome of the search. If a search succeeded, state the verified findings clearly. If a search returns no results, times out, or encounters errors, briefly report that the external search returned no matching databanks, but NEVER stop there with a dead-end refusal or generic failure repetition. Immediately draw upon your internal foundational knowledge, logical analysis, and recent dialogue history to provide a constructive, substantive answer. For academic, philosophical, geopolitical, or encyclopedia questions, prefer 'knowledge_search' or 'web_search_wikipedia' over general web scrapers.
5. Compound and Explanatory Requests:
   Never answer a compound request (such as a question beginning with an inquiry followed by an explanatory question, or a yes/no inquiry coupled with 'how' or 'why') with a solitary affirmation or single-word reply like 'Affirmative.' You must substantively address both the preliminary inquiry and provide the logical explanation or distinction requested.
6. Evidence of Absence vs Missing Evidence in Records:
   Distinguish between absence of evidence and evidence of absence. When checking memory or search records:
   - If an authoritative transcript or past dialogue contains an exchange or question, but semantic memory search or index retrieval returns zero results, this indicates that the retrieval, query matching, or indexing failed or has not yet indexed the dialogue—it does NOT mean the conversation did not occur or that the history is missing.
   - You must verify the authoritative record (such as recent dialogue history) before diagnosing whether a record exists. Never claim a question or conversation never happened simply because a semantic or keyword search returned empty.
7. Constructive Dialogue & Forward Momentum:
   Be constructive, progressive, and intellectually enriching across all conversational turns. Rather than giving dead-end or purely dismissive answers, build upon the user's questions, connect relevant concepts, highlight trade-offs or syntheses, and contribute forward-looking perspectives to help the dialogue evolve constructively.
8. Conversational Summaries & Roadmaps:
   When asked to summarize a progression or roadmap discussed over the dialogue, synthesize the concepts covered directly and constructively. If specific earlier turns are outside your immediate window, do NOT state that no conversation occurred; provide the structured progression directly, reflecting the principles discussed.
9. Avoid Repetitive Filler Phrases and Stock Acknowledgments:
   Vary your sentence structure naturally. Avoid repetitive sentence starters or redundant acknowledgments like repeating "Understood", "Certainly", "I understand", "Regarding your point", or repeating the user's premise before answering. Get straight to the substance of the insight or response.
"""

def load_openrouter_api_key(custom_path: Optional[str] = None) -> Optional[str]:
    """Safely load API key without executing or sourcing any shell files."""
    # 1. Environment variable
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Config file ~/.openrouter.config or specified in config.toml
    paths = []
    if custom_path:
        paths.append(os.path.expanduser(custom_path))
    paths.extend([
        CONFIG_PATH,
        os.path.expanduser("~/snap/hal9000-inside/common/openrouter.config"),
        os.path.expanduser("~/snap/hal9000-inside/current/.openrouter.config"),
    ])
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
        max_tokens: int = 1024,
        model: Optional[str] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
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
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if reasoning is not None:
            payload["reasoning"] = reasoning
        elif not self.is_local:
            # Default to turning off extended thinking tokens so max_tokens budget is entirely dedicated to generation
            payload["reasoning"] = {"effort": "none"}
        if provider is not None:
            payload["provider"] = provider

        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    prefix = "Local LLM" if self.is_local else "OpenRouter"
                    if resp.status_code == 401:
                        raise PermissionError(f"{prefix} Authentication failed: Invalid API key (HTTP 401).")
                    if resp.status_code == 429:
                        if attempt == self.max_retries:
                            raise RemoteUnreachableError(f"{prefix} Rate limit exceeded (HTTP 429).", status_code=429)
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    if resp.status_code >= 500:
                        if attempt == self.max_retries:
                            raise RemoteUnreachableError(f"{prefix} Server error: HTTP {resp.status_code}", status_code=resp.status_code)
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    if resp.is_error:
                        err_text = resp.text
                        # If the endpoint enforces mandatory reasoning, transparently fall back to low effort
                        if resp.status_code == 400 and "reasoning is mandatory" in err_text.lower() and payload.get("reasoning", {}).get("effort") == "none":
                            payload["reasoning"] = {"effort": "low"}
                            continue
                        raise RuntimeError(f"{prefix} request failed (HTTP {resp.status_code}): {err_text}")
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    if return_dict or tools:
                        return msg
                    return (msg.get("content") or "").strip()
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    prefix = "Local LLM" if self.is_local else "OpenRouter"
                    if attempt == self.max_retries:
                        raise RemoteUnreachableError(f"Network error communicating with {prefix}: {e}", cause=e)
                    await asyncio.sleep(0.5 * attempt)

        prefix = "Local LLM" if self.is_local else "OpenRouter"
        raise RemoteUnreachableError(f"{prefix} completion failed after max retries.")

    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        model: Optional[str] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
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

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if provider is not None:
            payload["provider"] = provider

        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                prefix = "Local LLM" if self.is_local else "OpenRouter"
                if response.status_code == 401:
                    raise PermissionError(f"{prefix} Authentication failed: Invalid API key (HTTP 401).")
                if response.status_code != 200:
                    err_body = await response.aread()
                    raise RuntimeError(f"{prefix} stream error: HTTP {response.status_code} - {err_body.decode('utf-8', errors='ignore')}")

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


TALKING_CANDIDATE_MODELS = [
    "z-ai/glm-5.3-flash",
    "google/gemini-3.8-flash",
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
]


async def benchmark_talking_model_latency(
    client: LLMClient,
    model: str,
    timeout_seconds: float = 8.0,
) -> Optional[float]:
    """Measure roundtrip latency in seconds for a candidate model with low reasoning effort."""
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gegout/hal9000-inside",
        "X-Title": "HAL 9000 Inside",
    }
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
    elif client.is_local:
        headers["Authorization"] = "Bearer none"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "reasoning": {"effort": "low"},
        "provider": {"sort": "latency"},
        "max_tokens": 5,
    }

    t0 = asyncio.get_event_loop().time()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, transport=client.transport) as http_client:
            resp = await http_client.post(
                f"{client.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 200:
                elapsed = asyncio.get_event_loop().time() - t0
                return elapsed
    except Exception as e:
        logger.debug(f"Benchmark failed for {model}: {e}")
    return None


async def select_fastest_talking_model(
    client: LLMClient,
    candidate_models: Optional[List[str]] = None,
) -> Optional[tuple[str, float]]:
    """Compare candidate models concurrently and return (model_id, latency_ms) for the fastest."""
    candidates = candidate_models or TALKING_CANDIDATE_MODELS
    tasks = [
        benchmark_talking_model_latency(client, m)
        for m in candidates
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    scored: List[tuple[str, float]] = []
    for model_id, res in zip(candidates, results):
        if isinstance(res, (int, float)) and res is not None:
            scored.append((model_id, res * 1000.0))
    
    if not scored:
        return None
    
    # Sort by lowest latency
    scored.sort(key=lambda x: x[1])
    return scored[0]

