"""LLM Router with connectivity monitoring, latency tracking, and automatic fallback.

Inputs:
- connectivity state (probe with backoff)
- last N remote latencies and errors
- per-turn deadline to first audio
- user overrides / local-only flag
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from hal9000_inside.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    LLMClient,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8336/v3"
DEFAULT_LOCAL_MODEL = "gemma4-e4b-ov"


class ConnectivityMonitor:
    """Non-blocking background probe for remote endpoint connectivity."""

    def __init__(self, endpoint: str = DEFAULT_BASE_URL, probe_interval_seconds: float = 15.0):
        self.endpoint = endpoint
        self.probe_interval = probe_interval_seconds
        self.is_connected = True
        self.last_probe_time: float = 0.0
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._probe_loop())

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def probe_once(self, timeout_seconds: float = 2.0) -> bool:
        """Cheap probe checking remote endpoint health."""
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.get(f"{self.endpoint}/models")
                self.is_connected = (resp.status_code < 500)
                self.last_probe_time = time.time()
                return self.is_connected
        except Exception:
            self.is_connected = False
            self.last_probe_time = time.time()
            return False

    async def _probe_loop(self):
        while self._running:
            await self.probe_once()
            await asyncio.sleep(self.probe_interval)


class LLMRouter:
    """Selects remote or local LLM execution per turn with deadline monitoring and fallback."""

    def __init__(
        self,
        remote_client: Optional[LLMClient] = None,
        local_client: Optional[LLMClient] = None,
        connectivity_monitor: Optional[ConnectivityMonitor] = None,
        first_audio_deadline_seconds: float = 8.0,
        force_local: bool = False,
        force_remote: bool = False,
    ):
        self.remote_client = remote_client
        self.local_client = local_client
        self.connectivity = connectivity_monitor or ConnectivityMonitor()
        self.first_audio_deadline = first_audio_deadline_seconds
        self.force_local = force_local
        self.force_remote = force_remote

        self.remote_latencies: List[float] = []
        self.remote_errors: int = 0
        self.max_history = 10

    def record_remote_result(self, latency_ms: float, error: bool = False):
        self.remote_latencies.append(latency_ms)
        if len(self.remote_latencies) > self.max_history:
            self.remote_latencies.pop(0)
        if error:
            self.remote_errors += 1
            self.connectivity.is_connected = False
        else:
            self.remote_errors = max(0, self.remote_errors - 1)
            self.connectivity.is_connected = True

    def choose_route(self) -> str:
        """Decide 'remote' or 'local' based on state."""
        if self.force_local:
            return "local"
        if self.force_remote:
            return "remote"
        if not self.remote_client or not self.remote_client.is_configured:
            return "local"
        if not self.connectivity.is_connected:
            return "local"
        # If last 3 turns failed, route locally
        if self.remote_errors >= 3:
            return "local"
        return "remote"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        return_dict: bool = False,
        max_tokens: int = 256,
    ) -> tuple[Any, str, str]:
        """Generate response with routing, deadline, and fallback.
        
        Returns: (response, route_used, model_used)
        """
        route = self.choose_route()

        if route == "remote" and self.remote_client:
            t0 = time.perf_counter()
            try:
                # Wrap remote call in first_audio_deadline
                res = await asyncio.wait_for(
                    self.remote_client.generate_response(
                        messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
                    ),
                    timeout=self.first_audio_deadline,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.record_remote_result(latency_ms, error=False)
                return res, "remote", self.remote_client.model
            except Exception as e:
                logger.warning(f"Remote LLM call failed or timed out ({e}). Falling back to local model.")
                self.record_remote_result(self.first_audio_deadline * 1000.0, error=True)
                route = "local"

        # Local route execution or fallback
        if self.local_client:
            res = await self.local_client.generate_response(
                messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
            )
            return res, "local", self.local_client.model
        elif self.remote_client:
            # If no local client exists, re-raise or attempt remote fallback model
            res = await self.remote_client.generate_response(
                messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
            )
            return res, "remote", self.remote_client.model

        raise RuntimeError("No LLM client configured for generation.")
