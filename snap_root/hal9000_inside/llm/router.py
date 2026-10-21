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
    LocalModelExecutionError,
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
        cooldown_turns: int = 2,
    ):
        self.remote_client = remote_client
        self.local_client = local_client
        self.connectivity = connectivity_monitor or ConnectivityMonitor()
        self.first_audio_deadline = first_audio_deadline_seconds
        self.force_local = force_local
        self.force_remote = force_remote
        self.cooldown_turns = cooldown_turns

        self.remote_latencies: List[float] = []
        self.remote_errors: int = 0
        self.max_history = 10

        # Unreachable state & cooldown management
        self.is_remote_unreachable: bool = False
        self.cooldown_remaining: int = 0
        self.retry_in_progress: bool = False
        self._retry_lock = asyncio.Lock()
        self._just_fell_back: bool = False
        self.last_remote_error: Optional[Exception] = None
        self.last_local_error: Optional[Exception] = None
        self.local_backoff_until: float = 0.0

    def record_remote_result(self, latency_ms: float, error: bool = False, unreachable: bool = False):
        self.remote_latencies.append(latency_ms)
        if len(self.remote_latencies) > self.max_history:
            self.remote_latencies.pop(0)
        if error or unreachable:
            self.remote_errors += 1
            self.connectivity.is_connected = False
            if unreachable:
                self.is_remote_unreachable = True
                self.cooldown_remaining = self.cooldown_turns
                self._just_fell_back = True
        else:
            self.remote_errors = max(0, self.remote_errors - 1)
            self.connectivity.is_connected = True
            self.is_remote_unreachable = False
            self.cooldown_remaining = 0
            self._just_fell_back = False

    def record_turn_completed(self, route_used: str):
        """Call after an interaction turn finishes to decrement cooldown when routing locally."""
        if self._just_fell_back:
            self._just_fell_back = False
            return
        if self.is_remote_unreachable and self.cooldown_remaining > 0 and route_used == "local":
            self.cooldown_remaining -= 1
            logger.info(f"Cooldown interaction completed. Turns remaining before retry: {self.cooldown_remaining}")

    @property
    def model(self) -> str:
        if self.choose_route() == "local" and self.local_client:
            return self.local_client.model
        if self.remote_client:
            return self.remote_client.model
        if self.local_client:
            return self.local_client.model
        return "unknown"

    @property
    def is_local(self) -> bool:
        return self.choose_route() == "local"

    @property
    def is_configured(self) -> bool:
        return bool((self.remote_client and self.remote_client.is_configured) or (self.local_client and self.local_client.is_configured))

    def choose_route(self) -> str:
        """Decide 'remote' or 'local' based on state, cooldown, and connectivity."""
        if self.force_local:
            return "local"
        if self.force_remote:
            return "remote"
        if not self.remote_client or not self.remote_client.is_configured:
            return "local"
        # If local model is in backoff due to local execution error, prefer remote
        if time.time() < self.local_backoff_until and self.remote_client and self.remote_client.is_configured:
            return "remote"
        # If in unreachable cooldown, stay local
        if self.is_remote_unreachable and self.cooldown_remaining > 0:
            return "local"
        # If a retry is already in progress, prevent concurrent retries
        if self.retry_in_progress:
            return "local"
        if not self.connectivity.is_connected and not self.is_remote_unreachable:
            return "local"
        if self.remote_errors >= 3 and not (self.is_remote_unreachable and self.cooldown_remaining == 0):
            return "local"
        return "remote"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        return_dict: bool = False,
        max_tokens: int = 1024,
        on_fallback: Optional[Any] = None,
    ) -> tuple[Any, str, str]:
        """Generate response with routing, deadline, and fallback.
        
        Args:
            messages: Dialogue messages.
            tools: Optional tool definitions.
            return_dict: Return message dict if True.
            max_tokens: Token generation limit.
            on_fallback: Optional async or sync callback invoked immediately when remote
                         fails before local generation completes, e.g. on_fallback(error).
        
        Returns: (response, route_used, model_used)
        """
        route = self.choose_route()
        is_retry = bool(self.is_remote_unreachable and self.cooldown_remaining == 0 and route == "remote")

        if route == "remote" and self.remote_client:
            if is_retry:
                if self._retry_lock.locked():
                    route = "local"
                else:
                    await self._retry_lock.acquire()
                    self.retry_in_progress = True

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
                self.record_remote_result(latency_ms, error=False, unreachable=False)
                if is_retry:
                    self.retry_in_progress = False
                    if self._retry_lock.locked():
                        self._retry_lock.release()
                return res, "remote", self.remote_client.model
            except (asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError, Exception) as e:
                self.last_remote_error = e
                # Distinguish auth/config errors from unreachable
                is_auth_error = isinstance(e, PermissionError) or "401" in str(e)
                if is_auth_error:
                    logger.warning(f"Remote authentication error ({e}). Routing locally without unreachable status.")
                    self.record_remote_result(self.first_audio_deadline * 1000.0, error=True, unreachable=False)
                else:
                    logger.warning(f"Remote LLM unreachable or timed out ({e}). Triggering resilient local fallback.")
                    self.record_remote_result(self.first_audio_deadline * 1000.0, error=True, unreachable=True)

                if is_retry:
                    self.retry_in_progress = False
                    if self._retry_lock.locked():
                        self._retry_lock.release()

                # Trigger immediate acknowledgment callback if provided
                cb_task = None
                if on_fallback is not None and not is_auth_error:
                    try:
                        res_cb = on_fallback(e)
                        if asyncio.iscoroutine(res_cb):
                            cb_task = asyncio.create_task(res_cb)
                    except Exception as fe:
                        logger.debug(f"Error in on_fallback callback: {fe}")

                # Concurrently execute local inference while callback runs in background
                if self.local_client:
                    local_coro = self.local_client.generate_response(
                        messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
                    )
                    local_task = asyncio.create_task(local_coro)
                    try:
                        res = await local_task
                    except Exception as le:
                        self.last_local_error = le
                        self.local_backoff_until = time.time() + 60.0
                        logger.warning(f"Local LLM fallback failed ({le}).")
                        if cb_task and not cb_task.done():
                            try:
                                await cb_task
                            except Exception:
                                pass
                        # If remote client is configured and was not an auth error, give remote one direct attempt
                        if self.remote_client and not is_auth_error:
                            try:
                                res = await self.remote_client.generate_response(
                                    messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
                                )
                                return res, "remote", self.remote_client.model
                            except Exception:
                                pass
                        raise
                    if cb_task:
                        try:
                            await cb_task
                        except Exception:
                            pass
                    return res, "local", self.local_client.model
                elif self.remote_client:
                    res = await self.remote_client.generate_response(
                        messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
                    )
                    if cb_task:
                        try:
                            await cb_task
                        except Exception:
                            pass
                    return res, "remote", self.remote_client.model

        # Direct local route execution
        if self.local_client:
            try:
                res = await self.local_client.generate_response(
                    messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
                )
                return res, "local", self.local_client.model
            except Exception as le:
                self.last_local_error = le
                self.local_backoff_until = time.time() + 60.0
                logger.warning(f"Local LLM execution failed ({le}). Trying remote backup if available.")
                if self.remote_client and self.remote_client.is_configured:
                    try:
                        res = await self.remote_client.generate_response(
                            messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
                        )
                        return res, "remote", self.remote_client.model
                    except Exception as re:
                        self.last_remote_error = re
                        raise
                raise
        elif self.remote_client:
            res = await self.remote_client.generate_response(
                messages, tools=tools, return_dict=return_dict, max_tokens=max_tokens
            )
            return res, "remote", self.remote_client.model

        raise RuntimeError("No LLM client configured for generation.")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        return_dict: bool = False,
        max_tokens: int = 1024,
    ) -> Any:
        """Compatibility method matching LLMClient interface. Delegates to generate()."""
        res, _, _ = await self.generate(
            messages=messages,
            tools=tools,
            return_dict=return_dict,
            max_tokens=max_tokens,
        )
        return res
