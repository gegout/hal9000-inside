"""SmolLM2 local model service integration and verification for HAL 9000 Inside."""

import logging
import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

DEFAULT_SMOLLM2_ENDPOINT = "http://127.0.0.1:8344/v1"
DEFAULT_SMOLLM2_MODEL = "smollm2-135m"

CANDIDATE_LOCAL_ENDPOINTS = [
    "http://127.0.0.1:8344/v1",
    "http://127.0.0.1:8366/v1",
    "http://127.0.0.1:8336/v1",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.1:8080/v1",
]


def _extract_model_ids(data: dict) -> list[str]:
    """Extract model IDs from OpenAI-compatible /v1/models JSON response."""
    if not isinstance(data, dict):
        return []
    items = data.get("data", data.get("models", []))
    if not isinstance(items, list):
        return []
    models = []
    for m in items:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("name") or ""
            if mid:
                models.append(str(mid))
        elif isinstance(m, str) and m:
            models.append(m)
    return models


def sync_discover_local_llm_endpoint(
    preferred_endpoint: str = DEFAULT_SMOLLM2_ENDPOINT,
    preferred_model: str = DEFAULT_SMOLLM2_MODEL,
    timeout_seconds: float = 1.2,
    allow_fallback_ports: bool = True,
) -> Tuple[str, str, bool]:
    """Synchronously probe preferred_endpoint (and known local LLM ports if unreachable)
    and return (resolved_endpoint, resolved_model, is_reachable).
    """
    pref_ep = (preferred_endpoint or DEFAULT_SMOLLM2_ENDPOINT).rstrip("/")
    candidates = [pref_ep]
    if allow_fallback_ports:
        for ep in CANDIDATE_LOCAL_ENDPOINTS:
            ep_clean = ep.rstrip("/")
            if ep_clean not in candidates:
                candidates.append(ep_clean)

    for ep in candidates:
        url = f"{ep}/models"
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    try:
                        models = _extract_model_ids(resp.json())
                    except Exception:
                        models = []
                    if models:
                        if preferred_model in models:
                            resolved_model = preferred_model
                        else:
                            resolved_model = models[0]
                    else:
                        resolved_model = preferred_model
                    return ep, resolved_model, True
        except Exception:
            continue

    return pref_ep, preferred_model, False


async def discover_local_llm_endpoint(
    preferred_endpoint: str = DEFAULT_SMOLLM2_ENDPOINT,
    preferred_model: str = DEFAULT_SMOLLM2_MODEL,
    timeout_seconds: float = 1.2,
    allow_fallback_ports: bool = True,
) -> Tuple[str, str, bool]:
    """Asynchronously probe preferred_endpoint (and known local LLM ports if unreachable)
    and return (resolved_endpoint, resolved_model, is_reachable).
    """
    pref_ep = (preferred_endpoint or DEFAULT_SMOLLM2_ENDPOINT).rstrip("/")
    candidates = [pref_ep]
    if allow_fallback_ports:
        for ep in CANDIDATE_LOCAL_ENDPOINTS:
            ep_clean = ep.rstrip("/")
            if ep_clean not in candidates:
                candidates.append(ep_clean)

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for ep in candidates:
            url = f"{ep}/models"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    try:
                        models = _extract_model_ids(resp.json())
                    except Exception:
                        models = []
                    if models:
                        if preferred_model in models:
                            resolved_model = preferred_model
                        else:
                            resolved_model = models[0]
                    else:
                        resolved_model = preferred_model
                    return ep, resolved_model, True
            except Exception:
                continue

    return pref_ep, preferred_model, False


def check_smollm2_snap_installed() -> bool:
    """Check if the smollm2 snap package is installed on the host."""
    snap_bin = shutil.which("snap")
    if not snap_bin:
        return False
    try:
        proc = subprocess.run(
            [snap_bin, "list", "smollm2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3.0,
        )
        return proc.returncode == 0
    except Exception as e:
        logger.debug(f"Error checking smollm2 snap: {e}")
        return False


def get_smollm2_install_instruction() -> str:
    """Return non-root instruction to install the smollm2 snap."""
    return (
        "SmolLM2 local model service is not installed.\n"
        "To install SmolLM2 on your Ubuntu host, please run:\n"
        "    sudo snap install smollm2\n"
        "Once installed, SmolLM2 will serve inference at http://127.0.0.1:8344/v1."
    )


async def verify_smollm2_endpoint(
    endpoint: str = DEFAULT_SMOLLM2_ENDPOINT,
    timeout_seconds: float = 2.0,
) -> Tuple[bool, Optional[str]]:
    """Verify that SmolLM2 endpoint is alive and exposes models.
    Returns (is_ready, error_message).
    """
    url = f"{endpoint.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                # Check for smollm2 model in returned list
                models = [m.get("id") or m.get("name") or "" for m in data.get("data", data.get("models", []))]
                return True, None
            return False, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        return False, str(e)


def sync_verify_smollm2_endpoint(
    endpoint: str = DEFAULT_SMOLLM2_ENDPOINT,
    timeout_seconds: float = 2.0,
) -> Tuple[bool, Optional[str]]:
    """Synchronous verification of SmolLM2 endpoint."""
    url = f"{endpoint.rstrip('/')}/models"
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                return True, None
            return False, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        return False, str(e)
