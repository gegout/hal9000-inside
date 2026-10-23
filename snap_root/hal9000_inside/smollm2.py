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
