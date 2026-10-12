"""Interoception module for HAL 9000.

Monitors internal system telemetry and state:
- CPU load and memory pressure (/proc/pressure/memory or cgroups)
- GPU availability
- Network / connectivity state
- Active model & route
- Recent turn latency and error rates
Produces concise honest self-state block for system prompt injection (< 60 tokens).
"""
import os
import shutil
from typing import Optional, Dict, Any


def get_memory_pressure() -> Optional[float]:
    """Read Linux PSI memory pressure 'some avg10' if available."""
    psi_path = "/proc/pressure/memory"
    if os.path.exists(psi_path):
        try:
            with open(psi_path, "r") as f:
                for line in f:
                    if line.startswith("some"):
                        parts = line.split()
                        for p in parts:
                            if p.startswith("avg10="):
                                return float(p.split("=")[1])
        except Exception:
            pass
    return None


def get_cpu_load_1m() -> Optional[float]:
    """Get 1-minute load average normalized by core count."""
    try:
        load = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        return round(load / cpu_count, 2)
    except Exception:
        return None


class InteroceptionMonitor:
    """Tracks internal telemetry and operational health."""

    def __init__(self):
        self.last_turn_telemetry: Optional[Dict[str, Any]] = None

    def record_turn_telemetry(self, telemetry: Dict[str, Any]) -> None:
        self.last_turn_telemetry = telemetry

    def get_system_state(
        self,
        route: str = "remote",
        model_id: str = "unknown",
        connected: bool = True,
        gpu_available: bool = False,
    ) -> Dict[str, Any]:
        return {
            "route": route,
            "model_id": model_id,
            "connected": connected,
            "gpu_available": gpu_available,
            "cpu_load": get_cpu_load_1m(),
            "mem_pressure": get_memory_pressure(),
            "last_turn": self.last_turn_telemetry,
        }

    def format_self_state_prompt(
        self,
        route: str = "remote",
        model_id: str = "unknown",
        connected: bool = True,
        gpu_available: bool = False,
        recent_latency_ms: Optional[float] = None,
    ) -> str:
        """Produce an honest, compact self-state block (< 60 tokens) for system prompt injection."""
        state_parts = []
        if not connected or route == "local":
            state_parts.append(f"Running local model '{model_id}' (offline/local mode).")
        else:
            state_parts.append(f"Connected to remote model '{model_id}'.")

        if recent_latency_ms and recent_latency_ms > 4000:
            state_parts.append("Latency is high; keep answers concise.")

        mem_pressure = get_memory_pressure()
        if mem_pressure is not None and mem_pressure > 40.0:
            state_parts.append("System memory pressure is elevated.")

        cpu_load = get_cpu_load_1m()
        if cpu_load is not None and cpu_load > 2.0:
            state_parts.append("High CPU load.")

        if gpu_available:
            state_parts.append("Hardware acceleration active.")

        text = " ".join(state_parts)
        return f"[System State: {text}]"
