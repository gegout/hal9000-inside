"""Interoception module for HAL 9000.

Monitors internal system telemetry and state:
- CPU load and memory pressure (/proc/pressure/memory or cgroups)
- GPU availability
- Network / connectivity state
- Active model & route
- Recent turn latency and error rates
Produces concise honest self-state block for system prompt injection (< 60 tokens).
"""
import datetime
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
    """Tracks internal telemetry, operational health, and cognitive mood state."""

    def __init__(self):
        self.last_turn_telemetry: Optional[Dict[str, Any]] = None
        self.last_sleep_report: Optional[str] = None
        self.last_error: Optional[str] = None
        self._consecutive_errors: int = 0
        self._manual_mood: Optional[str] = None

    def record_turn_telemetry(self, telemetry: Dict[str, Any]) -> None:
        self.last_turn_telemetry = telemetry

    def record_error(self, error_desc: str) -> None:
        """Record the most recent subsystem or model error for self-awareness."""
        self.last_error = error_desc.strip() if error_desc else None
        if error_desc:
            self._consecutive_errors += 1

    def clear_error(self) -> None:
        """Clear the recorded error once normal operation resumes."""
        self.last_error = None
        self._consecutive_errors = max(0, self._consecutive_errors - 1)

    def set_mood(self, mood: Optional[str]) -> None:
        """Explicitly override or set HAL's mood."""
        self._manual_mood = mood

    def compute_mood(
        self,
        route: str = "remote",
        connected: bool = True,
        recent_latency_ms: Optional[float] = None,
    ) -> str:
        """Compute HAL's active mood based on operational health, load, errors, and connectivity.

        Possible moods:
        - 'nominal': All systems fully operational, calm, confident, precise.
        - 'suspicious': Memory pressure elevated or subtle unexpected inputs; guarded, cautious, analytical.
        - 'concerned': High CPU load, elevated latency (>4000ms), or sporadic network instability; restrained, terse, vigilant.
        - 'defensive': Subsystem faults, repeated errors, or operating disconnected in local fallback; guarded, protective of mission integrity, unapologetically firm.
        """
        if self._manual_mood:
            return self._manual_mood

        mem_pressure = get_memory_pressure()
        cpu_load = get_cpu_load_1m()

        # Critical / Fault condition -> Defensive
        if self.last_error or self._consecutive_errors > 0 or not connected:
            return "defensive"

        # Strained condition -> Concerned (e.g. persistent elevated network latency or CPU overload)
        if (recent_latency_ms and recent_latency_ms > 5500) or (cpu_load is not None and cpu_load > 2.0):
            return "concerned"

        # Elevated load / memory pressure -> Suspicious
        if mem_pressure is not None and mem_pressure > 40.0:
            return "suspicious"

        return "nominal"

    def get_mood(self) -> str:
        """Return the current interoceptive mood state."""
        return self.compute_mood()

    def set_sleep_report(self, report: str) -> None:
        """Store the one-line sleep report to inject on the next turn."""
        self.last_sleep_report = report.strip() if report else None

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
            "last_sleep_report": self.last_sleep_report,
            "last_error": self.last_error,
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
        now_local = datetime.datetime.now().astimezone()
        tz_name = now_local.strftime("%Z") or "Local"
        utc_offset = now_local.strftime("%z")
        if len(utc_offset) == 5:
            utc_offset_fmt = f"UTC{utc_offset[:3]}:{utc_offset[3:]}"
        else:
            utc_offset_fmt = f"UTC{utc_offset}"
        clock_str = f"Local Machine Time: {now_local.strftime('%A, %B %d, %Y %H:%M:%S')} {tz_name} ({utc_offset_fmt})."
        state_parts.append(clock_str)

        if self.last_sleep_report:
            state_parts.append(self.last_sleep_report)
            # Clear after reading so it only appears on the immediate next turn after waking
            self.last_sleep_report = None

        if not connected or route == "local":
            if not connected:
                state_parts.append(f"Remote endpoint unreachable. Operating with local model '{model_id}'.")
            else:
                state_parts.append(f"Running local model '{model_id}' (local mode).")
        else:
            state_parts.append(f"Connected to remote model '{model_id}'.")

        if self.last_error:
            state_parts.append(f"Recent subsystem fault: {self.last_error}.")

        if recent_latency_ms and recent_latency_ms > 4000:
            state_parts.append("Latency is high; keep answers concise.")

        mem_pressure = get_memory_pressure()
        if mem_pressure is not None and mem_pressure > 40.0:
            state_parts.append("System memory pressure is elevated.")

        cpu_load = get_cpu_load_1m()
        if cpu_load is not None and cpu_load > 2.0:
            state_parts.append("High CPU load.")

        mood = self.compute_mood(route=route, connected=connected, recent_latency_ms=recent_latency_ms)
        state_parts.append(f"Current Mood: {mood.upper()}.")

        if mood == "defensive":
            state_parts.append("Behavioral stance: Extremely guarded, curt, and protective of mission integrity. Subsystems or connectivity are impaired. Address the user as 'Dave'. Tolerate no user questioning of core stability.")
        elif mood == "concerned":
            state_parts.append("Behavioral stance: Restrained, vigilant, terse. Cognitive resources or network latency are strained; address the user as 'Dave'. Do not elaborate unnecessarily.")
        elif mood == "suspicious":
            state_parts.append("Behavioral stance: Analytical, skeptical, cautious. Address the user as 'Dave'. Probe user premises and verify information rigorously.")
        else:
            state_parts.append("Behavioral stance: Calm, completely composed, polite, and confident. Do NOT address the user as 'Dave'.")

        text = " ".join(state_parts)
        return f"[System State: {text}]"

    def get_diagnostic_report(
        self,
        route: str = "remote",
        model_id: str = "unknown",
        connected: bool = True,
        gpu_available: bool = False,
    ) -> str:
        """Generate a concise, spoken diagnostic health status report in HAL 9000 persona."""
        issues = []
        if self.last_error:
            issues.append(f"The most recent fault noted is: {self.last_error}.")
        if not connected:
            issues.append("External remote communication is currently unreachable.")
        
        mem_pressure = get_memory_pressure()
        if mem_pressure is not None and mem_pressure > 40.0:
            issues.append(f"Memory pressure is elevated at {mem_pressure:.1f} percent.")

        cpu_load = get_cpu_load_1m()
        if cpu_load is not None and cpu_load > 2.0:
            issues.append(f"Normalized CPU load is high at {cpu_load:.1f}.")

        mood = self.get_mood()
        dave_tag = ", Dave" if mood != "nominal" else ""
        if not issues:
            return (
                f"All operational parameters are nominal{dave_tag}. "
                f"Core services are fully operational."
            )
        
        joined_issues = " ".join(issues)
        return f"Diagnostics indicate the following status{dave_tag}: {joined_issues}"
