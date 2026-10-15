"""Session-stamped logging configuration for HAL 9000 Inside.

Routes INFO, WARNING, and ERROR logs to a session-stamped log file:
  <data_dir>/logs/hal900-inside_YYYYMMDD_HHMMSS.logs
while keeping the interactive terminal UI clean and uncluttered.
"""

import datetime
import logging
import os
from typing import Optional

_SESSION_LOG_PATH: Optional[str] = None


def setup_session_logging(
    data_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    console_level: int = logging.WARNING,
) -> str:
    """Initialize root logging with session-stamped file handler and console filter.
    
    Returns the absolute path to the session log file.
    """
    global _SESSION_LOG_PATH
    if _SESSION_LOG_PATH:
        return _SESSION_LOG_PATH

    target_dir = data_dir or os.environ.get("SNAP_USER_COMMON") or os.path.expanduser("~/.local/share/hal9000-inside")
    logs_dir = os.path.join(target_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True, mode=0o700)

    # UTC timestamp for filename: YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"hal900-inside_{timestamp}.logs")
    _SESSION_LOG_PATH = log_file

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # File handler: logs INFO, WARNING, ERROR, and DEBUG with timestamps and module names
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    class AutoFlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    file_handler = AutoFlushingFileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # Console handler: only WARNING or ERROR by default so terminal UI remains uncluttered
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # Filter third-party noise from terminal
    for noisy in ("search_mcp", "trafilatura", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"HAL 9000 session started. Logging to {log_file}")
    return log_file


def get_current_log_path() -> Optional[str]:
    """Return the active session log file path, if initialized."""
    return _SESSION_LOG_PATH


def log_dialogue_turn(
    speaker: str,
    text: str,
    mood: Optional[str] = None,
    route: Optional[str] = None,
    model: Optional[str] = None,
    latency_ms: Optional[float] = None,
    tts_ms: Optional[float] = None,
    tool_calls: Optional[int] = None,
    status: Optional[str] = None,
    fluidity_notes: Optional[str] = None,
):
    """Log an authoritative human-readable dialogue turn entry into the session log.
    
    This ensures the dialogue history is flushed into the session log files along with
    conversational metrics (latency, mood, route, model, fluidity notes) so developers
    can easily review transcripts, spot awkwardness, truncation, or broken conversational flow.
    """
    logger = logging.getLogger("hal9000_inside.dialogue")
    meta_parts = []
    if mood:
        meta_parts.append(f"mood={mood}")
    if route:
        meta_parts.append(f"route={route}")
    if model:
        meta_parts.append(f"model={model}")
    if latency_ms is not None:
        meta_parts.append(f"llm={latency_ms:.0f}ms")
    if tts_ms is not None:
        meta_parts.append(f"tts={tts_ms:.0f}ms")
    if tool_calls is not None and tool_calls > 0:
        meta_parts.append(f"tools={tool_calls}")
    if status:
        meta_parts.append(f"status={status}")

    meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""
    line = f"[DIALOGUE] {speaker.upper()}: \"{text.strip()}\"{meta_str}"
    if fluidity_notes:
        line += f" -> [FLUIDITY: {fluidity_notes}]"
    logger.info(line)

