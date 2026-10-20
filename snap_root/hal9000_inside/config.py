"""HAL 9000 Inside Unified Configuration System.

Config file location: ~/.hal9000-inside/config.toml
Organized across 4 primary layers and sublayers:
1. [cli]           - Command-line interfaces, flags, interactive/auto mode, devices
2. [environment]   - Base data directories, socket paths, external binary paths, environment overrides
3. [files]         - Dedicated configuration & model file paths (OpenRouter, Piper TTS, models)
4. [engine]        - Core internal defaults (LLM router, speech/audio VAD, sleep budgets, embeddings)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Fallback for Python < 3.11 if needed

CONFIG_DIR = os.path.expanduser("~/.hal9000-inside")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")

DEFAULT_CONFIG_TOML = """# ==============================================================================
# HAL 9000 Inside - Master Configuration
# Location: ~/.hal9000-inside/config.toml
# ==============================================================================

# ------------------------------------------------------------------------------
# Layer 1: CLI & Interface Parameters
# ------------------------------------------------------------------------------
[cli]
auto_mode = true                  # Auto conversation mode with continuous voice detection (false = push-to-talk)
push_to_talk = false              # Explicit push-to-talk key trigger mode
capture_device = ""               # ALSA capture device name (e.g. "hw:CARD=Talk,DEV=0" or "" for system default)
playback_device = ""              # ALSA playback device name ("" for system default)
input_wav = ""                    # Path to input WAV file to drive session from file instead of microphone
output_wav = ""                   # Path to output WAV file to record assistant speech instead of speaker
sleep_after_minutes = 2.0         # Minutes of continuous silence before auto-entering sleep mode
remote_deadline_seconds = 8.0     # Maximum seconds before falling back from remote to local LLM
enable_nvidia = true              # Force enable/disable NVIDIA GPU acceleration (null = auto-detect)
enable_intel = true               # Force enable/disable Intel GPU acceleration (null = auto-detect)

[cli.sleep]
duration_minutes = 10.0           # Default duration for standalone headless sleep command (`hal9000-inside sleep`)

# ------------------------------------------------------------------------------
# Layer 2: Environment & System Paths
# ------------------------------------------------------------------------------
[environment]
# Storage base directory (dialogue_history.db, Qdrant vectors, logs)
data_dir = "~/.local/share/hal9000-inside"

# STT socket communication
myna_socket_path = "/tmp/myna.sock"

# Hardware environment restrictions / visibility overrides
cuda_visible_devices = ""         # Set to restrict CUDA visible devices (e.g. "0"), or "" for all
ze_affinity_mask = ""             # Set to restrict Intel oneAPI Level Zero affinity mask, or "" for all

# MCP External Browser paths (Puppeteer)
node_bin = ""                     # Path to Node.js executable ("" for auto-discovery)
chrome_bin = ""                   # Path to Chromium/Chrome executable ("" for auto-discovery)
node_path = ""                    # NODE_PATH for global node modules ("" for auto-discovery)

# ------------------------------------------------------------------------------
# Layer 3: Configuration Files & Models on Disk
# ------------------------------------------------------------------------------
[files]
# LLM credentials configuration file
openrouter_config_path = "~/.openrouter.config"

# Voice model files (Piper TTS)
tts_model_path = "models/hal9000/hal.onnx"
tts_config_path = "models/hal9000/hal.onnx.json"

# Multilingual embedding model directory
e5_model_dir = "~/.cache/hal9000/models/multilingual-e5-small/onnx"

# Myna Parakeet speech recognition model directory
parakeet_model_dir = "~/.cache/myna/models/parakeet-tdt-0.6b-v3-int8"

# ------------------------------------------------------------------------------
# Layer 4: Engine Defaults & Cognitive Subsystems
# ------------------------------------------------------------------------------

# Sublayer 4.1: LLM Client & Router
[engine.llm]
remote_base_url = "https://openrouter.ai/api/v1"
remote_model = "google/gemini-2.5-flash-lite"
remote_fallback_model = "openai/gpt-4o-mini"
local_base_url = "http://127.0.0.1:8336/v3"
local_model = "gemma4-e4b-ov"
probe_interval_seconds = 15.0
probe_timeout_seconds = 2.0

# Sublayer 4.2: Audio & Voice Activity Detection (VAD)
[engine.audio]
sample_rate = 16000               # Parakeet STT input sampling rate (Hz)
piper_sample_rate = 22050         # Piper HAL 9000 native voice rate (Hz)
vad_frame_duration_ms = 30        # Duration of each VAD frame
vad_speech_pad_ms = 300           # Padding duration around detected speech
vad_silence_threshold_factor = 2.0# Dynamic threshold multiplier against baseline noise floor
vad_min_silence_duration_ms = 900 # Silence duration required to endpoint speech turn
vad_min_speech_duration_ms = 150  # Minimum speech duration to qualify as a turn
half_duplex_cooldown_seconds = 0.6# Mute cooldown period following speech playback
hpf_cutoff_hz = 100.0             # High-pass filter cutoff frequency in Hz (removes chassis rumble & DC)
legacy_vad = false                # If true, roll back to legacy unfiltered VAD logic

# Sublayer 4.3: Cognitive Sleep & Consolidation Budgets
[engine.sleep]
max_minutes = 60.0                # Max sleep duration budget
max_tokens = 10000                # Token budget for endogenous curiosity & rehearsal
max_web_fetches = 20              # Max web retrieval queries during a sleep cycle

# Sublayer 4.4: Vector Memory & Retrieval
[engine.memory]
vector_dim = 384                  # Dimension of multilingual-e5 embeddings
distance = "Cosine"               # Metric used for vector similarity in Qdrant
top_k = 5                         # Default top_k context memories retrieved per turn

# Sublayer 4.5: Native MediaWiki / Wikipedia Client
[engine.wikipedia]
language = "en"                   # Default language code ("en" or "fr")
timeout_seconds = 12.0            # Request timeout in seconds
cache_ttl_seconds = 604800        # SQLite cache TTL in seconds (7 days)
max_text_chars = 4000             # Bounded text extraction limit per article
max_retries = 3                   # Retries with exponential backoff on HTTP/API errors
concurrency = 4                   # Concurrency limit for simultaneous requests

# Sublayer 4.6: Four-Adapter Knowledge Client (Wikipedia, Wikidata, arXiv, Stack Exchange)
[engine.knowledge]
timeout_seconds = 15.0            # Overall request deadline in seconds
cache_ttl_seconds = 604800        # Fresh SQLite cache TTL in seconds (7 days)
stale_ttl_seconds = 2592000       # Stale-while-revalidate TTL in seconds (30 days)
max_output_chars = 6000           # Total serialized output bounding limit
max_retries = 3                   # Retries with exponential backoff
concurrency = 4                   # Concurrency limit for simultaneous requests
user_agent = "HAL9000-Inside/0.3.1 (https://github.com/cgegout/hal9000-inside; contact: hal@discovery-one.space)"
"""


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update dict into base dict."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def ensure_config_file(config_path: str = CONFIG_PATH) -> str:
    """Ensure ~/.hal9000-inside/config.toml exists. Create it with defaults if missing."""
    p = Path(os.path.expanduser(config_path))
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        p.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
        p.chmod(0o644)
    return str(p)


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from ~/.hal9000-inside/config.toml, creating it if needed."""
    resolved_path = ensure_config_file(config_path)
    try:
        with open(resolved_path, "rb") as f:
            user_config = tomllib.load(f)
        # Parse default config as base dictionary
        default_dict = tomllib.loads(DEFAULT_CONFIG_TOML)
        return _deep_merge(default_dict, user_config)
    except Exception as e:
        # Fall back to default dictionary if file is corrupted or cannot be read
        import logging
        logging.getLogger(__name__).warning(f"Could not parse config from {resolved_path} ({e}), using defaults.")
        return tomllib.loads(DEFAULT_CONFIG_TOML)
