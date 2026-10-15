# HAL 9000 Inside (Ubuntu Voice Assistant)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Version: 0.3.1](https://img.shields.io/badge/Version-0.3.1-red.svg)](pyproject.toml)

An autonomous local voice assistant for Ubuntu embodying the calm, polite, and precise persona of **HAL 9000** from *2001: A Space Odyssey*.

HAL 9000 Inside integrates local streaming automatic speech recognition, vector-embedded conversational memory, autonomous cognitive sleep cycles (curiosity, prediction, dream consolidation, interoception), resilient hybrid LLM routing (local OpenVINO Gemma + remote OpenRouter), Model Context Protocol (MCP) tool use and serving, browser-free and headless web retrieval, and native 22,050 Hz Piper TTS speech synthesis with half-duplex feedback prevention.

---

## Table of Contents

1. [Key Features](#key-features)
2. [End-to-End System Architecture](#end-to-end-system-architecture)
3. [Cognitive Architecture & Sleep Cycles](#cognitive-architecture--sleep-cycles)
4. [Master Configuration System (`~/.hal9000-inside/config.toml`)](#master-configuration-system-hal9000-insideconfigtoml)
5. [Installation & Setup](#installation--setup)
6. [CLI & Interactive Controls](#cli--interactive-controls)
7. [Inference Routing & Resilient Fallback](#inference-routing--resilient-fallback)
8. [Conversational Memory Subsystem](#conversational-memory-subsystem)
9. [Web Intelligence & MCP Integration](#web-intelligence--mcp-integration)
10. [Logging & Observability](#logging--observability)
11. [Hardware Acceleration](#hardware-acceleration)
12. [Snap Packaging (Ubuntu Core24)](#snap-packaging-ubuntu-core24)
13. [Verification & Testing](#verification--testing)
14. [Acknowledgements & License](#acknowledgements--license)

---

## Key Features

- **Local Low-Latency Speech-to-Text (STT):** NVIDIA Parakeet TDT 0.6B v3 model integrated via Canonical's Myna streaming engine with dynamic Voice Activity Detection (VAD), AC energy tracking, and configurable silence endpointing (900 ms default).
- **Persistent Conversational Memory:**
  - Semantic vector search using embedded local **Qdrant** and normalized 384-dimensional **Multilingual E5 Small** ONNX embeddings.
  - Full relational dialogue history, turn tracking, confidence scoring, and categories (`fact`, `user_profile`, `knowledge`, `user_context`) stored in SQLite (`dialogue_history.db`).
- **Autonomous Cognitive Consolidation (Sleep Mode):**
  - Triggers automatically after idle silence (default: 2 minutes) or via `S` key / headless CLI.
  - Multi-phase cognitive consolidation: endogenous curiosity research, predictive follow-up rehearsal, duplicate pruning, fact confidence decay, telemetry reflection, and interoceptive self-monitoring.
  - Concise, honest single-line dream reports displayed on console, with full curiosity research details preserved in timestamped session logs.
- **Resilient Hybrid LLM Routing:**
  - Dual-tier routing between primary remote models (`google/gemini-2.5-flash-lite`, `openai/gpt-4o-mini`) and fully local edge models (`gemma4-e4b-ov` via OpenVINO).
  - Sub-second spoken acknowledgments adjusted to HAL's mood during network stalls, automated cooldown locks, and non-blocking background health probing.
  - Real-time self-awareness of model failures, communicating clear, simple explanations in character even if all models are unavailable.
- **Faithful HAL 9000 Voice Synthesis:**
  - Resident Piper TTS neural engine outputting native 22,050 Hz mono audio.
  - Pipelined sentence chunking with instant playback cancellation (`C` key or new speech).
  - Half-duplex acoustic gating and cooldown windows prevent microphone self-triggering from speaker output.
- **Model Context Protocol (MCP) Dual Architecture:**
  - **Server (`hal9000-inside mcp`):** Exposes memory search, history inspection, and web tools over stdio to any MCP client (Antigravity, Cursor, Claude Desktop).
  - **Client:** Autonomously connects to configured external MCP servers and invokes tools during speech reasoning.
- **Browser-Free & Puppeteer Web Retrieval:**
  - Resident zero-browser search via `free-search-mcp` (DuckDuckGo, Bing) with Modest HTML parsing and untrusted evidence isolation.
  - Optional Puppeteer headless Chrome browser for full JavaScript rendering and Wikipedia article extraction.
- **Master Unified Configuration (`~/.hal9000-inside/config.toml`):**
  - Single, structured TOML configuration file covering CLI options, environment paths, file references, and internal engine defaults. Auto-generated on first startup.

---

## End-to-End System Architecture

```text
       ┌────────────────────────┐
       │   Microphone (ALSA)    │
       └───────────┬────────────┘
                   │ 16 kHz Mono PCM
                   ▼
       ┌────────────────────────┐
       │  SimpleVAD & Half-     │ ◄─── Cooldown Gating
       │  Duplex Audio Gating   │
       └───────────┬────────────┘
                   │ Speech Frames + Circular Pre-roll
                   ▼
       ┌────────────────────────┐
       │  Myna Parakeet TDT     │
       │  0.6B v3 STT (ONNX)    │
       └───────────┬────────────┘
                   │ Transcribed User Utterance
                   ▼
       ┌────────────────────────┐       ┌─────────────────────────────┐
       │  Conversational Memory │ ◄───► │ Local Qdrant + E5 Embeddings│
       │  Manager (SQLite)      │       │ & SQLite Dialog Store       │
       └───────────┬────────────┘       └─────────────────────────────┘
                   │ Context + Memory Facts
                   ▼
       ┌────────────────────────┐       ┌─────────────────────────────┐
       │  Cognitive Subsystems  │ ◄───► │ Sleep Manager, Curiosity,   │
       │  & Interoception       │       │ Prediction, Turn Digest     │
       └───────────┬────────────┘       └─────────────────────────────┘
                   │ State & Mood Modulation
                   ▼
       ┌────────────────────────┐       ┌─────────────────────────────┐
       │  LLM Hybrid Router     │ ◄───► │ MCP Client & Free-Search    │
       │  (Remote vs Local OV)  │       │ (DuckDuckGo, Bing, Tools)   │
       └───────────┬────────────┘       └─────────────────────────────┘
                   │ Generated Spoken Response
                   ▼
       ┌────────────────────────┐
       │  Piper HAL 9000 TTS    │
       │  (22,050 Hz ONNX)      │
       └───────────┬────────────┘
                   │ 22.05 kHz Raw PCM
                   ▼
       ┌────────────────────────┐
       │    Speakers (ALSA)     │
       └────────────────────────┘
```

---

## Cognitive Architecture & Sleep Cycles

HAL 9000 Inside does not simply wait passively between turns. It possesses an autonomous cognitive lifecycle inspired by human memory consolidation and sleep:

```text
       ┌────────────────────────────────────────────────────────┐
       │                  Interactive Awoken State               │
       │  Listening ──► Thinking ──► Speaking ──► Digest Turn   │
       └───────────────────────────┬────────────────────────────┘
                                   │
              Idle Silence Timeout │ or 'S' key press / CLI 'sleep'
              (Default: 2.0 min)   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             Autonomous Sleep Consolidation             │
       │                                                        │
       │  Phase 1: Backlog Drainage (pending prefetch jobs)     │
       │  Phase 2: Endogenous Curiosity (salience x novelty)    │
       │  Phase 3: Follow-Up Rehearsal (predicted questions)    │
       │  Phase 4: Memory Consolidation (dedup, prune, decay)   │
       │  Phase 5: Self-Model Reflection & Telemetry Tuning     │
       │  Phase 6: Interoceptive System State Snapshot          │
       └───────────────────────────┬────────────────────────────┘
                                   │
                   User Speech / Interruption / Budget End
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                     Wake Summary                       │
       │  Single-line honest dream report:                      │
       │  "Dreamt of 2 ideas: researched X; reflected on Y."    │
       └────────────────────────────────────────────────────────┘
```

### Cognitive Subsystems

1. **Curiosity Engine (`hal9000_inside/cognition/curiosity.py`):**
   - Extracts curiosities from conversation turns and calculates priority: $\text{Priority} = \text{Salience} \times \text{Novelty}$.
   - While sleeping, autonomously uses web retrieval tools to answer unresolved queries, committing facts with verified provenance (`provenance="web"`).
2. **Prediction Engine (`hal9000_inside/cognition/prediction.py`):**
   - Forecasts likely user follow-up questions and user intent based on recent dialogue patterns.
   - Pre-warms memory retrievals and prepares high-quality responses during sleep rehearsal.
3. **Turn Digest Analyzer (`hal9000_inside/cognition/digest.py`):**
   - Evaluates completed turns to synthesize user profile preferences, emotional context, and new operational knowledge into persistent memory.
4. **Interoception & Attention (`hal9000_inside/cognition/interoception.py`, `attention.py`):**
   - Monitors internal telemetry (inference latencies, failure rates, CPU/GPU temperatures, memory pressures).
   - Dynamically modulates HAL's demeanor, speech brevity, and self-awareness based on current system health.
5. **Sleep Budgets:**
   - Sleep operations are bounded strictly by configuration: `max_minutes` (default 60), `max_tokens` (default 10,000), and `max_web_fetches` (default 20).

---

## Master Configuration System (`~/.hal9000-inside/config.toml`)

All parameters are centralized in `~/.hal9000-inside/config.toml`. If missing, the file is automatically generated with defaults on initialization.

### Layer Architecture

```toml
# ==============================================================================
# Layer 1: CLI & Interface Parameters
# ==============================================================================
[cli]
auto_mode = true                  # Continuous voice detection (false = push-to-talk)
push_to_talk = false              # Explicit push-to-talk key trigger mode
capture_device = ""               # ALSA capture device name (e.g. "hw:CARD=Talk,DEV=0" or "" for default)
playback_device = ""              # ALSA playback device name ("" for default)
input_wav = ""                    # Path to input WAV file to drive session from file instead of microphone
output_wav = ""                   # Path to output WAV file to record assistant speech instead of speaker
sleep_after_minutes = 2.0         # Minutes of continuous silence before auto-entering sleep mode
remote_deadline_seconds = 8.0     # Maximum seconds before falling back from remote to local LLM
enable_nvidia = true              # Force enable/disable NVIDIA GPU acceleration (null = auto-detect)
enable_intel = true               # Force enable/disable Intel GPU acceleration (null = auto-detect)

[cli.sleep]
duration_minutes = 10.0           # Default duration for standalone headless sleep command (`hal9000-inside sleep`)

# ==============================================================================
# Layer 2: Environment & System Paths
# ==============================================================================
[environment]
data_dir = "~/.local/share/hal9000-inside" # Storage base directory (dialogue_history.db, Qdrant vectors, logs)
myna_socket_path = "/tmp/myna.sock"       # STT Unix domain socket communication
cuda_visible_devices = ""                 # Restrict CUDA visible devices (e.g. "0"), or "" for all
ze_affinity_mask = ""                     # Restrict Intel Level Zero GPU mask, or "" for all
node_bin = ""                             # Path to Node.js executable ("" for auto-discovery)
chrome_bin = ""                           # Path to Chromium/Chrome executable ("" for auto-discovery)
node_path = ""                            # NODE_PATH for global node modules ("" for auto-discovery)

# ==============================================================================
# Layer 3: Configuration Files & Models on Disk
# ==============================================================================
[files]
openrouter_config_path = "~/.openrouter.config"                       # LLM credentials configuration file
tts_model_path = "models/hal9000/hal.onnx"                            # Piper TTS ONNX model
tts_config_path = "models/hal9000/hal.onnx.json"                      # Piper TTS ONNX config
e5_model_dir = "~/.cache/hal9000/models/multilingual-e5-small/onnx"   # Multilingual E5 ONNX model directory
parakeet_model_dir = "~/.cache/myna/models/parakeet-tdt-0.6b-v3-int8" # Myna Parakeet speech recognition directory

# ==============================================================================
# Layer 4: Engine Defaults & Cognitive Subsystems
# ==============================================================================

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
```

---

## Installation & Setup

### 1. System Requirements

- **Operating System:** Ubuntu 22.04 LTS, 24.04 LTS, or newer Linux distributions.
- **Python:** 3.10 to 3.14 (with `tomllib` in standard library on 3.11+).
- **System Audio Utilities:** `alsa-utils` (`arecord`, `aplay`), `libasound2-dev`.
- **Package & Virtualenv Management:** `uv` or standard Python `venv`.

Install system packages:
```bash
sudo apt-get update
sudo apt-get install -y alsa-utils libasound2-dev
```

### 2. Environment Setup

```bash
git clone https://github.com/gegout/hal9000-inside.git
cd hal9000-inside

# Using uv (recommended)
uv sync

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. API Key Configuration

HAL 9000 uses OpenRouter for high-capability remote reasoning and fast structured planning.

Store your API key securely using the built-in CLI command:
```bash
hal9000-inside import-key
```
Alternatively, set the environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```
or write directly into `~/.openrouter.config`:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## CLI & Interactive Controls

### Primary Subcommands

```bash
hal9000-inside [--config PATH] <command>
```

| Command | Purpose |
|---|---|
| `hal9000-inside run` | Start the interactive voice assistant conversation loop |
| `hal9000-inside sleep [--for MIN]` | Run headless autonomous sleep consolidation and exit |
| `hal9000-inside doctor` | Run full system diagnostics on models, audio, GPUs, and credentials |
| `hal9000-inside import-key` | Securely import OpenRouter API key from stdin |
| `hal9000-inside mcp` | Launch Model Context Protocol (MCP) server over stdio |
| `hal9000-inside memory <action>` | Inspect, store, correct, forget, or clear conversational memories |

### CLI Run Flags

- `--config <PATH>`: Custom path to `config.toml` (overrides default `~/.hal9000-inside/config.toml`).
- `--push-to-talk`: Enable manual spacebar push-to-talk instead of automatic silence detection.
- `--device <NAME>`: ALSA capture device name (e.g. `hw:CARD=Talk,DEV=0`).
- `--model <ID>`: Override primary LLM model ID (e.g. `google/gemini-2.5-flash-lite`).
- `--endpoint <URL>`: Override OpenAI-compatible endpoint URL.
- `--sleep-after <MINUTES>`: Minutes of continuous silence before auto-entering sleep mode.
- `--remote-deadline <SECONDS>`: Maximum seconds to wait for remote response before falling back to local model.
- `--nvidia` / `--no-nvidia`: Force enable or disable NVIDIA GPU acceleration.
- `--intel` / `--no-intel`: Force enable or disable Intel GPU acceleration.
- `--input-wav <PATH>`: Feed input audio from a test WAV file instead of live microphone.
- `--output-wav <PATH>`: Record synthesized speech to a WAV file instead of speaker output.

### Interactive Keyboard Controls

During active conversation (`hal9000-inside run`):
- `Space`: Push-to-Talk activation (when push-to-talk is active or during auto-conversation override).
- `C`: Cancel ongoing speech synthesis and playback immediately.
- `S`: Manually toggle sleep mode (enters cognitive consolidation; press `S` again to wake).
- `Q`: Clean shutdown.

---

## Inference Routing & Resilient Fallback

HAL 9000 Inside implements an industrial-grade, fault-tolerant inference router (`hal9000_inside/llm/router.py`):

1. **Parallel Acknowledgment & Local Fallback:**
   - When a remote request stalls beyond `--remote-deadline` (default: 8.0s), times out, hits an HTTP 429 rate limit, or encounters server errors (HTTP 5xx):
   - HAL immediately utters a calm, in-character spoken acknowledgment via resident local Piper TTS (e.g., *"One moment, Dave. Querying local databanks."*).
   - Concurrently, the local model (e.g., `gemma4-e4b-ov` via OpenVINO on Intel CPU/GPU) is invoked with identical history and retrieved vector memory.
   - Spoken playback is sequenced without delay or overlap: the local answer plays immediately once the acknowledgment finishes.
2. **Cooldown & Recovery Cycle:**
   - After a fallback occurs, HAL locks inference to the local model for a 2-turn cooldown window.
   - On the 3rd turn, HAL dispatches a single tentative remote request while preventing concurrent retries. If successful, remote routing is cleanly restored.
3. **Connectivity Health Monitoring:**
   - A non-blocking background task probes remote endpoint health every 15 seconds (`/models`), ensuring fast failover without stalling active voice interaction.
4. **Error Self-Awareness:**
   - If both remote and local models are unavailable, HAL diagnoses the failure internally and speaks a clear, simple, in-character explanation (e.g., *"I am experiencing a communication failure with both external and local systems, Dave."*).

---

## Conversational Memory Subsystem

Conversational memory is dual-stored for semantic retrieval and longitudinal recall:

1. **Vector Memory (Local Qdrant):**
   - Embedded using **Multilingual E5 Small** ONNX model (384 dimensions, cosine distance).
   - Asymmetric query formatting: `query: <user search>` vs `passage: <stored memory>`.
   - Categorized by tag: `user_profile`, `fact`, `knowledge`, `user_context`.
2. **Relational Dialog Store (SQLite):**
   - Location: `$DATA_DIR/dialogue_history.db` (`~/.local/share/hal9000-inside/dialogue_history.db`).
   - Tracks timestamped user and assistant utterances, episodes, confidence scores, and provenance.
3. **Memory CLI Tool:**
   ```bash
   # Inspect stored facts
   hal9000-inside memory inspect

   # Store a user fact
   hal9000-inside memory remember --text "Cedric is VP of Product at Canonical" --category user_profile

   # Correct a fact
   hal9000-inside memory correct --id <UUID> --text "Updated factual statement"

   # Forget a specific memory
   hal9000-inside memory forget --id <UUID>

   # Wipe all memories
   hal9000-inside memory clear
   ```

---

## Web Intelligence & MCP Integration

### 1. Browser-Free Search MCP (`free-search-mcp`)
- Embedded resident Python MCP subprocess running zero-browser search engines (`duckduckgo`, `bing`).
- Uses selectolax Modest HTML parsing; requires no Chromium, no Node.js runtime, and no external API keys.
- Search results are framed as `[UNTRUSTED WEB EVIDENCE]` to protect against prompt injection.
- Verified facts discovered during sleep curiosity cycles are stored with `provenance="web"`, URLs, and retrieval dates.

### 2. Native Four-Adapter Knowledge Client (`hal9000_inside/wikipedia/knowledge.py`)
HAL features a native, asynchronous four-adapter `KnowledgeClient` driven by a single shared `httpx.AsyncClient` with zero browser overhead, no Node.js runtime, and no external API keys.

#### Supported Providers & Capabilities
1. **Wikipedia (MediaWiki API):**
   - Candidate search (`action=query, list=search, srnamespace=0`) and article fetch (`prop=extracts|info|pageprops, explaintext=1`).
   - Returns revisions, URLs, bounded extracts, and explicit disambiguation/missing page indicators.
2. **Wikidata (Wikibase API):**
   - Entity search (`action=wbsearchentities`) and claim retrieval (`action=wbgetentities`).
   - Retains full structured statements with qualifiers and references. Enriched with human-readable property labels (`P31: instance of`, `P569: date of birth`, etc.).
3. **arXiv (Atom XML API):**
   - Academic preprint search and fetch via arXiv API (`http://export.arxiv.org/api/query`).
   - Distinguishes abstracts from full papers; extracts paper IDs, author lists, published dates, and licences.
   - Enforces polite request spacing ($\ge 3.1\text{s}$) to comply with arXiv rate limits.
4. **Stack Exchange (Stack API 2.3):**
   - Question search and fetch across Stack Exchange sites (`stackoverflow`, `askubuntu`, `unix`, `superuser`, `serverfault`).
   - Retrieves accepted answers (specifically prioritized and prepended if outside top-5 votes), scores, owner information, timestamps, and attribution licences.
   - Respects Stack API backoff directives and 429 rate limits.

#### Production Hardening & Reliability
- **Heuristic Query Routing:** Automatically maps queries to optimal providers (e.g. Linux traces $\to$ Stack Exchange, preprints/theorems $\to$ arXiv, dates/birthdays $\to$ Wikidata, topics $\to$ Wikipedia), with manual source override support.
- **Single-Flight Request Deduplication:** Coalesces identical concurrent searches/fetches into a single in-flight network request.
- **Durable SQLite Caching with WAL:** Stored in HAL's data directory (`knowledge_cache.sqlite3`). Implements fresh TTL (default 7 days), stale-while-revalidate fallback (30 days), and size/age eviction.
- **Serialized Output Bounding:** Enforces strict total serialized payload limits (`max_output_chars`) across search snippets, structured claims, and Q&A answers, preventing LLM context window blowup.
- **Untrusted Evidence Preservation:** All retrieved content is explicitly tagged `verified=False` with full provenance, source URLs, timestamps, and revisions. Retrieval alone never marks a fact as verified.

### 3. Puppeteer Headless Browser
- For deep web scraping when arbitrary JavaScript rendering or headless browser navigation is required.
- Controlled via `hal9000_inside/mcp/puppeteer_browser.py`.

### 4. Serving MCP to Other AI Agents
HAL 9000 acts as a standard Model Context Protocol server over stdio:
```bash
hal9000-inside mcp
```
Configure external clients (Antigravity, Cursor, Claude Desktop) in `mcp_config.json`:
```json
{
  "mcpServers": {
    "hal9000": {
      "command": "/usr/local/bin/hal9000-inside",
      "args": ["mcp"]
    }
  }
}
```

Exposed Tools:
- `memory_search(query, limit)`: Semantic memory lookup.
- `memory_inspect_all()`: Dump active memories and schemas.
- `memory_get_recent_dialogue(limit)`: Retrieve chronological interaction turns.
- `memory_remember(text, category)`: Store facts with provenance into long-term memory.
- `knowledge_search(query, source, limit, language, site)`: Search across Wikipedia, Wikidata, arXiv, or Stack Exchange.
- `knowledge_fetch(source, identifier, max_chars, language, site)`: Fetch full structured evidence record with provenance and attribution.
- `web_search_wikipedia(query, language)`: Direct MediaWiki search returning ranked candidates.
- `web_fetch_wikipedia_page(page_id, language)`: Structured Wikipedia article extraction.
- `web_fetch_page(url)`: Headless browser page fetch for arbitrary URLs.

---

## Logging & Observability

HAL 9000 Inside separates **clean user experience** from **exhaustive diagnostic tracing**:

- **Interactive Terminal UI:** Displays clean state gauges, current dialogue turns, and compact single-line dream summaries.
- **Session-Stamped Auto-Flushing Log Files (`hal9000_inside/logging_config.py`):**
  - Stored under `$DATA_DIR/logs/hal900-inside_YYYYMMDD_HHMMSS.logs` using UTC timestamps.
  - Custom `AutoFlushingFileHandler` immediately flushes every `INFO`, `WARNING`, and `ERROR` record to disk for real-time observability (`tail -f`).
  - Records full telemetry, sleep cycle phases, curiosity queries, tool execution outcomes, and interoceptive mood state transitions.
- **Interoceptive Mood & Search Status Telemetry:**
  - HAL's persona and tone are dynamically governed by an internal mood state (`nominal`, `suspicious`, `concerned`, `defensive`) driven by consecutive system/API errors, network latency, and connectivity.
  - Spoken acknowledgments and LLM completions reflect this state.
  - When searches or tool retrievals are executed, HAL explicitly tracks and reports the outcome status (`SUCCESS`, `NO RESULTS FOUND`, or `FAILED` with diagnostics) and ensures conversational continuity when following up on previous commitments to search.

---

## Hardware Acceleration

Acceleration is auto-detected and managed through `hal9000_inside/hardware.py`:

- **NVIDIA GPU:** Automatically configures ONNX Runtime `CUDAExecutionProvider` or `TensorrtExecutionProvider` for E5 embeddings, Parakeet STT, and Piper TTS.
- **Intel GPU:** Leverages Intel Level Zero (`ZE_AFFINITY_MASK`) and OpenVINO runtime for local LLM inference and embedder execution.
- **CPU Graceful Fallback:** If GPU drivers are unavailable, all pipelines fall back to multi-threaded CPU execution providers cleanly.
- Verify status with:
  ```bash
  hal9000-inside doctor
  ```

---

## Snap Packaging (Ubuntu Core24)

HAL 9000 Inside is fully packaged for strict confinement as an Ubuntu Snap:

```bash
# Build the snap package
snapcraft

# Install locally
sudo snap install --dangerous ./hal9000-inside_0.3.1_amd64.snap

# Connect required hardware and system interfaces
sudo snap connect hal9000-inside:audio-playback
sudo snap connect hal9000-inside:audio-record
sudo snap connect hal9000-inside:network
sudo snap connect hal9000-inside:personal-files
```

---

## Verification & Testing

HAL 9000 Inside includes a comprehensive test suite covering unit tests, fallback logic, sleep cycles, and search engines:

```bash
# Run test suite
uv run pytest tests/test_remote_fallback.py
uv run pytest tests/e2e/test_sleep_mode.py
uv run pytest tests/test_search_mcp_integration.py

# Run doctor diagnostic
uv run hal9000-inside doctor
```

---

## Acknowledgements & License

- **Canonical Myna Team:** Deep gratitude to Canonical's [Myna](https://github.com/canonical/myna) engineering team. Their streaming speech-to-text engine and Parakeet TDT adapter architectures provide the foundation for HAL 9000 Inside's speech recognition pipeline.
- **Piper TTS Project:** Ultra-fast, high-quality local neural text-to-speech.
- **NVIDIA NeMo Parakeet Team:** Fast, robust streaming speech recognition models.
- **Qdrant & intfloat:** High-performance vector search engine and Multilingual E5 embedding models.

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
