# HAL 9000 Inside (Ubuntu Voice Assistant)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version: 0.2.0](https://img.shields.io/badge/Version-0.2.0-red.svg)](pyproject.toml)

An autonomous local voice assistant for Ubuntu embodying the calm, polite, and precise persona of **HAL 9000** from *2001: A Space Odyssey*.

HAL 9000 Inside integrates local streaming automatic speech recognition, vector-embedded conversational memory, Model Context Protocol (MCP) tool use and serving, Puppeteer headless web retrieval, OpenRouter LLM reasoning, and native 22,050 Hz Piper TTS speech synthesis with half-duplex feedback prevention.

---

## Key Features

- **Local Low-Latency Speech-to-Text (STT):** NVIDIA Parakeet TDT 0.6B v3 model integrated via Canonical's Myna streaming engine with dynamic voice activity detection (VAD) and silence endpointing.
- **Persistent Conversational Memory:**
  - Semantic vector search using embedded local **Qdrant** and normalized 384-dimensional **Multilingual E5 Small** ONNX embeddings.
  - Complete timestamped conversational history and turn tracking stored in SQLite.
- **Model Context Protocol (MCP) Dual Architecture:**
  - **MCP Server (`hal9000-inside mcp`):** Exposes HAL's embedded memory and Puppeteer web tools over stdio to any MCP client (Antigravity, Cursor, Claude Desktop).
  - **MCP Client:** HAL autonomously connects to configured MCP servers (defined in `~/.gemini/config/mcp_config.json`) and invokes external tools during speech reasoning.
- **Puppeteer Headless Web Intelligence:** Integrated headless Chrome browsing and text extraction to fetch articles, documentation, and Wikipedia entries.
- **Faithful HAL 9000 Voice Synthesis:** Resident Piper TTS neural engine outputting native 22,050 Hz mono audio with instant playback cancellation on interruption.
- **Half-Duplex Audio Coordination:** Acoustic gating and cooldown windows prevent microphone self-triggering from speaker audio.
- **Hardware Acceleration:** Auto-detection and seamless offload to NVIDIA GPUs (CUDA/TensorRT) and Intel GPUs, with graceful CPU fallback.
- **Compact Terminal UI:** Clean single-line live VU/energy gauge, status emojis (👂 Listening, 🧠 Thinking, 🔴 Responding), and formatted dialog display.

---

## Architecture Overview

```text
       ┌────────────────────────┐
       │   Microphone (ALSA)    │
       └───────────┬────────────┘
                   │ 16 kHz Mono PCM
                   ▼
       ┌────────────────────────┐
       │  SimpleVAD & Half-     │
       │  Duplex Audio Gating   │
       └───────────┬────────────┘
                   │ Speech Frames + Pre-roll
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
                   │ Augmented Context
                   ▼
       ┌────────────────────────┐       ┌─────────────────────────────┐
       │  HAL 9000 Reasoning    │ ◄───► │ MCP Client & Server         │
       │  (OpenRouter / Gemma)  │       │ (Memory, Puppeteer, Web)    │
       └───────────┬────────────┘       └─────────────────────────────┘
                   │ Generated Response
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

## Installation & Setup

### 1. System Requirements

- **Operating System:** Ubuntu 22.04 LTS, 24.04 LTS, or newer Linux distributions.
- **Python:** 3.10 or higher.
- **System Utilities:** `alsa-utils` (`arecord`, `aplay`).
- **Web Retrieval Dependencies:** Node.js (v18+), `google-chrome` or Chromium, and `puppeteer-core`.

Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y alsa-utils libasound2-dev google-chrome-stable
```

### 2. Environment Setup

Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/gegout/hal9000-inside.git
cd hal9000-inside
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Install Puppeteer for web retrieval:
```bash
npm install -g puppeteer-core
```

### 3. API Key Configuration

HAL 9000 uses OpenRouter (or any OpenAI-compatible endpoint) for reasoning.

Store your API key securely using the built-in CLI command:
```bash
hal9000-inside import-key
```
Alternatively, set environment variables or edit `~/.openrouter.config`:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

---

## Usage

### System Diagnostics
Verify all models, checksums, hardware acceleration, audio devices, and credentials:
```bash
hal9000-inside doctor
```

### Start Voice Assistant
Run interactive speech loop in auto voice-activity detection mode:
```bash
hal9000-inside run
```

Command-line options:
- `--push-to-talk`: Enable manual spacebar push-to-talk instead of automatic silence detection.
- `--device <NAME>`: Specify ALSA capture device (e.g. `hw:CARD=Audio,DEV=0`).
- `--model <ID>`: Override reasoning model (e.g. `gemma4-e4b-ov` or `openai/gpt-4o-mini`).
- `--endpoint <URL>`: Override OpenAI-compatible endpoint (e.g. `http://127.0.0.1:8336/v3`).
- `--nvidia` / `--no-nvidia`: Force enable or disable NVIDIA GPU acceleration.
- `--intel` / `--no-intel`: Force enable or disable Intel GPU acceleration.

Interactive Keyboard Controls during execution:
- `Space`: Push-to-Talk trigger
- `C`: Cancel ongoing speech synthesis and playback immediately
- `M`: Toggle microphone mute
- `Q`: Clean shutdown

---

## Memory Management CLI

Manage HAL 9000's embedded long-term memory directly from the terminal:

- **Inspect all memories:**
  ```bash
  hal9000-inside memory inspect
  ```
- **Store a new fact or preference:**
  ```bash
  hal9000-inside memory remember --text "Commander Dave Bowman prefers concise status updates." --category fact
  ```
- **Update/correct a memory:**
  ```bash
  hal9000-inside memory correct --id <memory-id> --text "Commander Dave Bowman is mission pilot."
  ```
- **Delete a memory:**
  ```bash
  hal9000-inside memory forget --id <memory-id>
  ```
- **Reset all conversational memory:**
  ```bash
  hal9000-inside memory clear
  ```

---

## Model Context Protocol (MCP) Support

### Running the MCP Server
HAL 9000 can serve its memory and web retrieval tools to external AI agents and IDEs (such as Antigravity, Claude Desktop, Cursor):

```bash
hal9000-inside mcp
```

Exposed MCP Tools:
- `memory_search(query, limit)`: Semantic retrieval over conversational memory.
- `memory_inspect_all()`: List all stored long-term memories and facts.
- `memory_get_recent_dialogue(limit)`: Retrieve previous dialog turns.
- `memory_remember(text, category)`: Store facts or learned knowledge into vector store.
- `web_search_wikipedia(query, language)`: Wikipedia headless retrieval via Puppeteer.
- `web_fetch_page(url)`: Extract full readable markdown text from any web URL.

### Client Configuration (`mcp_config.json`)
To connect Antigravity or other MCP clients to HAL 9000, add to `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "hal9000": {
      "command": "/home/cgegout/projects/hal9000-inside/.venv/bin/hal9000-inside",
      "args": ["mcp"],
      "env": {
        "NODE_PATH": "/home/cgegout/.local/node_modules",
        "CHROME_BIN": "/usr/bin/google-chrome"
      }
    }
  }
}
```

---

## Snap Packaging

HAL 9000 Inside is configured for strict confinement packaging with Ubuntu `core24`:

```bash
# Sync package contents and build Snap
cp -r hal9000_inside snap_root/
snap pack snap_root .
```

Install and connect interfaces:
```bash
sudo snap install --dangerous ./hal9000-inside_0.2.0_amd64.snap
sudo snap connect hal9000-inside:audio-playback
sudo snap connect hal9000-inside:audio-record
sudo snap connect hal9000-inside:network
sudo snap connect hal9000-inside:personal-files
```

---

## Testing

Run the test suite:
```bash
python3 -m unittest discover tests
```

---

## Acknowledgements

- **Canonical Myna Developer Team:** We would like to express our deepest gratitude to Canonical's [Myna](https://github.com/canonical/myna) team. Their outstanding work on low-latency streaming speech-to-text, WebSocket/Unix socket IPC architectures, and Parakeet TDT adapters served as the foundational building blocks for HAL 9000 Inside's speech recognition pipeline.
- **Piper TTS Project:** High-quality, fast local neural text-to-speech engine.
- **NVIDIA NeMo Parakeet Team:** Fast, robust streaming speech recognition models.
- **Qdrant & HuggingFace / intfloat:** Vector similarity search engine and Multilingual E5 embedding models.

---

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
