# HAL 9000 Inside — Investigation Report

**Repository:** `gegout/hal9000-inside`
**Base commit:** `2662153` — "Release v0.2.0" (2026-10-12)
**Method:** code reading only. Nothing was executed, as requested.
Anything marked **(inferred)** or **(verify)** was not confirmed by running code.

---

## 1. What the application is today

One Python process. One asyncio event loop. One spoken turn at a time.

| Component | File | Role |
|---|---|---|
| Core loop | `hal9000_inside/assistant.py` | Capture → VAD → STT → memory → LLM → TTS → playback |
| LLM client | `hal9000_inside/llm/client.py` | One OpenAI-compatible endpoint, retries, optional streaming |
| Memory | `hal9000_inside/memory/store.py`, `embeddings.py` | SQLite turns + embedded Qdrant + E5 small (384-d) |
| MCP server | `hal9000_inside/mcp/server.py` | 6 tools over stdio for external agents |
| MCP client | `hal9000_inside/mcp/client_manager.py` | Connects to servers in `~/.gemini/config/mcp_config.json` |
| Web | `hal9000_inside/mcp/puppeteer_browser.py`, `puppeteer_runner.js` | Headless Chrome fetch / Wikipedia search |
| Audio | `hal9000_inside/audio/manager.py` | `arecord` capture, `aplay` playback, VAD, half-duplex gate |
| STT | vendored `hal9000_inside/stt/myna/**` | Parakeet TDT 0.6B int8 ONNX adapter (Canonical Myna) |
| TTS | `hal9000_inside/tts/hal_piper.py` | Piper HAL voice, 22,050 Hz, sentence splitter |
| UI | `hal9000_inside/ui/terminal.py` | Single live status line + committed dialog lines |
| CLI | `hal9000_inside/cli/main.py` | `run`, `doctor`, `memory`, `import-key`, `mcp` |
| Hardware | `hal9000_inside/hardware.py` | NVIDIA / Intel detection, ONNX provider list |

`snap_root/hal9000_inside` is a byte-identical manual copy of the package.
Only `tests/test_assistant_e2e.py` differs between the two trees (one assertion).

---

## 2. The turn pipeline, step by step

All in `assistant.py`. Line numbers refer to the base commit.

1. **Frame loop** — `async for frame in capture.stream_frames()` (`assistant.py:161`).
   Frames come from an `arecord` subprocess read with `readexactly` (`audio/manager.py:190-200`).
2. **Half-duplex gate** — capture skipped while speaking or in 0.6 s cooldown (`assistant.py:166-170`).
3. **VAD** — `SimpleVAD.process_frame` returns `(is_speaking, utterance_finished, energy)` (`assistant.py:172`).
4. **End of utterance** — if buffer > 0.4 s of audio, `await self._handle_turn(...)` (`assistant.py:197-204`).
   The frame loop **blocks here** until the whole turn is done.
5. **STT** — in-process `ParakeetAdapter.run_session` (`assistant.py:325-344`).
6. **Record user turn** in SQLite (`assistant.py:233`). Session id is always `"default"`.
7. **Cancel words** — "cancel" / "stop talking" stop TTS (`assistant.py:236-240`).
8. **Memory retrieval** — `memory.search(user_text, limit=4)` with cosine threshold 0.5 (`assistant.py:243`).
9. **Context** — system prompt + memory block + last 6 turns (`assistant.py:248-257`).
10. **LLM call** — `generate_response(..., max_tokens=256)` with MCP tools (`assistant.py:262`).
11. **One tool round** — executes every requested tool, then a second completion (`assistant.py:265-294`).
12. **Error fallback** — fixed line "I am sorry, Dave..." (`assistant.py:299-301`).
13. **Record assistant turn**, then synthesize the **whole reply** in one executor call and play it (`assistant.py:306-323`).

Key property: the pipeline is **strictly sequential**. Nothing runs between turns.
There are no background tasks anywhere in the package.

---

## 3. Models and routing today

- One `LLMClient`, created once at startup (`assistant.py:64`).
- Default remote model `openai/gpt-4o-mini`; `FALLBACK_MODEL` is defined but **never used** (`llm/client.py:14-15`).
- Local mode is just a URL string check for `127.0.0.1` / `localhost` (`llm/client.py:120-123`).
- Local path from CLI help: `--model gemma4-e4b-ov --endpoint http://127.0.0.1:8336/v3` (`cli/main.py:261-262`).
  The `/v3` prefix and `-ov` suffix suggest OpenVINO Model Server **(inferred)**.
- No connectivity probe. Startup only checks that a key exists (`assistant.py:116-119`).
- Timeout 30 s × 3 retries (`llm/client.py:111-112`). A dead network can stall a turn for well over a minute.
- `stream_response` exists (`llm/client.py:202-258`) but is **unused**.
- `speak_sentences` sentence pipelining exists (`tts/hal_piper.py:120-147`) but is **unused**.
- No notion of "which model answered" is stored anywhere.

So today "local vs remote" is a static choice at launch. There is no routing, no fallback, no learning across the two.

---

## 4. Memory today

**SQLite** (`memory/store.py:47-66`):

| Table | Columns |
|---|---|
| `turns` | id, timestamp, role, content, session_id |
| `memories` | id, created_at, updated_at, text, category, source_turn_id, is_active |

**Qdrant** (embedded, `path=` mode, `memory/store.py:71`): collection `hal9000_memories`, 384-d cosine.
Payload: `text, category, created_at, is_active` (`memory/store.py:119`).

**Who writes long-term memory:**

- `remember()` is called only by the CLI (`cli/main.py:153`) and the MCP server tool (`mcp/server.py:72`).
- **The assistant itself never calls `remember()`.** It only records raw turns.

**Retrieval:** one semantic search on the raw utterance, top 4, threshold 0.5. No query expansion, no recency or salience weighting.

**Missing for the target features:** provenance (who/what produced a memory), confidence, salience, access counts, expiry, model id, session boundaries, episodes.

---

## 5. Tools

- **MCP server** exposes `memory_search`, `memory_inspect_all`, `memory_get_recent_dialogue`, `memory_remember`, `web_search_wikipedia`, `web_fetch_page` (`mcp/server.py:44-89`).
- **MCP client** reads `~/.gemini/config/mcp_config.json` or `~/.config/hal9000/mcp_config.json` (`client_manager.py:20-23`) and exposes tools as OpenAI functions (`client_manager.py:108-122`).
- **In-process, the LLM has no direct access to HAL's own memory or web tools.** They reach the LLM only if HAL's own MCP server is listed in `mcp_config.json`. That spawns a **second process** with a second `MemoryStore` on the **same Qdrant path**. Embedded Qdrant is single-process **(verify qdrant-client lock behaviour)**.
- **Puppeteer** launches a Node process per call. Paths are hardcoded to `/home/cgegout/...` (`puppeteer_browser.py:20-22`). Output capped at 4,000 chars (`puppeteer_runner.js:36`), 40 s timeout.

---

## 6. Audio, STT, TTS — what matters for a cognitive layer

Good news: the asyncio structure already allows "thinking while listening or speaking".

- Capture awaits a subprocess pipe, so the loop is free between frames.
- TTS synthesis runs in an executor thread (`assistant.py:317`). Playback runs `aplay` in an executor (`audio/manager.py:154-166`). The loop is free during speech.
- `SimpleVAD.is_speaking` is a ready-made "user started talking" signal to preempt background work.
- `HalfDuplexAudioController` already tracks playing / cooldown / mute (`audio/manager.py:110-145`).
- STT is in-process. `MynaSTTClient` (`stt/client.py`) and `MynaServerDaemon` (`stt/server_wrapper.py`) are unused by the assistant.
- Myna ships a `MemoryPressureMonitor` (`stt/myna/server/lifecycle.py:190`) reading `/proc/pressure/memory`. Reusable idea for interoception.

---

## 7. UI, CLI, packaging, tests

- **UI:** states LISTENING / VOICE_DETECTED / THINKING / RESPONDING / ERROR (`ui/terminal.py:33-40`). No place for background activity.
- **CLI:** `run`, `doctor`, `memory inspect|remember|correct|forget|clear`, `import-key`, `mcp`.
- **Packaging:** `pyproject.toml` lists `piper-tts, onnxruntime, tokenizers, numpy, websockets, sounddevice, httpx, pydantic, qdrant-client`. It does **not** list `mcp` or `myna`, both imported at runtime.
  There is **no `hal9000_inside/__init__.py`**, so it is an implicit namespace package. `packages.find` may discover nothing **(verify `pip install -e .`)**. The snap wrapper sidesteps this by running `python3 -m hal9000_inside.cli.main` with PYTHONPATH set.
- **Tests:** 8 `unittest` files. Seven need live models, an API key, or audio. Only `tests/test_stt_contract.py` is a pure unit test with mocks. There are no fakes for the LLM or the embedder.

---

## 8. Latent bugs found by reading (not executed)

| # | Where | Issue | Confidence |
|---|---|---|---|
| 1 | `llm/client.py:7` vs `:133-139` | `Any` used in annotations but never imported. Python evaluates annotations at `def` time and there is no `from __future__ import annotations`, so importing the module should raise `NameError`. | High |
| 2 | `llm/client.py:182,187,198` | `asyncio.sleep` is used but `asyncio` is not imported. Fails on any 429 / 5xx / network retry. | High |
| 3 | `memory/embeddings.py:13` vs `:29` | `Optional` used but not imported. Same import-time `NameError`. | High |
| 4 | `memory/store.py:166` | `correct()` upserts a new payload and drops `category` and `created_at`. | High |
| 5 | `stt/server_wrapper.py:46-48` | Imports `myna.server.core.MynaServer` and `transport_ws.unix_serve`. Neither exists in the vendored tree (only `serve_unix` at `transport_ws.py:278`). Dead code. | High |
| 6 | `mcp/puppeteer_browser.py:20-22` | Hardcoded home paths. | High |
| 7 | `memory/store.py:71` + `mcp/server.py:33` | Two processes can open the same embedded Qdrant path (assistant + MCP server). | Medium — verify lock behaviour |
| 8 | `assistant.py:161-205` | Audio buffered in the `arecord` pipe during a long turn is consumed afterwards. Stale speech may be transcribed. | Medium |
| 9 | `tests/test_assistant_e2e.py` vs `snap_root/tests/...` | Diverged assertion. | High |
| 10 | `pyproject.toml` | `mcp` and `myna` not declared. | High |

Note on 1–3: the author presumably runs this app, so the GitHub copy may lag their working tree.
First action for any implementer: one import check (`python -c "import hal9000_inside.assistant"`).

---

## 9. Gaps versus the seven target capabilities

| Capability | Exists today | Missing |
|---|---|---|
| Endogenous curiosity | Web tools exist (via MCP only) | Question extraction, queue, deep-search worker, knowledge notes with provenance |
| Attention | Raw-utterance retrieval, top 4 | Focus state across turns, query expansion, salience-aware ranking, background enrichment while speaking |
| Interoception | GPU detection at startup | Per-turn telemetry (latencies, model, tokens, errors, interruptions), system load, honest self-state in the prompt |
| Prediction | — | Predicted follow-ups / topics, prefetch, rehearsal with remote models while online |
| Surprise | — | Prediction resolution, contradiction detection, calibration tracking |
| Metacognition | — | Utility tracking of self-generated knowledge, waste ratio, answer-confidence, governor |
| Self-improvement | — | Versioned self-model, bounded strategy tuning, behavioural addendum, rollback |
| Online / offline continuity | Static model choice at launch | Connectivity monitor, router with deadlines, fallback, consolidation of remote knowledge into local memory |

---

## 10. Design implications that shape the prompt

1. **A background scheduler that always yields to the foreground** is the backbone. The asyncio layout already permits it. Preempt on VAD speech start.
2. **One structured "turn digest" call per turn** can feed memory writes, curiosity, prediction and attention at once. That keeps cost under control.
3. **Embeddings do the cheap work.** Surprise scoring and attention updates need only E5 vectors, which are local and fast. Use the LLM only where judgement is needed.
4. **Memory schema v2** (provenance, confidence, salience, access stats, expiry) is a prerequisite for all seven capabilities.
5. **Router + connectivity monitor + streaming sentence TTS** is what delivers "fluid, connected or not".
6. **Consolidation** (remote answers and web findings distilled into local notes, plus style exemplars) is the concrete mechanism for "learn while connected, stay smart offline".
7. **The self-model must be bounded, versioned and reversible.** HAL tunes parameters within ranges and a short prompt addendum. It never edits code or its core persona.
8. **Phase 0 must fix the import bugs** and add LLM / embedder fakes, or nothing else can be tested.
