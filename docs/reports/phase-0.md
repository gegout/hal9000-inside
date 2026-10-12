# Phase 0 Report — Importable, Testable, and Drivable from Files

**Date:** 2026-10-12
**Status:** Completed & Verified

## Summary of Accomplishments

1. **Import Bugs & Latent Defect Resolution:**
   - Fixed missing `Any` and `asyncio` imports in `hal9000_inside/llm/client.py`.
   - Fixed missing `Optional` import in `hal9000_inside/memory/embeddings.py`.
   - Corrected `MemoryStore.correct()` to preserve `category` and `created_at` in both SQLite and Qdrant payloads.
   - Fixed dead imports in `hal9000_inside/stt/server_wrapper.py`.
   - Replaced hardcoded paths in `hal9000_inside/mcp/puppeteer_browser.py` with environment variable fallbacks (`NODE_BIN`, `CHROME_BIN`, `NODE_PATH`).
   - Declared `mcp` in `pyproject.toml` dependencies.
   - Added `hal9000_inside/__init__.py`. Verified editable install `pip install -e .`.
   - Handled Qdrant multi-process lock conflict gracefully with clear actionable runtime error.
   - Implemented `purge_buffered()` in `ArecordSource` to discard stale audio buffers accumulated during long turns.

2. **Test Harness & Fakes:**
   - Abstracted audio I/O with `AudioSource` (`ArecordSource`, `WavFileSource`) and `AudioSink` (`AplaySink`, `WavFileSink`, `NullSink`).
   - Added CLI support for `--input-wav` and `--output-wav`.
   - Built full suite of test fakes in `tests/fakes/`:
     - `FakeClock`
     - `FakeEmbedder` (deterministic 384-dimensional normalized vectors)
     - `FakeSTT` (scriptable transcript responses)
     - `FakeTTS` (silent PCM generation)
     - `FakeOpenAIServer` (in-process `httpx.MockTransport` supporting streaming, tool calls, error simulation)
   - Updated E2E test suite to execute hermetically without physical hardware or paid external APIs.

3. **CI & Verification:**
   - Created `scripts/check.sh` executing pip install, import check, and full unittest discovery.
   - Created `.github/workflows/ci.yml`.
   - All tests pass cleanly.
