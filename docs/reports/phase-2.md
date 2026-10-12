# Phase 2 Report — Router, Connectivity, and Streaming Pipelining

**Date:** 2026-10-12
**Status:** Completed & Verified

## Summary of Accomplishments

1. **LLM Router & Fallback:**
   - Implemented `LLMRouter` in `hal9000_inside/llm/router.py`. Dynamically selects `remote` or `local` execution per turn.
   - Incorporated `ConnectivityMonitor` for non-blocking health checks of remote endpoints.
   - Enforced per-turn deadline (`first_audio_deadline_seconds=8.0s`) with automatic mid-turn fallback to local model upon timeout or 5xx/network errors.
   - Recorded `route` and `model_id` in `turn_telemetry` and SQLite.

2. **Streaming & Audio Pipelining:**
   - Connected `speak_sentences` in `HalPiperTTS` to synthesize and stream sentence-by-sentence PCM audio.
   - Began audio playback as soon as the first sentence completes synthesis without waiting for full response completion.

3. **Verification:**
   - Added unit and E2E tests for remote routing, deadline failure fallback to local model, and forced local overrides.
   - `scripts/check.sh` passes 19 tests with zero errors.
