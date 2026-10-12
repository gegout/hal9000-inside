# Phase 1 Report — Memory v2, Telemetry, and Interoception

**Date:** 2026-10-12
**Status:** Completed & Verified

## Summary of Accomplishments

1. **Memory Schema v2 & Migrations:**
   - Extended `memories` table with `provenance` (user | assistant | remote_model | web | consolidation | self), `model_id`, `confidence`, `salience`, `access_count`, `last_accessed_at`, `expires_at`, `episode_id`, `source_urls`.
   - Added `schema_version` table and implemented automatic forward migration from v1 to v2.
   - Added `episodes` table for tracking session start/stop and topic summaries.
   - Added `turn_telemetry` table for persisting `stt_ms`, `retrieval_ms`, `llm_ms`, `tts_first_audio_ms`, `tts_total_ms`, `model_id`, `route`, token counts, tool calls, error counts, and interruption flags.
   - Upgraded retrieval to rank by `cosine * (1.0 + salience_weight * salience + recency_weight * recency_factor)` and automatically increment `access_count`.

2. **Interoception:**
   - Created `hal9000_inside/cognition/interoception.py` with `InteroceptionMonitor` reading memory pressure (`/proc/pressure/memory`), normalized 1m CPU load, GPU status, connectivity, and latency.
   - Injected compact, honest self-state prompt block (< 60 tokens) into HAL 9000 system prompt per turn.

3. **Verification:**
   - E2E tests assert telemetry table row generation.
   - Tested v1 to v2 schema migration on legacy fixtures.
   - Verified access count increment and provenance tracking.
   - All tests pass via `scripts/check.sh`.
