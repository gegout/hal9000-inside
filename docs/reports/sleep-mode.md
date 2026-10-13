# HAL 9000 Inside — Sleep Mode Verification Report

## 1. Overview & Objectives
Sleep mode provides HAL 9000 with dedicated background consolidation and learning intervals:
- Audio capture and playback are muted/stopped.
- Background cognitive processing runs with no preemption and configurable budgets (`max_minutes`, `max_tokens`, `max_web_fetches`).
- Looping across the 6 cognitive phases:
  1. **Backlog**: drain open questions and pending prefetch candidates.
  2. **Endogenous curiosity**: pick top queued questions by salience x novelty (Online only: answers via web MCP and/or remote LLM, writes notes with `provenance='web'|'remote_model'`).
  3. **Rehearsal**: synthesize dialogue exemplars for predicted follow-ups (Online only).
  4. **Consolidation**: merge duplicate memories, link related facts, deactivate stale notes (Online and Offline).
  5. **Reflection**: parse telemetry, update `self_model.json` stats, tune parameters within safe bounds (Online and Offline).
  6. **Interoception**: sample CPU, memory, GPU, and connection telemetry every cycle (Online and Offline).
- **Offline Rule**: *Restructure, never invent*. Steps 2 and 3 are skipped when offline.
- **Wake Behavior**: Unmute microphone, purge buffered audio, log sleep summary, and inject a one-line sleep report (`Slept <duration>, learned <N> notes, <online|offline the whole time>.`) into the immediate next turn's self-state block.

## 2. Gate Verification (`tests/e2e/test_sleep_mode.py`)

All 6 required gate tests passed hermetically without network access or physical audio devices:

| Test Case | Description & Verification Result |
|---|---|
| `test_sleep_mutes_io` | Entering sleep mutes the audio controller. Audio sent during sleep is not transcribed and produces no TTS. Waking un-mutes and resumes normal turn processing. **PASS** |
| `test_sleep_runs_jobs_online` | Queued questions and predictions are answered online via fake MCP and remote LLM. Notes are stored with `provenance='web'` / `'remote_model'`, and `self_model.json` is updated. **PASS** |
| `test_sleep_offline_no_invention` | When offline, zero requests hit the remote LLM, zero web/remote notes are created, questions remain queued, while memory deduplication and self-model reflection still execute. **PASS** |
| `test_sleep_budget_stops` | Sleep manager strictly halts when budget (`max_minutes`) is hit, stopping loop cleanly and saving partial progress. **PASS** |
| `test_sleep_report_on_wake` | After waking, the next turn prompt contains the one-line sleep report in the system state block. **PASS** |
| `test_sleep_headless_cli` | `hal9000-inside sleep --for <minutes> --data-dir <dir>` runs headlessly with no audio devices opened and exits with code 0. **PASS** |

## 3. Test Suite Run Output

```text
=== 1. Checking package installation ===
=== 2. Checking imports ===
All core modules imported successfully.
=== 3. Running unit test suite ===
.............................
----------------------------------------------------------------------
Ran 29 tests in 11.013s

OK
=== All checks passed successfully. ===
```
