# Phase 3 Report: Cognitive Scheduler, Turn Digest, Curiosity, Attention & Prediction

## 1. Objectives & Scope
The goal of Phase 3 was to introduce autonomous cognitive processing without adding any latency to spoken turns:
1. **Cognitive Scheduler (`hal9000_inside/cognition/scheduler.py`)**:
   - Asynchronous priority job queue (`digest`, `curiosity_search`, `prefetch`, `rehearse`, `consolidate`, `reflect`).
   - Hard constraints: preemptible immediately on VAD speech detection (`on_speech_start()`), pauses execution during active spoken turns.
   - Resource bounded: limits on concurrency (`max_concurrent=2`).
2. **Turn Digest (`hal9000_inside/cognition/digest.py`)**:
   - Structured JSON extraction from turn transcripts via Pydantic model (`TurnDigestResult`).
   - Extracts: `facts_to_remember` (with salience/confidence), `open_questions`, `focus_topics`, `predicted_followups`, `user_corrections`, and `answer_confidence`.
   - Non-blocking: enqueued as background task upon completion of assistant turn.
3. **Endogenous Curiosity (`hal9000_inside/cognition/curiosity.py`)**:
   - Queues open questions surfaced during conversations.
   - Resolves questions during idle periods via MCP tools (e.g. Wikipedia / Puppeteer search) or local cognitive reflection.
   - Stores synthesized discoveries in long-term memory with `provenance='curiosity'`.
4. **Attention Tracking (`hal9000_inside/cognition/attention.py`)**:
   - Tracks active focus topics across conversation turns with exponential decay.
   - Computes dynamic salience boost factors for contextual retrieval.
5. **Prediction & Prefetching (`hal9000_inside/cognition/prediction.py`)**:
   - Anticipates user follow-ups based on the turn digest.
   - Pre-fetches candidate memories into warm cache during idle intervals.

---

## 2. Implementation Details

- **Scheduler Preemption**:
  - Connected `CognitiveScheduler` directly to `Hal9000Assistant` capture loop.
  - On `is_speaking = True`, immediately calls `self.scheduler.on_speech_start()` which cancels all active background tasks to free CPU/GPU resources.
  - When idle silence is detected, `scheduler.run_once()` executes queued tasks.
- **Turn Digest Integration**:
  - `_enqueue_post_turn_cognition()` in `assistant.py` launches turn digest analysis as a high-priority background job.
  - Facts extracted from the user turn are stored in `MemoryStore` with metadata (`provenance='assistant'`, `confidence`, `salience`).
  - Extracted open questions are queued in `CuriosityEngine`.
- **Attention & Prediction**:
  - `AttentionTracker` updates topic weights dynamically after each turn.
  - `PredictionEngine` prefetches matching candidate memories into memory cache.

---

## 3. Verification & Gate Results
- **Unit and Hermetic Tests (`tests/test_cognition_phase3.py`)**:
  - `test_scheduler_priority_and_preemption`: Verified task execution order by priority and immediate task cancellation upon `on_speech_start()`.
  - `test_turn_digest_extraction`: Verified strict Pydantic validation of turn digest payloads.
  - `test_attention_and_prediction`: Verified topic weight decay, salience boost calculation, and candidate prefetching.
  - `test_curiosity_exploration_and_memory`: Verified curiosity question processing, LLM reflection, and persistent storage with `provenance='curiosity'`.
- **Full Test Suite (`scripts/check.sh`)**:
  - All 23 tests passed cleanly (`test_audio`, `test_hardware`, `test_llm`, `test_memory`, `test_parakeet_live`, `test_stt_contract`, `test_tts`, `test_assistant_e2e`, `test_cognition_phase3`).
