# HAL 9000 Inside — Architecture & Engineering Decisions

## Phase 0 Decisions

### 1. Embedded Qdrant Concurrency & Multi-Process Handling
- **Problem:** Embedded Qdrant (`qdrant_client.QdrantClient(path=...)`) exclusively locks the SQLite/WAL/storage directory. Spawning multiple processes pointing to the same data directory (e.g., the assistant and an out-of-process MCP server) triggers storage locking errors.
- **Decision:** Embedded Qdrant file lock is explicitly handled with a clear descriptive error (`RuntimeError("Embedded Qdrant database at ... is already open by another process.")`). In Phase 1+, the MCP server and external tools query the running assistant via socket/in-process memory client or distinct paths.

### 2. Audio Capture Pipe & Buffer Management
- **Problem:** When speech synthesis or a long turn is running, `arecord` continues capturing PCM frames into the OS pipe. Reading immediately afterwards would feed stale background noise or old speech into the STT engine.
- **Decision:** Added `purge_buffered()` to `AudioSource` / `ArecordSource`. After each spoken turn completes, `capture.purge_buffered()` discards all queued frames in `StreamReader._buffer` before the assistant resumes VAD listening.

### 3. File-Driven Audio Testing & Abstraction
- **Problem:** Tests and headless runs cannot rely on ALSA hardware (`arecord` / `aplay`).
- **Decision:** Introduced `AudioSource` (`ArecordSource`, `WavFileSource`) and `AudioSink` (`AplaySink`, `WavFileSink`, `NullSink`) in `hal9000_inside/audio/manager.py`. CLI supports `--input-wav` and `--output-wav`. Tests run completely hermetic with `FakeSTT`, `FakeEmbedder`, `FakeTTS`, `FakeClock`, and `FakeOpenAIServer`.

## Sleep Mode Decisions

### 1. Offline Mode: Restructure, Never Invent
- **Rule:** When offline (local model or no network), HAL must strictly avoid hallucinating or inventing new facts during sleep cycles.
- **Decision:** Steps 2 (endogenous curiosity web search) and 3 (remote model dialogue rehearsal) are skipped entirely when offline. Offline cycles exclusively execute consolidation (merging duplicates, pruning stale memories), reflection (telemetry stats, bounded parameter tuning), and interoception.

### 2. Immediate Mute and Purge on Wake
- **Problem:** Audio captured while asleep or buffered in pipes could trigger unintended turns upon waking.
- **Decision:** Entering sleep sets `audio_ctrl.set_muted(True)`, stops VAD, and cancels any running TTS. Waking un-mutes audio, calls `capture.purge_buffered()` to discard all stale audio accumulated during sleep, and returns cleanly to `LISTENING`.

### 3. One-Line Sleep Report Injection
- **Decision:** On wake, a concise summary is logged (`Slept <X> min, learned <Y> notes, <online/offline the whole time>.`) and stored in `InteroceptionMonitor.last_sleep_report`. On the very next spoken turn, it is injected into the system prompt's self-state block and then cleared.
