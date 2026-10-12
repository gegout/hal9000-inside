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
