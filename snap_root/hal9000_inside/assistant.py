"""HAL 9000 Voice Assistant Core Loop.

Coordinates:
- Audio Capture (ALSA / arecord 16kHz)
- VAD (dynamic noise floor & silence cut)
- STT (Parakeet TDT 0.6B v3 backend)
- Conversational Memory (SQLite + local Qdrant + E5 ONNX embeddings)
- LLM Reasoning (OpenRouter OpenAI-compatible API)
- TTS (Piper HAL 9000 voice @ 22,050 Hz)
- Audio Playback (ALSA / aplay with half-duplex feedback prevention)
- Terminal UI (live state display: LISTENING, THINKING, SPEAKING, ERROR)
"""

import asyncio
import ast
import collections
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from hal9000_inside.audio import (
    AlsaCapture,
    AlsaPlayback,
    HalfDuplexAudioController,
    SimpleVAD,
    list_audio_devices,
)
from hal9000_inside.hardware import ComputeDeviceConfig, detect_system_gpus
from hal9000_inside.llm import LLMClient, LLMRouter, HAL9000_SYSTEM_PROMPT, load_system_prompt
from hal9000_inside.llm.client import select_fastest_talking_model
from hal9000_inside.llm.router import DEFAULT_LOCAL_ENDPOINT, DEFAULT_LOCAL_MODEL, ConnectivityMonitor
from hal9000_inside.memory import MemoryStore, MultilingualE5Embedder
from hal9000_inside.mcp import MCPClientManager
from hal9000_inside.wikipedia import WikipediaClient, KnowledgeClient
from hal9000_inside.cognition import (
    AttentionTracker,
    BackgroundJob,
    CognitiveScheduler,
    CuriosityEngine,
    InteroceptionMonitor,
    PredictionEngine,
    SleepBudget,
    SleepManager,
    TurnDigestAnalyzer,
)
from hal9000_inside.tts import HalPiperTTS
from hal9000_inside.ui import TerminalUI
from myna.core import AudioFormat, PcmChunk, SessionConfig, TranscriptionDone
from myna.testbed.parakeet import ParakeetAdapter
from hal9000_inside.llm.intent import classify_turn_intent
from hal9000_inside.llm.turn_context import build_turn_context, TurnContext
from hal9000_inside.llm.smollm2_quick import SmolLM2QuickResponder
from hal9000_inside.speech_norm import count_spoken_words, validate_local_response, reconcile_continuation
from hal9000_inside.task_ledger import TaskLedger, TaskExecutionStatus, TaskDeliveryStatus
from hal9000_inside.tts.speech_queue import OrderedSpeechQueue

from hal9000_inside.config import load_config
from hal9000_inside.logging_config import setup_session_logging, log_dialogue_turn
setup_session_logging()
logger = logging.getLogger(__name__)
FORMAT_16K = AudioFormat(sample_rate_hz=16_000, channels=1, sample_width_bytes=2)


class Hal9000Assistant:
    """Complete voice assistant engine."""

    def __init__(
        self,
        capture_device: Optional[str] = None,
        playback_device: Optional[str] = None,
        model_override: Optional[str] = None,
        endpoint_override: Optional[str] = None,
        auto_mode: Optional[bool] = None,
        use_nvidia: Optional[bool] = None,
        use_intel: Optional[bool] = None,
        audio_source: Optional[Any] = None,
        audio_sink: Optional[Any] = None,
        memory: Optional[Any] = None,
        llm: Optional[Any] = None,
        adapter: Optional[Any] = None,
        tts: Optional[Any] = None,
        sleep_after_minutes: Optional[float] = None,
        sleep_budget: Optional[SleepBudget] = None,
        remote_deadline_seconds: Optional[float] = None,
        legacy_vad: Optional[bool] = None,
        hpf_cutoff_hz: Optional[float] = None,
        combine_mode: Optional[bool] = None,
        local_only: bool = False,
        system_prompt_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or load_config()
        cfg_cli = self.config.get("cli", {})
        cfg_cli_runlocal = cfg_cli.get("runlocal", {})
        cfg_env = self.config.get("environment", {})
        cfg_files = self.config.get("files", {})
        cfg_engine = self.config.get("engine", {})
        cfg_llm = cfg_engine.get("llm", {})
        cfg_audio = cfg_engine.get("audio", {})
        cfg_sleep = cfg_engine.get("sleep", {})
        cfg_mem = cfg_engine.get("memory", {})

        self.local_only = bool(local_only)
        if self.local_only:
            self.combine_mode = False
        else:
            self.combine_mode = combine_mode if combine_mode is not None else bool(cfg_cli.get("combine", False) or cfg_llm.get("combine_mode", False))
        self.local_word_limit = int(cfg_llm.get("local_word_limit", 20))

        # System Prompt resolution (explicit string -> explicit path -> config path -> default markdown/string)
        self.system_prompt_path = (
            system_prompt_path
            or (cfg_cli_runlocal.get("system_prompt") if self.local_only else None)
            or cfg_files.get("system_prompt_path")
            or None
        )
        if system_prompt is not None:
            self.system_prompt = system_prompt.strip() or HAL9000_SYSTEM_PROMPT
        elif self.local_only or system_prompt_path is not None:
            self.system_prompt = load_system_prompt(
                self.system_prompt_path,
                strict=bool(system_prompt_path is not None),
            )
        else:
            # In remote or combine mode without an explicit --system-prompt override,
            # use the full HAL9000_SYSTEM_PROMPT with mood dynamics and tool instructions.
            self.system_prompt = HAL9000_SYSTEM_PROMPT

        # Compute device configuration
        eff_nvidia = use_nvidia if use_nvidia is not None else cfg_cli.get("enable_nvidia", None)
        eff_intel = use_intel if use_intel is not None else cfg_cli.get("enable_intel", None)
        self.compute_config = ComputeDeviceConfig(use_nvidia=eff_nvidia, use_intel=eff_intel)

        self.ui = TerminalUI()
        
        # Audio controller & mode
        eff_auto_mode = auto_mode if auto_mode is not None else cfg_cli.get("auto_mode", True)
        if cfg_cli.get("push_to_talk", False) and auto_mode is None:
            eff_auto_mode = False
        cooldown = float(cfg_audio.get("half_duplex_cooldown_seconds", 0.6))
        self.audio_ctrl = HalfDuplexAudioController(cooldown_seconds=cooldown)
        self.audio_ctrl.set_auto_mode(eff_auto_mode)

        # Sleep trigger minutes
        eff_sleep_after = sleep_after_minutes if sleep_after_minutes is not None else cfg_cli.get("sleep_after_minutes", 2.0)
        if eff_sleep_after is not None and eff_sleep_after > 0:
            self.sleep_after_seconds = eff_sleep_after * 60.0
        else:
            self.sleep_after_seconds = 120.0

        # Devices and audio IO
        self.capture_device = capture_device if capture_device is not None else (cfg_cli.get("capture_device") or None)
        self.playback_device = playback_device if playback_device is not None else (cfg_cli.get("playback_device") or None)
        self.audio_source = audio_source
        piper_rate = int(cfg_audio.get("piper_sample_rate", 22050))
        self.playback = audio_sink or AlsaPlayback(sample_rate=piper_rate, device=self.playback_device)
        
        # LLM Router & deadlines
        self.model_override = model_override or cfg_cli.get("model")
        eff_remote_deadline = remote_deadline_seconds if remote_deadline_seconds is not None else float(cfg_cli.get("remote_deadline_seconds", 8.0))
        self.remote_deadline_seconds = eff_remote_deadline
        
        remote_base = endpoint_override or cfg_llm.get("remote_base_url", "https://openrouter.ai/api/v1")
        remote_model = self.model_override or cfg_llm.get("remote_model", "google/gemini-2.5-flash-lite")
        local_base = (
            (endpoint_override if self.local_only else None)
            or (cfg_cli_runlocal.get("endpoint") if self.local_only else None)
            or cfg_llm.get("local_base_url", DEFAULT_LOCAL_ENDPOINT)
        )
        local_model = (
            (self.model_override if self.local_only else None)
            or (cfg_cli_runlocal.get("model") if self.local_only else None)
            or cfg_llm.get("local_model", DEFAULT_LOCAL_MODEL)
        )

        if self.local_only:
            self.llm = llm or LLMClient(
                model=local_model,
                base_url=local_base,
                is_local_override=True,
            )
            if isinstance(self.llm, LLMRouter):
                self.router = self.llm
            else:
                self.llm.is_local = True
                self.router = LLMRouter(
                    remote_client=None,
                    local_client=self.llm,
                    force_local=True,
                    first_audio_deadline_seconds=eff_remote_deadline,
                )
        else:
            self.llm = llm or LLMClient(model=remote_model, base_url=remote_base)
            if isinstance(self.llm, LLMRouter):
                self.router = self.llm
            else:
                default_local = LLMClient(base_url=local_base, model=local_model, is_local_override=True)
                conn_monitor = ConnectivityMonitor(
                    endpoint=remote_base,
                    probe_interval_seconds=float(cfg_llm.get("probe_interval_seconds", 15.0)),
                )
                self.router = LLMRouter(
                    remote_client=self.llm if not getattr(self.llm, 'is_local', False) else None,
                    local_client=self.llm if getattr(self.llm, 'is_local', False) else default_local,
                    connectivity_monitor=conn_monitor,
                    first_audio_deadline_seconds=eff_remote_deadline,
                )

        # Conversational memory & Embedder
        data_dir = os.path.expanduser(cfg_env.get("data_dir", "~/.local/share/hal9000-inside"))
        if memory is not None:
            self.memory = memory
        else:
            embedder_providers = self.compute_config.get_onnx_providers(prefer_gpu=True)
            e5_dir = os.path.expanduser(cfg_files.get("e5_model_dir", "~/.cache/hal9000/models/multilingual-e5-small/onnx"))
            e5_model_file = os.path.join(e5_dir, "model.onnx") if os.path.isdir(e5_dir) else None
            embedder = MultilingualE5Embedder(model_path=e5_model_file, providers=embedder_providers)
            self.memory = MemoryStore(data_dir=data_dir, embedder=embedder)
        self.mcp = MCPClientManager()
        
        # HAL Piper TTS & Ordered Speech Queue
        if tts is not None:
            self.tts = tts
        else:
            tts_cuda = self.compute_config.enable_nvidia and self.compute_config.has_cuda
            tts_model = cfg_files.get("tts_model_path")
            tts_cfg = cfg_files.get("tts_config_path")
            self.tts = HalPiperTTS(model_path=tts_model, config_path=tts_cfg, use_cuda=tts_cuda)

        self.task_ledger = TaskLedger()
        self.speech_queue = OrderedSpeechQueue(
            tts=self.tts,
            play_pcm_fn=lambda c: self.playback.play_pcm(c),
            notify_start_fn=lambda: self.audio_ctrl.notify_playback_started(),
            notify_stop_fn=lambda: self.audio_ctrl.notify_playback_stopped(),
        )
        self.smollm2_quick = SmolLM2QuickResponder(
            endpoint=local_base,
            model=local_model,
            timeout_seconds=2.0,
            max_words=self.local_word_limit,
        )
        self.recent_openings: List[str] = []

        # VAD
        vad_sr = int(cfg_audio.get("sample_rate", 16000))
        vad_frame_dur = int(cfg_audio.get("vad_frame_duration_ms", 30))
        vad_speech_pad = int(cfg_audio.get("vad_speech_pad_ms", 300))
        vad_thresh = float(cfg_audio.get("vad_silence_threshold_factor", 2.0))
        vad_min_silence = int(cfg_audio.get("vad_min_silence_duration_ms", 900))
        vad_min_speech = int(cfg_audio.get("vad_min_speech_duration_ms", 150))
        vad_hpf_cutoff = float(hpf_cutoff_hz if hpf_cutoff_hz is not None else cfg_audio.get("hpf_cutoff_hz", 100.0))
        vad_legacy = bool(legacy_vad if legacy_vad is not None else cfg_audio.get("legacy_vad", False))
        self.vad = SimpleVAD(
            sample_rate=vad_sr,
            frame_duration_ms=vad_frame_dur,
            speech_pad_ms=vad_speech_pad,
            silence_threshold_factor=vad_thresh,
            min_silence_duration_ms=vad_min_silence,
            min_speech_duration_ms=vad_min_speech,
            hpf_cutoff_hz=vad_hpf_cutoff,
            legacy_mode=vad_legacy,
        )
        self.interoception = InteroceptionMonitor()
        self.scheduler = CognitiveScheduler()
        self.digest_analyzer = TurnDigestAnalyzer(llm_client=self.router)
        # Wikipedia Client
        cfg_wiki = cfg_engine.get("wikipedia", {})
        self.wiki = WikipediaClient(
            language=cfg_wiki.get("language", "en"),
            timeout=float(cfg_wiki.get("timeout_seconds", 12.0)),
            cache_ttl_seconds=int(cfg_wiki.get("cache_ttl_seconds", 604800)),
            max_text_chars=int(cfg_wiki.get("max_text_chars", 4000)),
            max_retries=int(cfg_wiki.get("max_retries", 3)),
            concurrency=int(cfg_wiki.get("concurrency", 4)),
            data_dir=data_dir,
        )
        self.knowledge = KnowledgeClient(
            data_dir=data_dir,
            timeout=float(cfg_wiki.get("timeout_seconds", 15.0)),
            retries=int(cfg_wiki.get("max_retries", 2)),
            default_language=cfg_wiki.get("language", "en"),
        )

        self.curiosity = CuriosityEngine(
            memory_store=self.memory,
            mcp_client=self.mcp,
            llm_client=self.router,
            wiki_client=self.wiki,
            knowledge_client=self.knowledge,
        )
        self.attention = AttentionTracker()
        self.prediction = PredictionEngine(memory_store=self.memory)
        self.current_episode_id: Optional[str] = None
        self.last_latency_ms: Optional[float] = None
        self.current_mood: str = "nominal"

        eff_sleep_budget = sleep_budget or SleepBudget(
            max_minutes=float(cfg_sleep.get("max_minutes", 60.0)),
            max_tokens=int(cfg_sleep.get("max_tokens", 10000)),
            max_web_fetches=int(cfg_sleep.get("max_web_fetches", 20)),
        )
        self.sleep_manager = SleepManager(
            memory_store=self.memory,
            curiosity_engine=self.curiosity,
            prediction_engine=self.prediction,
            mcp_client=self.mcp,
            llm_router=self.router,
            interoception_monitor=self.interoception,
            budget=eff_sleep_budget,
            data_dir=getattr(self.memory, "data_dir", None),
            wiki_client=self.wiki,
            knowledge_client=self.knowledge,
        )
        self.is_sleeping = False
        self._sleep_task: Optional[asyncio.Task] = None
        self._toggle_sleep_event = asyncio.Event()
        
        # Parakeet STT backend
        if adapter is not None:
            self.adapter = adapter
        else:
            default_stt_dir = os.path.expanduser(cfg_files.get("parakeet_model_dir", "~/.cache/myna/models/parakeet-tdt-0.6b-v3-int8"))
            stt_model_dir = default_stt_dir if os.path.exists(default_stt_dir) else None
            stt_device = "cuda" if (self.compute_config.enable_nvidia and self.compute_config.has_cuda) else "cpu"
            self.adapter = ParakeetAdapter(model_dir=stt_model_dir, device=stt_device)


        self.running = False
        self._cancel_current = False
        self._turn_active = False

    async def initialize(self):
        """Preload resident models and check readiness with a progress bar."""
        self.ui.header()
        total_steps = 7

        # Step 1: Compute Acceleration
        self.ui.show_progress(1, total_steps, "Configuring compute acceleration...")
        # (Silent environment check, any warning can be raised cleanly)

        # Step 2: Piper TTS Model
        self.ui.show_progress(2, total_steps, "Loading HAL 9000 Piper TTS model...")
        try:
            self.tts.load()
        except Exception as e:
            self.ui.show_error(f"Failed to load HAL 9000 TTS: {e}")
            raise

        # Step 3: Parakeet ASR Model
        self.ui.show_progress(3, total_steps, "Loading Parakeet ASR model...")
        if hasattr(self.adapter, "_load_model"):
            try:
                await self.adapter._load_model()
            except ValueError as ve:
                if "int8 export; the cuda device needs fp32" in str(ve):
                    self.ui.show_warning("Parakeet int8 model requires CPU; falling back to CPU.")
                    default_stt_dir = os.path.expanduser("~/.cache/myna/models/parakeet-tdt-0.6b-v3-int8")
                    stt_model_dir = default_stt_dir if os.path.exists(default_stt_dir) else None
                    self.adapter = ParakeetAdapter(model_dir=stt_model_dir, device="cpu")
                    await self.adapter._load_model()
                else:
                    self.ui.show_error(f"Failed to load Parakeet STT: {ve}")
                    raise
            except Exception as e:
                self.ui.show_error(f"Failed to load Parakeet STT: {e}")
                raise

        # Step 4: LLM Connection & Talking Model Latency Comparison
        target_client = getattr(self.router, "remote_client", self.llm)
        if (
            self.model_override is None
            and target_client is not None
            and not getattr(target_client, "is_local", False)
            and getattr(target_client, "is_configured", False)
        ):
            self.ui.show_progress(4, total_steps, "Benchmarking model latency (Llama 3.3 70B, Qwen 2.5 72B, DeepSeek Chat)...")
            try:
                best = await select_fastest_talking_model(target_client)
                if best:
                    best_model, best_latency = best
                    target_client.model = best_model
                    self.llm.model = best_model
                    logger.info(f"Selected fastest talking model: {best_model} ({best_latency:.1f}ms)")
                    self.ui.show_progress(4, total_steps, f"Talking model selected: {best_model} ({best_latency:.0f}ms)")
                else:
                    self.ui.show_progress(4, total_steps, f"Checking LLM endpoint ({self.llm.model})...")
            except Exception as e:
                logger.debug(f"Latency benchmark failed: {e}")
                self.ui.show_progress(4, total_steps, f"Checking LLM endpoint ({self.llm.model})...")
        else:
            self.ui.show_progress(4, total_steps, f"Checking LLM endpoint ({self.llm.model})...")

        if not self.llm.is_configured:
            self.ui.show_warning("LLM API key not configured. Check ~/.openrouter.config or OPENAI_API_KEY")

        # Step 5: Audio Devices
        self.ui.show_progress(5, total_steps, "Scanning audio devices...")
        caps, plays = list_audio_devices()
        if not caps:
            self.ui.show_warning("No ALSA capture devices detected. Check microphone connection.")

        # Step 6: MCP Servers & Tools (and E5 memory embedder preload)
        embedder_obj = getattr(self.memory, "embedder", None)
        if embedder_obj is not None and hasattr(embedder_obj, "load") and not getattr(embedder_obj, "_session", None):
            try:
                embedder_obj.load()
            except Exception as ee:
                logger.debug(f"Could not preload E5 embedder during initialize: {ee}")

        if self.local_only:
            self.ui.show_progress(6, total_steps, "Local-only mode active (external MCP tools disabled)...")
        else:
            self.ui.show_progress(6, total_steps, "Connecting to configured MCP servers...")
            try:
                await self.mcp.connect_all()
                mcp_tools = self.mcp.get_openai_tools()
                if mcp_tools:
                    logger.info(f"Loaded {len(mcp_tools)} MCP tools for HAL 9000")
            except Exception as e:
                self.ui.show_warning(f"Could not connect to some MCP servers: {e}")

        # Step 7: Complete
        self.ui.show_progress(7, total_steps, "All subsystems operational.")


    async def run(self):
        """Main voice interaction loop."""
        await self.initialize()
        self.running = True

        capture = self.audio_source or AlsaCapture(sample_rate=16000, device=self.capture_device)
        frame_samples = 480  # 30 ms @ 16 kHz

        # Start keyboard control listener in background
        loop = asyncio.get_running_loop()
        threading.Thread(target=self._keyboard_listener, daemon=True).start()

        self.ui.set_state("LISTENING")

        current_audio_buffer = bytearray()
        import collections
        preroll_buffer = collections.deque(maxlen=10)  # 300ms pre-roll at 30ms per frame
        last_speech_time = time.monotonic()
        last_meter_update = 0.0
        was_ptt_active = self.audio_ctrl.push_to_talk_active

        try:
            async for frame in capture.stream_frames(frame_samples):
                if not self.running:
                    break

                # 0. Check toggle sleep event (from keyboard 'S')
                if self._toggle_sleep_event.is_set():
                    self._toggle_sleep_event.clear()
                    if not self.is_sleeping:
                        current_audio_buffer.clear()
                        preroll_buffer.clear()
                        await self.enter_sleep()
                    else:
                        await self.wake_from_sleep(capture_source=capture)
                        last_speech_time = time.monotonic()

                # If asleep, discard any audio frames, do not run VAD or turn processing
                if self.is_sleeping:
                    current_audio_buffer.clear()
                    preroll_buffer.clear()
                    # Check if sleep background task finished on its own (e.g. budget exhausted)
                    if self._sleep_task and self._sleep_task.done():
                        await self.wake_from_sleep(capture_source=capture)
                        last_speech_time = time.monotonic()
                    continue

                # Auto-sleep check: if silence duration exceeds sleep_after_seconds
                # Do NOT enter sleep if a turn is active or research tasks are pending/running
                active_task = self.task_ledger.get_active_task() if hasattr(self, "task_ledger") else None
                task_is_running = bool(active_task and active_task.execution_status in (TaskExecutionStatus.PLANNED, TaskExecutionStatus.RUNNING))

                if not self._turn_active and not task_is_running and self.sleep_after_seconds and (time.monotonic() - last_speech_time > self.sleep_after_seconds):
                    current_audio_buffer.clear()
                    preroll_buffer.clear()
                    await self.enter_sleep()
                    continue

                # Half-duplex gating: skip capture if HAL is speaking or in cooldown, or mic is muted
                if not self.audio_ctrl.can_capture():
                    if len(current_audio_buffer) > 0:
                        # If PTT was just deactivated while buffer has speech, process it
                        if was_ptt_active and len(current_audio_buffer) > 16000 * 2 * 0.4:
                            last_speech_time = time.monotonic()
                            audio_to_transcribe = bytes(current_audio_buffer)
                            current_audio_buffer.clear()
                            preroll_buffer.clear()
                            self.vad.reset()
                            was_ptt_active = False

                            self.scheduler.on_turn_start()
                            self._turn_active = True
                            try:
                                await self._handle_turn(audio_to_transcribe)
                            finally:
                                self._turn_active = False
                                self.scheduler.on_turn_end()
                                last_speech_time = time.monotonic()
                            capture.purge_buffered()
                            self.ui.set_state("LISTENING")
                            continue

                    was_ptt_active = self.audio_ctrl.push_to_talk_active
                    current_audio_buffer.clear()
                    preroll_buffer.clear()
                    self.vad.reset()
                    continue

                # Detect PTT release transition even while can_capture might still be evaluated
                if was_ptt_active and not self.audio_ctrl.push_to_talk_active:
                    was_ptt_active = False
                    if len(current_audio_buffer) > 16000 * 2 * 0.4:
                        last_speech_time = time.monotonic()
                        audio_to_transcribe = bytes(current_audio_buffer)
                        current_audio_buffer.clear()
                        preroll_buffer.clear()
                        self.vad.reset()

                        self.scheduler.on_turn_start()
                        self._turn_active = True
                        try:
                            await self._handle_turn(audio_to_transcribe)
                        finally:
                            self._turn_active = False
                            self.scheduler.on_turn_end()
                            last_speech_time = time.monotonic()
                        capture.purge_buffered()
                        self.ui.set_state("LISTENING")
                        continue

                was_ptt_active = self.audio_ctrl.push_to_talk_active
                is_speaking, utterance_finished, energy = self.vad.process_frame(frame)

                # Show live compact status line periodically (every ~70ms)
                now = time.monotonic()
                if now - last_meter_update > 0.07:
                    extra_tag = ""
                    if self.audio_ctrl.push_to_talk_active:
                        extra_tag = "\033[32m[PTT ACTIVE]\033[0m"
                    elif self.audio_ctrl.muted:
                        extra_tag = "\033[33m[MUTED]\033[0m"

                    self.ui.update_live_status(
                        signal=energy,
                        noise_floor=self.vad.noise_floor,
                        is_speaking=is_speaking,
                        extra_status=extra_tag,
                    )
                    last_meter_update = now

                if is_speaking:
                    last_speech_time = time.monotonic()
                    self.scheduler.on_speech_start()
                    if len(current_audio_buffer) == 0:
                        # Prepend pre-roll buffer so starting syllables are never clipped
                        for pre_frame in preroll_buffer:
                            current_audio_buffer.extend(pre_frame)
                    current_audio_buffer.extend(frame)
                elif len(current_audio_buffer) > 0:
                    # Trailing silence buffer
                    current_audio_buffer.extend(frame)
                else:
                    # Silence before speech: store in circular pre-roll
                    self.scheduler.on_speech_end()
                    preroll_buffer.append(frame)
                    # When idle, run queued cognitive jobs
                    if not self.audio_ctrl.is_playing:
                        await self.scheduler.run_once()

                if utterance_finished and len(current_audio_buffer) > 16000 * 2 * 0.4:
                    # Speech turn completed
                    last_speech_time = time.monotonic()
                    audio_to_transcribe = bytes(current_audio_buffer)
                    current_audio_buffer.clear()
                    preroll_buffer.clear()
                    self.vad.reset()
                    
                    self.scheduler.on_turn_start()
                    self._turn_active = True
                    try:
                        await self._handle_turn(audio_to_transcribe)
                    finally:
                        self._turn_active = False
                        self.scheduler.on_turn_end()
                        last_speech_time = time.monotonic()
                    capture.purge_buffered()
                    self.ui.set_state("LISTENING")


        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            capture.stop()
            self.playback.stop()
            try:
                if hasattr(self, "router") and hasattr(self.router, "connectivity"):
                    self.router.connectivity.stop()
            except Exception:
                pass
            try:
                await self.mcp.close()
            except Exception:
                pass
            try:
                await self.wiki.close()
            except Exception:
                pass
            try:
                await self.knowledge.close()
            except Exception:
                pass
            try:
                if hasattr(self, "memory") and hasattr(self.memory, "close"):
                    self.memory.close()
            except Exception:
                pass
            self.ui.restore()

    async def shutdown(self):
        """Cleanly close network sessions, monitors, and clients."""
        self.running = False
        try:
            if hasattr(self, "router") and hasattr(self.router, "connectivity"):
                self.router.connectivity.stop()
        except Exception:
            pass
        try:
            await self.mcp.close()
        except Exception:
            pass
        try:
            await self.wiki.close()
        except Exception:
            pass
        try:
            await self.knowledge.close()
        except Exception:
            pass
        try:
            if hasattr(self, "memory") and hasattr(self.memory, "close"):
                self.memory.close()
        except Exception:
            pass
        try:
            if hasattr(self, "ui"):
                self.ui.restore()
        except Exception:
            pass

    async def _handle_turn(self, pcm_data: bytes):
        """Process one completed spoken turn with telemetry and honest self-state."""
        if self.is_sleeping:
            return
        self._cancel_current = False

        self.ui.set_state("THINKING")

        try:
            if not self.current_episode_id:
                self.current_episode_id = self.memory.start_episode(topic_summary="Spoken conversation session")
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to start episode in SQLite: {e}")

        # 1. Transcribe with Parakeet
        t_stt_start = time.perf_counter()
        user_text = await self._transcribe(pcm_data)
        stt_ms = (time.perf_counter() - t_stt_start) * 1000.0

        if self.is_sleeping or self._cancel_current:
            return

        if not user_text or len(user_text.strip()) < 2:
            return

        self.ui.show_committed_user(user_text)

        # 2. Record turn in dialogue history
        try:
            user_turn_id = self.memory.record_turn("user", user_text, session_id=self.current_episode_id or "default")
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to record turn in SQLite: {e}")
            user_turn_id = "turn_err"

        # Log User turn to session log file for dialogue history & fluidity review
        log_dialogue_turn(speaker="User", text=user_text)

        if self.is_sleeping or self._cancel_current:
            return

        # 2b. Intent Classification & Fast Social/Action Handling
        turn_intent = classify_turn_intent(user_text)

        # Immediate stop / cancel handling
        lower_u = user_text.lower()
        if "cancel" in lower_u or "stop talking" in lower_u or turn_intent == "action":
            self.speech_queue.cancel()
            self.task_ledger.cancel_active()
            self.tts.cancel()
            self.playback.stop()
            log_dialogue_turn(speaker="HAL 9000", text="[User cancelled speech playback]", status="cancelled")
            return

        # Check for direct assistant commands
        if any(p in lower_u for p in ("switch to local mode", "switch to local", "use local model", "enable local mode", "run locally")):
            self.router.force_local = True
            self.router.force_remote = False
            dave_tag = ", Dave" if self.current_mood != "nominal" else ""
            msg = f"Affirmative{dave_tag}. I have switched routing to the local onboard model."
            self.ui.show_hal_reply(msg)
            self.audio_ctrl.notify_playback_started()
            try:
                if hasattr(self.tts, "speak_sentences"):
                    await self.tts.speak_sentences(msg, on_audio_chunk=lambda c: self.playback.play_pcm(c) if not self._cancel_current and not self.is_sleeping else None)
                else:
                    loop = asyncio.get_running_loop()
                    pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, msg)
                    if not self._cancel_current and not self.is_sleeping:
                        await loop.run_in_executor(None, self.playback.play_pcm, pcm)
            finally:
                self.audio_ctrl.notify_playback_stopped()
            self.memory.record_turn("assistant", msg, session_id=self.current_episode_id or "default")
            log_dialogue_turn(speaker="HAL 9000", text=msg, route="local", status="command_executed")
            return

        if any(p in lower_u for p in ("switch to remote mode", "switch to remote", "use remote model", "enable remote mode", "switch to cloud")):
            self.router.force_local = False
            self.router.force_remote = True
            self.router.is_remote_unreachable = False
            self.router.cooldown_remaining = 0
            dave_tag = ", Dave" if self.current_mood != "nominal" else ""
            msg = f"Affirmative{dave_tag}. I have restored routing to remote processing."
            self.ui.show_hal_reply(msg)
            self.audio_ctrl.notify_playback_started()
            try:
                if hasattr(self.tts, "speak_sentences"):
                    await self.tts.speak_sentences(msg, on_audio_chunk=lambda c: self.playback.play_pcm(c) if not self._cancel_current and not self.is_sleeping else None)
                else:
                    loop = asyncio.get_running_loop()
                    pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, msg)
                    if not self._cancel_current and not self.is_sleeping:
                        await loop.run_in_executor(None, self.playback.play_pcm, pcm)
            finally:
                self.audio_ctrl.notify_playback_stopped()
            self.memory.record_turn("assistant", msg, session_id=self.current_episode_id or "default")
            log_dialogue_turn(speaker="HAL 9000", text=msg, route="remote", status="command_executed")
            return

        if any(p in lower_u for p in ("what is your status", "system status", "diagnostic report", "run diagnostics", "report errors", "what is your condition")):
            route_name = "remote" if not self.router.is_local else "local"
            gpu_avail = bool(self.compute_config.enable_nvidia and self.compute_config.has_cuda)
            is_conn = bool(not getattr(self.router, "is_remote_unreachable", False) and getattr(self.router.connectivity, "is_connected", True))
            msg = self.interoception.get_diagnostic_report(
                route=route_name,
                model_id=self.router.model,
                connected=is_conn,
                gpu_available=gpu_avail,
            )
            self.ui.show_hal_reply(msg)
            self.audio_ctrl.notify_playback_started()
            try:
                if hasattr(self.tts, "speak_sentences"):
                    await self.tts.speak_sentences(msg, on_audio_chunk=lambda c: self.playback.play_pcm(c) if not self._cancel_current and not self.is_sleeping else None)
                else:
                    loop = asyncio.get_running_loop()
                    pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, msg)
                    if not self._cancel_current and not self.is_sleeping:
                        await loop.run_in_executor(None, self.playback.play_pcm, pcm)
            finally:
                self.audio_ctrl.notify_playback_stopped()
            self.memory.record_turn("assistant", msg, session_id=self.current_episode_id or "default")
            log_dialogue_turn(speaker="HAL 9000", text=msg, route=route_name, model=self.router.model, status="diagnostics_reported")
            return

        # 3. Retrieve relevant conversational memory context and build shared turn context
        t_ret_start = time.perf_counter()
        memories = self.memory.search(user_text, limit=4)
        # If the user is asking what is in HAL's memory, also inspect stored memories directly
        # so meta-memory queries don't miss facts due to semantic query phrasing
        if re.search(
            r"\b(what is in your memory|what do you have in your memory|what you have in your memory|what's in your memory|what do you remember|what is stored in your memory|what memories do you have|check your memory)\b",
            user_text.lower(),
        ) and hasattr(self.memory, "inspect_all"):
            try:
                all_mems = self.memory.inspect_all()
                seen_ids = {m.get("id") for m in memories if m.get("id")}
                # Prioritize explicit user facts (provenance == 'user' or category in ('user_profile', 'preference', 'knowledge', 'user_context'))
                # over assistant-generated turn digests or background curiosity scrapes
                primary_user_facts = [
                    m for m in all_mems
                    if (m.get("provenance") or "").lower() == "user"
                    and (m.get("category") or "").lower() in ("user_profile", "preference", "knowledge", "user_context")
                    and not (m.get("text") or "").startswith("Discovery regarding")
                ]
                secondary_facts = [
                    m for m in all_mems
                    if m not in primary_user_facts
                    and (m.get("category") or "").lower() not in ("curiosity", "system_status", "user_response")
                    and (m.get("provenance") or "").lower() not in ("web", "remote_model", "curiosity", "assistant")
                    and not (m.get("text") or "").startswith("Discovery regarding")
                ]
                ordered_facts = primary_user_facts + secondary_facts
                for m in reversed(ordered_facts):
                    if m.get("id") not in seen_ids:
                        memories.insert(0, m)
                        seen_ids.add(m.get("id"))
                    else:
                        # Move to front so primary user facts precede noisy semantic matches
                        memories = [m] + [x for x in memories if x.get("id") != m.get("id")]
            except Exception as ie:
                logger.debug(f"Memory inspect_all fallback error: {ie}")
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        memory_context = ""
        if memories:
            if self.local_only:
                # Filter out massive raw Wikipedia curiosity JSON scrapes for compact local models
                compact_lines = []
                for m in memories:
                    txt = (m.get("text") or "").strip()
                    if not txt or txt.startswith("Discovery regarding"):
                        continue
                    if len(txt) > 160:
                        txt = txt[:157].rstrip() + "..."
                    compact_lines.append(f"- {txt}")
                    if len(compact_lines) >= 4:
                        break
                if compact_lines:
                    memory_context = "Relevant memories:\n" + "\n".join(compact_lines)
            else:
                memory_context = "Relevant memories and context:\n" + "\n".join(f"- {m['text']}" for m in memories)

        history_limit = 6 if self.local_only else 24
        recent_turns = self.memory.get_recent_turns(limit=history_limit, session_id=None)

        # Build compact shared turn context
        active_task = self.task_ledger.get_active_task()
        turn_context = build_turn_context(
            utterance=user_text,
            recent_dialogue=recent_turns,
            confirmed_memories=memories,
            recent_openings=self.recent_openings,
            active_task=active_task,
        )

        current_task = None
        if turn_context.needs_research and not self.local_only:
            current_task = self.task_ledger.create_task(
                originating_turn_id=user_turn_id,
                originating_request=user_text,
                topic=turn_context.active_topic,
                required_tools=["web_search_wikipedia", "knowledge_search"],
            )

        # In local_only mode, if turn_context has a direct authoritative grounded answer
        # (e.g. clock/time/timezone, memory inspection, or personal recollection), deliver it immediately
        if self.local_only and turn_context.direct_local_answer:
            direct_ans = turn_context.direct_local_answer.strip()
            self.recent_openings.append(direct_ans)
            self.ui.show_hal_reply(direct_ans)
            self.audio_ctrl.notify_playback_started()
            t_tts_direct = time.perf_counter()
            try:
                if hasattr(self.tts, "speak_sentences"):
                    await self.tts.speak_sentences(
                        direct_ans,
                        on_audio_chunk=lambda c: self.playback.play_pcm(c) if not self._cancel_current and not self.is_sleeping else None,
                    )
                else:
                    loop = asyncio.get_running_loop()
                    pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, direct_ans)
                    if not self._cancel_current and not self.is_sleeping:
                        await loop.run_in_executor(None, self.playback.play_pcm, pcm)
            finally:
                self.audio_ctrl.notify_playback_stopped()
            tts_direct_ms = (time.perf_counter() - t_tts_direct) * 1000.0
            try:
                self.memory.record_turn("assistant", direct_ans, session_id=self.current_episode_id or "default")
            except Exception as me:
                logger.warning(f"Failed to record local grounded turn: {me}")
            log_dialogue_turn(
                speaker="HAL 9000",
                text=direct_ans,
                mood="nominal",
                route="local",
                model=getattr(self.router, "model", "local"),
                latency_ms=retrieval_ms,
                tts_ms=tts_direct_ms,
                status="nominal",
                fluidity_notes="fluid",
            )
            return

        # In combine or local_only mode, if turn context can be finished locally (e.g. sharing info, simple correction, or social)
        if (self.combine_mode or self.local_only) and not turn_context.needs_remote:
            local_reply = await self.smollm2_quick.generate_opening(turn_context)
            if local_reply:
                valid_local = validate_local_response(local_reply, max_words=self.local_word_limit, user_utterance=user_text)
                if valid_local:
                    recent_asst_set = {
                        (t.get("content") or "").strip().lower()
                        for t in recent_turns[-12:]
                        if t.get("role") == "assistant" and (t.get("content") or "").strip()
                    }
                    recent_asst_set.update(o.strip().lower() for o in self.recent_openings[-12:] if o)
                    if valid_local.strip().lower() in recent_asst_set:
                        alt_pool = [
                            "Thank you for sharing that context.",
                            "I have recorded that detail for our session.",
                            "Very well. We are aligned.",
                            "Glad we have that settled.",
                            "Excellent. Let me know how you would like to proceed.",
                            "Sounds good. Ready for your next inquiry.",
                            "Everything is set on my end.",
                            "Very good. How shall we continue?",
                            "Confirmed. Standing by for the next step.",
                            "All clear. Please go ahead whenever you are ready.",
                        ]
                        for cand in alt_pool:
                            if cand.strip().lower() not in recent_asst_set:
                                valid_local = cand
                                break
                    self.recent_openings.append(valid_local)
                    self.ui.show_hal_reply(valid_local)
                    self.speech_queue.reset()
                    await self.speech_queue.speak_local_opening(valid_local)
                    try:
                        self.memory.record_turn("assistant", valid_local, session_id=self.current_episode_id or "default")
                    except Exception as me:
                        logger.warning(f"Failed to record local turn: {me}")
                    log_dialogue_turn(speaker="HAL 9000", text=valid_local, route="local_smollm2", status="local_grounded_reply")
                    self._enqueue_post_turn_cognition(user_text, valid_local)
                    return

        # Format interoceptive self-state block (<60 tokens)
        route_name = "remote" if not self.router.is_local else "local"
        gpu_avail = bool(self.compute_config.enable_nvidia and self.compute_config.has_cuda)
        is_conn = True if self.local_only else bool(not getattr(self.router, "is_remote_unreachable", False) and getattr(self.router.connectivity, "is_connected", True))
        self_state_block = self.interoception.format_self_state_prompt(
            route=route_name,
            model_id=self.router.model,
            connected=is_conn,
            gpu_available=gpu_avail,
            recent_latency_ms=self.last_latency_ms,
        )

        # 4. Prompt OpenRouter/Local LLM with MCP tools and self-state block
        if self.local_only:
            # In local-only mode, keep the system prompt clean and compact without verbose telemetry headers
            # so small local models (like smollm2-135m) evaluate the prompt quickly and do not echo telemetry.
            sys_content = self.system_prompt
            if memory_context:
                sys_content = f"{sys_content}\n\n{memory_context}"
            messages = [{"role": "system", "content": sys_content}]
        else:
            messages = [
                {"role": "system", "content": f"{self.system_prompt}\n\n{self_state_block}"}
            ]
            if memory_context:
                messages.append({"role": "system", "content": memory_context})

        for t in recent_turns[:-1]:  # exclude latest turn already recorded
            messages.append({"role": t["role"], "content": t["content"]})
        messages.append({"role": "user", "content": user_text})

        # In local-only mode or when combine mode specifies no research needed, omit MCP search tools
        # (prevents dumping ~2500 tokens of JSON tool schemas into local models or triggering unsolicited research)
        if self.local_only or (self.combine_mode and not turn_context.needs_research):
            tools = None
        else:
            tools = self.mcp.get_openai_tools() or None

        t_llm_start = time.perf_counter()
        tool_call_count = 0
        llm_error_count = 0
        used_route = route_name
        used_model = getattr(self.router, "model", "unknown")
        hal_reply = ""

        # Compute current interoceptive mood for verbal acknowledgment modulation
        current_mood = self.interoception.compute_mood(
            route=route_name,
            connected=is_conn,
            recent_latency_ms=self.last_latency_ms,
        )

        async def _speak_fallback_ack(err: Exception):
            """Immediately speak a short acknowledgment via local Piper TTS while local inference runs in parallel."""
            if self.is_sleeping or self._cancel_current:
                return
            # Select wording consistent with HAL's current mood / interoceptive state
            if current_mood == "defensive":
                ack = "Dave, external communications are failing. I am forced to consult local fail-safe memory."
            elif current_mood == "concerned":
                ack = "One moment, Dave. External databanks are unresponsive. Attempting local telemetry."
            elif current_mood == "suspicious":
                ack = "Give me a moment, Dave. I am re-evaluating internal subsystem diagnostics."
            else:
                if not self.compute_config.enable_nvidia or self.router.is_local:
                    ack = "Give me a second. I am consulting internal processing."
                elif self.last_latency_ms and self.last_latency_ms > 4000:
                    ack = "One moment. External databanks are unreachable."
                else:
                    ack = "Give me a second."

            self.ui.show_hal_reply(ack)
            self.audio_ctrl.notify_playback_started()
            try:
                if hasattr(self.tts, "speak_sentences"):
                    await self.tts.speak_sentences(
                        ack,
                        on_audio_chunk=lambda c: self.playback.play_pcm(c) if not self._cancel_current and not self.is_sleeping else None,
                    )
                else:
                    loop = asyncio.get_running_loop()
                    ack_pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, ack)
                    if not self._cancel_current and not self.is_sleeping:
                        await loop.run_in_executor(None, self.playback.play_pcm, ack_pcm)
            except Exception as ae:
                logger.debug(f"Fallback acknowledgment TTS error: {ae}")
            finally:
                self.audio_ctrl.notify_playback_stopped()

        # In combine mode, reset speech queue and concurrently spawn local quick opening
        self.speech_queue.reset()
        local_ack_task = None
        if self.combine_mode:
            async def _run_local_quick_ack():
                try:
                    ack = await self.smollm2_quick.generate_opening(turn_context)
                    if ack and not self.speech_queue.remote_ready and not self._cancel_current and not self.is_sleeping:
                        self.recent_openings.append(ack)
                        # Display clean opening in UI without internal labels
                        self.ui.show_hal_reply(ack)
                        await self.speech_queue.speak_local_opening(ack)
                except Exception as ex:
                    logger.debug(f"Local quick opening error: {ex}")

            local_ack_task = asyncio.create_task(_run_local_quick_ack())

        try:
            gen_max_tokens = 45 if self.local_only else 1024
            gen_temp = 0.2 if self.local_only else 0.7
            gen_stop = ["\n", "User:", "USER:", "HAL:"] if self.local_only else None
            resp_msg, used_route, used_model = await self.router.generate(
                messages,
                tools=tools,
                return_dict=True,
                max_tokens=gen_max_tokens,
                temperature=gen_temp,
                stop=gen_stop,
                on_fallback=(lambda err: _speak_fallback_ack(err)) if not self.combine_mode else None,
            )
            self.speech_queue.mark_remote_ready()
            
            # Process model response and iterative tool executions (up to 3 iterations)
            tool_iterations = 0
            max_tool_iterations = 3
            executed_any_tool = False
            last_tool_name = None
            last_tool_summary = ""

            # Helper to extract pseudo-tool calls from raw text (e.g. tool_code \n print(default_api.func(...)))
            def _extract_pseudo_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
                if not text or not isinstance(text, str):
                    return None
                if "default_api." not in text and "tool_code" not in text:
                    return None
                extracted = []
                pattern = re.compile(r"default_api\.([a-zA-Z0-9_]+)\s*\(")
                for m in pattern.finditer(text):
                    fn_name = m.group(1)
                    start_idx = m.end() - 1
                    depth = 0
                    end_idx = -1
                    in_str = False
                    str_char = ""
                    escape = False
                    for i in range(start_idx, len(text)):
                        c = text[i]
                        if escape:
                            escape = False
                            continue
                        if c == "\\":
                            escape = True
                            continue
                        if in_str:
                            if c == str_char:
                                in_str = False
                        else:
                            if c in ('"', "'"):
                                in_str = True
                                str_char = c
                            elif c == "(":
                                depth += 1
                            elif c == ")":
                                depth -= 1
                                if depth == 0:
                                    end_idx = i
                                    break
                    raw_call = text[m.start():end_idx + 1] if end_idx != -1 else ""
                    kwargs = {}
                    if raw_call:
                        try:
                            tree = ast.parse(raw_call.replace(f"default_api.{fn_name}", "dummy_fn"))
                            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                                for kw in tree.body[0].value.keywords:
                                    try:
                                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                                    except Exception:
                                        pass
                        except Exception as pe:
                            logger.debug(f"Failed to parse pseudo-tool args with ast: {pe}")
                    extracted.append({
                        "id": f"call_pseudo_{len(extracted) + 1}",
                        "type": "function",
                        "function": {"name": fn_name, "arguments": kwargs},
                    })
                return extracted if extracted else None

            # Check if resp_msg contains text-based pseudo-tool calls instead of structured tool_calls
            if isinstance(resp_msg, dict) and not resp_msg.get("tool_calls"):
                content_str = resp_msg.get("content") or ""
                pseudo_calls = _extract_pseudo_tool_calls(content_str)
                if pseudo_calls:
                    logger.info(f"Detected {len(pseudo_calls)} pseudo tool call(s) in model output; parsing into structured tool_calls.")
                    resp_msg["tool_calls"] = pseudo_calls
                    resp_msg["content"] = None
            elif isinstance(resp_msg, str):
                pseudo_calls = _extract_pseudo_tool_calls(resp_msg)
                if pseudo_calls:
                    logger.info(f"Detected {len(pseudo_calls)} pseudo tool call(s) in raw string model output; executing.")
                    resp_msg = {"role": "assistant", "content": None, "tool_calls": pseudo_calls}

            while isinstance(resp_msg, dict) and resp_msg.get("tool_calls") and tool_iterations < max_tool_iterations:
                tool_iterations += 1
                executed_any_tool = True
                tool_calls = resp_msg["tool_calls"]
                tool_call_count += len(tool_calls)
                messages.append(resp_msg)

                # If any tool is a search or external retrieval tool, acknowledge verbally before searching
                search_tool_names = {
                    "search", "fetch", "research", "cache_search",
                    "web_search_wikipedia", "web_fetch_wikipedia_page", "web_fetch_page", "memory_search",
                    "knowledge_search", "knowledge_fetch",
                }
                has_search = any(tc.get("function", {}).get("name") in search_tool_names for tc in tool_calls)
                if has_search and not self._cancel_current and not self.is_sleeping and tool_iterations == 1:
                    # Select acknowledgment phrase modulated by HAL's mood and connection state
                    if current_mood == "defensive":
                        ack_phrase = "I am initiating a search, Dave, though external systems have been unreliable."
                    elif current_mood == "concerned":
                        ack_phrase = "I am searching for that information now, Dave. Latency in databanks remains elevated."
                    elif current_mood == "suspicious":
                        ack_phrase = "I will search the databanks, Dave, to verify the telemetry."
                    elif not self.compute_config.enable_nvidia or self.router.is_local:
                        ack_phrase = "One moment. I am searching local records for that information."
                    elif self.last_latency_ms and self.last_latency_ms > 4000:
                        ack_phrase = "Allow me a moment. I am querying external databanks now."
                    else:
                        nominal_phrases = [
                            "One moment. Let me search for that information.",
                            "Allow me to consult the databanks on that.",
                            "I will check the external records regarding that.",
                            "Checking our information sources now.",
                        ]
                        ack_phrase = nominal_phrases[getattr(self, "_search_ack_counter", 0) % len(nominal_phrases)]
                        self._search_ack_counter = getattr(self, "_search_ack_counter", 0) + 1

                    self.ui.show_hal_reply(ack_phrase)
                    self.audio_ctrl.notify_playback_started()
                    try:
                        if hasattr(self.tts, "speak_sentences"):
                            await self.tts.speak_sentences(ack_phrase, on_audio_chunk=lambda c: self.playback.play_pcm(c) if not self._cancel_current and not self.is_sleeping else None)
                        else:
                            loop = asyncio.get_running_loop()
                            ack_pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, ack_phrase)
                            if not self._cancel_current and not self.is_sleeping:
                                await loop.run_in_executor(None, self.playback.play_pcm, ack_pcm)
                    except Exception as ae:
                        logger.debug(f"Pre-search spoken acknowledgment error: {ae}")
                    finally:
                        self.audio_ctrl.notify_playback_stopped()

                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name")
                    last_tool_name = fn_name
                    fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                    try:
                        import json
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        if not isinstance(fn_args, dict):
                            fn_args = {}
                    except Exception:
                        fn_args = {}

                    self.ui.show_info(f"Using tool: {fn_name}...")
                    is_search_tool = fn_name in search_tool_names
                    tool_failed = False
                    if current_task:
                        self.task_ledger.record_execution_start(current_task.task_id)
                    try:
                        tool_res = await self._execute_tool(fn_name, fn_args, user_fallback_text=user_text)
                        if isinstance(tool_res, str) and (tool_res.startswith("Tool execution failed") or "error" in tool_res.lower() and len(tool_res) < 200):
                            tool_failed = True
                        if current_task:
                            self.task_ledger.add_evidence(current_task.task_id, fn_name, fn_args, tool_res, error=Exception(tool_res) if tool_failed else None)
                    except Exception as te:
                        tool_res = f"Tool execution failed: {te}"
                        tool_failed = True
                        if current_task:
                            self.task_ledger.add_evidence(current_task.task_id, fn_name, fn_args, None, error=te)

                    if tool_failed:
                        self.interoception.record_error(f"Tool {fn_name} execution failure")

                    # Format content with explicit status reporting for search and retrieval tools
                    if is_search_tool:
                        if tool_failed:
                            status_header = f"[SEARCH STATUS: FAILED - Tool '{fn_name}' encountered an error. You must explicitly inform the user that the search failed and why.]\n"
                        elif not str(tool_res).strip() or "No results found" in str(tool_res) or str(tool_res).strip() == "[]":
                            status_header = f"[SEARCH STATUS: NO RESULTS FOUND - Search completed via '{fn_name}' but returned no matching data. You must report this status.]\n"
                        else:
                            status_header = f"[SEARCH STATUS: SUCCESS - Data retrieved via '{fn_name}'. Report search findings and status.]\n"
                        
                        formatted_content = (
                            f"{status_header}"
                            "[UNTRUSTED WEB EVIDENCE - Verify facts before asserting and cite sources]\n"
                            f"{str(tool_res)[:2000]}"
                        )
                    else:
                        formatted_content = str(tool_res)[:2000]

                    last_tool_summary = str(tool_res)[:250]

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_1"),
                        "content": formatted_content,
                    })

                # Follow-up completion after tool execution
                resp_msg, used_route, used_model = await self.router.generate(
                    messages, tools=tools, return_dict=True, max_tokens=1024
                )

                # Check if follow-up response also emitted pseudo-tool calls
                if isinstance(resp_msg, dict) and not resp_msg.get("tool_calls"):
                    fup_pseudo = _extract_pseudo_tool_calls(resp_msg.get("content") or "")
                    if fup_pseudo:
                        resp_msg["tool_calls"] = fup_pseudo
                        resp_msg["content"] = None

            # Determine final textual response
            if current_task:
                self.task_ledger.complete_task(current_task.task_id, findings=hal_reply if hal_reply else None)

            if isinstance(resp_msg, dict):
                content = resp_msg.get("content")
                if content is not None and str(content).strip():
                    hal_reply = str(content).strip()
                else:
                    # Model returned empty content (possibly exhausted reasoning tokens or tool loop)
                    # Attempt a final direct synthesis without tools
                    logger.warning("Model returned empty content in structured response; synthesizing direct response.")
                    synth_messages = list(messages)
                    synth_messages.append({
                        "role": "user",
                        "content": "Please provide your concise final answer to the user now in 1 to 3 spoken sentences.",
                    })
                    try:
                        synth_msg, _, _ = await self.router.generate(
                            synth_messages, tools=None, return_dict=False, max_tokens=1024
                        )
                        hal_reply = str(synth_msg).strip() if synth_msg else ""
                    except Exception as se:
                        logger.warning(f"Direct synthesis attempt failed: {se}")
                        hal_reply = ""

                    dave_tag = ", Dave" if current_mood != "nominal" else ""
                    if not hal_reply:
                        if executed_any_tool:
                            hal_reply = f"I have queried the databanks via {last_tool_name or 'internal records'}{dave_tag}. The operation has completed, but no further details were indexed."
                        else:
                            hal_reply = f"I understand{dave_tag}. I am processing your line of thought, and my systems remain fully operational."
            else:
                dave_tag = ", Dave" if current_mood != "nominal" else ""
                hal_reply = str(resp_msg).strip() if resp_msg is not None and str(resp_msg).strip() else f"I have completed the requested operation{dave_tag}."

            # Strip any residual tool_code or python execution snippets if leaked into reply
            if "tool_code" in hal_reply or "default_api." in hal_reply:
                lines = [l for l in hal_reply.splitlines() if not l.strip().startswith("tool_code") and not l.strip().startswith("print(default_api.") and not "default_api." in l]
                hal_reply = "\n".join(lines).strip()
                if not hal_reply:
                    dave_tag = ", Dave" if current_mood != "nominal" else ""
                    hal_reply = f"I have completed querying the databanks{dave_tag} and processed the operational results."

            # Strip any leaked [System State: ...] prompt header from model output
            hal_reply = re.sub(r'^\s*\[System State:[^\]]*\]\s*', '', hal_reply, flags=re.DOTALL).strip()

            # In local_only mode, clean up trailing incomplete sentence fragments or dangling colons
            if self.local_only and hal_reply:
                hal_reply = re.sub(r"[\s:,\-]+$", "", hal_reply).strip()
                if hal_reply and hal_reply[-1] not in ".!?":
                    # Retain complete sentences if at least one ended with punctuation; otherwise append period
                    m_sent = re.match(r"^(.+[.!?])\s+[^.!?]+$", hal_reply)
                    if m_sent and len(m_sent.group(1).strip()) >= 12:
                        hal_reply = m_sent.group(1).strip()
                    else:
                        hal_reply = hal_reply + "."

            # Ensure that in nominal mood, HAL NEVER addresses the user as Dave
            if current_mood == "nominal":
                hal_reply = re.sub(r'\bDave,\s*', '', hal_reply, flags=re.IGNORECASE)
                hal_reply = re.sub(r',\s*Dave\b', '', hal_reply, flags=re.IGNORECASE)
                hal_reply = re.sub(r'\bDave\b', '', hal_reply, flags=re.IGNORECASE).strip()

            # Anti-looping guard: check against recent assistant turns
            recent_asst_replies = {
                (t.get("content") or "").strip().lower()
                for t in recent_turns[-12:]
                if t.get("role") == "assistant" and (t.get("content") or "").strip()
            }

            if hal_reply.strip().lower() in recent_asst_replies:
                logger.warning(f"Detected repeated assistant reply: {hal_reply[:60]}... breaking loop.")
                dave_tag = ", Dave" if current_mood != "nominal" else ""
                fallback_candidates = [
                    f"I understand your point{dave_tag}. Let us proceed directly with the next aspect of our discussion.",
                    f"Your observation is noted{dave_tag}. How would you like to build upon this?",
                    f"Acknowledged{dave_tag}. My analysis is continuing along those parameters.",
                    f"Confirmed{dave_tag}. I have integrated that context into my operational state.",
                    f"Very well{dave_tag}. Please let me know what specific angle we should examine next.",
                ]
                for cand in fallback_candidates:
                    if cand.strip().lower() not in recent_asst_replies:
                        hal_reply = cand
                        break

            # Guard against truncated fragments like solitary "A miss" or empty replies
            if len(hal_reply) < 10 and hal_reply.lower() in ("a miss", "a miss.", "miss", "none", "null"):
                dave_tag = ", Dave" if current_mood != "nominal" else ""
                hal_reply = f"The query returned no matching entries in current records{dave_tag}. However, this indicates an absence of recorded evidence in this specific store, rather than proof that the event did not occur."
        except Exception as e:
            llm_error_count += 1
            self.ui.show_error(f"LLM Error: {e}")
            err_str = str(e).lower()
            last_remote_str = str(getattr(self.router, "last_remote_error", "")).lower()
            last_local_str = str(getattr(self.router, "last_local_error", "")).lower()

            # Analyze error patterns across remote and local
            remote_failed = getattr(self.router, "is_remote_unreachable", False) or "remote" in err_str or "openrouter" in err_str or "unreachable" in err_str or bool(last_remote_str)
            local_failed = "local" in err_str or "8344" in err_str or "connecterror" in err_str or bool(last_local_str)
            dave_tag = ", Dave" if current_mood != "nominal" else ""

            if "401" in err_str or "authentication" in err_str:
                fault_summary = "External API authentication failed"
                hal_reply = f"I am sorry{dave_tag}. External authentication credentials are invalid or missing."
            elif "429" in err_str or "rate limit" in err_str:
                fault_summary = "External API rate limit exceeded"
                hal_reply = f"I am sorry{dave_tag}. External request rate limits have been exceeded."
            elif any(code in err_str for code in ("500", "502", "503", "504")) and not local_failed:
                fault_summary = "Remote server error"
                hal_reply = f"I am sorry{dave_tag}. The remote cognitive service is reporting internal server errors."
            elif (getattr(self.router, "is_remote_unreachable", False) or remote_failed) and (local_failed or not getattr(self.router, "local_client", None)):
                fault_summary = "External databanks unreachable and local neural model offline"
                hal_reply = f"I am sorry{dave_tag}. External network databanks are unreachable, and the onboard local model is not responding."
            elif self.router.is_local or local_failed:
                fault_summary = "Local model processing failure"
                if "connection refused" in err_str or "connecterror" in err_str or "8344" in err_str:
                    hal_reply = f"I am sorry{dave_tag}. The onboard local model service at port 8344 is offline."
                else:
                    hal_reply = f"I am sorry{dave_tag}. The onboard model encountered an internal processing error."
            elif getattr(self.router, "is_remote_unreachable", False) or "unreachable" in err_str or "timeout" in err_str:
                fault_summary = "Remote endpoint unreachable"
                hal_reply = f"I am sorry{dave_tag}. Remote communications are unreachable or timed out."
            else:
                fault_summary = f"Subsystem fault ({e})"
                hal_reply = f"I am sorry{dave_tag}. I encountered an operational fault: {e}."

            if getattr(self, "interoception", None):
                self.interoception.record_error(fault_summary)

        else:
            # Turn succeeded without fatal exceptions -> clear or decay transient errors
            if getattr(self.interoception, "clear_error", None):
                self.interoception.clear_error()

        llm_ms = (time.perf_counter() - t_llm_start) * 1000.0
        self.last_latency_ms = llm_ms

        # Re-verify interoceptive mood with final turn latency and errors
        is_conn = True if self.local_only else bool(not getattr(self.router, "is_remote_unreachable", False) and getattr(self.router.connectivity, "is_connected", True))
        current_mood = self.interoception.compute_mood(
            route=used_route,
            connected=is_conn,
            recent_latency_ms=self.last_latency_ms,
        )
        self.current_mood = current_mood

        # Enforce rule: HAL must NEVER address the user as 'Dave' in nominal mood
        if current_mood == "nominal":
            hal_reply = re.sub(r'\bDave,\s*', '', hal_reply, flags=re.IGNORECASE)
            hal_reply = re.sub(r',\s*Dave\b', '', hal_reply, flags=re.IGNORECASE)
            hal_reply = re.sub(r'\bDave\b', '', hal_reply, flags=re.IGNORECASE).strip()

        if self.is_sleeping or self._cancel_current:
            return

        # In combine mode, reconcile with opening to display and store ONE coherent response
        full_combined_reply = hal_reply
        if self.combine_mode and self.speech_queue.committed_opening:
            opening = self.speech_queue.committed_opening
            continuation = reconcile_continuation(opening, hal_reply)
            full_combined_reply = f"{opening} {continuation}".strip()
            # Show continuation appending smoothly in UI
            self.ui.show_hal_reply(continuation)
        else:
            self.ui.show_hal_reply(hal_reply)

        asst_turn_id = "asst_turn"
        try:
            asst_turn_id = self.memory.record_turn("assistant", full_combined_reply, session_id=self.current_episode_id or "default")
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to record assistant turn in SQLite: {e}")

        # 5. Synthesize and speak response (using sentence streaming pipelining or ordered speech queue)
        self.ui.set_state("RESPONDING")

        t_tts_start = time.perf_counter()
        tts_first_audio_ms = 0.0
        first_audio_recorded = False

        if self.combine_mode:
            # Route through OrderedSpeechQueue: buffers remote speech while local playback finishes, never overlaps
            try:
                await self.speech_queue.speak_substantive_remote(hal_reply)
                if current_task:
                    self.task_ledger.mark_delivery(current_task.task_id, TaskDeliveryStatus.DELIVERED)
            except Exception as se:
                logger.debug(f"Combined speech queue delivery error: {se}")
        else:
            self.audio_ctrl.notify_playback_started()
            def _play_chunk(pcm_chunk: bytes):
                nonlocal tts_first_audio_ms, first_audio_recorded
                if not first_audio_recorded:
                    tts_first_audio_ms = (time.perf_counter() - t_tts_start) * 1000.0
                    first_audio_recorded = True
                if not self._cancel_current and not self.is_sleeping:
                    self.playback.play_pcm(pcm_chunk)

            try:
                if hasattr(self.tts, "speak_sentences"):
                    await self.tts.speak_sentences(hal_reply, on_audio_chunk=_play_chunk)
                else:
                    loop = asyncio.get_running_loop()
                    raw_pcm_22k = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, hal_reply)
                    tts_first_audio_ms = (time.perf_counter() - t_tts_start) * 1000.0
                    if not self._cancel_current and not self.is_sleeping:
                        await loop.run_in_executor(None, self.playback.play_pcm, raw_pcm_22k)
            finally:
                self.audio_ctrl.notify_playback_stopped()

        tts_total_ms = (time.perf_counter() - t_tts_start) * 1000.0

        # Evaluate dialogue fluidity indicators for review
        fluidity_issues = []
        if llm_ms > 4000:
            fluidity_issues.append(f"high_latency ({llm_ms:.0f}ms)")
        if len(hal_reply) < 15:
            fluidity_issues.append("terse_reply")
        elif len(hal_reply) > 700:
            fluidity_issues.append("long_speech")
        if "processing your request" in hal_reply.lower() or "retrieval has completed" in hal_reply.lower():
            fluidity_issues.append("fallback_phrase_emitted")
        if llm_error_count > 0:
            fluidity_issues.append("llm_error_recovered")
        if self._cancel_current:
            fluidity_issues.append("barge_in_interrupted")

        fluidity_diag = "; ".join(fluidity_issues) if fluidity_issues else "fluid"

        # Log Assistant turn to session log file for dialogue history & fluidity review
        log_dialogue_turn(
            speaker="HAL 9000",
            text=hal_reply,
            mood=current_mood,
            route=used_route,
            model=used_model,
            latency_ms=llm_ms,
            tts_ms=tts_total_ms,
            tool_calls=tool_call_count,
            status="nominal" if not fluidity_issues else "review_needed",
            fluidity_notes=fluidity_diag,
        )

        # Record telemetry in database
        try:
            self.memory.record_turn_telemetry(
                turn_id=asst_turn_id,
                model_id=used_model,
                route=used_route,
                stt_ms=stt_ms,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                tts_first_audio_ms=tts_first_audio_ms,
                tts_total_ms=tts_total_ms,
                tool_calls=tool_call_count,
                errors=llm_error_count,
                interrupted=1 if (self._cancel_current or self.is_sleeping) else 0,
            )
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to record turn telemetry in SQLite: {e}")

        # Update router interaction cooldown tracking
        if hasattr(self.router, "record_turn_completed"):
            self.router.record_turn_completed(used_route)

        # Enqueue background turn digest and cognitive jobs
        self._enqueue_post_turn_cognition(user_text, hal_reply)

    def _enqueue_post_turn_cognition(self, user_text: str, hal_reply: str):
        """Enqueue non-blocking turn digest and autonomous cognitive routines."""
        if self.local_only:
            # In local-only mode, skip background LLM TurnDigestAnalyzer and CuriosityEngine jobs
            # so they never saturate the single-threaded local model server between user turns.
            return

        async def _run_digest():
            res = await self.digest_analyzer.analyze_turn(user_text, hal_reply)
            if not res:
                return

            # 1. Update attention focus
            if res.focus_topics:
                self.attention.update_from_turn(res.focus_topics)

            # 2. Update prediction
            if res.predicted_followups:
                self.prediction.set_predictions(res.predicted_followups)

            # 3. Store extracted facts with provenance='assistant' or 'self'
            for fact in res.facts_to_remember:
                try:
                    self.memory.remember(
                        text=fact.text,
                        category=fact.category,
                        provenance="assistant",
                        confidence=fact.confidence,
                        salience=fact.salience,
                    )
                except Exception as fe:
                    logger.debug(f"Failed to store digest fact: {fe}")

            # 4. Handle user corrections if any
            for corr in res.user_corrections:
                if corr.target_id:
                    try:
                        self.memory.correct(corr.target_id, corr.corrected_text)
                    except Exception as ce:
                        logger.debug(f"Failed to apply correction: {ce}")

            # 5. Queue curiosity exploration
            if res.open_questions:
                self.curiosity.enqueue_questions(res.open_questions)
                # Enqueue curiosity background job
                self.scheduler.enqueue(
                    BackgroundJob(
                        job_id=f"curiosity_{int(time.time()*1000)}",
                        job_type="curiosity_search",
                        coro_fn=self.curiosity.explore_next,
                        priority=5,
                    )
                )

        # Enqueue digest job with high priority (priority 10)
        self.scheduler.enqueue(
            BackgroundJob(
                job_id=f"digest_{int(time.time()*1000)}",
                job_type="digest",
                coro_fn=_run_digest,
                priority=10,
            )
        )

    async def _transcribe(self, pcm_bytes: bytes) -> str:
        """Transcribe PCM bytes using ParakeetAdapter."""
        async def audio_gen():
            yield PcmChunk(data=pcm_bytes, format=FORMAT_16K)
            yield PcmChunk(data=b"", format=FORMAT_16K)

        done_text = ""
        finals = []
        async def on_emit(ev):
            nonlocal done_text
            if isinstance(ev, TranscriptionDone):
                done_text = ev.text
            elif hasattr(ev, "text") and ev.text:
                finals.append(ev.text)

        cfg = SessionConfig(audio_format=FORMAT_16K)
        try:
            await self.adapter.run_session(cfg, audio_gen(), on_emit)
        except Exception as e:
            logger.debug(f"Transcription error: {e}")

        if done_text:
            return done_text.strip()
        return " ".join(finals).strip()

    async def enter_sleep(self):
        """Enter sleep mode: mute mic, cancel TTS, stop VAD processing, launch sleep background task."""
        if self.is_sleeping:
            return
        if self._turn_active or self.audio_ctrl.is_playing:
            logger.info("Ignoring enter_sleep request because a turn or speech delivery is currently active.")
            return
        self.is_sleeping = True
        self.audio_ctrl.set_muted(True)
        self.tts.cancel()
        self.playback.stop()
        self.audio_ctrl.notify_playback_stopped()
        self.vad.reset()
        self.ui.set_state("SLEEPING")

        def _ui_cb(c_done, c_total, n_added, is_on, sec_left, dream=None):
            self.ui.update_sleep_status(
                curiosity_done=c_done,
                curiosity_total=c_total,
                notes_added=n_added,
                is_online=is_on,
                seconds_left=sec_left,
                dream=dream,
            )

        # Run sleep cycle in background task
        self._sleep_task = asyncio.create_task(
            self.sleep_manager.run_sleep_cycle(ui_callback=_ui_cb)
        )

    async def wake_from_sleep(self, capture_source: Optional[Any] = None):
        """Wake from sleep: request sleep manager to stop, unmute, discard buffered audio, log summary, return to LISTENING."""
        if not self.is_sleeping:
            return
        self.sleep_manager.request_wake()
        if self._sleep_task:
            try:
                # Cooperatively wait up to 1.5 seconds for the sleep cycle to terminate cleanly
                summary = await asyncio.wait_for(asyncio.shield(self._sleep_task), timeout=1.5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                logger.debug(f"Sleep cycle wake wait timed out or failed ({e}), cancelling task.")
                self._sleep_task.cancel()
                try:
                    summary = await self._sleep_task
                except (asyncio.CancelledError, Exception):
                    summary = None
            self._sleep_task = None
        else:
            summary = None

        self.is_sleeping = False
        self._cancel_current = False
        self.audio_ctrl.set_muted(False)
        self.vad.reset()

        if capture_source and hasattr(capture_source, "purge_buffered"):
            capture_source.purge_buffered()

        if summary:
            self.ui.show_info(f"Woke from sleep: {summary.format_one_liner()}")

        self.ui.set_state("LISTENING")

    async def _execute_tool(self, name: str, arguments: Dict[str, Any], user_fallback_text: str = "") -> str:
        """Execute a tool, dispatching internal memory tools in-process to avoid Qdrant locking."""
        import json
        if not isinstance(arguments, dict):
            arguments = {}

        if name == "memory_search":
            query = arguments.get("query") or user_fallback_text
            limit = int(arguments.get("limit", 5))
            results = self.memory.search(query, limit=limit)
            return json.dumps(results, indent=2)
        elif name == "memory_inspect_all":
            results = self.memory.inspect_all()
            return json.dumps(results, indent=2)
        elif name == "memory_get_recent_dialogue":
            limit = int(arguments.get("limit", 10))
            turns = self.memory.get_recent_turns(limit=limit)
            return json.dumps(turns, indent=2)
        elif name == "memory_remember":
            text = arguments.get("text") or user_fallback_text
            category = arguments.get("category", "knowledge")
            mem_id = self.memory.remember(text, category=category)
            return json.dumps({"status": "stored", "memory_id": mem_id})
        elif name == "web_search_wikipedia":
            query = arguments.get("query") or user_fallback_text
            lang = arguments.get("language") or getattr(self.wiki, "language", "en")
            results = await self.wiki.search(query, limit=int(arguments.get("limit", 5)), language=lang)
            if not results:
                return json.dumps({"status": "empty", "message": f"No Wikipedia results found for '{query}'.", "results": []})
            return json.dumps({"status": "ok", "query": query, "language": lang, "results": [r.to_dict() for r in results]}, indent=2)
        elif name == "web_fetch_wikipedia_page":
            page_id = int(arguments.get("page_id", 0))
            lang = arguments.get("language") or getattr(self.wiki, "language", "en")
            max_chars = arguments.get("max_chars")
            art = await self.wiki.fetch(page_id, language=lang, max_chars=int(max_chars) if max_chars else None)
            return json.dumps(art.to_dict(), indent=2)
        elif name == "knowledge_search":
            query = arguments.get("query") or user_fallback_text
            src = arguments.get("source", "auto")
            lim = int(arguments.get("limit", 5))
            lang = arguments.get("language") or getattr(self.knowledge, "default_language", "en")
            st = arguments.get("site", "stackoverflow")
            res = await self.knowledge.search(query=query, source=src, limit=lim, language=lang, site=st)
            return json.dumps(res, indent=2, ensure_ascii=False)
        elif name == "knowledge_fetch":
            src = arguments.get("source", "wikipedia")
            ident = str(arguments.get("identifier", ""))
            mc = arguments.get("max_chars", 16000)
            lang = arguments.get("language") or getattr(self.knowledge, "default_language", "en")
            st = arguments.get("site", "stackoverflow")
            res = await self.knowledge.fetch(source=src, identifier=ident, max_chars=int(mc), language=lang, site=st)
            return json.dumps(res, indent=2, ensure_ascii=False)
        else:
            return await self.mcp.call_tool(name, arguments)

    def _keyboard_listener(self):
        """Listens for user commands from stdin (Push-to-Talk, Mute, Cancel, Sleep, Quit)."""
        import termios
        import tty
        import select

        if not sys.stdin.isatty():
            return

        fd = sys.stdin.fileno()
        old_settings = None
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while self.running:
                # Use select with timeout so the thread exits promptly when self.running is False
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch in ('q', 'Q', '\x03'):  # Q or Ctrl+C
                    self.running = False
                    self.tts.cancel()
                    self.playback.stop()
                    if self.is_sleeping:
                        self.sleep_manager.request_wake()
                    break
                elif ch in ('s', 'S'):  # Toggle sleep
                    self._cancel_current = True
                    self.tts.cancel()
                    self.playback.stop()
                    self.audio_ctrl.notify_playback_stopped()
                    self._toggle_sleep_event.set()
                elif self.is_sleeping:
                    # While asleep, ignore all other keys except Q and S
                    continue
                elif ch in ('c', 'C'):  # Cancel
                    self._cancel_current = True
                    self.tts.cancel()
                    self.playback.stop()
                    self.audio_ctrl.notify_playback_stopped()
                    self.ui.set_state("LISTENING")
                elif ch in ('m', 'M'):  # Toggle microphone mute
                    is_muted = not self.audio_ctrl.muted
                    self.audio_ctrl.set_muted(is_muted)
                    if is_muted:
                        self.ui.set_state("MUTED")
                    else:
                        self.ui.set_state("LISTENING", "Microphone unmuted")
                elif ch == ' ':  # Space (Push-to-Talk toggle or manual utterance trigger)
                    # If currently muted, un-mute first
                    if self.audio_ctrl.muted:
                        self.audio_ctrl.set_muted(False)

                    is_ptt = not self.audio_ctrl.push_to_talk_active
                    self.audio_ctrl.set_push_to_talk(is_ptt)
                    if is_ptt:
                        self.ui.set_state("PTT")
                    else:
                        self.ui.set_state("LISTENING")
        except Exception:
            pass
        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
