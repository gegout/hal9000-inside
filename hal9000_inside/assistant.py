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
import collections
import logging
import os
import sys
import threading
import time
from typing import Optional

from hal9000_inside.audio import (
    AlsaCapture,
    AlsaPlayback,
    HalfDuplexAudioController,
    SimpleVAD,
    list_audio_devices,
)
from hal9000_inside.hardware import ComputeDeviceConfig, detect_system_gpus
from hal9000_inside.llm import LLMClient, LLMRouter, HAL9000_SYSTEM_PROMPT
from hal9000_inside.memory import MemoryStore, MultilingualE5Embedder
from hal9000_inside.mcp import MCPClientManager
from hal9000_inside.cognition.interoception import InteroceptionMonitor
from hal9000_inside.tts import HalPiperTTS
from hal9000_inside.ui import TerminalUI
from myna.core import AudioFormat, PcmChunk, SessionConfig, TranscriptionDone
from myna.testbed.parakeet import ParakeetAdapter

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
        auto_mode: bool = True,
        use_nvidia: Optional[bool] = None,
        use_intel: Optional[bool] = None,
        audio_source: Optional[Any] = None,
        audio_sink: Optional[Any] = None,
        memory: Optional[Any] = None,
        llm: Optional[Any] = None,
        adapter: Optional[Any] = None,
        tts: Optional[Any] = None,
    ):
        self.compute_config = ComputeDeviceConfig(use_nvidia=use_nvidia, use_intel=use_intel)
        self.ui = TerminalUI()
        self.audio_ctrl = HalfDuplexAudioController(cooldown_seconds=0.6)
        self.audio_ctrl.set_auto_mode(auto_mode)
        
        self.capture_device = capture_device
        self.audio_source = audio_source
        self.playback = audio_sink or AlsaPlayback(sample_rate=22050)
        
        self.llm = llm or LLMClient(model=model_override, base_url=endpoint_override)
        if isinstance(self.llm, LLMRouter):
            self.router = self.llm
        else:
            self.router = LLMRouter(remote_client=self.llm if not getattr(self.llm, 'is_local', False) else None,
                                    local_client=self.llm if getattr(self.llm, 'is_local', False) else None)
        if memory is not None:
            self.memory = memory
        else:
            embedder_providers = self.compute_config.get_onnx_providers(prefer_gpu=True)
            embedder = MultilingualE5Embedder(providers=embedder_providers)
            self.memory = MemoryStore(embedder=embedder)
        self.mcp = MCPClientManager()
        
        # HAL Piper TTS with CUDA only if onnxruntime has CUDA provider available
        if tts is not None:
            self.tts = tts
        else:
            tts_cuda = self.compute_config.enable_nvidia and self.compute_config.has_cuda
            self.tts = HalPiperTTS(use_cuda=tts_cuda)

        self.vad = SimpleVAD(sample_rate=16000, frame_duration_ms=30, min_silence_duration_ms=800)
        self.interoception = InteroceptionMonitor()
        self.current_episode_id: Optional[str] = None
        self.last_latency_ms: Optional[float] = None
        
        # Parakeet STT backend
        if adapter is not None:
            self.adapter = adapter
        else:
            stt_device = "cuda" if (self.compute_config.enable_nvidia and self.compute_config.has_cuda) else "cpu"
            self.adapter = ParakeetAdapter(device=stt_device)

        self.running = False
        self._cancel_current = False

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
                    self.adapter = ParakeetAdapter(device="cpu")
                    await self.adapter._load_model()
                else:
                    self.ui.show_error(f"Failed to load Parakeet STT: {ve}")
                    raise
            except Exception as e:
                self.ui.show_error(f"Failed to load Parakeet STT: {e}")
                raise

        # Step 4: LLM Connection
        self.ui.show_progress(4, total_steps, f"Checking LLM endpoint ({self.llm.model})...")
        if not self.llm.is_configured:
            self.ui.show_warning("LLM API key not configured. Check ~/.openrouter.config or OPENAI_API_KEY")

        # Step 5: Audio Devices
        self.ui.show_progress(5, total_steps, "Scanning audio devices...")
        caps, plays = list_audio_devices()
        if not caps:
            self.ui.show_warning("No ALSA capture devices detected. Check microphone connection.")

        # Step 6: MCP Servers & Tools
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
        last_meter_update = 0.0

        try:
            async for frame in capture.stream_frames(frame_samples):
                if not self.running:
                    break

                # Half-duplex gating: skip capture if HAL is speaking or in cooldown
                if not self.audio_ctrl.can_capture():
                    current_audio_buffer.clear()
                    preroll_buffer.clear()
                    self.vad.reset()
                    continue

                is_speaking, utterance_finished, energy = self.vad.process_frame(frame)

                # Show live compact status line periodically (every ~70ms)
                now = time.monotonic()
                if now - last_meter_update > 0.07:
                    self.ui.update_live_status(
                        signal=energy,
                        noise_floor=self.vad.noise_floor,
                        is_speaking=is_speaking,
                    )
                    last_meter_update = now

                if is_speaking:
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
                    preroll_buffer.append(frame)

                if utterance_finished and len(current_audio_buffer) > 16000 * 2 * 0.4:
                    # Speech turn completed
                    audio_to_transcribe = bytes(current_audio_buffer)
                    current_audio_buffer.clear()
                    preroll_buffer.clear()
                    self.vad.reset()
                    
                    await self._handle_turn(audio_to_transcribe)
                    capture.purge_buffered()
                    self.ui.set_state("LISTENING")


        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            capture.stop()
            self.playback.stop()
            try:
                await self.mcp.close()
            except Exception:
                pass
            self.ui.restore()

    async def _handle_turn(self, pcm_data: bytes):
        """Process one completed spoken turn with telemetry and honest self-state."""
        self.ui.set_state("THINKING")
        self._cancel_current = False

        if not self.current_episode_id:
            self.current_episode_id = self.memory.start_episode(topic_summary="Spoken conversation session")

        # 1. Transcribe with Parakeet
        t_stt_start = time.perf_counter()
        user_text = await self._transcribe(pcm_data)
        stt_ms = (time.perf_counter() - t_stt_start) * 1000.0

        if not user_text or len(user_text.strip()) < 2:
            return

        self.ui.show_committed_user(user_text)

        # 2. Record turn in dialogue history
        user_turn_id = self.memory.record_turn("user", user_text, session_id=self.current_episode_id)

        # Check for direct assistant commands
        lower_u = user_text.lower()
        if "cancel" in lower_u or "stop talking" in lower_u:
            self.tts.cancel()
            self.playback.stop()
            return

        # 3. Retrieve relevant conversational memory context
        t_ret_start = time.perf_counter()
        memories = self.memory.search(user_text, limit=4)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        memory_context = ""
        if memories:
            memory_context = "Relevant memories and context:\n" + "\n".join(f"- {m['text']}" for m in memories)

        recent_turns = self.memory.get_recent_turns(limit=6, session_id=self.current_episode_id)

        # Format interoceptive self-state block (<60 tokens)
        route_name = "remote" if not self.llm.is_local else "local"
        gpu_avail = bool(self.compute_config.enable_nvidia and self.compute_config.has_cuda)
        self_state_block = self.interoception.format_self_state_prompt(
            route=route_name,
            model_id=self.llm.model,
            connected=True,
            gpu_available=gpu_avail,
            recent_latency_ms=self.last_latency_ms,
        )

        # 4. Prompt OpenRouter/Local LLM with MCP tools and self-state block
        messages = [
            {"role": "system", "content": f"{HAL9000_SYSTEM_PROMPT}\n\n{self_state_block}"}
        ]
        if memory_context:
            messages.append({"role": "system", "content": memory_context})

        for t in recent_turns[:-1]:  # exclude latest turn already recorded
            messages.append({"role": t["role"], "content": t["content"]})
        messages.append({"role": "user", "content": user_text})

        tools = self.mcp.get_openai_tools() or None

        t_llm_start = time.perf_counter()
        tool_call_count = 0
        llm_error_count = 0
        used_route = route_name
        used_model = getattr(self.llm, "model", "unknown")
        hal_reply = ""

        try:
            resp_msg, used_route, used_model = await self.router.generate(
                messages, tools=tools, return_dict=True, max_tokens=256
            )
            
            # Check if LLM requested MCP tool execution
            if isinstance(resp_msg, dict) and resp_msg.get("tool_calls"):
                tool_calls = resp_msg["tool_calls"]
                tool_call_count = len(tool_calls)
                messages.append(resp_msg)
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name")
                    fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                    try:
                        import json
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except Exception:
                        fn_args = {}

                    self.ui.show_info(f"Using tool: {fn_name}...")
                    try:
                        tool_res = await self.mcp.call_tool(fn_name, fn_args)
                    except Exception as te:
                        tool_res = f"Tool execution failed: {te}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_1"),
                        "content": str(tool_res)[:2000]
                    })

                # Follow-up completion after tool execution
                second_resp, used_route, used_model = await self.router.generate(
                    messages, max_tokens=256
                )
                if isinstance(second_resp, dict):
                    hal_reply = second_resp.get("content") or "Action completed."
                else:
                    hal_reply = str(second_resp)
            elif isinstance(resp_msg, dict):
                hal_reply = resp_msg.get("content") or "Affirmative."
            else:
                hal_reply = str(resp_msg)
        except Exception as e:
            llm_error_count += 1
            self.ui.show_error(f"LLM Error: {e}")
            hal_reply = "I am sorry, Dave. I am experiencing a temporary communication difficulty."

        llm_ms = (time.perf_counter() - t_llm_start) * 1000.0
        self.last_latency_ms = llm_ms

        if self._cancel_current:
            return

        self.ui.show_hal_reply(hal_reply)
        asst_turn_id = self.memory.record_turn("assistant", hal_reply, session_id=self.current_episode_id)

        # 5. Synthesize and speak response (using sentence streaming pipelining)
        self.ui.set_state("RESPONDING")
        self.audio_ctrl.notify_playback_started()

        t_tts_start = time.perf_counter()
        tts_first_audio_ms = 0.0
        first_audio_recorded = False

        def _play_chunk(pcm_chunk: bytes):
            nonlocal tts_first_audio_ms, first_audio_recorded
            if not first_audio_recorded:
                tts_first_audio_ms = (time.perf_counter() - t_tts_start) * 1000.0
                first_audio_recorded = True
            if not self._cancel_current:
                self.playback.play_pcm(pcm_chunk)

        try:
            if hasattr(self.tts, "speak_sentences"):
                await self.tts.speak_sentences(hal_reply, on_audio_chunk=_play_chunk)
            else:
                loop = asyncio.get_running_loop()
                raw_pcm_22k = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, hal_reply)
                tts_first_audio_ms = (time.perf_counter() - t_tts_start) * 1000.0
                if not self._cancel_current:
                    await loop.run_in_executor(None, self.playback.play_pcm, raw_pcm_22k)
        finally:
            self.audio_ctrl.notify_playback_stopped()

        tts_total_ms = (time.perf_counter() - t_tts_start) * 1000.0

        # Record telemetry in database
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
            interrupted=1 if self._cancel_current else 0,
        )

    async def _transcribe(self, pcm_bytes: bytes) -> str:
        """Transcribe PCM bytes using ParakeetAdapter."""
        async def audio_gen():
            yield PcmChunk(data=pcm_bytes, format=FORMAT_16K)
            yield PcmChunk(data=b"", format=FORMAT_16K)

        results = []
        async def on_emit(ev):
            if isinstance(ev, TranscriptionDone):
                results.append(ev.text)
            elif hasattr(ev, "text") and ev.text:
                results.append(ev.text)

        cfg = SessionConfig(audio_format=FORMAT_16K)
        try:
            await self.adapter.run_session(cfg, audio_gen(), on_emit)
        except Exception as e:
            logger.debug(f"Transcription error: {e}")

        return " ".join(results).strip()

    def _keyboard_listener(self):
        """Listens for user commands from stdin (Push-to-Talk, Mute, Cancel, Quit)."""
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
                    break
                elif ch in ('c', 'C'):  # Cancel
                    self._cancel_current = True
                    self.tts.cancel()
                    self.playback.stop()
                    self.audio_ctrl.notify_playback_stopped()
                    self.ui.set_state("LISTENING")
                elif ch in ('m', 'M'):  # Mute toggle
                    current = self.audio_ctrl._muted
                    self.audio_ctrl.set_muted(not current)
                    status = "Muted" if not current else "Unmuted"
                    self.ui.show_info(f"Microphone {status}")
                elif ch == ' ':  # Space (Push-to-Talk or trigger)
                    pass
        except Exception:
            pass
        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
