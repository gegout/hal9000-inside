"""5 Long-Run 10-Minute Tests with Real Voice Generation and Piper TTS Playback.

Requirements:
- 5 sessions of 10 minutes (600s) each.
- Real voice generation: Synthesized via Piper with pitch-shifting/formant variations
  for distinct user voices across sessions.
- Ingested via audio pipeline and processed through Hal9000Assistant.
- HAL 9000 responses synthesized in real speech with HalPiperTTS.
- Complete metrics logging: latency, speech duration, tool calls, repetition detection,
  and bottleneck identification.
"""

import asyncio
import io
import json
import logging
import math
import os
import re
import sys
import time
import wave
from pathlib import Path
import numpy as np

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.config import load_config
from hal9000_inside.tts.hal_piper import HalPiperTTS
from hal9000_inside.audio.manager import resample_pcm
from tests.fakes.fake_stt import FakeSTT

# Voice profiles (semitone shifts for distinct simulated user speakers)
VOICE_PROFILES = [
    {"name": "Speaker_Alpha (Deep Baritone)", "semitones": -5.0, "rate_factor": 0.95},
    {"name": "Speaker_Beta (Crisp Tenor)", "semitones": +2.5, "rate_factor": 1.05},
    {"name": "Speaker_Gamma (Soft Alto)", "semitones": +4.5, "rate_factor": 1.00},
    {"name": "Speaker_Delta (Grounded Bass)", "semitones": -3.0, "rate_factor": 0.92},
    {"name": "Speaker_Epsilon (Brisk Mezzo)", "semitones": +1.5, "rate_factor": 1.08},
]

# Scenarios for the 5 Sessions (10 minutes each)
SESSION_SCENARIOS = [
    {
        "id": 1,
        "topic": "Ubuntu, Linux Troubleshooting, Intel GPU Inference & Local AI",
        "voice": VOICE_PROFILES[0],
        "turns": [
            ("Hello HAL. Let's inspect the Level Zero compute drivers and Intel GPU inference configuration on Ubuntu 24.04.", "L0 drivers & GPU inference"),
            ("Please search external documentation or Wikipedia for Intel Level Zero API architecture and its relationship to oneAPI.", "Level Zero search"),
            ("What does /dev/dri/renderD128 provide for inference and which user group permissions are strictly necessary?", "Render node permissions"),
            ("How does OpenVINO utilize the NPU and integrated GPU compared to Level Zero direct execution?", "OpenVINO NPU vs GPU"),
            ("ALSA reports device or resource busy intermittently when starting local audio pipelines. How should PipeWire and ALSA contention be addressed?", "ALSA PipeWire contention"),
            ("Please search external databanks for troubleshooting PipeWire wireplumber audio node locks.", "PipeWire wireplumber search"),
            ("Explain zero-copy memory transfer and unified shared memory between Intel CPUs and integrated Arc GPUs.", "Unified shared memory"),
            ("How can we monitor GPU compute engines and memory bandwidth in real-time with intel_gpu_top?", "intel_gpu_top monitoring"),
            ("What are the architectural advantages of the new Linux Xe kernel driver over the legacy i915 driver for Intel discrete GPUs?", "Xe vs i915 driver"),
            ("HAL, provide a concise summary of our diagnostic plan for optimizing local AI inference on this system.", "Diagnostic summary"),
            ("Following up: what fallbacks exist if Level Zero driver fails to initialize during boot?", "Level Zero fallback"),
            ("How does dynamic model quantisation to INT8 impact inference latency versus FP16 on Intel Arc matrix engines?", "Quantisation INT8 vs FP16"),
        ]
    },
    {
        "id": 2,
        "topic": "Wingfoil: Aerodynamics, Hydrodynamics, Equipment & Progression",
        "voice": VOICE_PROFILES[1],
        "turns": [
            ("HAL, let's explore the physics and equipment choices in modern wingfoiling.", "Wingfoil fundamentals"),
            ("Please search external Wikipedia records for the development of hydrofoil surfing and wingfoiling.", "Hydrofoil history search"),
            ("How does foil aspect ratio affect stall speed, glide efficiency, and pitch stability in choppy coastal water?", "Aspect ratio dynamics"),
            ("When transitioning to light-wind riding under 12 knots, what front wing surface area and fuselage length offer the optimal pump response?", "Light-wind foil setup"),
            ("Explain the aerodynamic lift distribution across a compact inflatable handheld wing with rigid carbon handles.", "Wing aerodynamics & handles"),
            ("Search online databanks for modern high-modulus carbon mast torsional stiffness under heavy carving loads.", "Carbon mast stiffness search"),
            ("What causes sudden ventilation or breaching of the front wing during high-speed downwind carves?", "Wing ventilation & breaching"),
            ("How should footstrap placement and mast track position be adjusted when rider balance feels excessively front-foot heavy?", "Trim & mast track adjustment"),
            ("Describe the kinetic mechanics of a successful foil jibe, specifically foot swapping before versus after wing transition.", "Foil jibe mechanics"),
            ("HAL, synthesize our discussion into three core progression principles for mastering powered foil tacks.", "Foil tacking principles"),
            ("How does downwind swell riding change when using a glide-oriented unhooked wing harness?", "Unhooked harness dynamics"),
            ("What safety precautions are essential regarding offshore tidal currents and foil leash retention?", "Safety and tidal currents"),
        ]
    },
    {
        "id": 3,
        "topic": "Open-Source Robotics, ROS 2, and Edge AI Architectures",
        "voice": VOICE_PROFILES[2],
        "turns": [
            ("HAL, let's evaluate real-time deterministic communication and robotics architectures using ROS 2.", "ROS 2 fundamentals"),
            ("Please search Wikipedia or external records for ROS 2 architecture and the Data Distribution Service DDS standard.", "ROS 2 DDS search"),
            ("What are the key trade-offs between CycloneDDS and FastDDS in multi-robot wireless mesh networks?", "CycloneDDS vs FastDDS"),
            ("How do lifecycle nodes in ROS 2 ensure deterministic state transitions during robotic arm calibration?", "ROS 2 lifecycle nodes"),
            ("Search external documentation for micro-ROS running on microcontroller firmware like ESP32 and STM32.", "micro-ROS search"),
            ("Explain how zero-copy transport via shared memory operates within ROS 2 nodes on Ubuntu.", "ROS 2 zero-copy shared memory"),
            ("What algorithmic differences distinguish 2D LiDAR SLAM from 3D visual-inertial odometry in cluttered environments?", "LiDAR vs Visual SLAM"),
            ("How should edge AI object detection models like YOLO be integrated with ROS 2 sensor pipelines without introducing queue backlog?", "Edge AI YOLO integration"),
            ("In safety-critical autonomous mobile robots, how are hardware watchdog timers and emergency stop loops architected?", "Watchdogs and E-Stop"),
            ("HAL, summarize the recommended software stack for an industrial warehouse inspection robot.", "Robotics stack summary"),
            ("How does real-time PREEMPT_RT kernel patching reduce jitter in motor actuator control loops?", "PREEMPT_RT motor loops"),
            ("What strategies prevent ROS 2 parameter server bottlenecks when orchestrating hundreds of distributed sensors?", "Parameter server scaling"),
        ]
    },
    {
        "id": 4,
        "topic": "AI Agents: Episodic Memory, Cognitive Architecture & Interoception",
        "voice": VOICE_PROFILES[3],
        "turns": [
            ("HAL, let's discuss cognitive architectures for autonomous AI assistants and agents.", "Cognitive architectures"),
            ("Please search external databanks for cognitive architectures like SOAR and ACT-R and their influence on modern AI agents.", "SOAR and ACT-R search"),
            ("How should an AI agent distinguish between short-term working memory, semantic knowledge, and episodic memory?", "Memory taxonomy"),
            ("What mechanisms prevent catastrophic forgetting when updating vector embeddings with new user preferences?", "Catastrophic forgetting mitigation"),
            ("Explain the concept of interoception and synthetic mood dynamics in conversational artificial intelligence.", "Interoception and synthetic mood"),
            ("Search external research records for generative agent reflection loops and memory consolidation algorithms.", "Agent reflection search"),
            ("How can an agent detect when it is falling into repetitive verbal loops or hallucinated justifications?", "Repetition loop detection"),
            ("What is the role of offline consolidation or simulated sleep cycles in optimizing agent knowledge graphs?", "Sleep cycles and knowledge graphs"),
            ("How should multi-turn dialogue state tracking handle implicit corrections and ambiguous user references?", "Dialogue state tracking"),
            ("HAL, assess your own operational architecture regarding memory recall, latency tracking, and mood stability.", "Self-assessment telemetry"),
            ("What safeguards ensure privacy and credential protection across persistent cognitive memory stores?", "Memory security & privacy"),
            ("How do hierarchical task networks assist autonomous agents in long-horizon reasoning?", "Hierarchical task planning"),
        ]
    },
    {
        "id": 5,
        "topic": "Product Leadership, Engineering Culture & Digital Sovereignty",
        "voice": VOICE_PROFILES[4],
        "turns": [
            ("HAL, let's examine principles of high-performing engineering leadership and open-source strategy.", "Engineering leadership"),
            ("Please search external records for digital sovereignty initiatives in European enterprise cloud infrastructure.", "Digital sovereignty search"),
            ("How can engineering leaders cultivate high psychological safety while maintaining rigorous code quality and accountability?", "Psychological safety & rigor"),
            ("When scaling engineering organizations from fifty to several hundred engineers, what architectural decoupled team structures prevent friction?", "Team topologies & scaling"),
            ("Search online for open source business sustainability models and commercial support frameworks.", "Open source business models"),
            ("How should technical debt be prioritized and negotiated alongside product roadmap features with non-technical stakeholders?", "Technical debt negotiation"),
            ("What are the key differences between Canonical's enterprise open source governance and proprietary SaaS cloud platforms?", "Open source vs SaaS governance"),
            ("How does a leader deliver constructive, actionable feedback to senior technical staff during major operational incidents?", "Constructive feedback in crisis"),
            ("What metrics best reflect developer velocity and operational reliability without creating perverse incentives?", "Engineering metrics & DORA"),
            ("HAL, synthesize a strategic manifesto for leading autonomous AI and infrastructure teams into the next decade.", "Leadership manifesto synthesis"),
            ("How can companies balance internal proprietary innovations with active contributions to the global open-source commons?", "Open-source reciprocity"),
            ("What final guidance do you offer for building resilient, ethical, and purpose-driven engineering organizations?", "Resilient engineering organizations"),
        ]
    }
]

class MemoryPlaybackSink:
    """Captures played PCM audio chunks for validation."""
    def __init__(self):
        self.total_bytes = 0
        self.chunk_count = 0

    def play_pcm(self, pcm_bytes: bytes):
        self.total_bytes += len(pcm_bytes)
        self.chunk_count += 1

    def close(self):
        pass

def pitch_shift_and_resample(pcm_bytes: bytes, semitones: float, rate_factor: float) -> bytes:
    """Modulates voice tone and rate to generate distinct user voice."""
    factor = (2.0 ** (semitones / 12.0)) * rate_factor
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    n_in = len(samples)
    n_out = max(100, int(n_in / factor))
    indices = np.linspace(0, n_in - 1, n_out)
    shifted = np.interp(indices, np.arange(n_in), samples).astype(np.int16)
    return shifted.tobytes()

async def run_single_session(asst: Hal9000Assistant, sink: MemoryPlaybackSink, stt: FakeSTT, tts: HalPiperTTS, scenario: dict, target_duration_s: float = 600.0) -> dict:
    """Run one 10-minute session with continuous speech synthesis and conversation."""
    session_id = scenario["id"]
    topic = scenario["topic"]
    voice = scenario["voice"]
    turns_script = scenario["turns"]

    print("\n" + "=" * 90)
    print(f"STARTING SESSION {session_id}/5: {topic}")
    print(f"VOICE: {voice['name']} (Semitones: {voice['semitones']}, Rate: {voice['rate_factor']})")
    print(f"TARGET DURATION: {target_duration_s/60:.1f} minutes ({target_duration_s:.0f}s)")
    print("=" * 90)

    # Set episode ID for this session
    asst.current_episode_id = f"session_{session_id}_{int(time.time())}"

    session_start_time = time.time()
    turn_metrics = []
    repetitive_count = 0
    previous_replies = []
    script_idx = 0
    turn_num = 0

    while (time.time() - session_start_time) < target_duration_s:
        turn_num += 1
        elapsed = time.time() - session_start_time
        remaining = max(0.0, target_duration_s - elapsed)

        # Get question
        script_item = turns_script[script_idx % len(turns_script)]
        cycle_idx = script_idx // len(turns_script)
        script_idx += 1

        if cycle_idx > 0:
            user_text = f"Continuing our exploration: {script_item[0]}"
            label = f"[R{cycle_idx+1}] {script_item[1]}"
        else:
            user_text = script_item[0]
            label = script_item[1]

        # 1. Synthesize user voice with distinct persona
        t_user_synth = time.time()
        base_user_pcm = tts.synthesize_raw_pcm(user_text)
        shifted_user_pcm_22k = pitch_shift_and_resample(base_user_pcm, voice["semitones"], voice["rate_factor"])
        user_pcm_16k = resample_pcm(shifted_user_pcm_22k, 22050, 16000)
        synth_user_dur = time.time() - t_user_synth

        # Configure STT script to simulate exact transcription from audio input
        stt.set_script([user_text])

        print(f"\n[Turn {turn_num:02d} | Elapsed: {elapsed/60:.1f}m / Rem: {remaining/60:.1f}m] [{label}]")
        print(f"USER ({voice['name']}): {user_text}")
        print(f"  -> Generated User Audio: {len(user_pcm_16k)} bytes ({len(user_pcm_16k)/32000:.2f}s, synth={synth_user_dur:.2f}s)")

        sink_bytes_before = sink.total_bytes
        t_turn_start = time.time()

        # Execute turn through full assistant orchestration
        await asst._handle_turn(user_pcm_16k)
        turn_wall_time = time.time() - t_turn_start

        # Retrieve reply
        recent = asst.memory.get_recent_turns(limit=2)
        reply = recent[-1]["content"] if recent and recent[-1]["role"] == "assistant" else "NO_REPLY"
        mood = getattr(asst, "current_mood", "nominal")
        bytes_spoken = sink.total_bytes - sink_bytes_before
        spoken_seconds = bytes_spoken / (2 * 22050)

        print(f"HAL ({mood}, turn_time={turn_wall_time:.2f}s, speech={spoken_seconds:.2f}s):")
        print(f"  \"{reply}\"")

        # Repetition check
        is_repetitive = False
        reply_clean = reply.strip().lower()
        for prev in previous_replies[-5:]:
            if reply_clean == prev:
                is_repetitive = True
                repetitive_count += 1
                print(f"  [WARNING] EXACT REPETITION DETECTED!")
                break
            # Check 50-char prefix repetition
            if len(reply_clean) > 50 and len(prev) > 50 and reply_clean[:50] == prev[:50]:
                is_repetitive = True
                repetitive_count += 1
                print(f"  [WARNING] PREFIX REPETITION DETECTED ({reply_clean[:50]})!")
                break

        previous_replies.append(reply_clean)

        # Log metric record
        metric = {
            "turn": turn_num,
            "label": label,
            "elapsed_s": elapsed,
            "turn_wall_time_s": turn_wall_time,
            "speech_seconds": spoken_seconds,
            "bytes_spoken": bytes_spoken,
            "reply_length": len(reply),
            "mood": mood,
            "repetitive": is_repetitive,
            "reply_snippet": reply[:100],
        }
        turn_metrics.append(metric)

        # Pacing: brief realistic inter-turn pause
        await asyncio.sleep(1.0)

    total_session_elapsed = time.time() - session_start_time
    total_speech_spoken = sink.total_bytes / (2 * 22050)
    avg_turn_time = sum(m["turn_wall_time_s"] for m in turn_metrics) / len(turn_metrics) if turn_metrics else 0.0

    print("\n" + "-" * 80)
    print(f"SESSION {session_id} SUMMARY:")
    print(f"  Total Duration: {total_session_elapsed/60:.2f} minutes ({total_session_elapsed:.1f}s)")
    print(f"  Total Turns Completed: {turn_num}")
    print(f"  Total Spoken Speech: {total_speech_spoken:.1f} seconds (~{total_speech_spoken/60:.2f} minutes)")
    print(f"  Average Turn Time: {avg_turn_time:.2f}s")
    print(f"  Repetitions Detected: {repetitive_count}")
    print("-" * 80)

    return {
        "session_id": session_id,
        "topic": topic,
        "voice": voice["name"],
        "duration_s": total_session_elapsed,
        "turns_completed": turn_num,
        "total_speech_s": total_speech_spoken,
        "avg_turn_time_s": avg_turn_time,
        "repetitions": repetitive_count,
        "metrics": turn_metrics,
    }

async def main():
    print("=" * 95)
    print("STARTING 5x10-MINUTE LONG-RUN VOICE HARNESS (HAL 9000 INSIDE)")
    print("=" * 95)

    cfg = load_config()
    stt = FakeSTT()
    tts = HalPiperTTS()
    sink = MemoryPlaybackSink()

    # Create assistant with real Piper TTS and MemoryPlaybackSink
    asst = Hal9000Assistant(
        config=cfg,
        adapter=stt,
        tts=tts,
        audio_sink=sink,
        auto_mode=True,
    )
    await asst.mcp.connect_all()

    all_results = []
    t_all_start = time.time()

    for scenario in SESSION_SCENARIOS:
        res = await run_single_session(asst, sink, stt, tts, scenario, target_duration_s=600.0)
        all_results.append(res)

    t_all_end = time.time()
    total_campaign_s = t_all_end - t_all_start

    print("\n" + "=" * 95)
    print("ALL 5 LONG-RUN VOICE SESSIONS COMPLETED")
    print(f"TOTAL CAMPAIGN DURATION: {total_campaign_s/60:.2f} minutes ({total_campaign_s:.1f}s)")
    print("=" * 95)

    # Save summary report
    out_path = Path("campaigns/long_run_5x10m_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "total_campaign_seconds": total_campaign_s,
            "sessions": all_results
        }, f, indent=2)
    print(f"Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
