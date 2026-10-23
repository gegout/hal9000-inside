"""5x30-Minute Long-Run Voice Campaign on Canonical & Open Source.

Requirements:
- 5 sessions of 30 minutes (1800s) each (~2.5 hours total).
- Real voice generation: Synthesized via Piper with distinct pitch-shifting/formant
  variations simulating 5 distinct user personas.
- Ingested through Hal9000Assistant orchestration pipeline.
- Real spoken output with HalPiperTTS into MemoryPlaybackSink.
- Topics focused on Canonical, Ubuntu, Debian, Snaps, Linux Kernel, Open Source governance & economics.
- Continuous metrics logging: turn time, latency, speech duration, repetition detection.
"""

import asyncio
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.config import load_config
from hal9000_inside.tts.hal_piper import HalPiperTTS
from hal9000_inside.audio.manager import resample_pcm
from tests.fakes.fake_stt import FakeSTT

# 5 Distinct Personas
VOICE_PROFILES = [
    {"name": "Mark_Persona (Reflective Visionary)", "semitones": -4.0, "rate_factor": 0.96},
    {"name": "Kernel_Maintainer (Gruff Pragmatist)", "semitones": -2.0, "rate_factor": 1.02},
    {"name": "App_Developer (Enthusiastic Innovator)", "semitones": +3.0, "rate_factor": 1.05},
    {"name": "SysAdmin_SRE (Analytical Operator)", "semitones": +1.0, "rate_factor": 0.94},
    {"name": "OpenSource_Steward (Diplomatic Advocate)", "semitones": -3.0, "rate_factor": 0.98},
]

# 5 Sessions of 30 Minutes on Canonical & Open Source
CANONICAL_SCENARIOS = [
    {
        "id": 1,
        "topic": "Canonical History, Ubuntu Philosophy, and Debian Roots",
        "voice": VOICE_PROFILES[0],
        "turns": [
            ("Hello HAL. Let us examine Canonical's founding in 2004 and the original mission behind Ubuntu.", "Canonical founding & mission"),
            ("Please search external databanks or Wikipedia for the history of Mark Shuttleworth and the 'Bug #1' declaration.", "Mark Shuttleworth Bug #1 search"),
            ("How does Ubuntu maintain its package synchronization and divergence cadence with Debian Sid and Testing?", "Debian sync & divergence"),
            ("Explain the Ubuntu governance model, including the Community Council and Technical Board.", "Ubuntu governance structure"),
            ("What philosophical differences distinguish Debian's Social Contract from Ubuntu's 'Linux for Human Beings' ethos?", "Debian DFSG vs Ubuntu ethos"),
            ("Please search Wikipedia for the history of Ubuntu release codenames and the six-month time-based release schedule.", "Release cadence & codenames search"),
            ("How did Ubuntu's early live-CD and free ShipIt disc distribution change Linux adoption globally?", "ShipIt & LiveCD impact"),
            ("Following up: how does Canonical balance upstream Debian contributions with its own product priorities?", "Debian upstream balance"),
            ("What role did Launchpad and Bazaar play in the early Canonical developer ecosystem versus modern git and GitHub?", "Launchpad & Bazaar history"),
            ("HAL, summarize how Canonical's open source approach evolved from desktop disruption to enterprise server and cloud infrastructure.", "Evolution to enterprise cloud"),
            ("How does the Ubuntu Code of Conduct serve as a pioneer template for community standards across the open source world?", "Ubuntu Code of Conduct"),
            ("What is the current relationship between Canonical and the Debian project regarding security CVE patches?", "Debian security collaboration"),
        ]
    },
    {
        "id": 2,
        "topic": "Snap Confinement, AppArmor, Snapcraft, and Ubuntu Core",
        "voice": VOICE_PROFILES[1],
        "turns": [
            ("HAL, let us investigate the architectural design of Snap packages, squashfs mounts, and AppArmor profiles.", "Snap & AppArmor architecture"),
            ("Please search Wikipedia or external records for AppArmor LSM mandatory access control in the Linux kernel.", "AppArmor LSM search"),
            ("Why does Canonical enforce strict confinement in snaps and how does the Snapd plug and slot interface mechanism work?", "Plugs, slots & strict confinement"),
            ("What is Ubuntu Core and how does its all-snap immutable operating system ensure transactional rollback during updates?", "Ubuntu Core & transactional updates"),
            ("How does snapd handle classic confinement versus strict confinement for developer toolchains like Python and compilers?", "Classic vs strict confinement"),
            ("Please search external documentation for Snapcraft build lifecycle stages: pull, build, stage, and prime.", "Snapcraft lifecycle search"),
            ("How do delta downloads in snap updates save bandwidth for embedded IoT and edge deployments?", "Delta downloads & bandwidth"),
            ("Following up: what are the security trade-offs of mounting read-only squashfs images compared to Flatpak OCI layers?", "Squashfs vs Flatpak OCI security"),
            ("How does snap confinement interact with user namespace unprivileged containers and seccomp system call filtering?", "Seccomp & user namespaces"),
            ("HAL, describe how enterprise software vendors leverage the Snap Store for automated channel tracks and staged rollouts.", "Enterprise tracks & rollouts"),
            ("What are the criticisms of the closed-source nature of the Canonical Snap Store backend and how does Canonical address them?", "Snap store backend debate"),
            ("How does the snap sandbox integrate with Wayland and PulseAudio or PipeWire for desktop applications?", "Desktop portals & audio sandboxing"),
        ]
    },
    {
        "id": 3,
        "topic": "Canonical Cloud & Infrastructure: MAAS, Juju, LXD, and MicroK8s",
        "voice": VOICE_PROFILES[2],
        "turns": [
            ("HAL, explain Canonical's cloud infrastructure stack, starting with Metal-as-a-Service, or MAAS.", "MAAS architecture"),
            ("Please search external databanks for Canonical MAAS bare-metal provisioning and IPAM capabilities.", "MAAS provisioning search"),
            ("How does Juju application modeling and Charmed Operators differ from traditional Ansible or Terraform automation?", "Juju & Charmed Operators"),
            ("What is the architectural role of LXD system containers compared to Docker application micro-containers?", "LXD system containers vs Docker"),
            ("Please search Wikipedia or documentation for the history of LXD and its transition into the Canonical project portfolio.", "LXD history search"),
            ("How does MicroK8s provide zero-ops Kubernetes clusters for developers and edge computing?", "MicroK8s architecture & zero-ops"),
            ("Following up: what is Canonical's approach to OpenStack deployments via Charms versus TripleO or Kolla?", "Charmed OpenStack vs alternatives"),
            ("How does Ceph integration work within Canonical's software-defined storage portfolio?", "Canonical Ceph storage"),
            ("What are the benefits of running production Kubernetes workloads on Ubuntu Pro with Kernel Livepatch?", "Ubuntu Pro & Kernel Livepatch"),
            ("HAL, synthesize how Canonical positions its infrastructure stack against Red Hat OpenShift and VMware.", "Canonical vs Red Hat OpenShift"),
            ("How does Canonical address hybrid cloud portability across AWS, Azure, Google Cloud, and bare-metal data centers?", "Hybrid cloud portability"),
            ("What telemetry, observability, and logging tools form Canonical's Canonical Observability Stack, or COS?", "Canonical Observability Stack COS"),
        ]
    },
    {
        "id": 4,
        "topic": "Linux Kernel, Ubuntu Hardware Enablement (HWE), and Silicon Partnerships",
        "voice": VOICE_PROFILES[3],
        "turns": [
            ("HAL, explain the Ubuntu Hardware Enablement, or HWE, kernel rolling model in LTS releases.", "Ubuntu HWE kernel model"),
            ("Please search external databanks for the Linux kernel release cycle and Ubuntu kernel rebasing process.", "Kernel rebase & lifecycle search"),
            ("How does Canonical collaborate with Intel, AMD, and ARM for day-zero silicon support in new Ubuntu releases?", "Silicon partnerships Intel AMD ARM"),
            ("What is the significance of the real-time Linux kernel, PREEMPT_RT, in Ubuntu 24.04 LTS for robotics and industrial automation?", "PREEMPT_RT in Ubuntu 24.04"),
            ("How do Level Zero compute drivers and Intel Arc GPU acceleration integrate with the Ubuntu Xe kernel driver?", "Level Zero & Intel Arc Xe driver"),
            ("Please search Wikipedia for the Linux PREEMPT_RT patchset and its official merge status into mainline Linux.", "PREEMPT_RT mainline merge search"),
            ("Following up: how does Canonical ensure power management and thermal efficiency on modern laptops through thermald and tlp?", "Power & thermals on Ubuntu"),
            ("What validation and certification testing does Canonical perform for Ubuntu Certified hardware with OEM partners like Lenovo and Dell?", "Ubuntu OEM hardware certification"),
            ("How does Canonical support RISC-V architecture on Ubuntu, such as StarFive VisionFive and SiFive developer boards?", "Ubuntu on RISC-V"),
            ("HAL, explain how confidential computing, AMD SEV-SNP, and Intel TDX are supported in Ubuntu virtualization.", "Confidential computing SEV TDX"),
            ("What mechanisms does Canonical use to backport fixes for zero-day vulnerabilities like Spectre, Meltdown, or Downfall into legacy kernels?", "Kernel vulnerability backporting"),
            ("How do DKMS, Dynamic Kernel Module Support, and Secure Boot signature validation work together in Ubuntu?", "DKMS & Secure Boot signing"),
        ]
    },
    {
        "id": 5,
        "topic": "Open Source Economics, Dual Licensing, Ubuntu Pro, and the Future of FLOSS",
        "voice": VOICE_PROFILES[4],
        "turns": [
            ("HAL, let us analyze the economic model of commercial open source companies like Canonical, Red Hat, and SUSE.", "Open source business models"),
            ("Please search external databanks for the history and controversy surrounding license changes from open source to BSL or SSPL.", "BSL SSPL license controversy search"),
            ("What is Ubuntu Pro, and how does Canonical monetize expanded security maintenance, ESM, for 10 to 12 years?", "Ubuntu Pro & ESM economics"),
            ("How does Canonical maintain free community access to Ubuntu Pro for up to 5 personal machines while charging enterprises?", "Ubuntu Pro personal vs enterprise"),
            ("What role does the Open Source Initiative, OSI, play in defining open source, and how are AI model weights currently being categorized?", "OSI & Open Source AI Definition"),
            ("Please search Wikipedia for the Open Source Definition and the 10 criteria established by Bruce Perens and OSI.", "Open Source Definition criteria search"),
            ("Following up: how do public cloud provider re-packaging and managed services affect open source project sustainability?", "Cloud hyperscalers vs FLOSS projects"),
            ("How does Canonical approach copyleft licenses like GPLv3 and AGPL versus permissive licenses like Apache 2.0 and MIT?", "GPL copyleft vs permissive licenses"),
            ("What is Canonical's stance on software supply chain security, SLSA compliance, and reproducible builds in Ubuntu?", "Software supply chain & SLSA"),
            ("HAL, provide a comprehensive evaluation of how Canonical can balance public commercial growth with open source community fidelity.", "Commercial growth vs community fidelity"),
            ("How does the rise of open-weight LLMs and local AI agents change operating system software stacks over the next decade?", "Local AI agents in next-gen OS"),
            ("What closing reflections can you share, HAL, on the enduring value of human collaboration through free and open-source software?", "Enduring value of FLOSS collaboration"),
        ]
    }
]

class MemoryPlaybackSink:
    """In-memory playback sink measuring synthesized speech audio bytes."""
    def __init__(self):
        self.total_bytes = 0
        self.chunks = []

    def play_pcm(self, data: bytes):
        self.total_bytes += len(data)
        self.chunks.append(data)

    def write(self, data: bytes):
        self.play_pcm(data)

    def stop(self):
        pass

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

async def run_single_session(asst: Hal9000Assistant, sink: MemoryPlaybackSink, stt: FakeSTT, tts: HalPiperTTS, scenario: dict, target_duration_s: float = 1800.0) -> dict:
    """Run one 30-minute session with continuous speech synthesis and conversation."""
    session_id = scenario["id"]
    topic = scenario["topic"]
    voice = scenario["voice"]
    turns_script = scenario["turns"]

    print("\n" + "=" * 95)
    print(f"STARTING SESSION {session_id}/5: {topic}")
    print(f"VOICE PERSONA: {voice['name']} (Pitch: {voice['semitones']:+.1f} st, Rate: {voice['rate_factor']:.2f}x)")
    print(f"TARGET DURATION: {target_duration_s/60:.1f} minutes ({target_duration_s:.0f} seconds)")
    print("=" * 95)

    # Update assistant episode context for session separation
    asst.current_episode_id = f"canonical_session_{session_id}_{int(time.time())}"

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

        # Retrieve next turn prompt
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

        # Configure STT adapter to simulate exact recognition
        stt.set_script([user_text])

        print(f"\n[Turn {turn_num:03d} | Elapsed: {elapsed/60:.1f}m / Rem: {remaining/60:.1f}m] [{label}]")
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

        # Repetition check (exact or 50-char prefix)
        is_repetitive = False
        reply_clean = reply.strip().lower()
        for prev in previous_replies[-5:]:
            if reply_clean == prev:
                is_repetitive = True
                repetitive_count += 1
                print(f"  [WARNING] EXACT REPETITION DETECTED!")
                break
            if len(reply_clean) > 50 and len(prev) > 50 and reply_clean[:50] == prev[:50]:
                is_repetitive = True
                repetitive_count += 1
                print(f"  [WARNING] PREFIX REPETITION DETECTED ({reply_clean[:50]})!")
                break

        previous_replies.append(reply_clean)

        # Log turn metric
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

        # Realistic pacing
        await asyncio.sleep(1.0)

    total_session_elapsed = time.time() - session_start_time
    total_speech_spoken = sink.total_bytes / (2 * 22050)
    avg_turn_time = sum(m["turn_wall_time_s"] for m in turn_metrics) / len(turn_metrics) if turn_metrics else 0.0

    print("\n" + "-" * 85)
    print(f"SESSION {session_id} SUMMARY:")
    print(f"  Total Duration: {total_session_elapsed/60:.2f} minutes ({total_session_elapsed:.1f}s)")
    print(f"  Total Turns Completed: {turn_num}")
    print(f"  Total Spoken Speech: {total_speech_spoken:.1f} seconds (~{total_speech_spoken/60:.2f} minutes)")
    print(f"  Average Turn Time: {avg_turn_time:.2f}s")
    print(f"  Repetitions Detected: {repetitive_count}")
    print("-" * 85)

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
    print("STARTING 5x30-MINUTE LONG-RUN VOICE HARNESS ON CANONICAL & OPEN SOURCE")
    print("=" * 95)

    cfg = load_config()
    stt = FakeSTT()
    tts = HalPiperTTS()
    sink = MemoryPlaybackSink()

    # Create assistant instance with full MCP tool connections
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

    # Run all 5 sessions for 30 minutes (1800s) each
    for scenario in CANONICAL_SCENARIOS:
        res = await run_single_session(asst, sink, stt, tts, scenario, target_duration_s=1800.0)
        all_results.append(res)

    t_all_end = time.time()
    total_campaign_s = t_all_end - t_all_start

    print("\n" + "=" * 95)
    print("ALL 5 LONG-RUN 30-MINUTE CANONICAL VOICE SESSIONS COMPLETED")
    print(f"TOTAL CAMPAIGN DURATION: {total_campaign_s/60:.2f} minutes ({total_campaign_s:.1f}s)")
    print("=" * 95)

    out_path = Path("campaigns/long_run_5x30m_canonical_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "total_campaign_seconds": total_campaign_s,
            "sessions": all_results
        }, f, indent=2)
    print(f"Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
