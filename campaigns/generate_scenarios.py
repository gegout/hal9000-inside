import json

def make_session_1():
    # Session 1: Ubuntu, Linux troubleshooting, Intel GPU inference and local AI. All < 20 words.
    turns = [
        "Let's troubleshoot Ubuntu 24.04 audio and AI acceleration.",
        "My system has an integrated Intel Arc GPU alongside CPU.",
        "Check how Linux handles Level Zero drivers for Intel GPUs.",
        "What does /dev/dri/renderD128 provide for userspace inference?",
        "Does standard user group need video or render permissions?",
        "I noticed ALSA reports device or resource busy intermittently.",
        "Why would pipewire and pulseaudio conflict over the sound card?",
        "How can I trace which process holds the audio device?",
        "Can fuser or lsof identify the audio conflict?",
        "Good. Now explain how OpenVINO leverages Level Zero on Linux.",
        "Does OpenVINO require specific environment variables for GPU discovery?",
        "What about ZE_AFFINITY_MASK when multiple GPUs exist?",
        "If I set ZE_AFFINITY_MASK to zero, what happens?",
        "Can you search online for recent Intel Arc compute driver issues on Ubuntu 24.04?", # Research request 1
        "What did the search reveal about recent compute driver stability?",
        "Earlier I mentioned ALSA conflicts. Could GPU driver loading affect audio over HDMI?",
        "That connects to earlier audio busy errors. How to verify HDMI audio?",
        "Let's test local AI LLM quantization. What is INT4 vs INT8?",
        "Which format offers better throughput on Intel Arc graphics?",
        "Does Gemma 2B or Gemma 4B run smoothly in INT4?",
        "What memory bandwidth bottleneck occurs during token generation?",
        "Is prompt evaluation compute-bound while token generation is memory-bound?",
        "Explain why KV caching consumes significant memory during long contexts.",
        "Can we offload parts of KV cache to system RAM?",
        "Let's check systemd services for our local inference daemon.",
        "How do I create a user systemd service with restart policies?",
        "What happens if the service crashes due to an out-of-memory error?",
        "Can systemd OOMScoreAdjust protect critical inference services?",
        "Remember earlier we checked render permissions? Does systemd service inherit those groups?",
        "Actually, in my earlier question I said Arc GPU, but it's an integrated Xe GPU.", # Correction
        "How does integrated Xe GPU differ in shared memory architecture?",
        "Does integrated Xe share unified memory directly with host DDR5 RAM?",
        "That means zero-copy buffer sharing is possible through Level Zero?",
        "Search documentation for Intel oneAPI level-zero USM shared memory performance.", # Research request 2
        "Summarize what those benchmarks reported regarding USM allocations.",
        "Could USM memory fragmentation cause latency spikes in speech synthesis?",
        "How do we monitor GPU memory usage on Linux for Intel?",
        "Is intel_gpu_top part of the intel-gpu-tools package?",
        "Can intel_gpu_top output JSON or CSV telemetry directly?",
        "How does that compare to nvidia-smi query telemetry?",
        "Let's return to pipewire. How does wireplumber handle default audio sinks?",
        "If HDMI audio disappears after suspend, how to restart pipewire cleanly?",
        "Does systemctl --user restart pipewire pipewire-pulse wireplumber work reliably?",
        "What about ALSA dmix plugin for sharing raw hardware devices?",
        "Why is dmix rarely needed when pipewire-pulse is active?",
        "In turn 14 you mentioned driver issues. Did kernel 6.8 address them?",
        "How do Ubuntu HWE kernels get rolled out to LTS users?",
        "If I pin a custom OEM kernel, will it break DKMS modules?",
        "What is the best way to verify DKMS status after updates?",
        "Give me a three-step summary of our troubleshooting strategy for GPU and audio."
    ]
    return turns

def make_session_2():
    # Session 2: Wingfoil: technique, light-wind sessions, equipment and progression. All < 20 words.
    turns = [
        "Let's talk about wingfoiling technique, gear, and light-wind progression.",
        "What is the minimum wind speed required to foil comfortably?",
        "How does a high aspect front foil change light wind takeoff?",
        "Does a larger front wing always guarantee an easier start?",
        "What is the trade-off in drag when using 2000cm2 wings?",
        "Explain the pumping technique with the wing and legs together.",
        "Which comes first: board speed or foil lift during pumping?",
        "Can you search online for the latest wingfoil world cup light wind pumping records?", # Research request 1
        "What did you find about the latest equipment used in light wind?",
        "How does mast placement in the track affect front foot pressure?",
        "If the board dives when getting on foil, move mast forward or back?",
        "Wait, earlier I said board dives, but actually it breaches unexpectedly.", # Correction
        "If the foil breaches, moving the mast backward lowers front lift?",
        "How does fuselage length influence pitch stability during carves?",
        "What is the advantage of a short fuselage for wave riding?",
        "Let's discuss wing size. Is an 8-meter wing too cumbersome?",
        "How does boom versus soft handles affect pumping efficiency?",
        "Are rigid carbon handles better for direct power transfer?",
        "What about downwind paddle foiling boards for ultra-light wind winging?",
        "Why do long, narrow mid-length boards take off with less power?",
        "Explain the hydrodynamic displacement hump on narrow boards.",
        "Once on foil, does board shape still matter for drag?",
        "How does apparent wind build once you accelerate on foil?",
        "Does apparent wind move the power pocket forward or backward?",
        "How should you adjust your hands as apparent wind shifts forward?",
        "Let's analyze gybing. Why do most beginners drop off foil mid-turn?",
        "Is speed maintenance or foot change the primary cause of falling?",
        "Should you switch feet before or after turning the wing?",
        "Explain the heelside to toeside carve without switching feet first.",
        "Remember earlier we talked about mast position? How does it affect gybes?",
        "Does a forward mast make foot switches more sensitive to weight shifts?",
        "What stabilizer shim angle increases glide during turns?",
        "Does negative shim on the stabilizer increase top speed?",
        "Search online for recent foil stabilizer shimming guides and aerodynamic theory.", # Research request 2
        "What did the search reveal about stabilizer downforce and pitch trim?",
        "How does carbon stiffness in the mast prevent ventilation and flutter?",
        "What causes ventilation along the leading edge of the mast?",
        "Can sanding the trailing edge eliminate high-pitched mast whistling?",
        "What wet-sanding grit is recommended for mast tuning?",
        "Let's discuss safety. What impact vest and helmet standards apply?",
        "Are coiled leashes safer on the waist than calf leashes?",
        "Why is a quick-release harness line essential in gusty offshore winds?",
        "Earlier we discussed an 8-meter wing. What wind range suits a 5-meter?",
        "If the wind drops to 8 knots, can a 5-meter wing get on foil?",
        "What rider weight is realistic for 8 knots with standard gear?",
        "How does chop and swell affect the initial planing phase?",
        "Can using ocean swell help pump up on foil in marginal wind?",
        "What is the best exercise for building calf and shoulder endurance?",
        "How do you safely handle the wing during a deep water pack-down?",
        "Summarize our progression roadmap from water-starts to carve gybes."
    ]
    return turns

print("Generator imported successfully")
