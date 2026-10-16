"""Main CLI entrypoint for HAL 9000 Inside.

Subcommands:
- run: start the conversational assistant (speech loop)
- doctor: check models, devices, credentials, and dependencies
- memory: inspect, remember, correct, forget, clear memory
- import-key: securely import OpenRouter API key into application config
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

def cmd_doctor(args):
    """Run diagnostics on models, dependencies, devices, and configuration."""
    print("Running HAL 9000 Inside Doctor...")
    issues = 0

    from hal9000_inside.config import load_config
    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()
    cfg_files = cfg.get("files", {})
    cfg_env = cfg.get("environment", {})

    # 1. OpenRouter API Key
    from hal9000_inside.llm import LLMClient, load_openrouter_api_key
    custom_key_path = os.path.expanduser(cfg_files.get("openrouter_config_path", "~/.openrouter.config"))
    key = load_openrouter_api_key(custom_path=custom_key_path) if "custom_path" in load_openrouter_api_key.__code__.co_varnames else load_openrouter_api_key()
    if key and key.startswith("sk-or-"):
        redacted = key[:8] + "..." + key[-4:]
        print(f"  [PASS] OpenRouter API Key configured: {redacted}")
    else:
        print("  [FAIL] OpenRouter API Key not found or invalid.")
        issues += 1

    # 2. HAL 9000 Voice Model Checksums
    from hal9000_inside.tts import (
        MODEL_PATH,
        CONFIG_PATH,
        EXPECTED_ONNX_SHA256,
        EXPECTED_JSON_SHA256,
        resolve_model_path,
        resolve_config_path,
    )
    import hashlib
    model_path = resolve_model_path(cfg_files.get("tts_model_path"))
    config_path = resolve_config_path(cfg_files.get("tts_config_path"))
    if os.path.isfile(model_path):
        with open(model_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h == EXPECTED_ONNX_SHA256:
            print(f"  [PASS] HAL 9000 ONNX verified: {model_path}")
        else:
            print(f"  [FAIL] HAL 9000 ONNX checksum mismatch: {h}")
            issues += 1
    else:
        print(f"  [FAIL] HAL 9000 ONNX missing: {model_path}")
        issues += 1

    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h == EXPECTED_JSON_SHA256:
            print(f"  [PASS] HAL 9000 Config verified: {config_path}")
        else:
            print(f"  [FAIL] HAL 9000 Config checksum mismatch: {h}")
            issues += 1
    else:
        print(f"  [FAIL] HAL 9000 Config missing: {config_path}")
        issues += 1

    # 3. Myna Parakeet Model
    parakeet_dir = os.path.expanduser(cfg_files.get("parakeet_model_dir", "~/.cache/myna/models/parakeet-tdt-0.6b-v3-int8"))
    parakeet_enc = os.path.join(parakeet_dir, "encoder-model.int8.onnx")
    if os.path.isfile(parakeet_enc):
        print(f"  [PASS] Myna Parakeet TDT 0.6B model present: {parakeet_dir}")
    else:
        print(f"  [FAIL] Myna Parakeet TDT 0.6B model missing at: {parakeet_dir}")
        issues += 1

    # 4. Multilingual E5 Model
    e5_dir = os.path.expanduser(cfg_files.get("e5_model_dir", "~/.cache/hal9000/models/multilingual-e5-small/onnx"))
    e5_model = os.path.join(e5_dir, "model.onnx")
    if os.path.isfile(e5_model):
        print(f"  [PASS] Multilingual E5 ONNX model present: {e5_dir}")
    else:
        print(f"  [FAIL] Multilingual E5 ONNX model missing at: {e5_dir}")
        issues += 1

    # 5. Audio Subsystem (ALSA & sound utilities)
    from hal9000_inside.audio import list_audio_devices
    caps, plays = list_audio_devices()
    if caps or plays:
        print(f"  [PASS] ALSA Audio devices detected: {len(caps)} capture, {len(plays)} playback")
    else:
        print("  [WARN] No ALSA capture/playback devices detected.")

    # 6. Compute Hardware & GPU Acceleration
    from hal9000_inside.hardware import detect_system_gpus
    gpu_info = detect_system_gpus()
    nv = gpu_info["nvidia"]
    intel = gpu_info["intel"]
    if nv["hardware_present"]:
        drv = "driver ready" if nv["driver_ready"] else "driver not detected"
        print(f"  [PASS] NVIDIA GPU detected: {nv['device_name']} ({drv})")
    else:
        print("  [INFO] NVIDIA GPU: None detected")

    if intel["hardware_present"]:
        dri = "render node available" if intel["dri_present"] else "no dri node"
        print(f"  [PASS] Intel GPU detected: {intel['device_name']} ({dri})")
    else:
        print("  [INFO] Intel GPU: None detected")

    # 7. Storage & Memory Paths
    storage_dir = os.path.expanduser(cfg_env.get("data_dir", "~/.local/share/hal9000-inside"))
    print(f"  [PASS] Memory storage directory: {storage_dir}")


    if issues == 0:
        print("\nAll core system checks passed.")
        return 0
    else:
        print(f"\n{issues} check(s) failed.")
        return 1


def cmd_import_key(args):
    """Securely import OpenRouter API key from stdin."""
    print("Enter OpenRouter API key (input hidden or paste):")
    try:
        key = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1

    if not key.startswith("sk-or-"):
        print("Error: Invalid key format. Expected key starting with 'sk-or-'.")
        return 1

    # Save to user snap or local directory
    target_dir = os.environ.get("SNAP_USER_COMMON") or os.path.expanduser("~/.local/share/hal9000-inside")
    os.makedirs(target_dir, exist_ok=True, mode=0o700)
    config_file = os.path.join(target_dir, "openrouter.config")
    
    with open(config_file, "w") as f:
        f.write(f"OPENROUTER_API_KEY={key}\n")
    os.chmod(config_file, 0o600)
    print(f"Successfully saved API key to {config_file}")
    return 0


def cmd_memory(args):
    """Memory commands: inspect, remember, correct, forget, clear."""
    from hal9000_inside.memory import MemoryStore
    store = MemoryStore()

    if args.action == "inspect":
        memories = store.inspect_all()
        if not memories:
            print("No memories stored.")
        for m in memories:
            print(f"[{m['id'][:8]}] ({m['category']}) {m['text']}")
    elif args.action == "remember":
        if not args.text:
            print("Error: --text required for remember.")
            return 1
        mid = store.remember(args.text, category=args.category or "fact")
        print(f"Remembered [{mid[:8]}]: {args.text}")
    elif args.action == "correct":
        if not args.id or not args.text:
            print("Error: --id and --text required for correct.")
            return 1
        store.correct(args.id, args.text)
        print(f"Updated [{args.id[:8]}]: {args.text}")
    elif args.action == "forget":
        if not args.id:
            print("Error: --id required for forget.")
            return 1
        store.forget(args.id)
        print(f"Forgotten memory: {args.id}")
    elif args.action == "clear":
        store.clear()
        print("All conversational memory and dialogue cleared.")
    return 0


def cmd_run(args):
    """Run interactive voice assistant."""
    import atexit
    import termios
    import sys

    # Save terminal attributes at command start if connected to a tty
    orig_termios = None
    if sys.stdin.isatty():
        try:
            orig_termios = termios.tcgetattr(sys.stdin.fileno())
            def _restore_term():
                if orig_termios:
                    try:
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, orig_termios)
                    except Exception:
                        pass
            atexit.register(_restore_term)
        except Exception:
            pass

    from hal9000_inside.config import load_config
    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()

    from hal9000_inside.logging_config import setup_session_logging
    data_dir = os.path.expanduser(cfg.get("environment", {}).get("data_dir", "~/.local/share/hal9000-inside"))
    log_file = setup_session_logging(data_dir=data_dir)
    
    from hal9000_inside.assistant import Hal9000Assistant
    
    # Resolve GPU flags (CLI flag overrides config file)
    use_nvidia = None
    if getattr(args, "nvidia", False):
        use_nvidia = True
    elif getattr(args, "no_nvidia", False):
        use_nvidia = False

    use_intel = None
    if getattr(args, "intel", False):
        use_intel = True
    elif getattr(args, "no_intel", False):
        use_intel = False

    from hal9000_inside.audio import WavFileSource, WavFileSink

    input_wav_path = getattr(args, "input_wav", None) or cfg.get("cli", {}).get("input_wav")
    audio_source = None
    if input_wav_path:
        audio_source = WavFileSource(input_wav_path, real_time=False)

    output_wav_path = getattr(args, "output_wav", None) or cfg.get("cli", {}).get("output_wav")
    audio_sink = None
    if output_wav_path:
        audio_sink = WavFileSink(output_wav_path)

    auto_mode = None
    if getattr(args, "push_to_talk", False):
        auto_mode = False

    assistant = Hal9000Assistant(
        capture_device=args.device,
        playback_device=cfg.get("cli", {}).get("playback_device") or None,
        model_override=args.model,
        endpoint_override=getattr(args, "endpoint", None),
        auto_mode=auto_mode,
        use_nvidia=use_nvidia,
        use_intel=use_intel,
        audio_source=audio_source,
        audio_sink=audio_sink,
        sleep_after_minutes=getattr(args, "sleep_after", None),
        remote_deadline_seconds=getattr(args, "remote_deadline", None),
        config=cfg,
    )
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nHAL 9000 voice assistant stopped.")
    finally:
        if orig_termios and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, orig_termios)
            except Exception:
                pass
    return 0


def cmd_sleep(args):
    """Run sleep mode headless, no audio devices opened, then exit."""
    from hal9000_inside.config import load_config
    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()

    from hal9000_inside.cognition import SleepBudget, SleepManager
    from hal9000_inside.memory import MemoryStore, MultilingualE5Embedder
    from hal9000_inside.llm import LLMClient, LLMRouter
    from hal9000_inside.mcp import MCPClientManager
    from hal9000_inside.hardware import ComputeDeviceConfig
    from hal9000_inside.cognition import CuriosityEngine, PredictionEngine, InteroceptionMonitor

    cfg_cli_sleep = cfg.get("cli", {}).get("sleep", {})
    cfg_env = cfg.get("environment", {})
    cfg_files = cfg.get("files", {})
    cfg_sleep = cfg.get("engine", {}).get("sleep", {})

    duration_min = float(getattr(args, "duration", None) or cfg_cli_sleep.get("duration_minutes", 10.0))
    print(f"HAL 9000 entering headless sleep mode for {duration_min} minutes...")

    compute_cfg = ComputeDeviceConfig()
    embedder_providers = compute_cfg.get_onnx_providers(prefer_gpu=False)
    e5_dir = os.path.expanduser(cfg_files.get("e5_model_dir", "~/.cache/hal9000/models/multilingual-e5-small/onnx"))
    e5_model_file = os.path.join(e5_dir, "model.onnx") if os.path.isdir(e5_dir) else None
    try:
        embedder = MultilingualE5Embedder(model_path=e5_model_file, providers=embedder_providers)
    except Exception:
        embedder = None

    data_dir = getattr(args, "data_dir", None) or os.path.expanduser(cfg_env.get("data_dir", "~/.local/share/hal9000-inside"))
    from hal9000_inside.logging_config import setup_session_logging
    setup_session_logging(data_dir=data_dir)

    memory = MemoryStore(data_dir=data_dir, embedder=embedder)
    llm = LLMClient()
    router = LLMRouter(remote_client=llm if not getattr(llm, 'is_local', False) else None)
    mcp = MCPClientManager()
    curiosity = CuriosityEngine(memory_store=memory, mcp_client=mcp, llm_client=router)
    prediction = PredictionEngine(memory_store=memory)
    interoception = InteroceptionMonitor()

    budget = SleepBudget(
        max_minutes=duration_min,
        max_tokens=int(cfg_sleep.get("max_tokens", 10000)),
        max_web_fetches=int(cfg_sleep.get("max_web_fetches", 20)),
    )
    sleep_mgr = SleepManager(
        memory_store=memory,
        curiosity_engine=curiosity,
        prediction_engine=prediction,
        mcp_client=mcp,
        llm_router=router,
        interoception_monitor=interoception,
        budget=budget,
        data_dir=memory.data_dir,
    )

    summary = asyncio.run(sleep_mgr.run_sleep_cycle())
    print(f"Sleep complete: {summary.format_one_liner()}")
    return 0


def cmd_mcp(args):
    """Run HAL 9000 Model Context Protocol (MCP) server over stdio."""
    from hal9000_inside.mcp import run_stdio
    run_stdio()
    return 0


def main():
    parser = argparse.ArgumentParser(description="HAL 9000 Inside - Ubuntu Voice Assistant")
    parser.add_argument("--config", help="Custom path to config.toml (default: ~/.hal9000-inside/config.toml)")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # doctor
    subparsers.add_parser("doctor", help="Run system diagnostics")

    # import-key
    subparsers.add_parser("import-key", help="Securely import OpenRouter API key from stdin")

    # mcp
    subparsers.add_parser("mcp", help="Run Model Context Protocol (MCP) stdio server for memory & web retrieval")

    # memory
    mem_parser = subparsers.add_parser("memory", help="Manage conversational memory")
    mem_parser.add_argument("action", choices=["inspect", "remember", "correct", "forget", "clear"])
    mem_parser.add_argument("--id", help="Memory ID")
    mem_parser.add_argument("--text", help="Memory content")
    mem_parser.add_argument("--category", default="fact", help="Memory category")

    # sleep
    sleep_parser = subparsers.add_parser("sleep", help="Run sleep mode headless, no audio devices opened, then exit")
    sleep_parser.add_argument("--for", dest="duration", type=float, default=10.0, help="Duration in minutes to sleep")
    sleep_parser.add_argument("--data-dir", help="Custom data directory for SQLite and Qdrant")

    # run
    run_parser = subparsers.add_parser("run", help="Start voice assistant interactive loop")
    run_parser.add_argument("--push-to-talk", action="store_true", help="Enable push-to-talk instead of auto conversation mode")
    run_parser.add_argument("--device", help="ALSA capture device name (e.g. hw:CARD=Talk,DEV=0 or default)")
    run_parser.add_argument("--model", help="OpenAI-compatible model ID override (e.g. gemma4-e4b-ov)")
    run_parser.add_argument("--endpoint", help="OpenAI-compatible base URL override (e.g. http://127.0.0.1:8336/v3)")
    run_parser.add_argument("--input-wav", help="Path to input WAV file to drive session from file instead of microphone")
    run_parser.add_argument("--output-wav", help="Path to output WAV file to record assistant speech instead of speaker")
    run_parser.add_argument("--sleep-after", type=float, default=2.0, help="Auto-enter sleep mode after specified minutes of silence (default: 2.0 minutes)")
    run_parser.add_argument("--remote-deadline", type=float, default=8.0, help="Remote request deadline in seconds before local fallback (default: 8.0s)")

    # GPU options
    nvidia_group = run_parser.add_mutually_exclusive_group()
    nvidia_group.add_argument("--nvidia", dest="nvidia", action="store_true", default=None, help="Force enable NVIDIA GPU acceleration (CUDA/TensorRT)")
    nvidia_group.add_argument("--no-nvidia", dest="no_nvidia", action="store_true", default=None, help="Disable NVIDIA GPU acceleration")

    intel_group = run_parser.add_mutually_exclusive_group()
    intel_group.add_argument("--intel", dest="intel", action="store_true", default=None, help="Force enable Intel GPU acceleration")
    intel_group.add_argument("--no-intel", dest="no_intel", action="store_true", default=None, help="Disable Intel GPU acceleration")

    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "import-key":
        sys.exit(cmd_import_key(args))
    elif args.command == "mcp":
        sys.exit(cmd_mcp(args))
    elif args.command == "memory":
        sys.exit(cmd_memory(args))
    elif args.command == "sleep":
        sys.exit(cmd_sleep(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
