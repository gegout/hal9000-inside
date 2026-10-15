import asyncio
import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

# Prevent audio device contention & disable sleep during rapid campaign turns
os.environ["HAL9000_DISABLE_AUDIO"] = "1"
os.environ["OPENROUTER_TIMEOUT"] = "30"
os.environ["HAL_DISABLE_SLEEP"] = "1"

import campaigns.session_bank as sb
from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.config import load_config
from hal9000_inside.memory.store import MemoryStore
from tests.fakes.fake_stt import FakeSTT
from tests.fakes.fake_tts import FakeTTS

MONITORED_FILLERS = [
    "affirmative",
    "one moment",
    "let me search",
    "i would be pleased",
    "is there anything else",
    "certainly",
    "as an ai",
    "i am hal",
    "dave",
]

async def run_session(session_num: int):
    print(f"\n=======================================================")
    print(f"STARTING CAMPAIGN SESSION {session_num}")
    print(f"=======================================================\n")

    turns = sb.get_session(session_num)
    assert len(turns) >= 50, f"Session {session_num} has only {len(turns)} turns"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("campaigns/transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"session_{session_num}_{timestamp}.json"

    with tempfile.TemporaryDirectory() as tmp_data_dir:
        print(f"Using isolated data directory: {tmp_data_dir}")
        os.environ["HAL9000_DATA_DIR"] = tmp_data_dir

        config = load_config()
        if "environment" in config:
            config["environment"]["data_dir"] = tmp_data_dir

        isolated_memory = MemoryStore(data_dir=tmp_data_dir)
        assistant = Hal9000Assistant(config=config, memory=isolated_memory)
        fake_stt = FakeSTT()
        fake_tts = FakeTTS()
        assistant.adapter = fake_stt
        assistant.tts = fake_tts

        session_records = []
        filler_counts = {f: 0 for f in MONITORED_FILLERS}
        latencies = []

        try:
            await assistant.initialize()
            print("HAL9000 Assistant initialized successfully.")

            for turn_idx, user_input in enumerate(turns):
                t_start = time.perf_counter()
                turn_num = turn_idx + 1

                print(f"[{turn_num:02d}/50] USER: {user_input}")
                fake_stt.set_script([user_input])
                fake_tts.synthesized_texts.clear()

                # Process turn via real application orchestration
                await assistant._handle_turn(b"\x00" * 3200)

                duration = time.perf_counter() - t_start
                latencies.append(duration)

                # Get the assistant's last response from memory
                recent = assistant.memory.get_recent_turns(limit=2)
                assistant_response = ""
                for r in reversed(recent):
                    if r.get("role") == "assistant":
                        assistant_response = r.get("content", "")
                        break

                if not assistant_response and fake_tts.synthesized_texts:
                    assistant_response = " ".join(fake_tts.synthesized_texts)

                resp_lower = assistant_response.lower()
                detected_fillers = []
                for filler in MONITORED_FILLERS:
                    if filler in resp_lower:
                        filler_counts[filler] += 1
                        detected_fillers.append(filler)

                # Spoken audio verification
                spoken_texts = list(fake_tts.synthesized_texts)

                preview = assistant_response.replace("\n", " ")[:120]
                print(f"       HAL ({duration:.2f}s): {preview}...")
                if detected_fillers:
                    print(f"       [FILLERS: {detected_fillers}]")
                if spoken_texts:
                    print(f"       [TTS SYNTHESIZED: {len(spoken_texts)} chunks]")

                record = {
                    "turn": turn_num,
                    "user_input": user_input,
                    "user_words": len(user_input.split()),
                    "hal_response": assistant_response,
                    "hal_words": len(assistant_response.split()),
                    "duration_seconds": round(duration, 3),
                    "fillers_detected": detected_fillers,
                    "synthesized_chunks": spoken_texts,
                    "mood": getattr(assistant, "current_mood", "nominal")
                }
                session_records.append(record)

            # Metrics
            latencies_sorted = sorted(latencies)
            median_latency = latencies_sorted[len(latencies_sorted) // 2]
            p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            total_duration = sum(latencies)

            summary = {
                "session_num": session_num,
                "timestamp": timestamp,
                "total_turns": len(session_records),
                "total_duration_seconds": round(total_duration, 2),
                "median_latency_seconds": round(median_latency, 3),
                "p95_latency_seconds": round(p95_latency, 3),
                "filler_counts": filler_counts,
                "records": session_records
            }

            with open(out_file, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n=======================================================")
            print(f"SESSION {session_num} COMPLETED")
            print(f"Total turns: {len(session_records)}")
            print(f"Total time: {total_duration:.1f}s | Median turn: {median_latency:.2f}s | P95 turn: {p95_latency:.2f}s")
            print(f"Fillers detected: {filler_counts}")
            print(f"Transcript written to {out_file}")
            print(f"=======================================================")
            return summary

        finally:
            if hasattr(assistant, "mcp") and assistant.mcp:
                try:
                    await assistant.mcp.close()
                except Exception:
                    pass
            if hasattr(assistant, "router") and hasattr(assistant.router, "close"):
                try:
                    await assistant.router.close()
                except Exception:
                    pass

if __name__ == "__main__":
    s_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(run_session(s_num))
