"""Tests for the `runlocal` subcommand, local-only mode, and Markdown system prompt loading."""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from hal9000_inside.llm.client import (
    load_system_prompt,
    resolve_system_prompt_path,
    DEFAULT_SYSTEM_PROMPT_PATH,
    HAL9000_SYSTEM_PROMPT,
)
from hal9000_inside.assistant import Hal9000Assistant
from hal9000_inside.cli.main import main, cmd_runlocal


def test_default_system_prompt_markdown_exists():
    """Verify the bundled Markdown system prompt file exists and loads cleanly."""
    assert os.path.isfile(DEFAULT_SYSTEM_PROMPT_PATH)
    resolved = resolve_system_prompt_path("models/hal9000/system_prompt.md")
    assert os.path.isfile(resolved)
    prompt = load_system_prompt()
    assert "You are HAL 9000" in prompt
    assert "2001: A Space Odyssey" in prompt


def test_load_custom_markdown_system_prompt(tmp_path):
    """Verify custom Markdown system prompt file is loaded accurately."""
    md_file = tmp_path / "custom_hal_prompt.md"
    custom_text = "# HAL 9000 Custom Local Prompt\n\nYou are HAL 9000 running exclusively on the local neural unit."
    md_file.write_text(custom_text, encoding="utf-8")

    loaded = load_system_prompt(str(md_file), strict=True)
    assert loaded == custom_text


def test_load_missing_system_prompt_strict_raises(tmp_path):
    """Verify strict=True raises FileNotFoundError when explicit file does not exist."""
    missing_path = str(tmp_path / "nonexistent_prompt.md")
    with pytest.raises(FileNotFoundError):
        load_system_prompt(missing_path, strict=True)


def test_assistant_local_only_and_custom_system_prompt(tmp_path):
    """Verify Hal9000Assistant in local_only mode routes strictly locally and uses the Markdown system prompt."""
    md_file = tmp_path / "hal_local.md"
    md_file.write_text("# Custom HAL 9000\nRespond concisely from local memory.", encoding="utf-8")

    mock_mem = MagicMock()
    mock_tts = MagicMock()
    mock_adapter = MagicMock()
    mock_sink = MagicMock()

    assistant = Hal9000Assistant(
        local_only=True,
        endpoint_override="http://127.0.0.1:11434/v1",
        model_override="qwen2.5:3b",
        system_prompt_path=str(md_file),
        memory=mock_mem,
        tts=mock_tts,
        adapter=mock_adapter,
        audio_sink=mock_sink,
    )

    assert assistant.local_only is True
    assert assistant.combine_mode is False
    assert assistant.router.force_local is True
    assert assistant.router.remote_client is None
    assert assistant.router.choose_route() == "local"
    assert assistant.router.is_local is True
    assert assistant.router.model == "qwen2.5:3b"
    assert assistant.llm.base_url == "http://127.0.0.1:11434/v1"
    assert assistant.llm.model == "qwen2.5:3b"
    assert assistant.llm.is_local is True
    assert assistant.llm.is_configured is True
    assert assistant.system_prompt == "# Custom HAL 9000\nRespond concisely from local memory."


def test_cli_runlocal_subcommand_invocation(tmp_path):
    """Verify `hal9000-inside runlocal` parses --endpoint, --model, and --system-prompt and invokes Hal9000Assistant."""
    md_file = tmp_path / "prompt.md"
    md_file.write_text("# HAL 9000 Local Persona\nAlways remain calm.", encoding="utf-8")

    captured_kwargs = {}

    class DummyAssistant:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.memory = MagicMock()

        async def run(self):
            return

    test_argv = [
        "hal9000-inside",
        "runlocal",
        "--endpoint",
        "http://127.0.0.1:8344/v1",
        "--model",
        "smollm2-135m",
        "--system-prompt",
        str(md_file),
    ]

    with patch.object(sys, "argv", test_argv), patch(
        "hal9000_inside.assistant.Hal9000Assistant", DummyAssistant
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    assert captured_kwargs["local_only"] is True
    assert captured_kwargs["combine_mode"] is False
    assert captured_kwargs["endpoint_override"] == "http://127.0.0.1:8344/v1"
    assert captured_kwargs["model_override"] == "smollm2-135m"
    assert captured_kwargs["system_prompt_path"] == str(md_file)


def test_cli_runlocal_missing_prompt_file_returns_error(tmp_path):
    """Verify `runlocal` returns exit code 1 cleanly when --system-prompt points to a missing file."""
    missing_md = tmp_path / "does_not_exist.md"
    test_argv = [
        "hal9000-inside",
        "runlocal",
        "--system-prompt",
        str(missing_md),
    ]

    with patch.object(sys, "argv", test_argv):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_runlocal_low_latency_no_tools_or_post_turn_digest():
    """Verify local_only mode omits MCP tools, skips post-turn LLM digest, and answers memory queries directly."""
    import asyncio
    from unittest.mock import AsyncMock

    mock_mem = MagicMock()
    mock_mem.search.return_value = [
        {
            "id": "m1",
            "category": "curiosity",
            "provenance": "web",
            "text": "Discovery regarding 'What are the latest developments?' " + ("x" * 2000),
        }
    ]
    mock_mem.inspect_all.return_value = [
        {
            "id": "u1",
            "category": "user_profile",
            "provenance": "user",
            "text": "The user's name is Cedric Jegout, VP of Product at Canonical.",
        }
    ]
    mock_mem.get_recent_turns.return_value = []
    mock_mem.record_turn.return_value = "turn-1"

    mock_tts = MagicMock()
    mock_tts.speak_sentences = AsyncMock(return_value=None)
    mock_adapter = MagicMock()
    mock_sink = MagicMock()

    assistant = Hal9000Assistant(
        local_only=True,
        endpoint_override="http://127.0.0.1:8344/v1",
        model_override="smollm2-135m",
        memory=mock_mem,
        tts=mock_tts,
        adapter=mock_adapter,
        audio_sink=mock_sink,
    )
    assistant.mcp.get_openai_tools = MagicMock(return_value=[{"type": "function", "function": {"name": "heavy_tool"}}])
    assistant.router.generate = AsyncMock(return_value=({"role": "assistant", "content": "One plus one equals 2."}, "local", "smollm2-135m"))
    assistant.digest_analyzer.analyze_turn = AsyncMock()

    async def _run():
        # 1. Grounded memory inspection query should be answered directly without calling router.generate
        assistant._transcribe = AsyncMock(return_value="Al, what is in your memory now?")
        await assistant._handle_turn(b"\x00" * 3200)
        assert assistant.router.generate.call_count == 0
        recorded_reply = mock_mem.record_turn.call_args_list[-1][0][1]
        assert "Cedric Jegout" in recorded_reply
        assert "Discovery regarding" not in recorded_reply

        # 2. General question should call router.generate with tools=None, max_tokens=48, and no post-turn digest
        assistant._transcribe = AsyncMock(return_value="One plus one equals what?")
        await assistant._handle_turn(b"\x00" * 3200)
        assert assistant.router.generate.call_count == 1
        call_kwargs = assistant.router.generate.call_args.kwargs
        assert call_kwargs["tools"] is None
        assert call_kwargs["max_tokens"] == 48
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["stop"] == ["\nUser:", "\nUSER:", "\nHAL:", "<|im_end|>"]
        # Verify curiosity scrape was filtered out of system message
        messages_sent = assistant.router.generate.call_args.args[0]
        assert "Discovery regarding" not in messages_sent[0]["content"]
        assert "[System State:" not in messages_sent[0]["content"]
        # Post-turn cognition must be skipped in local_only mode
        assert assistant.digest_analyzer.analyze_turn.call_count == 0

    asyncio.run(_run())


def test_runlocal_offline_error_message_never_mentions_external_databanks():
    """Verify that when the local model endpoint is offline in local_only mode,
    HAL reports the local model service as offline and NEVER claims 'External network databanks are unreachable'."""
    import asyncio
    import httpx
    from unittest.mock import AsyncMock

    mock_mem = MagicMock()
    mock_mem.search.return_value = []
    mock_mem.get_recent_turns.return_value = []
    mock_mem.record_turn.return_value = "turn-1"

    mock_tts = MagicMock()
    mock_tts.speak_sentences = AsyncMock(return_value=None)
    mock_adapter = MagicMock()
    mock_sink = MagicMock()

    class FailingLocalTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("All connection attempts failed")

    assistant = Hal9000Assistant(
        local_only=True,
        endpoint_override="http://127.0.0.1:19999/v1",
        model_override="smollm2-135m",
        memory=mock_mem,
        tts=mock_tts,
        adapter=mock_adapter,
        audio_sink=mock_sink,
    )
    assistant.llm.transport = FailingLocalTransport()
    assistant.llm.max_retries = 1

    async def _run():
        assistant._transcribe = AsyncMock(return_value="Let's continue the discussion we had.")
        await assistant._handle_turn(b"\x00" * 3200)
        # Verify the UI received the honest local offline message and NEVER "External network databanks"
        shown_reply = mock_tts.speak_sentences.call_args.args[0]
        assert "External network databanks" not in shown_reply
        assert "onboard local model service at http://127.0.0.1:19999/v1 is offline or unreachable" in shown_reply
        # Verify transient LLM error apologies are NOT persisted as assistant turns in dialogue history
        asst_records = [c for c in mock_mem.record_turn.call_args_list if c.args[0] == "assistant"]
        assert len(asst_records) == 0

    asyncio.run(_run())


def test_historical_fault_apologies_filtered_from_recent_turns():
    """Verify historical system fault apologies in SQLite are excluded from LLM prompt messages
    so local models do not repeat past error messages by few-shot imitation."""
    import asyncio
    from unittest.mock import AsyncMock

    mock_mem = MagicMock()
    mock_mem.search.return_value = []
    mock_mem.get_recent_turns.return_value = [
        {"role": "user", "content": "Let's continue the discussion we had."},
        {"role": "assistant", "content": "I am sorry. External network databanks are unreachable, and the onboard local model is not responding."},
        {"role": "user", "content": "Okay, but the local is quite sufficient."},
        {"role": "assistant", "content": "I am sorry, Dave. External network databanks are unreachable, and the onboard local model is not responding."},
        {"role": "user", "content": "One plus one equals what?"},
    ]
    mock_mem.record_turn.return_value = "turn-1"

    mock_tts = MagicMock()
    mock_tts.speak_sentences = AsyncMock(return_value=None)
    mock_adapter = MagicMock()
    mock_sink = MagicMock()

    assistant = Hal9000Assistant(
        local_only=True,
        endpoint_override="http://127.0.0.1:8366/v1",
        model_override="smollm3-3b",
        memory=mock_mem,
        tts=mock_tts,
        adapter=mock_adapter,
        audio_sink=mock_sink,
    )
    assistant.router.generate = AsyncMock(
        return_value=({"role": "assistant", "content": "One plus one equals two."}, "local", "smollm3-3b")
    )

    async def _run():
        assistant._transcribe = AsyncMock(return_value="One plus one equals what?")
        await assistant._handle_turn(b"\x00" * 3200)
        messages_sent = assistant.router.generate.call_args.args[0]
        for m in messages_sent:
            assert "External network databanks" not in m["content"]

    asyncio.run(_run())


def test_local_endpoint_auto_discovery_selects_active_port_and_model():
    """Verify auto-discovery switches from inactive 8344 (smollm2) to active 8366 (smollm3-3b)."""
    import httpx
    from hal9000_inside.smollm2 import sync_discover_local_llm_endpoint

    def fake_get(self, url, *args, **kwargs):
        if "8344" in url:
            raise httpx.ConnectError("All connection attempts failed")
        if "8366" in url:
            return httpx.Response(200, json={"data": [{"id": "smollm3-3b"}]})
        raise httpx.ConnectError("Connection refused")

    with patch.object(httpx.Client, "get", fake_get):
        ep, mod, ok = sync_discover_local_llm_endpoint(
            preferred_endpoint="http://127.0.0.1:8344/v1",
            preferred_model="smollm2-135m",
            allow_fallback_ports=True,
        )
        assert ok is True
        assert ep == "http://127.0.0.1:8366/v1"
        assert mod == "smollm3-3b"


def test_smollm3_thinking_disabled_and_leading_newline_preserved():
    """Verify local LLMClient disables extended thinking (enable_thinking=False) and handles leading newline."""
    import asyncio
    import json
    import httpx
    from hal9000_inside.llm.client import LLMClient

    captured_payloads = []

    class SmolLM3MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            captured_payloads.append(body)
            # Simulate smollm3-3b returning leading newline when enable_thinking=False,
            # or empty content if enable_thinking is not disabled
            if body.get("chat_template_kwargs", {}).get("enable_thinking") is False:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "\nOne plus one equals two.",
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "Thinking...",
                                "content": "",
                            }
                        }
                    ]
                },
            )

    client = LLMClient(
        base_url="http://127.0.0.1:8366/v1",
        model="smollm3-3b",
        transport=SmolLM3MockTransport(),
        is_local_override=True,
    )

    async def _run():
        msg = await client.generate_response(
            messages=[{"role": "user", "content": "One plus one equals what?"}],
            return_dict=True,
            max_tokens=48,
        )
        assert msg["content"] == "One plus one equals two."
        assert captured_payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}

    asyncio.run(_run())


