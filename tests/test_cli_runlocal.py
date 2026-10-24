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
