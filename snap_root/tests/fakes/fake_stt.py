"""Fake STT adapter for test isolation."""

import asyncio
from typing import AsyncGenerator, Callable, List, Optional
from myna.core import PcmChunk, SessionConfig, TranscriptionDone


class FakeSTT:
    """Scriptable STT adapter yielding configured transcripts for input audio."""

    def __init__(self, script: Optional[List[str]] = None, default_text: str = "Good afternoon, HAL."):
        self.script = list(script or [])
        self.default_text = default_text
        self.transcribed_calls: List[bytes] = []

    def set_script(self, utterances: List[str]):
        self.script = list(utterances)

    async def run_session(
        self,
        config: SessionConfig,
        audio_stream: AsyncGenerator[PcmChunk, None],
        on_emit: Callable,
    ):
        received_pcm = bytearray()
        async for chunk in audio_stream:
            if chunk.data:
                received_pcm.extend(chunk.data)

        self.transcribed_calls.append(bytes(received_pcm))
        text = self.script.pop(0) if self.script else self.default_text

        # Emit result
        res = on_emit(TranscriptionDone(text=text))
        if asyncio.iscoroutine(res):
            await res
