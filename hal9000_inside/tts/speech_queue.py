"""Ordered, non-overlapping audio queue for combined local and remote speech."""

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class OrderedSpeechQueue:
    """Manages ordered speech synthesis and playback:
    - Plays local quick-response first (if available and not suppressed).
    - Buffers remote speech during local playback so audio never overlaps.
    - If the remote answer is ready before local speech starts, suppresses the redundant local acknowledgment.
    - Handles immediate cancellation and discard of all buffered speech.
    """

    def __init__(
        self,
        tts: any,
        play_pcm_fn: Callable[[bytes], None],
        notify_start_fn: Callable[[], None],
        notify_stop_fn: Callable[[], None],
    ):
        self.tts = tts
        self.play_pcm_fn = play_pcm_fn
        self.notify_start_fn = notify_start_fn
        self.notify_stop_fn = notify_stop_fn

        self.is_playing = False
        self.local_started_playing = False
        self.local_finished_playing = False
        self.remote_ready = False
        self._cancelled = False
        self._lock = asyncio.Lock()

    def cancel(self):
        """Immediately cancel active playback and purge queues."""
        self._cancelled = True
        if hasattr(self.tts, "cancel"):
            self.tts.cancel()

    def reset(self):
        self._cancelled = False
        self.is_playing = False
        self.local_started_playing = False
        self.local_finished_playing = False
        self.remote_ready = False
        if hasattr(self.tts, "reset_cancellation"):
            self.tts.reset_cancellation()

    async def speak_local_ack(self, ack_text: str) -> bool:
        """Play local acknowledgement speech if not cancelled and remote answer hasn't already started."""
        if not ack_text or self._cancelled or self.remote_ready:
            return False

        async with self._lock:
            if self._cancelled or self.remote_ready:
                return False
            self.is_playing = True
            self.local_started_playing = True

        self.notify_start_fn()
        try:
            if hasattr(self.tts, "speak_sentences"):
                await self.tts.speak_sentences(
                    ack_text,
                    on_audio_chunk=lambda c: self.play_pcm_fn(c) if not self._cancelled else None,
                )
            else:
                loop = asyncio.get_running_loop()
                pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, ack_text)
                if not self._cancelled:
                    await loop.run_in_executor(None, self.play_pcm_fn, pcm)
            return True
        finally:
            async with self._lock:
                self.local_finished_playing = True
                self.is_playing = False
            self.notify_stop_fn()

    def mark_remote_ready(self):
        """Mark remote answer as ready. If local speech hasn't started yet, suppresses it."""
        self.remote_ready = True

    async def speak_substantive_remote(self, remote_text: str) -> bool:
        """Play substantive remote answer.
        If local speech is actively playing, waits for it to complete without overlapping.
        """
        self.mark_remote_ready()
        if not remote_text or self._cancelled:
            return False

        # If local playback is currently in progress, wait until it finishes
        while self.is_playing and not self._cancelled:
            await asyncio.sleep(0.02)

        if self._cancelled:
            return False

        async with self._lock:
            if self._cancelled:
                return False
            self.is_playing = True

        self.notify_start_fn()
        try:
            if hasattr(self.tts, "speak_sentences"):
                await self.tts.speak_sentences(
                    remote_text,
                    on_audio_chunk=lambda c: self.play_pcm_fn(c) if not self._cancelled else None,
                )
            else:
                loop = asyncio.get_running_loop()
                pcm = await loop.run_in_executor(None, self.tts.synthesize_raw_pcm, remote_text)
                if not self._cancelled:
                    await loop.run_in_executor(None, self.play_pcm_fn, pcm)
            return True
        finally:
            async with self._lock:
                self.is_playing = False
            self.notify_stop_fn()
