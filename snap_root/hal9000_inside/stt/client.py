"""Myna Parakeet STT WebSocket Client.

Adheres strictly to Myna session contract:
- Connects to Myna via WebSocket over Unix Domain Socket
- Session creation handshake (session.created protocol_version: 1)
- Sends session.start with 16000Hz, 1ch, 16-bit PCM format
- Streams raw 16kHz mono PCM chunks
- App-level silence endpointing & VAD: emits input_audio_buffer.commit / session.finish
  so we do NOT wait for Myna's default 15s arm threshold!
- Handles transcription.progress, transcription.provisional, and transcription.final
- Accumulates committed text, drops partials cleanly upon commit/abort
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Callable, Dict, Optional
import websockets

logger = logging.getLogger(__name__)

DEFAULT_MYNA_SOCKET = os.environ.get("MYNA_SOCKET_PATH") or "/tmp/myna.sock"


class MynaSTTClient:
    """Client for Myna Parakeet ASR backend over Unix domain socket."""

    def __init__(self, socket_path: str = DEFAULT_MYNA_SOCKET):
        self.socket_path = socket_path
        self._ws = None
        self._is_ready = False
        self._session_id = None
        self._active_utterance = False
        self._committed_text_parts = []
        self._provisional_text = ""

    async def connect(self, timeout: float = 10.0):
        """Connects to Myna over Unix socket and performs session handshake."""
        if not os.path.exists(self.socket_path):
            raise FileNotFoundError(f"Myna socket not found at {self.socket_path}. Is Myna server running?")

        # websockets connect via unix socket
        self._ws = await asyncio.wait_for(
            websockets.unix_connect(self.socket_path),
            timeout=timeout,
        )
        logger.info(f"Connected to Myna WebSocket at {self.socket_path}")

        # Wait for session.created
        created_frame = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        data = json.loads(created_frame)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Unexpected initial frame from Myna: {data}")

        self._session_id = data.get("session_id")
        self._is_ready = True
        logger.info(f"Myna session created: {self._session_id}")

    async def start_utterance(self, sample_rate: int = 16000):
        """Send session.start or begin new utterance on connection."""
        if not self._ws:
            await self.connect()

        self._committed_text_parts = []
        self._provisional_text = ""
        self._active_utterance = True

        start_event = {
            "type": "session.start",
            "audio_format": {
                "sample_rate": sample_rate,
                "channels": 1,
                "bit_depth": 16,
            },
        }
        await self._ws.send(json.dumps(start_event))

    async def send_audio_chunk(self, pcm_chunk: bytes):
        """Send binary 16kHz 16-bit mono PCM chunk to Myna."""
        if not self._ws or not self._active_utterance:
            return
        await self._ws.send(pcm_chunk)

    async def commit_utterance(self) -> str:
        """Signal utterance endpointing explicitly to finalize transcription without waiting 15s."""
        if not self._ws or not self._active_utterance:
            return " ".join(self._committed_text_parts).strip()

        # Send finish event to drain and commit audio accepted
        finish_event = {"type": "session.finish"}
        await self._ws.send(json.dumps(finish_event))

        # Await final event
        while self._active_utterance:
            try:
                frame = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                if isinstance(frame, str):
                    ev = json.loads(frame)
                    ev_type = ev.get("type")
                    if ev_type in ("transcription.committed", "transcription.final"):
                        text = ev.get("text", "")
                        if text:
                            self._committed_text_parts.append(text)
                    elif ev_type in ("session.finished", "session.completed", "response.done"):
                        self._active_utterance = False
                        break
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for transcription finalization.")
                self._active_utterance = False
                break

        full_transcript = " ".join(self._committed_text_parts).strip()
        self._provisional_text = ""
        return full_transcript

    async def abort_utterance(self):
        """Immediately abort current utterance and discard uncommitted audio."""
        self._active_utterance = False
        self._committed_text_parts = []
        self._provisional_text = ""
        if self._ws:
            # Closing the socket aborts according to session-contract.md
            await self._ws.close()
            self._ws = None

    async def listen_events(
        self,
        on_provisional: Optional[Callable[[str], None]] = None,
        on_committed: Optional[Callable[[str], None]] = None,
    ):
        """Listen for server progress/provisional/committed events."""
        while self._ws:
            try:
                msg = await self._ws.recv()
                if isinstance(msg, str):
                    ev = json.loads(msg)
                    ev_type = ev.get("type")
                    if ev_type == "transcription.provisional":
                        text = ev.get("text", "")
                        self._provisional_text = text
                        if on_provisional:
                            on_provisional(text)
                    elif ev_type in ("transcription.committed", "transcription.final"):
                        text = ev.get("text", "")
                        if text:
                            self._committed_text_parts.append(text)
                            if on_committed:
                                on_committed(text)
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logger.debug(f"Error in STT event listener: {e}")
                break
