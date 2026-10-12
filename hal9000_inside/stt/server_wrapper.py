"""Myna Parakeet Server In-Process / Daemon Runner.

Spawns the pinned Myna Parakeet TDT 0.6B v3 backend using Myna's core server
and UNIX socket transport (`transport_ws.unix_serve`).
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PARAKEET_DEFAULT_DIR = Path.home() / ".cache" / "myna" / "models" / "parakeet-tdt-0.6b-v3-int8"

class MynaServerDaemon:
    """Manages the lifecycle of an embedded or local Myna Parakeet server instance."""

    def __init__(
        self,
        socket_path: str = "/tmp/myna.sock",
        model_dir: str | None = None,
        device: str = "cpu",
        stream_arm_s: float = 0.5,
        stream_silence_cut_s: float = 0.3,
    ):
        self.socket_path = socket_path
        self.model_dir = model_dir or str(PARAKEET_DEFAULT_DIR)
        self.device = device
        self.stream_arm_s = stream_arm_s
        self.stream_silence_cut_s = stream_silence_cut_s
        self._server = None
        self._adapter = None

    async def start(self):
        """Initialise Myna ParakeetAdapter and bind to Unix socket."""
        # Clean up stale socket if present
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        # Dynamically import Myna server and adapter
        from myna.testbed.parakeet import ParakeetAdapter
        from myna.core import serve_unix
        from myna.server.lifecycle import LifecycleService

        logger.info(f"Loading Myna ParakeetAdapter from {self.model_dir} (device={self.device})...")
        self._adapter = ParakeetAdapter(
            model_dir=self.model_dir,
            device=self.device,
            streaming=True,
            stream_arm_s=self.stream_arm_s,
            stream_silence_cut_s=self.stream_silence_cut_s,
        )

        lifecycle = LifecycleService(self._adapter)
        logger.info(f"Starting Myna WebSocket server on {self.socket_path}...")
        self._server = await serve_unix(lifecycle, self.socket_path)
        logger.info("Myna server started and listening.")

    async def stop(self):
        """Stops Myna server and removes unix socket."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
