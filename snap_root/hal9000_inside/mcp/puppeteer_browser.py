"""Puppeteer headless browser web fetcher and searcher for HAL 9000."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

RUNNER_SCRIPT = os.path.join(os.path.dirname(__file__), "puppeteer_runner.js")


class PuppeteerBrowser:
    """Manages headless browser operations using Puppeteer and Google Chrome."""

    def __init__(self):
        user_home = Path.home()
        self.node_path = (
            os.environ.get("NODE_BIN")
            or shutil.which("node")
            or str(user_home / ".local" / "bin" / "node")
        )
        self.chrome_path = (
            os.environ.get("CHROME_BIN")
            or shutil.which("google-chrome")
            or shutil.which("chromium")
            or shutil.which("chromium-browser")
            or "/usr/bin/google-chrome"
        )
        self.node_modules = (
            os.environ.get("NODE_PATH")
            or str(user_home / ".local" / "node_modules")
        )

    async def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch and extract readable text from a URL using headless Chrome."""
        return await self._run_action("fetch", url)

    async def _run_action(self, action: str, target: str, extra: Optional[str] = None) -> Dict[str, Any]:
        args = [self.node_path, RUNNER_SCRIPT, action, target]
        if extra:
            args.append(extra)

        env = os.environ.copy()
        env["NODE_PATH"] = self.node_modules
        env["CHROME_BIN"] = self.chrome_path

        def _exec():
            proc = subprocess.Popen(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = proc.communicate(timeout=40)
            return out, err

        loop = asyncio.get_running_loop()
        try:
            out, err = await loop.run_in_executor(None, _exec)
            for line in reversed(out.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)
            return {"status": "error", "error": f"Invalid output from puppeteer: {out} {err}"}
        except Exception as e:
            logger.error(f"Puppeteer browser execution failed: {e}")
            return {"status": "error", "error": str(e)}
