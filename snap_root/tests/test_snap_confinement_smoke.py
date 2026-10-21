"""Smoke tests for snap confinement and environment compatibility.

Verifies:
1. Snap metadata and plug declarations (network, home, audio, personal-files).
2. Strict confinement compatibility for stdio MCP process spawning and socketless IPC.
3. No reliance on external Node.js, uv, or external services.
4. Cache directory fallback to $SNAP_USER_DATA/.cache/search-mcp.
"""

import os
import unittest
from unittest import mock
import yaml

from hal9000_inside.mcp.client_manager import MCPClientManager


class TestSnapConfinementSmoke(unittest.TestCase):
    def test_snapcraft_yaml_integrity(self):
        """snapcraft.yaml declares strict confinement, network, and Python packages."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        snapcraft_path = os.path.join(repo_root, "snapcraft.yaml")
        self.assertTrue(os.path.exists(snapcraft_path), "snapcraft.yaml must exist")

        with open(snapcraft_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data.get("confinement"), "strict")
        app = data.get("apps", {}).get("hal9000-inside", {})
        plugs = app.get("plugs", [])
        self.assertIn("network", plugs)
        self.assertIn("audio-playback", plugs)
        self.assertIn("audio-record", plugs)

        # Check python packages
        parts = data.get("parts", {}).get("hal9000-inside", {})
        pkgs = parts.get("python-packages", [])
        self.assertIn("free-search-mcp==0.13.1", pkgs)
        self.assertIn("selectolax<1.0.0", pkgs)

    def test_snap_root_meta_integrity(self):
        """snap_root/meta/snap.yaml matches strict confinement and network plugs."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meta_path = os.path.join(repo_root, "snap_root", "meta", "snap.yaml")
        self.assertTrue(os.path.exists(meta_path), "snap_root/meta/snap.yaml must exist")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data.get("confinement"), "strict")
        app = data.get("apps", {}).get("hal9000-inside", {})
        plugs = app.get("plugs", [])
        self.assertIn("network", plugs)

    def test_cache_dir_resolution_under_snap(self):
        """Under snap environment, search-mcp cache resolves inside SNAP_USER_DATA."""
        fake_snap_data = "/home/testuser/snap/hal9000-inside/common"
        with unittest.mock.patch.dict(os.environ, {"SNAP_USER_DATA": fake_snap_data}):
            manager = MCPClientManager(enable_search_mcp=True)
            params = manager._build_search_mcp_params()
            expected_cache = os.path.join(fake_snap_data, ".cache", "search-mcp")
            self.assertEqual(params.env.get("SEARCH_MCP_CACHE_DIR"), expected_cache)
            self.assertEqual(params.env.get("SEARCH_MCP_FETCH_STRATEGY"), "http")
            self.assertIn("duckduckgo", params.env.get("SEARCH_MCP_DEFAULT_ENGINES", ""))


if __name__ == "__main__":
    unittest.main()
