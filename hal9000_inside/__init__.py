"""HAL 9000 Inside package."""

import os
import sys

# Ensure bundled myna STT engine is discoverable on python path
_myna_path = os.path.join(os.path.dirname(__file__), "stt")
if _myna_path not in sys.path:
    sys.path.insert(0, _myna_path)

__version__ = "0.4.0"

from .config import load_config, ensure_config_file, CONFIG_PATH

