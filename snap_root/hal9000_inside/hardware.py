"""Hardware and GPU acceleration detection and configuration.

Supports enabling/disabling:
- NVIDIA GPUs (CUDA / TensorRT)
- Intel GPUs (OpenVINO / Level-Zero / OpenCL)
"""

import logging
import os
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def detect_system_gpus() -> Dict[str, Any]:
    """Detect available NVIDIA and Intel GPU hardware on the system."""
    has_nvidia = False
    has_intel = False
    nvidia_name = None
    intel_name = None

    # 1. Check via lspci
    try:
        out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            line_l = line.lower()
            if any(term in line_l for term in ("vga compatible", "3d controller", "display controller")):
                if "nvidia" in line_l:
                    has_nvidia = True
                    nvidia_name = line.split(":", 2)[-1].strip()
                elif "intel" in line_l:
                    has_intel = True
                    intel_name = line.split(":", 2)[-1].strip()
    except Exception:
        pass

    # 2. Check via /dev/dri render nodes
    dri_nodes = []
    if os.path.isdir("/dev/dri"):
        try:
            dri_nodes = os.listdir("/dev/dri")
        except Exception:
            pass

    # 3. Check nvidia-smi
    nvidia_driver_ok = False
    try:
        subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        nvidia_driver_ok = True
    except Exception:
        pass

    return {
        "nvidia": {
            "hardware_present": has_nvidia,
            "driver_ready": nvidia_driver_ok,
            "device_name": nvidia_name or ("NVIDIA GPU" if has_nvidia else None),
        },
        "intel": {
            "hardware_present": has_intel,
            "device_name": intel_name or ("Intel GPU" if has_intel else None),
            "dri_present": any("render" in n for n in dri_nodes),
        },
    }


class ComputeDeviceConfig:
    """Manages GPU enablement flags and execution provider selections."""

    def __init__(
        self,
        use_nvidia: Optional[bool] = None,
        use_intel: Optional[bool] = None,
    ):
        """
        Args:
            use_nvidia: True to force enable, False to disable, None for auto-detect.
            use_intel: True to force enable, False to disable, None for auto-detect.
        """
        system_info = detect_system_gpus()
        self.system_info = system_info

        # NVIDIA resolution
        if use_nvidia is None:
            self.enable_nvidia = system_info["nvidia"]["hardware_present"] and system_info["nvidia"]["driver_ready"]
        else:
            self.enable_nvidia = bool(use_nvidia)

        # Intel resolution
        if use_intel is None:
            self.enable_intel = system_info["intel"]["hardware_present"] and system_info["intel"]["dri_present"]
        else:
            self.enable_intel = bool(use_intel)

        self._configure_environment()

    @property
    def has_cuda(self) -> bool:
        """Check if CUDA runtime provider is available in onnxruntime."""
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def _configure_environment(self):

        """Apply environment restrictions according to configured flags."""
        if not self.enable_nvidia:
            # Mask NVIDIA devices
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("NVIDIA GPU acceleration disabled by configuration.")
        else:
            # Ensure visible
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            logger.info("NVIDIA GPU acceleration enabled.")

        if not self.enable_intel:
            # Mask OpenVINO / Level-Zero Intel GPU
            os.environ["ZE_AFFINITY_MASK"] = ""
            logger.info("Intel GPU acceleration disabled by configuration.")
        else:
            if os.environ.get("ZE_AFFINITY_MASK") == "":
                os.environ.pop("ZE_AFFINITY_MASK", None)
            logger.info("Intel GPU acceleration enabled.")

    def get_onnx_providers(self, prefer_gpu: bool = True):
        """Return ordered list of ONNX Runtime Execution Providers based on device config."""
        import onnxruntime as ort
        available = ort.get_available_providers()

        providers = []
        if self.enable_nvidia and prefer_gpu:
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")

        # OpenVINO execution provider if available in onnxruntime
        if self.enable_intel and prefer_gpu:
            if "OpenVINOExecutionProvider" in available:
                providers.append("OpenVINOExecutionProvider")

        # Always fallback to CPU
        providers.append("CPUExecutionProvider")
        return providers

    def summary(self) -> str:
        """User-friendly summary of compute device configuration."""
        nv_stat = "ENABLED" if self.enable_nvidia else "DISABLED"
        nv_dev = self.system_info["nvidia"]["device_name"] or "None"
        intel_stat = "ENABLED" if self.enable_intel else "DISABLED"
        intel_dev = self.system_info["intel"]["device_name"] or "None"

        return (
            f"Compute Acceleration:\n"
            f"  - NVIDIA GPU: [{nv_stat}] ({nv_dev})\n"
            f"  - Intel GPU:  [{intel_stat}] ({intel_dev})"
        )
