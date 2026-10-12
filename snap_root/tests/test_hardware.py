import unittest
import os
from hal9000_inside.hardware import detect_system_gpus, ComputeDeviceConfig

class TestHardwareGPU(unittest.TestCase):
    def test_detect_system_gpus(self):
        info = detect_system_gpus()
        self.assertIn("nvidia", info)
        self.assertIn("intel", info)
        self.assertIn("hardware_present", info["nvidia"])
        self.assertIn("hardware_present", info["intel"])

    def test_compute_config_explicit_disable(self):
        cfg = ComputeDeviceConfig(use_nvidia=False, use_intel=False)
        self.assertFalse(cfg.enable_nvidia)
        self.assertFalse(cfg.enable_intel)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertEqual(os.environ.get("ZE_AFFINITY_MASK"), "")
        providers = cfg.get_onnx_providers()
        self.assertEqual(providers, ["CPUExecutionProvider"])

    def test_compute_config_explicit_nvidia(self):
        cfg = ComputeDeviceConfig(use_nvidia=True, use_intel=False)
        self.assertTrue(cfg.enable_nvidia)
        self.assertFalse(cfg.enable_intel)
        self.assertNotEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

if __name__ == "__main__":
    unittest.main()
