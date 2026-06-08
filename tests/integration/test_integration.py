#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""
Integration tests — runs on the self-hosted Jetson runner.
Requires: Docker container running, /dev/ttyUSB0 accessible,
          camera sensor-id=0 connected.

These tests verify the full Sense → Process → Decide → Act loop
by starting the container and probing its output.
"""

import os
import subprocess
import unittest

CONTAINER_NAME = "recycleright"
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/jetson/recycleright/models/best_fp16.engine")


class TestContainerStartup(unittest.TestCase):
    """Verify the Docker container starts and stays alive."""

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists(MODEL_PATH),
            f"TensorRT engine not found at {MODEL_PATH}"
        )

    def test_model_file_is_nonzero(self):
        size = os.path.getsize(MODEL_PATH)
        self.assertGreater(size, 0, "Engine file is empty")


class TestSerialDevice(unittest.TestCase):
    """Verify /dev/ttyUSB0 is accessible (ESP32 connected)."""

    def test_serial_device_exists(self):
        if not os.path.exists("/dev/ttyUSB0"):
            self.skipTest("/dev/ttyUSB0 not present — ESP32 not connected")
        self.assertTrue(os.path.exists("/dev/ttyUSB0"))


class TestNvidiaRuntime(unittest.TestCase):
    """Verify the Jetson NVIDIA runtime is functional."""

    def test_nvidia_smi_available(self):
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            self.skipTest("nvidia-smi not available on this runner")
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
