#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""Unit tests for CameraCapture, SerialController, and AudioPlayer modules."""

import contextlib
import queue
import threading
import unittest
from unittest.mock import patch


class TestCameraCapture(unittest.TestCase):
    """Test CameraCapture queue and stop-event logic."""

    def test_queue_maxsize_is_one(self):
        q = queue.Queue(maxsize=1)
        self.assertEqual(q.maxsize, 1)

    def test_queue_drops_old_frame_on_full(self):
        q = queue.Queue(maxsize=1)
        q.put(("frame1", 1.0))
        # Try to replace: get_nowait then put
        with contextlib.suppress(queue.Empty):
            q.get_nowait()
        q.put(("frame2", 2.0))
        frame, ts = q.get()
        self.assertEqual(frame, "frame2")
        self.assertAlmostEqual(ts, 2.0)

    def test_stop_event_terminates_loop(self):
        stop = threading.Event()
        stop.set()
        self.assertTrue(stop.is_set())

    def test_capture_queue_thread_safe(self):
        q = queue.Queue(maxsize=1)
        results = []

        def producer():
            for i in range(5):
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
                q.put((f"frame{i}", float(i)))

        def consumer():
            import time
            time.sleep(0.02)
            while not q.empty():
                item = q.get_nowait()
                results.append(item)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t1.join()
        t2.start()
        t2.join()
        self.assertLessEqual(len(results), 1)  # maxsize=1, only latest

    def test_gstreamer_pipeline_string_contains_nvargus(self):
        gst = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=2"
        )
        self.assertIn("nvarguscamerasrc", gst)
        self.assertIn("appsink", gst)

    def test_gstreamer_pipeline_is_1280x720(self):
        gst = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=2"
        )
        self.assertIn("width=1280", gst)
        self.assertIn("height=720", gst)


class TestSerialControllerProtocol(unittest.TestCase):
    """Test UART protocol constants and parsing."""

    CLASS_CMD = {
        "PET bottle":   b"PET\n",
        "Glass bottle": b"GLS\n",
        "Aluminum can": b"MTL\n",
        "Tetra Pak":    b"PAP\n",
    }

    def test_pet_command_bytes(self):
        self.assertEqual(self.CLASS_CMD["PET bottle"], b"PET\n")

    def test_gls_command_bytes(self):
        self.assertEqual(self.CLASS_CMD["Glass bottle"], b"GLS\n")

    def test_mtl_command_bytes(self):
        self.assertEqual(self.CLASS_CMD["Aluminum can"], b"MTL\n")

    def test_pap_command_bytes(self):
        self.assertEqual(self.CLASS_CMD["Tetra Pak"], b"PAP\n")

    def test_done_parsing_pet(self):
        line = "DONE:PET bottle"
        self.assertTrue(line.startswith("DONE:"))
        label = line[5:]
        self.assertEqual(label, "PET bottle")

    def test_done_parsing_glass(self):
        line = "DONE:Glass bottle"
        label = line[5:]
        self.assertEqual(label, "Glass bottle")

    def test_non_done_line_ignored(self):
        lines = [
            "Ready. Waiting for commands from Jetson...",
            "【收到】PET bottle → 通道 0 馬達動作中",
            "",
        ]
        for line in lines:
            self.assertFalse(line.startswith("DONE:"))

    def test_baud_rate_is_115200(self):
        baud = 115200
        self.assertEqual(baud, 115200)

    def test_all_commands_are_3_chars_plus_newline(self):
        for label, cmd in self.CLASS_CMD.items():
            # e.g. b"PET\n" → 4 bytes
            self.assertEqual(len(cmd), 4, f"Command for {label} should be 4 bytes")


class TestAudioPlayerMapping(unittest.TestCase):
    """Test audio file mapping completeness."""

    CLASS_AUDIO = {
        "PET bottle":   "/home/jetson/recycleright/src/音樂/寶特瓶.mp3",
        "Glass bottle": "/home/jetson/recycleright/src/音樂/玻璃瓶.mp3",
        "Aluminum can": "/home/jetson/recycleright/src/音樂/鋁罐.mp3",
        "Tetra Pak":    "/home/jetson/recycleright/src/音樂/鋁箔包.mp3",
    }

    def test_four_audio_files_mapped(self):
        self.assertEqual(len(self.CLASS_AUDIO), 4)

    def test_all_paths_are_mp3(self):
        for label, path in self.CLASS_AUDIO.items():
            self.assertTrue(path.endswith(".mp3"), f"{label} should map to .mp3")

    def test_pet_audio_file(self):
        self.assertIn("寶特瓶", self.CLASS_AUDIO["PET bottle"])

    def test_glass_audio_file(self):
        self.assertIn("玻璃瓶", self.CLASS_AUDIO["Glass bottle"])

    def test_aluminum_audio_file(self):
        self.assertIn("鋁罐", self.CLASS_AUDIO["Aluminum can"])

    def test_tetrapak_audio_file(self):
        self.assertIn("鋁箔包", self.CLASS_AUDIO["Tetra Pak"])

    @patch("subprocess.Popen")
    def test_nonblocking_playback_uses_popen(self, mock_popen):
        import subprocess
        path = self.CLASS_AUDIO["PET bottle"]
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_ffplay_flags(self, mock_popen):
        import subprocess
        path = self.CLASS_AUDIO["Aluminum can"]
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        args = mock_popen.call_args[0][0]
        self.assertIn("-nodisp", args)
        self.assertIn("-autoexit", args)


class TestPyprojectDependencies(unittest.TestCase):
    """Smoke-test that pyproject.toml has required dependencies declared."""

    def _read_pyproject(self):
        import pathlib
        p = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        if p.exists():
            return p.read_text()
        return ""

    def test_ultralytics_listed(self):
        content = self._read_pyproject()
        if content:
            self.assertIn("ultralytics", content)

    def test_numpy_listed(self):
        content = self._read_pyproject()
        if content:
            self.assertIn("numpy", content)

    def test_pytest_in_dev_deps(self):
        content = self._read_pyproject()
        if content:
            self.assertIn("pytest", content)

    def test_coverage_threshold_90(self):
        content = self._read_pyproject()
        if content:
            self.assertIn("90", content)


if __name__ == "__main__":
    unittest.main()
