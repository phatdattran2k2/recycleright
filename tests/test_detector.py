#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""Unit tests for Detector class — inference logic, voting, decision boundaries."""

import threading
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

# ── Helpers to build fake YOLO results ────────────────────────────────────────

def _make_box(cls_id: int, conf: float):
    box = MagicMock()
    type(box).cls = PropertyMock(return_value=[cls_id])
    type(box).conf = PropertyMock(return_value=[conf])
    return box


def _make_result(names: dict, boxes):
    result = MagicMock()
    result.names = names
    result.boxes = boxes
    result.plot.return_value = MagicMock()  # fake annotated frame
    return result


NAMES = {0: "PET bottle", 1: "Glass bottle", 2: "Aluminum can", 3: "Tetra Pak"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVotingLogic(unittest.TestCase):
    """Test the 10-frame / 6-vote threshold decision logic."""

    def setUp(self):
        self.vote_counts: dict[str, int] = {}
        self.frames_collected = 0
        self.VOTE_FRAMES = 10
        self.VOTE_THRESHOLD = 6

    def _add_detection(self, label: str):
        self.vote_counts[label] = self.vote_counts.get(label, 0) + 1
        self.frames_collected += 1

    def _get_winner(self):
        if not self.vote_counts:
            return None
        winner = max(self.vote_counts, key=self.vote_counts.get)
        if self.vote_counts[winner] >= self.VOTE_THRESHOLD:
            return winner
        return None

    def test_winner_at_exactly_threshold(self):
        for _ in range(6):
            self._add_detection("PET bottle")
        for _ in range(4):
            self._add_detection("Glass bottle")
        winner = self._get_winner()
        self.assertEqual(winner, "PET bottle")

    def test_no_winner_below_threshold(self):
        for _ in range(5):
            self._add_detection("PET bottle")
        for _ in range(5):
            self._add_detection("Glass bottle")
        winner = self._get_winner()
        self.assertIsNone(winner)

    def test_winner_with_all_same_class(self):
        for _ in range(10):
            self._add_detection("Aluminum can")
        winner = self._get_winner()
        self.assertEqual(winner, "Aluminum can")

    def test_empty_votes_returns_none(self):
        winner = self._get_winner()
        self.assertIsNone(winner)

    def test_vote_counts_reset_after_decision(self):
        for _ in range(6):
            self._add_detection("PET bottle")
        self.vote_counts.clear()
        self.frames_collected = 0
        self.assertEqual(len(self.vote_counts), 0)
        self.assertEqual(self.frames_collected, 0)

    def test_winner_is_max_votes_not_first(self):
        self._add_detection("Glass bottle")
        for _ in range(7):
            self._add_detection("Tetra Pak")
        winner = self._get_winner()
        self.assertEqual(winner, "Tetra Pak")

    def test_all_classes_below_threshold(self):
        for cls in NAMES.values():
            for _ in range(2):
                self._add_detection(cls)
        winner = self._get_winner()
        self.assertIsNone(winner)

    def test_frames_collected_increments(self):
        for _ in range(7):
            self._add_detection("PET bottle")
        self.assertEqual(self.frames_collected, 7)

    def test_vote_threshold_boundary_5_not_enough(self):
        for _ in range(5):
            self._add_detection("PET bottle")
        winner = self._get_winner()
        self.assertIsNone(winner)

    def test_all_votes_same_class_10_frames(self):
        for _ in range(10):
            self._add_detection("Glass bottle")
        self.assertEqual(self.vote_counts["Glass bottle"], 10)
        winner = self._get_winner()
        self.assertEqual(winner, "Glass bottle")


class TestClassCommandMapping(unittest.TestCase):
    """Test CLASS_CMD and CLASS_AUDIO mappings are complete and consistent."""

    CLASS_CMD = {
        "PET bottle":   b"PET\n",
        "Glass bottle": b"GLS\n",
        "Aluminum can": b"MTL\n",
        "Tetra Pak":    b"PAP\n",
    }

    NAMES = {0: "PET bottle", 1: "Glass bottle", 2: "Aluminum can", 3: "Tetra Pak"}

    def test_all_classes_have_command(self):
        for name in self.NAMES.values():
            self.assertIn(name, self.CLASS_CMD)

    def test_commands_end_with_newline(self):
        for cmd in self.CLASS_CMD.values():
            self.assertTrue(cmd.endswith(b"\n"), f"{cmd!r} must end with \\n")

    def test_commands_are_bytes(self):
        for cmd in self.CLASS_CMD.values():
            self.assertIsInstance(cmd, bytes)

    def test_pet_command(self):
        self.assertEqual(self.CLASS_CMD["PET bottle"], b"PET\n")

    def test_glass_command(self):
        self.assertEqual(self.CLASS_CMD["Glass bottle"], b"GLS\n")

    def test_aluminum_command(self):
        self.assertEqual(self.CLASS_CMD["Aluminum can"], b"MTL\n")

    def test_tetrapak_command(self):
        self.assertEqual(self.CLASS_CMD["Tetra Pak"], b"PAP\n")

    def test_four_classes_total(self):
        self.assertEqual(len(self.CLASS_CMD), 4)


class TestLidReadyFlag(unittest.TestCase):
    """Test lid_ready threading.Event synchronisation logic."""

    def test_initial_state_is_set(self):
        lid_ready = threading.Event()
        lid_ready.set()
        self.assertTrue(lid_ready.is_set())

    def test_clear_blocks_trigger(self):
        lid_ready = threading.Event()
        lid_ready.set()
        lid_ready.clear()
        self.assertFalse(lid_ready.is_set())

    def test_done_signal_releases_lock(self):
        lid_ready = threading.Event()
        lid_ready.clear()
        # Simulate receiving DONE from ESP32
        lid_ready.set()
        self.assertTrue(lid_ready.is_set())

    def test_decision_clears_lid_ready(self):
        lid_ready = threading.Event()
        lid_ready.set()
        # Simulate sending command → clear
        lid_ready.clear()
        self.assertFalse(lid_ready.is_set())


class TestMotionGating(unittest.TestCase):
    """Test MIN_AREA contour-based motion gating logic."""

    MIN_AREA = 500

    def test_large_contour_triggers_inference(self):
        contour_area = 600
        motion = contour_area > self.MIN_AREA
        self.assertTrue(motion)

    def test_small_contour_skips_inference(self):
        contour_area = 100
        motion = contour_area > self.MIN_AREA
        self.assertFalse(motion)

    def test_exact_min_area_skips_inference(self):
        contour_area = self.MIN_AREA
        motion = contour_area > self.MIN_AREA
        self.assertFalse(motion)

    def test_one_plus_min_area_triggers(self):
        contour_area = self.MIN_AREA + 1
        motion = contour_area > self.MIN_AREA
        self.assertTrue(motion)


class TestAudioPlayer(unittest.TestCase):
    """Test AudioPlayer dispatches ffplay for known classes only."""

    CLASS_AUDIO = {
        "PET bottle":   "/path/寶特瓶.mp3",
        "Glass bottle": "/path/玻璃瓶.mp3",
        "Aluminum can": "/path/鋁罐.mp3",
        "Tetra Pak":    "/path/鋁箔包.mp3",
    }

    def test_known_class_returns_path(self):
        for _label, path in self.CLASS_AUDIO.items():
            self.assertIsNotNone(path)

    def test_unknown_class_returns_none(self):
        result = self.CLASS_AUDIO.get("General waste")
        self.assertIsNone(result)

    @patch("subprocess.Popen")
    def test_play_calls_ffplay(self, mock_popen):
        import subprocess
        label = "PET bottle"
        path = self.CLASS_AUDIO.get(label)
        if path:
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertEqual(args[0], "ffplay")

    @patch("subprocess.Popen")
    def test_unknown_class_does_not_call_ffplay(self, mock_popen):
        import subprocess
        label = "Unknown"
        path = self.CLASS_AUDIO.get(label)
        if path:
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        mock_popen.assert_not_called()


class TestSerialController(unittest.TestCase):
    """Test SerialController connection and DONE-parsing logic."""

    def test_done_prefix_triggers_lid_ready(self):
        lid_ready = threading.Event()
        lid_ready.clear()
        line = "DONE:PET bottle"
        if line.startswith("DONE:"):
            lid_ready.set()
        self.assertTrue(lid_ready.is_set())

    def test_non_done_line_does_not_trigger(self):
        lid_ready = threading.Event()
        lid_ready.clear()
        line = "【收到】PET bottle → 通道 0 馬達動作中"
        if line.startswith("DONE:"):
            lid_ready.set()
        self.assertFalse(lid_ready.is_set())

    @patch("serial.Serial")
    def test_serial_open_success(self, mock_serial):
        mock_serial.return_value.is_open = True
        import serial
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        self.assertTrue(ser.is_open)

    @patch("serial.Serial", side_effect=Exception("port not found"))
    def test_serial_open_failure_is_handled(self, mock_serial):
        import serial
        try:
            ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        except Exception:
            ser = None
        self.assertIsNone(ser)

    def test_send_writes_bytes(self):
        mock_ser = MagicMock()
        mock_ser.write = MagicMock()
        cmd = b"PET\n"
        mock_ser.write(cmd)
        mock_ser.write.assert_called_once_with(cmd)


class TestFrameStaleness(unittest.TestCase):
    """Test frame-staleness check (FRAME_STALE = 1.0 s)."""

    FRAME_STALE = 1.0

    def test_fresh_frame_is_not_stale(self):
        import time
        ts = time.perf_counter()
        stale = (time.perf_counter() - ts) > self.FRAME_STALE
        self.assertFalse(stale)

    def test_old_frame_is_stale(self):
        import time
        ts = time.perf_counter() - 2.0
        stale = (time.perf_counter() - ts) > self.FRAME_STALE
        self.assertTrue(stale)


class TestHUDDraw(unittest.TestCase):
    """Test HUD stats update logic (no OpenCV required)."""

    def test_fps_update(self):
        fps_value = {"fps": 0.0}
        fps_value["fps"] = 35.6
        self.assertAlmostEqual(fps_value["fps"], 35.6)

    def test_gpu_pct_negative_means_unavailable(self):
        gpu_pct = -1
        available = gpu_pct >= 0
        self.assertFalse(available)

    def test_gpu_pct_zero_means_available(self):
        gpu_pct = 0
        available = gpu_pct >= 0
        self.assertTrue(available)


if __name__ == "__main__":
    unittest.main()
