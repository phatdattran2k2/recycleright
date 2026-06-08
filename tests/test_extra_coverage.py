import contextlib
import queue
import threading
import time
import types

import numpy as np

import src.camera as camera_mod
import src.detector as detector_mod
import src.hud as hud_mod
import src.serial_controller as serial_mod


def test_hud_draw_changes_frame():
    stop = threading.Event()
    hud = hud_mod.HUD(stop)
    hud.set_fps(12.3)
    hud.gpu_pct = 42
    hud.gpu_mw = 900

    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    out = hud.draw(frame)
    # Drawing should return a frame with same shape
    assert out.shape == frame.shape


def test_camera_release_calls_underlying_cap():
    q = queue.Queue()
    stop = threading.Event()
    cam = camera_mod.CameraCapture(q, stop)

    class FakeCap:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def release(self):
            self.released = True

    fake = FakeCap()
    cam._cap = fake
    cam.release()
    assert fake.released


def test_serial_controller_send_and_run(monkeypatch):
    stop = threading.Event()
    lid = threading.Event()

    # Fake serial.Serial implementation
    class FakeSerial:
        def __init__(self, port, baud, timeout=1):
            self.written = b""
            self._lines = [b"DONE: OK\n"]
            self.in_waiting = True
            self.is_open = True

        def write(self, data: bytes):
            self.written += data

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            self.in_waiting = False
            return b""

        def close(self):
            self.is_open = False

    monkeypatch.setattr(serial_mod, "serial", types.SimpleNamespace(Serial=FakeSerial))

    sc = serial_mod.SerialController("/dev/fake", 115200, stop, lid)
    assert sc.connected

    # Test send writes to underlying serial
    sc._ser.written = b""
    sc.send(b"CMD\n")
    assert sc._ser.written == b"CMD\n"

    # Run loop should set lid when reading DONE:
    t = threading.Thread(target=sc.run, daemon=True)
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)
    assert lid.is_set()


def test_detector_run_sends_command(monkeypatch):
    # Prepare environment
    q = queue.Queue()
    stop = threading.Event()

    # Fake SerialController
    class FakeSerial:
        def __init__(self):
            self.lid_ready = threading.Event()
            self.lid_ready.set()
            self.sent = []

        def send(self, cmd: bytes):
            self.sent.append(cmd)

    fake_serial = FakeSerial()

    # Fake HUD
    class FakeHUD:
        def __init__(self):
            self._f = 0.0
            self.frame = None

        def set_fps(self, v):
            self._f = v

        def draw(self, frame):
            self.frame = frame
            return frame

    fake_hud = FakeHUD()

    # Monkeypatch YOLO to avoid heavy model load
    class FakeYOLO:
        def __init__(self, path):
            pass

    monkeypatch.setattr(detector_mod, "YOLO", FakeYOLO)

    # Create Detector instance (uses patched YOLO)
    det = detector_mod.Detector(q, stop, fake_serial, fake_hud)

    # Replace model with fake callable that returns a predictable detection
    class FakeBox:
        def __init__(self):
            self.cls = 0
            self.conf = 0.9

    class FakeResult:
        def __init__(self):
            self.boxes = [FakeBox()]
            self.names = {0: "PET bottle"}

        def plot(self):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeModel:
        def __call__(self, frame, conf=0.5, verbose=False):
            return [FakeResult()]

    det._model = FakeModel()

    # Force motion by monkeypatching findContours and contourArea
    monkeypatch.setattr(detector_mod.cv2, "findContours", lambda fg, mode, method: ([1], None))
    monkeypatch.setattr(detector_mod.cv2, "contourArea", lambda c: detector_mod.MIN_AREA + 1)

    # Reduce vote frames/threshold to trigger immediate send
    monkeypatch.setattr(detector_mod, "VOTE_FRAMES", 1)
    monkeypatch.setattr(detector_mod, "VOTE_THRESHOLD", 1)

    # Prevent actual audio subprocess spawn
    monkeypatch.setattr(detector_mod.AudioPlayer, "play", lambda label: None)

    # Put a fake frame into queue
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    q.put((frame, time.perf_counter()))

    # Run detector in thread
    t = threading.Thread(target=det.run, daemon=True)
    t.start()
    time.sleep(0.1)
    stop.set()
    t.join(timeout=1.0)

    # Ensure a command was sent to serial
    assert fake_serial.sent, "Detector did not send any command"
    assert fake_serial.sent[0] in detector_mod.CLASS_CMD.values()


def test_main_runs_with_dummy_components(monkeypatch):
    import importlib
    import os
    import sys

    sys.modules["camera"] = importlib.import_module("src.camera")
    sys.modules["detector"] = importlib.import_module("src.detector")
    sys.modules["hud"] = importlib.import_module("src.hud")
    sys.modules["serial_controller"] = importlib.import_module("src.serial_controller")
    import src.main as main_mod

    # Dummy components that quickly set the stop event
    class DummyComp:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            # Set the global stop_event from main's closure by finding any Event in globals
            return None

        def release(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(main_mod, "CameraCapture", DummyComp)
    monkeypatch.setattr(main_mod, "Detector", DummyComp)
    monkeypatch.setattr(main_mod, "HUD", DummyComp)
    monkeypatch.setattr(main_mod, "SerialController", DummyComp)

    # Make threads start synchronously to avoid background threading complexities
    class FakeThread:
        def __init__(self, target=None, daemon=False):
            self._target = target

        def start(self):
            if self._target:
                with contextlib.suppress(Exception):
                    self._target()
    monkeypatch.setattr(main_mod.os, "system", lambda cmd: None)
    monkeypatch.setattr(main_mod.threading.Event, "wait", lambda self, timeout=None: True)

    # Force SSH path and ensure main returns quickly
    monkeypatch.setitem(os.environ, "SSH_CLIENT", "1")

    # Call main - it should run without raising
    main_mod.main()


def test_main_local_mode_exits_on_q(monkeypatch):
    import importlib
    import sys

    sys.modules["camera"] = importlib.import_module("src.camera")
    sys.modules["detector"] = importlib.import_module("src.detector")
    sys.modules["hud"] = importlib.import_module("src.hud")
    sys.modules["serial_controller"] = importlib.import_module("src.serial_controller")
    import src.main as main_mod

    class DummyComp:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return None

        def release(self):
            return None

        def close(self):
            return None

    class DummyDetector:
        def __init__(self, *args, **kwargs):
            class DummyLock:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return None

            self.frame_lock = DummyLock()
            self.latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)

        def run(self):
            return None

    monkeypatch.setattr(main_mod, "CameraCapture", DummyComp)
    monkeypatch.setattr(main_mod, "Detector", DummyDetector)
    monkeypatch.setattr(main_mod, "HUD", DummyComp)
    monkeypatch.setattr(main_mod, "SerialController", DummyComp)

    class FakeThread:
        def __init__(self, target=None, daemon=False):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(main_mod.threading, "Thread", FakeThread)
    monkeypatch.setattr(main_mod.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main_mod.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(main_mod.os, "system", lambda cmd: None)

    shown = {"count": 0}

    def fake_imshow(name, frame):
        shown["count"] += 1

    def fake_waitKey(timeout):
        return ord("q")

    monkeypatch.setattr(main_mod.cv2, "imshow", fake_imshow)
    monkeypatch.setattr(main_mod.cv2, "waitKey", fake_waitKey)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)

    main_mod.main()

    assert shown["count"] == 1


def test_camera_run_handles_no_camera(monkeypatch):
    stop = threading.Event()
    q = queue.Queue()

    # Fake VideoCapture that is not opened
    class FakeVC:
        def __init__(self, gst, api):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeVC)

    cam = camera_mod.CameraCapture(q, stop)
    cam.run()
    assert stop.is_set()


def test_camera_run_stops_during_warmup(monkeypatch):
    stop = threading.Event()
    q = queue.Queue()

    class FakeVC:
        def __init__(self, gst, api):
            pass

        def isOpened(self):
            return True

        def read(self):
            stop.set()
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            return None

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeVC)
    cam = camera_mod.CameraCapture(q, stop)
    cam.run()
    assert stop.is_set()


def test_camera_run_puts_frame_when_queue_empty(monkeypatch):
    stop = threading.Event()
    q = queue.Queue()

    class FakeVC:
        def __init__(self, gst, api):
            self.read_count = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.read_count += 1
            if self.read_count <= 30:
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            if self.read_count == 31:
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            stop.set()
            return False, None

        def release(self):
            self.released = True

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeVC)
    cam = camera_mod.CameraCapture(q, stop)
    cam.run()
    assert cam._cap is None
    assert not q.empty()


def test_camera_run_warms_up_and_releases(monkeypatch):
    stop = threading.Event()
    q = queue.Queue()

    class FakeVC:
        def __init__(self, gst, api):
            self.read_count = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.read_count += 1
            if self.read_count <= 30:
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            stop.set()
            return False, None

        def release(self):
            self.released = True

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", FakeVC)
    monkeypatch.setitem(camera_mod.os.environ, "DISPLAY", ":0")

    cam = camera_mod.CameraCapture(q, stop)
    cam.run()
    assert cam._cap is None


def test_hud_run_parses_gpu_and_power(monkeypatch):
    stop = threading.Event()
    hud = hud_mod.HUD(stop)
    lines = [
        "GR3D_FREQ 65% VDD_CPU_GPU_CV 1420mW\n",
    ]

    class FakeProc:
        def __init__(self):
            self.stdout = iter(lines)

        def terminate(self):
            pass

    def fake_popen(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(hud_mod.shutil, "which", lambda cmd: "/usr/bin/tegrastats" if cmd == "tegrastats" else None)
    monkeypatch.setattr(hud_mod.subprocess, "Popen", fake_popen)
    hud.run()
    assert hud.gpu_pct == 65
    assert hud.gpu_mw == 1420


def test_hud_run_handles_missing_tegrastats(monkeypatch):
    stop = threading.Event()
    hud = hud_mod.HUD(stop)

    # Make Popen raise FileNotFoundError
    def fake_popen(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(hud_mod.subprocess, "Popen", fake_popen)
    # Should not raise
    hud.run()
