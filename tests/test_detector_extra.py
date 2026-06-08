import queue
import threading
import time

import numpy as np

import src.detector as detector_mod


class FakeYOLO:
    def __init__(self, path):
        pass


class FakeBox:
    def __init__(self):
        self.cls = 0
        self.conf = 0.9


class FakeResult:
    def __init__(self, name="PET bottle"):
        self.boxes = [FakeBox()]
        self.names = {0: name}

    def plot(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class FakeModel:
    def __call__(self, frame, conf=0.5, verbose=False):
        return [FakeResult()]


def make_detector(monkeypatch, serial):
    monkeypatch.setattr(detector_mod, "YOLO", FakeYOLO)
    det = detector_mod.Detector(queue.Queue(), threading.Event(), serial, type("H", (), {"draw": lambda self, f: f, "set_fps": lambda self, v: None})())
    det._model = FakeModel()
    return det


def test_detector_non_motion_calls_draw(monkeypatch):
    q = queue.Queue()
    stop = threading.Event()

    class FakeSerial:
        def __init__(self):
            self.lid_ready = threading.Event()

    fake_serial = FakeSerial()
    monkeypatch.setattr(detector_mod, "YOLO", FakeYOLO)

    fake_hud = type("H", (), {"draw": lambda self, f: f, "set_fps": lambda self, v: None})()
    det = detector_mod.Detector(q, stop, fake_serial, fake_hud)
    det._model = FakeModel()

    # No contours => motion False
    monkeypatch.setattr(detector_mod.cv2, "findContours", lambda fg, mode, method: ([], None))
    monkeypatch.setattr(detector_mod.cv2, "contourArea", lambda c: 0)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    q.put((frame, time.perf_counter()))

    t = threading.Thread(target=det.run, daemon=True)
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)

    assert det.latest_frame is not None


def test_detector_decision_lid_not_ready(monkeypatch):
    q = queue.Queue()
    stop = threading.Event()

    class FakeSerial:
        def __init__(self):
            self.lid_ready = threading.Event()
            # lid not set
            self.sent = []

        def send(self, cmd: bytes):
            self.sent.append(cmd)

    fake_serial = FakeSerial()
    fake_hud = type("H", (), {"draw": lambda self, f: f, "set_fps": lambda self, v: None})()

    monkeypatch.setattr(detector_mod, "YOLO", FakeYOLO)
    det = detector_mod.Detector(q, stop, fake_serial, fake_hud)
    det._model = FakeModel()

    # Force motion
    monkeypatch.setattr(detector_mod.cv2, "findContours", lambda fg, mode, method: ([1], None))
    monkeypatch.setattr(detector_mod.cv2, "contourArea", lambda c: detector_mod.MIN_AREA + 1)

    # Trigger immediate voting decision
    monkeypatch.setattr(detector_mod, "VOTE_FRAMES", 1)
    monkeypatch.setattr(detector_mod, "VOTE_THRESHOLD", 1)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    q.put((frame, time.perf_counter()))

    t = threading.Thread(target=det.run, daemon=True)
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)

    assert not fake_serial.sent


def test_detector_insufficient_votes(monkeypatch):
    q = queue.Queue()
    stop = threading.Event()

    class FakeSerial:
        def __init__(self):
            self.lid_ready = threading.Event()
            self.lid_ready.set()
            self.sent = []

        def send(self, cmd: bytes):
            self.sent.append(cmd)

    fake_serial = FakeSerial()
    fake_hud = type("H", (), {"draw": lambda self, f: f, "set_fps": lambda self, v: None})()

    monkeypatch.setattr(detector_mod, "YOLO", FakeYOLO)
    det = detector_mod.Detector(q, stop, fake_serial, fake_hud)
    det._model = FakeModel()

    # Force motion
    monkeypatch.setattr(detector_mod.cv2, "findContours", lambda fg, mode, method: ([1], None))
    monkeypatch.setattr(detector_mod.cv2, "contourArea", lambda c: detector_mod.MIN_AREA + 1)

    # Trigger immediate voting decision but threshold higher than votes
    monkeypatch.setattr(detector_mod, "VOTE_FRAMES", 1)
    monkeypatch.setattr(detector_mod, "VOTE_THRESHOLD", 2)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    q.put((frame, time.perf_counter()))

    t = threading.Thread(target=det.run, daemon=True)
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)

    assert not fake_serial.sent
