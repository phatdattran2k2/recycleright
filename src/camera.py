#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

import os
import queue
import threading
import time
import cv2


GST = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 max-buffers=2"
)


class CameraCapture:
    """負責從 IMX219 CSI 相機讀取幀，放進 capture_queue。"""

    def __init__(self, capture_queue: queue.Queue, stop_event: threading.Event):
        self._queue      = capture_queue
        self._stop       = stop_event
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        # SSH DISPLAY workaround：開相機前暫時移除 DISPLAY
        display = os.environ.pop("DISPLAY", None)
        self._cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
        if display:
            os.environ["DISPLAY"] = display

        if not self._cap.isOpened():
            print("ERROR: Cannot open camera — try: sudo service nvargus-daemon restart")
            self._stop.set()
            return

        print("Warming up camera...")
        for _ in range(30):
            if self._stop.is_set():
                return
            self._cap.read()
        print("Camera ready.\n")

        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put((frame, time.perf_counter()))

        self._cap.release()
        self._cap = None

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
