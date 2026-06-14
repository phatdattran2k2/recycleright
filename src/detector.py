#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

import queue
import threading
import time

import cv2
from ultralytics import YOLO

from audio import AudioPlayer
from hud import HUD
from serial_controller import SerialController

MODEL_PATH     = "/home/jetson/recycleright/models/best_fp16.engine"
MIN_AREA       = 500
FRAME_STALE    = 1.0
VOTE_FRAMES    = 10
VOTE_THRESHOLD = 6

CLASS_CMD = {
    "PET bottle":   b"PET\n",
    "Glass bottle": b"GLS\n",
    "Aluminum can": b"MTL\n",
    "Tetra Pak":    b"PAP\n",
}


class Detector:
    """MOG2 動態偵測 + YOLOv8 推論 + 投票機制，結果透過 SerialController 送出。"""

    def __init__(
        self,
        capture_queue: queue.Queue,
        stop_event: threading.Event,
        serial_ctrl: SerialController,
        hud: HUD,
    ):
        self._queue       = capture_queue
        self._stop        = stop_event
        self._serial      = serial_ctrl
        self._hud         = hud

        self.latest_frame: cv2.Mat | None = None
        self.frame_lock   = threading.Lock()

        self._model = YOLO(MODEL_PATH)

    def run(self) -> None:
        """Thread: 消費 capture_queue，執行推論與投票，更新 latest_frame。"""
        mog2          = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)
        kernel_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        last_annotated = None

        fps_count = 0
        fps_ts    = time.perf_counter()

        vote_counts: dict[str, int] = {}
        frames_collected = 0

        while not self._stop.is_set():
            try:
                frame, ts = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if time.perf_counter() - ts > FRAME_STALE:
                continue

            fg = mog2.apply(frame)
            fg = cv2.erode(fg, kernel_erode, iterations=1)
            fg = cv2.dilate(fg, kernel_dilate, iterations=1)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = any(cv2.contourArea(c) > MIN_AREA for c in contours)

            if motion:
                results = self._model(frame, conf=0.5, verbose=False)
                last_annotated = results[0].plot()

                detected_this_frame: set[str] = set()
                for box in results[0].boxes:
                    name = results[0].names[int(box.cls)]
                    if float(box.conf) >= 0.5:
                        detected_this_frame.add(name)

                for name in detected_this_frame:
                    vote_counts[name] = vote_counts.get(name, 0) + 1

                frames_collected += 1
                print(f"[VOTE] 幀 {frames_collected}/{VOTE_FRAMES}  本幀: {detected_this_frame or '無'}  累計: {vote_counts}")

                if frames_collected >= VOTE_FRAMES:
                    if not self._serial.lid_ready.is_set():
                        print("[DECISION] 蓋子尚未關閉，等待中…")
                    else:
                        winner = max(vote_counts, key=vote_counts.get) if vote_counts else None
                        if winner and vote_counts[winner] >= VOTE_THRESHOLD:
                            print(f"[DECISION] {winner}  ({vote_counts[winner]}/{VOTE_FRAMES} 幀通過)  → 送 ESP32")
                            self._serial.lid_ready.clear()
                            self._serial.send(CLASS_CMD[winner])
                            AudioPlayer.play(winner)
                        else:
                            print(f"[DECISION] 票數不足，不送出  累計={vote_counts}")

                    vote_counts.clear()
                    frames_collected = 0
            else:
                last_annotated = last_annotated if last_annotated is not None else frame

            # FPS tracking
            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_ts
            if elapsed >= 1.0:
                self._hud.set_fps(fps_count / elapsed)
                fps_count = 0
                fps_ts    = now

            with self.frame_lock:
                self.latest_frame = self._hud.draw(last_annotated)

        del self._model
