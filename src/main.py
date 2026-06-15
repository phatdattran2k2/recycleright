#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

import os
import queue
import shutil
import signal
import subprocess  # nosec
import threading
import time

import cv2

from src.camera import CameraCapture
from src.detector import Detector
from src.hud import HUD
from src.serial_controller import SerialController


def main() -> None:
    stop_event  = threading.Event()
    lid_ready   = threading.Event()
    lid_ready.set()

    capture_queue = queue.Queue(maxsize=1)

    hud    = HUD(stop_event)
    serial = SerialController("/dev/ttyUSB0", 115200, stop_event, lid_ready)
    camera = CameraCapture(capture_queue, stop_event)
    detect = Detector(capture_queue, stop_event, serial, hud)

    def cleanup() -> None:
        print("\nCleaning up...")
        stop_event.set()
        camera.release()
        serial.close()
        cv2.destroyAllWindows()
        sudo = shutil.which("sudo")
        if sudo:
            subprocess.run([sudo, "service", "nvargus-daemon", "restart"], check=False)  # nosec
        print("Done.")

    def signal_handler(sig, frame) -> None:
        cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)

    threading.Thread(target=camera.run,  daemon=True).start()
    threading.Thread(target=detect.run,  daemon=True).start()
    threading.Thread(target=serial.run,  daemon=True).start()
    threading.Thread(target=hud.run,     daemon=True).start()

    is_ssh = bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))

    if is_ssh:
        print("SSH 模式：不開啟視窗，按 Ctrl+C 結束\n")
        stop_event.wait()
    else:
        print("本機模式：開啟相機視窗，按 Q 結束\n")
        while not stop_event.is_set():
            with detect.frame_lock:
                frame = detect.latest_frame

            if frame is not None:
                cv2.imshow("RecycleRight", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cleanup()
                break

            time.sleep(0.005)


if __name__ == "__main__":
    main()
