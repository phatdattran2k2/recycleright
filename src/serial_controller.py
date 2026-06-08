#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

import threading
import time

import serial


class SerialController:
    """管理與 ESP32 的 USB 序列通訊及 lid_ready 同步旗標。"""

    def __init__(
        self,
        port: str,
        baud: int,
        stop_event: threading.Event,
        lid_ready: threading.Event,
    ):
        self._stop     = stop_event
        self.lid_ready = lid_ready
        self._ser: serial.Serial | None = None

        try:
            self._ser = serial.Serial(port, baud, timeout=1)
            print(f"Serial port opened: {port}")
        except Exception as e:
            print(f"Serial port not available: {e}")

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def send(self, cmd: bytes) -> None:
        if self._ser:
            self._ser.write(cmd)

    def run(self) -> None:
        """Thread: 持續讀 ESP32 回傳資料，收到 DONE 則釋放 lid_ready。"""
        while not self._stop.is_set():
            if self._ser and self._ser.in_waiting:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"[ESP32] {line}")
                    if line.startswith("DONE:"):
                        self.lid_ready.set()
                        print("[LID] 蓋子已關閉，系統就緒")
            time.sleep(0.01)

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
