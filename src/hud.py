#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

import re
import shutil
import subprocess  # nosec
import threading

import cv2


class HUD:
    """管理 tegrastats 解析與 HUD 畫面疊加。"""

    def __init__(self, stop_event: threading.Event):
        self._stop       = stop_event
        self._lock       = threading.Lock()
        self.fps         = 0.0
        self.gpu_pct     = -1   # -1 = tegrastats unavailable
        self.gpu_mw      = -1
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        """Thread: 每秒解析一次 tegrastats 輸出。"""
        try:
            tegrastats = shutil.which("tegrastats")
            if tegrastats is None:
                raise FileNotFoundError
            self._proc = subprocess.Popen(
                [tegrastats, "--interval", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )  # nosec
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                m_gpu   = re.search(r'GR3D_FREQ\s+(\d+)%', line)
                m_power = (
                    re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line) or
                    re.search(r'POM_5V_GPU\s+(\d+)(?:mW)?/', line)
                )
                with self._lock:
                    if m_gpu:
                        self.gpu_pct = int(m_gpu.group(1))
                    if m_power:
                        self.gpu_mw = int(m_power.group(1))
        except FileNotFoundError:
            print("tegrastats not found — GPU stats disabled")
        except Exception as e:
            print(f"tegrastats error: {e}")
        finally:
            if self._proc:
                self._proc.terminate()

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self.fps = fps

    def draw(self, frame: "cv2.Mat") -> "cv2.Mat":
        """在 frame 副本上疊加 FPS 與 GPU 資訊。"""
        out = frame.copy()
        with self._lock:
            fps     = self.fps
            gpu_pct = self.gpu_pct
            gpu_mw  = self.gpu_mw

        lines = [f"FPS: {fps:.1f}"]
        if gpu_pct >= 0:
            lines.append(f"GPU: {gpu_pct}%   Power: {gpu_mw} mW")

        x, y0, dy = 10, 32, 30
        pad = 8
        max_w = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0] for line in lines
        )
        box_h = dy * len(lines) + pad
        roi = out[y0 - dy + pad : y0 - dy + pad + box_h, x - pad : x + max_w + pad]
        if roi.size:
            dark = roi.copy()
            dark[:] = (0, 0, 0)
            cv2.addWeighted(dark, 0.5, roi, 0.5, 0, roi)

        for i, txt in enumerate(lines):
            colour = (0, 255, 0) if i == 0 else (0, 220, 255)
            cv2.putText(out, txt, (x, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, txt, (x, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour,  2, cv2.LINE_AA)
        return out
