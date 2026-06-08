#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""
benchmark.py — 離線推論性能量測（不開相機）
使用固定的 320 張真實圖片，每次跑相同順序，跨電源模式結果可直接比較。

用法：
  python benchmark.py              # 保持目前模式
  python benchmark.py --mode 0    # 切換 15W 再量
  python benchmark.py --mode 2    # 切換 MAXN_SUPER 再量
"""

import argparse
import csv
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── 設定 ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/jetson/recycleright/models/best_fp16.engine"
IMAGE_DIR  = Path("/home/jetson/recycleright/data/raw")
WARMUP_N   = 20   # 預熱次數（不計入統計）
LOOPS      = 3    # 把 320 張圖跑幾遍（增加樣本量，p99 才穩定）

POWER_MODES = {0: "15W", 1: "25W", 2: "MAXN_SUPER"}


# ── power mode ────────────────────────────────────────────────────────────────

def get_power_mode() -> tuple[int, str]:
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        lines = out.strip().splitlines()
        mode_id = int(lines[-1].strip())
        return mode_id, POWER_MODES.get(mode_id, lines[0].replace("NV Power Mode:", "").strip())
    except Exception:
        return -1, "unknown"


def set_power_mode(mode_id: int):
    name = POWER_MODES.get(mode_id, str(mode_id))
    print(f"[PowerMode] 切換至 Mode {mode_id} ({name})…")
    ret = os.system(f"sudo nvpmodel -m {mode_id}")
    if ret == 0:
        print(f"[PowerMode] 已切換，等待 2 秒生效…")
        time.sleep(2)
    else:
        print("[PowerMode] 切換失敗（需要 sudo 權限）")


# ── tegrastats ────────────────────────────────────────────────────────────────

class TegrastatsCollector:
    def __init__(self, interval_ms: int = 500):
        self._samples: list[dict] = []
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(interval_ms,), daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self, interval_ms):
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", str(interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    break
                cpu_pcts = [int(x) for x in re.findall(r'(\d+)%@\d+', line)]
                m_gpu = re.search(r'GR3D_FREQ\s+(\d+)%', line)
                m_pwr = re.search(r'VDD_IN\s+(\d+)mW', line)
                if not cpu_pcts and not m_gpu:
                    continue
                with self._lock:
                    self._samples.append({
                        "cpu": int(np.mean(cpu_pcts)) if cpu_pcts else -1,
                        "gpu": int(m_gpu.group(1))    if m_gpu   else -1,
                        "pwr": int(m_pwr.group(1))    if m_pwr   else -1,
                    })
        except FileNotFoundError:
            print("[tegrastats] 未找到，跳過硬體監控")

    def result(self) -> dict:
        with self._lock:
            s = list(self._samples)
        cpu_v = [x["cpu"] for x in s if x["cpu"] >= 0]
        gpu_v = [x["gpu"] for x in s if x["gpu"] >= 0]
        pwr_v = [x["pwr"] for x in s if x["pwr"] >= 0]
        return {
            "n":           len(s),
            "cpu_avg":     float(np.mean(cpu_v)) if cpu_v else -1,
            "gpu_avg":     float(np.mean(gpu_v)) if gpu_v else -1,
            "pwr_avg_mw":  float(np.mean(pwr_v)) if pwr_v else -1,
            "pwr_peak_mw": float(np.max(pwr_v))  if pwr_v else -1,
        }


# ── 圖片載入 ──────────────────────────────────────────────────────────────────

def load_images() -> list[np.ndarray]:
    """載入 IMAGE_DIR 下四個類別資料夾的所有圖片，排序後固定順序。"""
    paths = sorted(IMAGE_DIR.rglob("*.jpg")) + sorted(IMAGE_DIR.rglob("*.png"))
    if not paths:
        raise FileNotFoundError(f"找不到圖片：{IMAGE_DIR}")
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    print(f"載入 {len(imgs)} 張圖片（來自 {IMAGE_DIR}）")
    return imgs


# ── 推論計時 ──────────────────────────────────────────────────────────────────

def measure_latency(model: YOLO, images: list[np.ndarray]) -> dict:
    """
    預熱 WARMUP_N 張後，把 images 跑 LOOPS 遍，記錄每次推論時間。
    每次跑相同的圖片 → 跨電源模式完全可比較。
    """
    print(f"\n預熱 {WARMUP_N} 張…")
    for img in images[:WARMUP_N]:
        model(img, conf=0.5, verbose=False)

    print(f"計時：{len(images)} 張 × {LOOPS} 遍 = {len(images)*LOOPS} 次推論…")
    latencies_ms: list[float] = []
    t_all = time.perf_counter()

    for _ in range(LOOPS):
        for img in images:
            t0 = time.perf_counter()
            model(img, conf=0.5, verbose=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_sec = time.perf_counter() - t_all
    lat = np.array(latencies_ms)

    return {
        "fps":     len(latencies_ms) / total_sec,
        "n":       len(latencies_ms),
        "p50_ms":  float(np.percentile(lat, 50)),
        "p95_ms":  float(np.percentile(lat, 95)),
        "p99_ms":  float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()),
        "min_ms":  float(lat.min()),
        "max_ms":  float(lat.max()),
    }


# ── CSV 彙總 ──────────────────────────────────────────────────────────────────

def save_and_summary(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def col(key):
        return [float(r[key]) for r in rows if r.get(key)]

    print("\n" + "=" * 62)
    print(f"  彙總報告 — 共 {len(rows)} 次測試")
    print("=" * 62)
    print(f"  {'#':<3}  {'模式':<12}  {'FPS':>6}  {'p50':>7}  {'p95':>7}  {'p99':>7}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    for i, r in enumerate(rows, 1):
        print(f"  #{i:<2}  {r.get('power_mode',''):<12}  "
              f"{r.get('fps',''):>6}  "
              f"{r.get('p50_ms',''):>6}ms  "
              f"{r.get('p95_ms',''):>6}ms  "
              f"{r.get('p99_ms',''):>6}ms")

    def stat(v):
        return f"{np.mean(v):.1f} (min {np.min(v):.1f} / max {np.max(v):.1f})" if v else "N/A"

    print(f"  {'─'*62}")
    print(f"  FPS 平均:              {stat(col('fps'))}")
    print(f"  p50 latency 平均:      {stat(col('p50_ms'))} ms")
    print(f"  p95 latency 平均:      {stat(col('p95_ms'))} ms")
    print(f"  p99 latency 平均:      {stat(col('p99_ms'))} ms")
    print(f"  CPU 使用率平均:        {stat(col('cpu_avg_pct'))} %")
    print(f"  GPU 使用率平均:        {stat(col('gpu_avg_pct'))} %")
    print(f"  VDD_IN 平均功耗 平均:  {stat(col('vdd_in_avg_mw'))} mW")
    pwr_pk = col("vdd_in_peak_mw")
    print(f"  VDD_IN 峰值功耗 最大:  {max(pwr_pk):.0f} mW" if pwr_pk else "  VDD_IN: N/A")
    print("=" * 62)
    print(f"\n結果已附加至 {csv_path}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecycleRight 離線推論 benchmark")
    parser.add_argument("--mode", "-m", type=int, choices=[0, 1, 2], default=None,
                        help="切換電源模式：0=15W  1=25W  2=MAXN_SUPER")
    args = parser.parse_args()

    if args.mode is not None:
        set_power_mode(args.mode)

    mode_id, mode_name = get_power_mode()
    power_mode = f"Mode {mode_id} ({mode_name})"

    print("=" * 62)
    print("  RecycleRight — 離線推論 benchmark（固定圖片集）")
    print(f"  電源模式: {power_mode}")
    print("=" * 62)

    images = load_images()
    model  = YOLO(MODEL_PATH)

    teg = TegrastatsCollector(interval_ms=500)
    teg.start()

    lat = measure_latency(model, images)

    teg.stop()
    hw = teg.result()

    # ── 終端機輸出 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  本次結果")
    print("=" * 62)
    print(f"  電源模式:      {power_mode}")
    print(f"\n{'推論速度':─<40}")
    print(f"  FPS:           {lat['fps']:.1f}")
    print(f"  推論樣本數:    {lat['n']}")
    print(f"  p50 latency:   {lat['p50_ms']:.1f} ms")
    print(f"  p95 latency:   {lat['p95_ms']:.1f} ms")
    print(f"  p99 latency:   {lat['p99_ms']:.1f} ms")
    print(f"  mean latency:  {lat['mean_ms']:.1f} ms")
    print(f"  min / max:     {lat['min_ms']:.1f} / {lat['max_ms']:.1f} ms")
    print(f"\n{'硬體資源（tegrastats, n=' + str(hw['n']) + '）':─<40}")
    print(f"  CPU 平均使用率:  {hw['cpu_avg']:.1f} %" if hw['cpu_avg'] >= 0 else "  CPU: N/A")
    print(f"  GPU 平均使用率:  {hw['gpu_avg']:.1f} %" if hw['gpu_avg'] >= 0 else "  GPU: N/A")
    print(f"  VDD_IN 平均:     {hw['pwr_avg_mw']:.0f} mW" if hw['pwr_avg_mw'] >= 0 else "  VDD_IN: N/A")
    print(f"  VDD_IN 峰值:     {hw['pwr_peak_mw']:.0f} mW" if hw['pwr_peak_mw'] >= 0 else "")
    print("=" * 62)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = Path(__file__).parent / "benchmark_offline_result.csv"
    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "power_mode":     power_mode,
        "n_images":       len(images),
        "loops":          LOOPS,
        "fps":            f"{lat['fps']:.1f}",
        "n_samples":      lat['n'],
        "p50_ms":         f"{lat['p50_ms']:.1f}",
        "p95_ms":         f"{lat['p95_ms']:.1f}",
        "p99_ms":         f"{lat['p99_ms']:.1f}",
        "mean_ms":        f"{lat['mean_ms']:.1f}",
        "min_ms":         f"{lat['min_ms']:.1f}",
        "max_ms":         f"{lat['max_ms']:.1f}",
        "cpu_avg_pct":    f"{hw['cpu_avg']:.1f}"    if hw['cpu_avg']    >= 0 else "",
        "gpu_avg_pct":    f"{hw['gpu_avg']:.1f}"    if hw['gpu_avg']    >= 0 else "",
        "vdd_in_avg_mw":  f"{hw['pwr_avg_mw']:.0f}" if hw['pwr_avg_mw'] >= 0 else "",
        "vdd_in_peak_mw": f"{hw['pwr_peak_mw']:.0f}" if hw['pwr_peak_mw'] >= 0 else "",
    }
    save_and_summary(csv_path, row)
