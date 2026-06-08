#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""
parse_tegrastats.py — Parse tegrastats.log into utilization.csv

Usage:
    python scripts/parse_tegrastats.py --input tegrastats.log --output utilization.csv

Output columns:
    t, cpu_avg_pct, gpu_pct, ram_used_mb, vdd_in_mw, vdd_cpu_mw, vdd_gpu_mw,
    vdd_soc_mw, gpu_temp_c, cpu_temp_c
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_line(line: str, t: int) -> dict | None:
    """Parse one tegrastats line into a dict row."""
    # CPU: [21%@1190,15%@1190,off,off]
    cpu_pcts = [int(x) for x in re.findall(r"(\d+)%@\d+", line)]
    cpu_avg = sum(cpu_pcts) / len(cpu_pcts) if cpu_pcts else -1

    m_gpu  = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    m_ram  = re.search(r"RAM\s+(\d+)/", line)

    # Power rails — Jetson Orin Nano naming
    m_vdd_in  = re.search(r"VDD_IN\s+(\d+)mW", line)
    m_vdd_cpu = re.search(r"VDD_CPU_CV\s+(\d+)mW", line)
    m_vdd_gpu = re.search(r"VDD_GPU_CV\s+(\d+)mW", line)
    m_vdd_soc = re.search(r"VDD_SOC_CV\s+(\d+)mW", line)

    # Temps
    m_gpu_t = re.search(r"GPU@([\d.]+)C", line)
    m_cpu_t = re.search(r"CPU@([\d.]+)C", line)

    if not cpu_pcts and not m_gpu:
        return None

    return {
        "t":           t,
        "cpu_avg_pct": round(cpu_avg, 1) if cpu_avg >= 0 else "",
        "gpu_pct":     int(m_gpu.group(1)) if m_gpu else "",
        "ram_used_mb": int(m_ram.group(1)) if m_ram else "",
        "vdd_in_mw":   int(m_vdd_in.group(1))  if m_vdd_in  else "",
        "vdd_cpu_mw":  int(m_vdd_cpu.group(1)) if m_vdd_cpu else "",
        "vdd_gpu_mw":  int(m_vdd_gpu.group(1)) if m_vdd_gpu else "",
        "vdd_soc_mw":  int(m_vdd_soc.group(1)) if m_vdd_soc else "",
        "gpu_temp_c":  float(m_gpu_t.group(1)) if m_gpu_t else "",
        "cpu_temp_c":  float(m_cpu_t.group(1)) if m_cpu_t else "",
    }


def parse_log(input_path: Path, output_path: Path) -> None:
    rows = []
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        for t, line in enumerate(f):
            row = parse_line(line.strip(), t)
            if row:
                rows.append(row)

    if not rows:
        print(f"[warn] No valid tegrastats lines found in {input_path}", file=sys.stderr)
        return

    fields = ["t", "cpu_avg_pct", "gpu_pct", "ram_used_mb",
              "vdd_in_mw", "vdd_cpu_mw", "vdd_gpu_mw", "vdd_soc_mw",
              "gpu_temp_c", "cpu_temp_c"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    import statistics
    def _col(key):
        return [float(r[key]) for r in rows if r.get(key) != ""]

    cpu_v = _col("cpu_avg_pct")
    gpu_v = _col("gpu_pct")
    pwr_v = _col("vdd_in_mw")

    print(f"\n=== tegrastats summary ({len(rows)} samples) ===")
    if cpu_v:
        print(f"  CPU avg: {statistics.mean(cpu_v):.1f}%  "
              f"p95: {sorted(cpu_v)[int(len(cpu_v)*0.95)]:.1f}%  "
              f"max: {max(cpu_v):.1f}%")
    if gpu_v:
        print(f"  GPU avg: {statistics.mean(gpu_v):.1f}%  "
              f"max: {max(gpu_v):.1f}%")
    if pwr_v:
        print(f"  VDD_IN avg: {statistics.mean(pwr_v):.0f} mW  "
              f"peak: {max(pwr_v):.0f} mW")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse tegrastats.log → utilization.csv")
    parser.add_argument("--input",  "-i", default="tegrastats.log",   help="Input log file")
    parser.add_argument("--output", "-o", default="utilization.csv",  help="Output CSV file")
    args = parser.parse_args()
    parse_log(Path(args.input), Path(args.output))
