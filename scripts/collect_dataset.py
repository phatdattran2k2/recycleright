#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題
"""Dataset collection - chụp tự động, không cần display (SSH mode)."""

import argparse
import time
from pathlib import Path

import cv2

GST_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 max-buffers=2"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", required=True,
                        choices=["pet", "metal", "paper", "glass"])
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Giây giữa mỗi ảnh")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return 1

    # Warm-up 10 frame
    print("Warming up camera...")
    for _ in range(10):
        cap.read()

    print("Class     :", args.class_name)
    print("Count     :", args.count)
    print("Interval  :", args.interval, "s")
    print("Output    :", out_dir)
    print("Starting in 3s... (Ctrl+C to stop)")
    time.sleep(3)

    count = 0
    while count < args.count:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("WARNING: bad frame, skipping")
            continue

        ts = int(time.time() * 1000)
        filename = out_dir / (args.class_name + "_" + str(ts) + "_" + str(count).zfill(3) + ".jpg")
        cv2.imwrite(str(filename), frame)
        count += 1
        print("[" + str(count).rjust(3) + "/" + str(args.count) + "] Saved: " + filename.name)
        time.sleep(args.interval)

    cap.release()
    print("Done! " + str(count) + " images saved to " + str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
