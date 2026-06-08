#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題
"""Dataset collection helper - chụp ảnh từ IMX219 camera với GStreamer."""

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
    parser.add_argument("--count", type=int, default=80,
                        help="Số ảnh cần chụp")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return 1

    print("Press SPACE to capture, Q to quit")

    count = 0
    while count < args.count:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(
            display,
            f"{args.class_name}: {count}/{args.count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        cv2.imshow("Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            filename = out_dir / f"{args.class_name}_{int(time.time())}_{count:03d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved: {filename}")
            count += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
