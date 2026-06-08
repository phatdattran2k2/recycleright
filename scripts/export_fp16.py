#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""將 best.pt 匯出為 TensorRT FP16 引擎，輸出至 models/best_fp16.engine。"""

import shutil
from pathlib import Path

from ultralytics import YOLO

SRC_PT     = Path(__file__).parent.parent / "models" / "best.pt"
MODELS_DIR = Path(__file__).parent.parent / "models"
DST_ENGINE = MODELS_DIR / "best_fp16.engine"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"載入模型: {SRC_PT}")
    model = YOLO(str(SRC_PT))

    print("匯出 TensorRT FP16（首次約需 3–5 分鐘）…")
    exported = model.export(
        format="engine",
        half=True,       # FP16
        device=0,        # GPU 0
        imgsz=640,
        workspace=4,     # TensorRT workspace GB
        verbose=False,
    )

    # Ultralytics 會把 .engine 存在 best.pt 旁邊
    generated = Path(str(exported))
    shutil.copy(generated, DST_ENGINE)
    print(f"引擎已複製至: {DST_ENGINE}")
    print("完成！更新 src/detector.py 的 MODEL_PATH 後即可使用。")


if __name__ == "__main__":
    main()
