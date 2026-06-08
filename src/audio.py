#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

import shutil
import subprocess  # nosec

AUDIO_DIR = "/home/jetson/recycleright/src/音樂"

CLASS_AUDIO = {
    "PET bottle":   f"{AUDIO_DIR}/寶特瓶.mp3",
    "Glass bottle": f"{AUDIO_DIR}/玻璃瓶.mp3",
    "Aluminum can": f"{AUDIO_DIR}/鋁罐.mp3",
    "Tetra Pak":    f"{AUDIO_DIR}/鋁箔包.mp3",
}


class AudioPlayer:
    """非阻塞播放音檔，不影響推論 thread。"""

    @staticmethod
    def play(label: str) -> None:
        path = CLASS_AUDIO.get(label)
        ffplay = shutil.which("ffplay")
        if path and ffplay:
            subprocess.Popen(
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            )  # nosec
