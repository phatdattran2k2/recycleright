# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題
# Root conftest — chạy trước mọi import, mock toàn bộ hardware deps

import sys
from unittest.mock import MagicMock, patch

# ── 1. Mock ultralytics hoàn chỉnh ───────────────────────────────────────────
yolo_instance = MagicMock()
yolo_instance.return_value = yolo_instance

ultralytics_mock = MagicMock()
ultralytics_mock.YOLO = yolo_instance
sys.modules['ultralytics'] = ultralytics_mock

# ── 2. Mock cv2 hoàn chỉnh ───────────────────────────────────────────────────
cv2_mock = MagicMock()
cv2_mock.CAP_GSTREAMER        = 200
cv2_mock.MORPH_ELLIPSE        = 2
cv2_mock.RETR_EXTERNAL        = 0
cv2_mock.CHAIN_APPROX_SIMPLE  = 1
cv2_mock.FONT_HERSHEY_SIMPLEX = 0
cv2_mock.LINE_AA              = 16
cv2_mock.VideoCapture.return_value.isOpened.return_value = False
cv2_mock.createBackgroundSubtractorMOG2.return_value = MagicMock()
cv2_mock.getStructuringElement.return_value = MagicMock()
cv2_mock.getTextSize.return_value = ((100, 20), 5)
cv2_mock.findContours.return_value = ([], None)
cv2_mock.contourArea.return_value = 0
sys.modules['cv2'] = cv2_mock

# ── 3. Mock serial hoàn chỉnh ────────────────────────────────────────────────
serial_mock = MagicMock()

class FakeSerialError(Exception):
    pass

serial_mock.Serial.side_effect = FakeSerialError("no port in test env")
sys.modules['serial'] = serial_mock

# ── 4. Mock Jetson.GPIO ───────────────────────────────────────────────────────
sys.modules['Jetson']        = MagicMock()
sys.modules['Jetson.GPIO']   = MagicMock()

# ── 5. Mock paho-mqtt ─────────────────────────────────────────────────────────
sys.modules['paho']                = MagicMock()
sys.modules['paho.mqtt']           = MagicMock()
sys.modules['paho.mqtt.client']    = MagicMock()

# ── 6. Mock smbus2, Adafruit ─────────────────────────────────────────────────
sys.modules['smbus2']              = MagicMock()
sys.modules['Adafruit_PCA9685']    = MagicMock()

# ── 7. Block ROS2 plugin ─────────────────────────────────────────────────────
for ros_mod in [
    'launch', 'launch_ros', 'launch_testing',
    'launch.frontend', 'launch.actions',
    'launch_testing.tools', 'launch_testing.tools.process',
]:
    sys.modules[ros_mod] = MagicMock()
