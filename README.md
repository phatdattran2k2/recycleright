# RecycleRight
**AI-powered waste sorting system — by 陳發達 & 楊瑋竣 at Datung University**

An edge-AI pipeline running on the NVIDIA Jetson Orin Nano that classifies waste in real time using a YOLOv8n object detection model and physically routes each item via servo motors controlled by an ESP32 microcontroller.

---

## How the System Works

The system follows a strict **Sense → Process → Decide → Act** loop across two devices:

```
┌─────────────────────────────── Jetson Orin Nano ──────────────────────────────────┐
│                                                                                    │
│  [IMX219 Camera]                                                                   │
│       │  frame (1280×720 @ 60 fps)                                                 │
│       ▼                                                                            │
│  CameraCapture ──(Queue maxsize=1)──▶ Detector                                    │
│                                          │                                         │
│                                          ├─ MOG2 background subtraction            │
│                                          │   (skip YOLO if no motion)              │
│                                          │                                         │
│                                          ├─ YOLOv8n TensorRT FP16 inference        │
│                                          │   (conf ≥ 0.50)                         │
│                                          │                                         │
│                                          ├─ Voting: accumulate 10 frames           │
│                                          │   need ≥ 6 hits → winner label          │
│                                          │                                         │
│                                          ├─▶ AudioPlayer  (ffplay, non-blocking)   │
│                                          │                                         │
│                                          └─▶ SerialController ──USB──▶ ESP32       │
│                                                                           │        │
│  HUD (tegrastats thread)                                                  │        │
│  overlays FPS + GPU% + power on screen                                    │        │
└───────────────────────────────────────────────────────────────────────────┼────────┘
                                                                            │
                                                              ┌─────────────▼──────┐
                                                              │       ESP32        │
                                                              │  PCA9685 (I²C)     │
                                                              │  4× SG90 servos    │
                                                              │  4× status LEDs    │
                                                              │  sends DONE back   │
                                                              └────────────────────┘
```

### Waste Classes & Actuator Mapping

| Class | Serial Command | Servo Channel | LED GPIO |
|---|---|---|---|
| PET bottle | `PET\n` | CH 0 | GPIO 23 |
| Glass bottle | `GLS\n` | CH 1 | GPIO 5 |
| Aluminum can | `MTL\n` | CH 2 | GPIO 4 |
| Tetra Pak | `PAP\n` | CH 3 | GPIO 2 |

Each servo opens to **120°**, holds for 5 seconds, then returns to **0°**. The LED stays on during motor motion and turns off when the cycle completes. The ESP32 sends `DONE:<label>` back over USB, which releases the `lid_ready` lock so the next detection cycle can begin.

### Threading Model

The pipeline runs **4 concurrent threads** that never block each other:

- **Thread 1 — Camera:** continuously captures frames from the IMX219 and pushes them into a `Queue(maxsize=1)`, dropping stale frames so Thread 2 always gets the freshest image.
- **Thread 2 — Detector:** pulls frames from the queue, runs MOG2 motion gating, then YOLOv8 inference, and applies the voting mechanism. Once a class is confirmed, it sends a UART command to the ESP32 and locks the system until a `DONE` reply is received.
- **Thread 3 — Serial:** listens for the ESP32's `DONE:<label>` reply over USB and releases the `lid_ready` lock, allowing the next detection cycle to begin.
- **Thread 4 — HUD:** parses `tegrastats` output once per second and updates the on-screen FPS / GPU% / power overlay.

### Motion Gating (MOG2)

Before running the YOLOv8 model (~28 ms), every frame is first passed through a MOG2 background subtractor (~1 ms). If no foreground contour exceeds 500 px², the frame is skipped entirely. This keeps GPU utilization low when the bin is idle.

### Voting System

To prevent false triggers from a single frame, the system accumulates **10 consecutive motion frames** and counts per-class detections. Only if the winning class appears in **≥ 6 of the 10 frames** is a command sent to the ESP32.

---

## ESP32 Firmware (`src/motorcontroll/esp32_controller.ino`)

The Arduino sketch running on the ESP32 is the physical actuator layer of the pipeline. It listens for single-line ASCII commands from the Jetson over USB serial, drives the appropriate servo via the PCA9685 PWM driver, lights the matching status LED, and replies with a `DONE` acknowledgement so the Jetson knows the bin lid has finished moving.

### Communication Protocol

| Direction | Format | Example |
|---|---|---|
| Jetson → ESP32 | `<CLASS>\n` (3-letter code) | `PET\n` |
| ESP32 → Jetson | `DONE:<label>\n` | `DONE:PET bottle\n` |

The ESP32 also emits human-readable status lines (e.g. `【收到】PET bottle → 通道 0 馬達動作中`) for debugging via the Arduino Serial Monitor. Only the `DONE:` line is parsed by `SerialController` on the Jetson side.

### Servo Angle Conversion

```
angleToPulse(angle) = map(angle, 0°, 180°, SERVOMIN=150, SERVOMAX=600)
```

This maps a degree value to the PCA9685 12-bit PWM tick count that drives a standard SG90 servo.

### Sorting Sequence (`runSorter`)

Each time a valid command is received, `runSorter` executes the following sequence:

1. **LED ON** — the status LED for that bin turns on.
2. **Servo open** — servo moves to **120°** (bin lid opens).
3. **Hold 5 s** — item falls through; a fixed `delay(5000)` gives enough time.
4. **Servo close** — servo returns to **0°** (bin lid closes).
5. **LED OFF** — status LED turns off.
6. **Flush serial buffer** — any commands that accumulated during the 5 s motion are discarded to avoid queued ghost triggers.
7. **Send `DONE:<label>`** — notifies the Jetson that the lid cycle is complete.

### Busy Flag

A global `bool busy` flag prevents re-entrant execution. While a sort cycle is running, incoming serial bytes are ignored (the `loop()` exits immediately). The buffer flush in step 6 above clears any backlog so the system is clean for the next detection.

### Hardware Initialisation (`setup`)

| Parameter | Value |
|---|---|
| Serial baud rate | 115200 |
| I²C pins (Wire) | SDA = GPIO 21, SCL = GPIO 22 |
| PCA9685 I²C address | 0x40 |
| Oscillator frequency | 27 MHz (corrected for clone boards) |
| PWM frequency | 50 Hz (standard servo frequency) |

All four servo channels are driven to 0° on startup so the bin lids begin in the closed position.

---

## Repository Structure

```
recycleright/
├── src/                        # All Python source — one main class per file
│   ├── main.py                 # Entry point: wires all objects, starts 4 threads
│   ├── camera.py               # CameraCapture — GStreamer/IMX219 capture loop
│   ├── detector.py             # Detector — MOG2 + YOLOv8 + voting + decision
│   ├── serial_controller.py    # SerialController — USB comms with ESP32
│   ├── hud.py                  # HUD — tegrastats parser + on-screen overlay
│   ├── audio.py                # AudioPlayer — non-blocking ffplay playback
│   └── motorcontroll/
│       └── esp32_controller.ino  # Arduino firmware for ESP32 + PCA9685
│
├── models/
│   ├── best.pt                 # Original FP32 YOLOv8n weights
│   └── best_fp16.engine        # TensorRT FP16 engine (generated by export script)
│
├── scripts/
│   ├── export_fp16.py          # Exports best.pt → TensorRT FP16 engine
│   ├── collect_dataset.py      # Dataset collection via CSI camera
│   └── collect_dataset_screen.py
│
├── data/                       # Held-out test images (not tracked in git)
│   └── raw/
│       ├── pet/
│       ├── glass/
│       ├── aluminum/
│       └── tetrapak/
│
├── calibration/                # Reserved for INT8 calibration dataset + script
├── deploy/                     # docker-compose.yml, deploy.sh (to be added)
├── report/                     # FINAL_REPORT.pdf, PRESENTATION.pdf, DEMO.mp4
├── tests/
│   └── integration/            # Integration tests (run on self-hosted Jetson runner)
│
├── pyproject.toml              # PDM-managed; ruff + pytest + coverage configured
├── Dockerfile                  # Runtime image for Jetson (--runtime nvidia)
└── README.md
```

---

## How to Run

### Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6.x
- IMX219 CSI camera connected to `sensor-id=0`
- ESP32 connected via USB (`/dev/ttyUSB0`)
- Python environment managed by [PDM](https://pdm-project.org/)

### 1. Install dependencies

```bash
pdm install
```

### 2. Generate the TensorRT FP16 engine (first time only, ~5 min)

```bash
python3 scripts/export_fp16.py
```

This reads `models/best.pt` and writes `models/best_fp16.engine`.

### 3. Flash the ESP32 firmware

Open `src/motorcontroll/esp32_controller.ino` in the Arduino IDE and upload to the ESP32.

### 4. Run the inference pipeline

```bash
python3 src/main.py
```

- **Local mode** (monitor attached): opens a `cv2.imshow` window; press `Q` to quit.
- **SSH mode** (no monitor): window is suppressed automatically; press `Ctrl+C` to quit.

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8n |
| Input resolution | 640 × 640 |
| Classes | 4 (PET bottle, Glass bottle, Aluminum can, Tetra Pak) |
| FP32 weights | `models/best.pt` (6 MB) |
| FP16 engine | `models/best_fp16.engine` (generated locally on Jetson) |
| Confidence threshold | 0.50 |

---

## Hardware

| Component | Detail |
|---|---|
| Edge compute | NVIDIA Jetson Orin Nano (8 GB) |
| Camera | Raspberry Pi IMX219, connected via CSI |
| Microcontroller | ESP32 (USB serial at 115200 baud) |
| Servo driver | PCA9685 PWM driver (I²C, address 0x40) |
| Servos | 4× SG90, angles 0° (closed) / 120° (open) |
| Status LEDs | 4× GPIO-driven (GPIOs 23, 5, 4, 2) |
