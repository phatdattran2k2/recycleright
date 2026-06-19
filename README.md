# RecycleRight — AI-Powered Waste Sorting System

**by 陳發達 & 楊瑋竣 at Datung University**
*I4210 AI實務專題 — Capstone Final Report*
*Copyright (c) 2026 陳發達, 楊瑋竣 — Datung University — I4210 AI實務專題*

---

## §1 — Project Problem Statement

### The Problem

Improper waste sorting at the point of disposal is a persistent challenge in recycling programs worldwide. In Taiwan, four major recyclable categories — PET bottles, glass bottles, aluminum cans, and Tetra Pak cartons — are routinely mixed into the same bin by users who are either unsure of the category or simply inattentive. When a single contaminated bin is introduced into the recycling stream, it degrades downstream processing and increases the cost of manual re-sorting.

Manual sorting stations at schools, offices, and public venues require either dedicated staff (expensive) or rely entirely on user compliance (unreliable). The result is a recycling stream that is significantly less pure than it should be, reducing the economic value of recovered materials.

### Who Benefits

RecycleRight targets shared-space venues — classrooms, university common areas, cafeterias, and office floors — where a high volume of mixed recyclables is generated under low supervision. The primary beneficiaries are:

- **Venue operators**: reduced contamination in recycling bins without staffing costs.
- **Recycling processors**: cleaner input material, lower re-sort labor.
- **End-users**: a guided experience that also builds sorting awareness through audio feedback.

### Why Edge AI

A cloud-connected sorting system introduces two critical failure modes for a physical actuator loop: network latency and connectivity loss. A servo that waits 300 ms for a cloud inference result will physically drop the item before the lid opens. A sorting station with no internet is completely inoperable.

Running YOLOv8n inference locally on the NVIDIA Jetson Orin Nano eliminates both failure modes. The Jetson provides GPU-accelerated inference at sustained throughput (≥ 25 FPS target) without any network dependency. The complete Sense → Process → Decide → Act loop runs on-device: the IMX219 CSI camera captures the item, the TensorRT FP16 engine classifies it, the decision logic selects the correct bin, and the ESP32 microcontroller drives the servo to open the lid — all within the local hardware stack.

### Refinement from the Proposal

The proposal identified the core challenge correctly but underestimated the importance of **motion gating**: running full YOLO inference at 60 FPS on every frame, including frames showing an empty bin, would consume significant GPU headroom with no sorting benefit. The implementation adds a MOG2 background subtraction layer that skips YOLO when no object is present, preserving GPU and power budget for frames that actually matter.

---

## §2 — Final Architecture

### System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JETSON ORIN NANO (8 GB)                             │
│                                                                         │
│  ┌──────────────┐   capture_queue   ┌───────────────────────────────┐  │
│  │ CameraCapture│ ────────────────▶ │         Detector              │  │
│  │ (Thread 1)   │   maxsize=1       │  (Thread 2)                   │  │
│  │              │                   │                               │  │
│  │ IMX219 CSI   │                   │  1. MOG2 motion gate          │  │
│  │ 1280×720     │                   │  2. YOLOv8n TRT FP16 infer   │  │
│  │ 60 FPS       │                   │  3. 10-frame voting           │  │
│  │ GStreamer     │                   │  4. Threshold ≥ 6 / 10       │  │
│  │ NV12 → BGR   │                   │  5. Send serial + play audio  │  │
│  └──────────────┘                   └────────────┬──────────────────┘  │
│                                                  │                      │
│  ┌──────────────┐   lid_ready event              │ serial CMD           │
│  │SerialCtrl    │ ◀──────────────────────────────┘                      │
│  │ (Thread 3)   │                                                       │
│  │ /dev/ttyUSB0 │  ─── "PET\n" / "GLS\n" / "MTL\n" / "PAP\n" ──▶     │
│  │ 115200 baud  │                                                       │
│  └──────────────┘                                                       │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │ HUD          │   tegrastats                                          │
│  │ (Thread 4)   │ ◀──────────────── sudo tegrastats --interval 1000    │
│  │ FPS overlay  │                                                       │
│  │ GPU% / power │                                                       │
│  └──────────────┘                                                       │
│                                                                         │
│  Docker Container: ghcr.io/dinos611451001/recycleright:latest           │
│  Runtime: --runtime nvidia  │  Network: host  │  Device: /dev/ttyUSB0  │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       │ USB Serial (115200 baud)
                          ┌────────────▼──────────────────────────────┐
                          │          ESP32 Microcontroller             │
                          │                                            │
                          │  I²C (SDA=GPIO21, SCL=GPIO22)             │
                          │      ▼                                     │
                          │  PCA9685 (0x40) PWM Driver                │
                          │  50 Hz, 27 MHz oscillator                  │
                          │                                            │
                          │  CH0 → SG90 Servo (PET bin)               │
                          │  CH1 → SG90 Servo (Glass bin)             │
                          │  CH2 → SG90 Servo (Aluminum bin)          │
                          │  CH3 → SG90 Servo (Tetra Pak bin)         │
                          │                                            │
                          │  LED: GPIO23 / 5 / 4 / 2                  │
                          │                                            │
                          │  Sequence: LED ON → Servo 120° → 5s       │
                          │           → Servo 0° → LED OFF → "DONE:"  │
                          └────────────────────────────────────────────┘
```

### MQTT Topic Map

In this implementation, the Sense → Process → Decide → Act loop communicates internally via thread-safe Python primitives rather than MQTT (a design decision explained in §2 Changes below). The inter-thread communication map is:

| Channel | Type | From → To | Content |
|---------|------|-----------|---------|
| `capture_queue` | `queue.Queue(maxsize=1)` | CameraCapture → Detector | `(frame, timestamp)` tuple |
| `lid_ready` | `threading.Event` | SerialController → Detector | Set when ESP32 sends `DONE:` |
| `stop_event` | `threading.Event` | main() → all threads | Global shutdown signal |
| `serial.send()` | USB serial | Detector → ESP32 | `b"PET\n"` / `b"GLS\n"` / `b"MTL\n"` / `b"PAP\n"` |
| `frame_lock` | `threading.Lock` | Detector → HUD.draw() | Latest annotated frame |

### Docker Container Boundary

```
Host Jetson OS
├── /dev/ttyUSB0       ─────── mounted into container
├── /dev/video0        ─────── accessed via --privileged (GStreamer nvargus)
├── NVIDIA GPU runtime ─────── --runtime nvidia
└── models/ volume     ─────── ro mount into /app/models/

Container: dustynv/pytorch:2.7-r36.4.0 (base)
└── src/main.py → CameraCapture + Detector + SerialController + HUD
    └── models/best_fp16.engine (TensorRT FP16, mounted read-only)
```

### What Runs on Which Hardware

| Component | Hardware | Software |
|-----------|----------|----------|
| AI inference (YOLOv8n TRT FP16) | Jetson GPU (iGPU) | Ultralytics + TensorRT |
| MOG2 motion gating | Jetson CPU | OpenCV BackgroundSubtractorMOG2 |
| Camera capture | IMX219 CSI → Jetson ISP | GStreamer + nvarguscamerasrc |
| Audio feedback | Jetson audio out | ffplay (subprocess) |
| Servo actuation | ESP32 + PCA9685 | Arduino firmware (I²C) |
| HUD overlay + tegrastats | Jetson CPU | Python + subprocess |

### What Changed vs the Proposal Architecture

| Proposed | Shipped | Reason |
|----------|---------|--------|
| MQTT broker for inter-module messaging | Thread-safe Queue + threading.Event | Single-device deployment makes MQTT unnecessary overhead; direct Python primitives are lower latency and simpler to test |
| dashboard.py FastAPI WebSocket stream | Moved to optional (not in production path) | Prioritized reliability of the actuator loop; dashboard can be added without touching the inference path |
| FP32 inference | TensorRT FP16 engine | 9% FPS improvement (39.5 → 43.1 FPS), 14% power reduction (8.7 W → 7.5 W) with negligible accuracy loss |
| Single-frame classification | 10-frame majority voting (threshold 6/10) | Eliminates single-frame false positives from motion blur and partial occlusion |

---

## §3 — Implementation Highlights

### 1. MOG2 Motion-Gated Inference Pipeline

The most impactful architectural decision was adding a motion gate in front of the YOLO inference call. Instead of running the 23 ms TensorRT FP16 inference on every frame — including the majority of frames showing an empty, static bin — the detector first runs OpenCV's MOG2 background subtractor:

```python
# src/detector.py — motion gate logic
fgmask = self._mog.apply(frame)
fgmask = cv2.erode(fgmask, None, iterations=1)
fgmask = cv2.dilate(fgmask, None, iterations=3)
contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
motion_detected = any(cv2.contourArea(c) > MIN_AREA for c in contours)
```

`MIN_AREA = 500 px²` was chosen empirically: it passes items as small as a crushed aluminum can while filtering camera noise and minor lighting fluctuations. MOG2 runs entirely on CPU in under 2 ms, meaning YOLO (23 ms GPU) is skipped on idle frames. During a typical sorting session where items are presented every 5–10 seconds, this gate suppresses > 95% of inference calls and is the primary reason the live system's GPU utilization is ~55% rather than saturated.

### 2. 10-Frame Majority Voting with Lid Interlock

A single-frame classification is unreliable when an item is still in motion. The voting system accumulates predictions over `VOTE_FRAMES = 10` consecutive frames and requires `VOTE_THRESHOLD = 6` (≥ 60%) agreement before committing to a decision:

```python
# src/detector.py — voting logic
if self._vote_count < VOTE_FRAMES:
    for box in result.boxes:
        label = self._model.names[int(box.cls)]
        self._vote_counts[label] = self._vote_counts.get(label, 0) + 1
    self._vote_count += 1
else:
    winner = max(self._vote_counts, key=self._vote_counts.get)
    if self._vote_counts[winner] >= VOTE_THRESHOLD and self._lid_ready.is_set():
        self._lid_ready.clear()
        self._serial.send(CLASS_CMD[winner])
        AudioPlayer.play(winner)
    self._vote_counts.clear()
    self._vote_count = 0
```

The `lid_ready` event acts as a physical interlock: once a command is sent to the ESP32, the event is cleared and no new commands can be dispatched until the ESP32 completes its servo sequence and sends back `"DONE:<label>\n"`. This prevents command stacking if a second item is presented while the first bin lid is still open.

### 3. TensorRT FP16 Engine Export and Calibration

The FP32 PyTorch model (`best.pt`, 6 MB) was exported to a TensorRT FP16 engine using `scripts/export_fp16.py`:

```python
# scripts/export_fp16.py
from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="engine", half=True, imgsz=640, device=0)
```

The export runs on the Jetson itself (takes ~3–5 minutes for YOLOv8n at 640×640), which ensures the TensorRT engine is compiled for the exact GPU architecture of the Orin Nano (SM 8.7, Ampere). The resulting engine is copied to `models/best_fp16.engine` and mounted read-only into the Docker container. No INT8 calibration was applied because FP16 already delivered the target FPS with acceptable accuracy loss (< 0.5 mAP@50 points based on Ultralytics documented benchmarks for this model size).

### 4. Four-Thread Architecture with GStreamer CSI Camera

The complete pipeline runs across four daemon threads coordinated by shared events and a single-element queue:

| Thread | Class | Key responsibility |
|--------|-------|--------------------|
| Thread 1 | `CameraCapture` | GStreamer → OpenCV BGR frames, warm-up 30 frames |
| Thread 2 | `Detector` | MOG2 gate → TRT inference → voting → serial dispatch |
| Thread 3 | `SerialController` | UART read loop, `DONE:` detection → `lid_ready.set()` |
| Thread 4 | `HUD` | `tegrastats` parser → FPS/GPU% overlay on frame copy |

The `capture_queue` has `maxsize=1` with a non-blocking `put_nowait` that discards stale frames. This means the Detector always processes the most recent camera frame rather than queuing up a backlog during heavy inference periods — critical for a real-time sorting system where a 500 ms old frame could show an item that has already fallen.

The GStreamer pipeline string for the Jetson CSI camera:

```
nvarguscamerasrc sensor-id=0 !
video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 !
nvvidconv ! video/x-raw,format=BGRx !
videoconvert ! video/x-raw,format=BGR !
appsink drop=1
```

`nvarguscamerasrc` accesses the ISP directly for hardware-accelerated demosaicing, and `nvvidconv` converts NV12 from the ISP into BGR for OpenCV, all without CPU copy overhead.

### 5. ESP32 Servo Control with Busy Guard and Serial Flush

The ESP32 firmware (`src/motorcontroll/esp32_controller.ino`) implements a `busy` flag that ignores incoming serial bytes during a sort cycle:

```cpp
void loop() {
  if (Serial.available() && !busy) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "PET") runSorter(0, "PET");
    else if (cmd == "GLS") runSorter(1, "GLS");
    // ...
  }
}

void runSorter(int ch, String label) {
  busy = true;
  digitalWrite(LED_PIN[ch], HIGH);
  setAngle(ch, 120);   // open lid
  delay(5000);          // hold for item to fall
  setAngle(ch, 0);     // close lid
  digitalWrite(LED_PIN[ch], LOW);
  while (Serial.available()) Serial.read();  // flush backlog
  Serial.println("DONE:" + label);
  busy = false;
}
```

After the servo sequence, the firmware flushes the serial receive buffer before sending `DONE:`, ensuring that any spurious bytes that arrived during the 5-second hold do not trigger a second sort cycle.

---

## §4 — Test Set Description

### Data Source and Collection

The dataset was custom-collected for this project using `scripts/collect_dataset.py`, which captures images from the IMX219 CSI camera under real-world conditions. Items were photographed in the actual deployment environment (a university lab) under mixed artificial lighting with varying backgrounds.

| Class | Train | Validation | Test | Total |
|-------|------:|----------:|-----:|------:|
| PET bottle | 5,093 | 654 | 673 | 6,420 |
| Aluminum can | 2,557 | 304 | 297 | 3,158 |
| Glass bottle | 2,300 | 255 | 305 | 2,860 |
| Tetra Pak | 1,820 | 271 | 221 | 2,312 |
| **Total** | **11,770** | **1,484** | **1,496** | **14,750** |

Note: instance counts exceed image counts because some images contain multiple objects.

**Image counts**:
- Train: 10,056 images
- Validation: 1,257 images
- Test: **1,258 images** (held out, never seen during training or hyperparameter search)

### Train / Validation / Test Split

The split was performed at the image level (not instance level) using an approximate 80 / 10 / 10 ratio. The test set was fixed before any training began and was not used to select model checkpoints or tune thresholds — only `best.pt` (selected by validation mAP on the val set) was evaluated against the test set for the final accuracy numbers.

### Edge Cases Included

The collection script was run at multiple times of day to capture:

- **Varying illumination**: fluorescent overhead light at noon vs. evening tungsten
- **Partial occlusion**: items placed at the camera edge or behind the bin lip
- **Label ambiguity**: crushed PET bottles (could be confused with paper Tetra Pak at low resolution)
- **Multiple items**: two items in frame simultaneously (common in busy sorting stations)
- **Background variation**: white lab bench, dark lab countertop, outdoor concrete

### Accuracy Gate Threshold

The CI accuracy gate (`tests/accuracy_gate.py`) enforces `mAP@50 ≥ 0.70` (a conservative minimum), and any commit that drops accuracy more than 3 percentage points below the baseline (`accuracy_baseline.json`, mAP@50 = 0.9602) fails the pipeline. This ensures model regressions from weight updates or data augmentation changes are caught before deployment.

---

## §5 — Performance Requirements & Optimization Journey

### Numerical Targets Set Up Front

| Metric | Target | Rationale |
|--------|--------|-----------|
| FPS | ≥ 25 FPS | Items fall into bin in ~2 s; 25 FPS → 50 frames → reliable voting |
| p95 latency | ≤ 60 ms | Ensures the decision is made before the item hits the bin bottom |
| mAP@50 | ≥ 0.80 | Conservative threshold for real-world reliability |
| Power (VDD_IN) | ≤ 15 W | Jetson Orin Nano rated TDP |
| GPU% | ≥ 40% | Confirms GPU is actually being used (not CPU fallback) |
| CPU% | ≤ 60% | Leave headroom for the OS, camera, serial, audio threads |

### Baseline — FP32 PyTorch (Step 0)

The first end-to-end run used `best.pt` loaded directly by Ultralytics (FP32 PyTorch, no TensorRT). Benchmarked offline on 320 fixed images (3 loops, 15W power mode):

| Metric | Baseline (FP32) |
|--------|---------------:|
| FPS | 39.5 |
| p50 latency | 24.8 ms |
| p95 latency | 28.0 ms |
| p99 latency | 30.7 ms |
| CPU% | 39.2% |
| GPU% | 73.0% |
| Avg power (VDD_IN) | 8,697 mW (8.7 W) |
| Peak power | 8,998 mW (9.0 W) |

The baseline already beat the 25 FPS target, but GPU was at 73% with 8.7 W power — both above the targets we wanted for a thermally stable continuous deployment.

### Optimization Step 1 — TensorRT FP16 Export

Converted `best.pt` to `best_fp16.engine` using `scripts/export_fp16.py` (`model.export(format="engine", half=True)`). TensorRT compiles and fuses operations for the Orin Nano GPU at FP16 precision. Benchmarked with the same 320-image set:

| Metric | Step 0 (FP32) | Step 1 (FP16 TRT) | Delta |
|--------|-------------:|------------------:|------:|
| FPS | 39.5 | 43.1 | **+9.1%** |
| p50 latency | 24.8 ms | 22.9 ms | −1.9 ms |
| p95 latency | 28.0 ms | 30.2 ms | +2.2 ms |
| p99 latency | 30.7 ms | 35.2 ms | +4.5 ms |
| CPU% | 39.2% | 39.1% | ≈0 |
| GPU% | 73.0% | 66.5% | **−6.5 pp** |
| Avg power | 8,697 mW | 7,507 mW | **−1,190 mW (−13.7%)** |
| Peak power | 8,998 mW | 8,464 mW | −534 mW |

FP16 TRT delivered the expected gains: faster inference and significantly lower power consumption. The p95/p99 increase is an artifact of TensorRT's first-batch warmup — subsequent batches are consistently faster.

### Optimization Step 2 — Power Mode 25W

Switched `nvpmodel` from Mode 0 (15W) to Mode 1 (25W) to test whether unlocking the CPU/GPU clock ceiling improved throughput:

| Metric | Step 1 (15W) | Step 2 (25W) | Delta |
|--------|-------------:|-------------:|------:|
| FPS | 43.1 | 44.8 | +1.7 |
| p50 latency | 22.9 ms | 21.2 ms | −1.7 ms |
| CPU% | 39.1% | 36.6% | −2.5 pp |
| GPU% | 66.5% | 61.9% | −4.6 pp |
| Avg power | 7,507 mW | 7,459 mW | −48 mW |

Marginal improvement (~4% FPS). At 25W mode the GPU clocks are higher but the workload is already GPU-bound by the TRT kernel; the extra headroom mostly reduces queue waiting time.

### Optimization Step 3 — MAXN_SUPER Mode

Tested the maximum performance envelope:

| Metric | Step 2 (25W) | Step 3 (MAXN_SUPER) | Delta |
|--------|-------------:|--------------------:|------:|
| FPS | 44.8 | 44.4 | −0.4 |
| p50 latency | 21.2 ms | 21.8 ms | +0.6 ms |
| Avg power | 7,459 mW | 7,833 mW | +374 mW |
| Peak power | 7,888 mW | 9,066 mW | +1,178 mW |

No improvement over 25W mode, with higher peak power. MAXN_SUPER unlocks additional compute blocks that YOLOv8n is too small to saturate. Conclusion: **25W mode is the optimal operating point for this model.**

### Optimization Step 4 — MOG2 Motion Gate (Full Live Pipeline)

Introduced the MOG2 background subtraction gate and measured the full 4-thread live pipeline in MAXN_SUPER mode with the camera active and inference flowing:

| Metric | Offline (Step 3) | Live Pipeline (Step 4) | Delta |
|--------|----------------:|---------------------:|------:|
| FPS (inference) | 44.4 | N/A (motion-gated) | — |
| CPU% | 37.2% | 48.5% | +11.3 pp |
| GPU% | 65.6% | 54.8% | −10.8 pp |
| Avg power | 7,833 mW | 10,765 mW | +2,932 mW |
| Peak power | 9,066 mW | 11,573 mW | +2,507 mW |

CPU rises because 4 threads are running concurrently (camera, MOG2, serial, HUD + tegrastats subprocess). GPU drops because MOG2 skips YOLO on idle frames, so the GPU is not being kept continuously hot. Power climbs in the live system because CPU work is higher and the camera pipeline draws additional power from the ISP and DDR bandwidth.

This is the correct operating point for deployment: the GPU ~55% headroom means the system has capacity to handle bursty presentation of items without thermal throttling.

### Did We Hit the Targets?

| Target | Goal | Achieved | Result |
|--------|------|----------|--------|
| FPS | ≥ 25 FPS | 43.1 FPS (offline) | **PASS** |
| p95 latency | ≤ 60 ms | 30.2 ms (offline) | **PASS** |
| mAP@50 | ≥ 0.80 | 0.9602 | **PASS** |
| Power (VDD_IN) | ≤ 15 W | 7.5 W (offline), 10.8 W (live) | **PASS** |
| GPU% | ≥ 40% | 54.8% (live) | **PASS** |
| CPU% | ≤ 60% | 48.5% (live) | **PASS** |

All six targets met.

---

## §6 — System Performance Results

### Accuracy — Held-Out Test Set (1,258 images, 1,496 instances)

Model evaluated: `best.pt` (FP32, on training server). Deployed as `best_fp16.engine` (TRT FP16 on Jetson Orin Nano). FP16 quantization is expected to cause < 0.5 mAP@50 drop based on Ultralytics benchmarks for YOLOv8n.

| Metric | Value |
|--------|------:|
| mAP@50 | **0.9602** (96.02%) |
| mAP@50-95 | 0.8275 (82.75%) |
| Precision | 0.9695 (96.95%) |
| Recall | 0.9638 (96.38%) |
| F1 Score | 0.9666 (96.66%) |

**Per-class AP@50:**

| Class | AP@50 | Precision | Recall |
|-------|------:|----------:|-------:|
| Aluminum can | **0.9730** (97.30%) | 0.9797 | 0.9609 |
| PET bottle | 0.9619 (96.19%) | 0.9716 | 0.9688 |
| Tetra Pak | 0.9565 (95.65%) | 0.9487 | 0.9683 |
| Glass bottle | 0.9495 (94.95%) | 0.9780 | 0.9371 |

All four classes exceed 94% AP@50, well above the CI gate minimum of 70%.

**Evaluation config**: conf_threshold = 0.50, IoU_threshold = 0.70, eval_date = 2026-05-24.

---

### Latency — FP16 TRT, 15W Power Mode, 320-Image Offline Benchmark

| Metric | Value |
|--------|------:|
| FPS | 43.1 |
| p50 latency | 22.9 ms |
| p95 latency | 30.2 ms |
| p99 latency | 35.2 ms |

Precision shipped: **FP16** (TensorRT FP16 engine).
Trade-off accepted: FP16 was chosen over INT8 because the expected accuracy gain from INT8 calibration is < 1% FPS while the risk of silent accuracy degradation without a calibrated dataset is higher. FP16 delivers sufficient throughput (43.1 FPS >> 25 FPS target) without a calibration dataset.

---

### Resource & Power — Live Pipeline (tegrastats, ≥ 60-second sustained run)

Data source: `tegrastats.log` → parsed by `scripts/parse_tegrastats.py` → `utilization.csv` (449 samples, 1 Hz sampling interval).

| Metric | Mean | p50 | p95 | Max |
|--------|-----:|----:|----:|----:|
| CPU% (avg across cores) | 32.1% | 30.5% | 48.2% | 55.3% |
| GPU% | 45.3% | 44.0% | 57.8% | 65.4% |
| RAM used (MB) | 6,448 | 6,451 | 6,475 | 6,512 |
| VDD_IN (mW) | 6,712 | 6,580 | 7,812 | 8,124 |
| GPU temperature (°C) | 53.2 | 53.0 | 55.0 | 57.0 |
| CPU temperature (°C) | 53.8 | 54.0 | 55.5 | 57.5 |

**Average power draw: 6.7 W** (well within the Jetson Orin Nano 15 W TDP).
**Peak power draw: 8.1 W** (well within limits; no thermal throttling observed).

---

## §7 — Lessons Learned

### 1. TensorRT Engine Build Must Happen on the Target Device

We initially tried to export the TRT engine on a laptop (x86, CUDA 11.8) and copy the `.engine` file to the Jetson. The engine failed to load with a "mismatched GPU architecture" error. TensorRT engines are compiled for a specific GPU SM version and CUDA version. The correct workflow is to run `export_fp16.py` on the Jetson itself. This cost us a day of debugging.

### 2. GStreamer Pipeline Strings Are Not Portable

The `nvarguscamerasrc` GStreamer pipeline works only on JetPack-enabled Jetson systems. On CI's Ubuntu cloud runners (no CUDA, no `nvarguscamerasrc`), the pipeline string causes `cv2.VideoCapture()` to fail at import time if the camera is opened. We solved this by mocking `cv2.VideoCapture` in `conftest.py`, which allows 90%+ test coverage without hardware. The lesson: design the test boundary so hardware can be substituted from the outside.

### 3. MOG2 Sensitivity Requires Tuning in the Actual Environment

The initial `MIN_AREA` threshold was 200 px², which triggered false positives from fluorescent light flicker and small shadows. Raising it to 500 px² eliminated these without missing any real item. The right value depends on the camera resolution, distance to the bin opening, and ambient lighting — it cannot be determined from a dataset alone and must be calibrated in situ.

### 4. The `lid_ready` Interlock Was Not in the Original Design

During early integration testing, we discovered that if an item was held above the bin and removed before the sorting decision fired, the next item placed would sometimes trigger two serial commands in rapid succession (one from the tail of the previous vote window). The `lid_ready` event, which clears on dispatch and is only set again by the ESP32's `DONE:` reply, completely eliminates this race condition. This is a pattern we didn't anticipate during the proposal phase.

### 5. tegrastats Parsing Is Non-Trivial

The `tegrastats` output format on JetPack 6.x differs from the JetPack 5.x examples in the course lab materials. The `VDD_CPU_GPU_CV` power rail label replaced `POM_5V_GPU`, and the CPU frequency format changed. We wrote `scripts/parse_tegrastats.py` with regex fallbacks for both formats. Always test the parser against actual log output before relying on the numbers.

---

## §8 — What We'd Do Differently if We Did It Again

### 1. Decide on INT8 vs FP16 at Week 1, Not Week 13

We went back and forth on whether to implement INT8 calibration throughout the project. The decision to ship FP16 was correct given our throughput margin, but the uncertainty caused us to leave a `calibration/` directory scaffolded in the repo without populating it. If we started again, we would benchmark FP16 vs INT8 at the very beginning on a small test set and commit to one path.

### 2. Write the Integration Tests Before Wiring the Actuator

We wired the servo-ESP32 circuit in Week 12 and then spent Week 13 retrofitting integration tests around code that was already running. Writing `tests/integration/test_integration.py` first — as contracts for what the hardware must do — would have caught the serial buffer flush bug before we saw it as an intermittent double-sort in production.

### 3. Collect a More Balanced Dataset Earlier

Glass bottles are the hardest class (AP@50 = 94.95%, lowest of four). We only realized this in Week 14 when the first accuracy numbers came in. A class-distribution analysis in Week 9 would have pointed us toward collecting more glass bottle samples while we still had time to re-train.

### 4. Pin the `nvpmodel` Setting in the Docker Entrypoint

During demos, we forgot which power mode was active and got inconsistent FPS numbers. The `nvpmodel -m 1` (25W mode) call should be added to `deploy/deploy.sh` (it requires `sudo` on the host, not inside the container), so the power mode is always deterministic on deploy.

### 5. Set Up the Self-Hosted GitHub Actions Runner in Week 10, Not Week 14

The runner registration took less than 30 minutes once we followed HW6 Step 0.0, but we postponed it until the integration-test job was actually needed. Starting it in Week 10 would have let us catch the first real integration failures weeks earlier, when we had more time to fix them.

---

## §9 — Individual Reflections

### 楊瑋竣

In this capstone I was primarily responsible for the software architecture and CI/CD pipeline. I built `src/detector.py` (the core inference + voting loop), `src/camera.py` (GStreamer CSI pipeline), and the full `.github/workflows/ci.yml` 5-stage pipeline. I also wrote the majority of the test suite and the `scripts/parse_tegrastats.py` parser.

The most technically interesting part was designing the `capture_queue(maxsize=1)` drop policy: every thread textbook I had read recommended bounded queues to prevent memory growth, but the non-blocking `put_nowait` with frame discard is less commonly discussed as a latency optimization. Seeing the fresh-frame policy eliminate the "stale classification" bug in real-time testing was satisfying.

I learned that writing tests before hardware is available is not just good practice — it's the only way to maintain development velocity when the physical device is shared between team members. The `conftest.py` mock layer let me build and test the full inference pipeline on a laptop before touching the Jetson.

The team divided work naturally: I focused on the Python stack and DevOps; 陳發達 focused on hardware integration, the ESP32 firmware, and the dataset collection. We pair-reviewed each other's code in weekly integration sessions and resolved conflicts in the voting-to-serial handoff through live debugging.

---

### 陳發達

My primary contributions in this capstone were the hardware integration layer, the ESP32 firmware, the dataset collection pipeline, and the model training workflow. I wrote `src/serial_controller.py`, `src/audio.py`, `src/motorcontroll/esp32_controller.ino`, and `scripts/collect_dataset.py`, and I ran all the `nvpmodel` benchmarking experiments.

The most important thing I learned is that a microcontroller's serial buffer does not automatically clear between commands. The `busy` flag and the `while (Serial.available()) Serial.read()` flush in the ESP32 firmware were added after observing the servo open twice on a single item — a bug that only appears under real timing conditions, never in unit tests. Physical debugging (watching the LED and servo with my own eyes while logging serial output) is irreplaceable for finding this class of bug.

I also underestimated how much variation there is in the same item category: a full glass bottle looks completely different from an empty one lying on its side. The collect_dataset script's 0.5-second interval was sometimes too fast for me to reposition items between shots, which introduced near-duplicate frames in the training set. A slower collection interval (1–2 seconds) with a pause-on-keypress mode would produce higher diversity per shot.

The team collaboration worked well because we established a clear interface boundary early: the Python stack sends a 4-byte command over serial; the ESP32 does the rest. As long as both sides honored that contract, we could develop independently. The `DONE:` reply pattern (which I defined in the firmware) turned out to be the key piece that enabled the `lid_ready` interlock on the Python side.

---

## §10 — Acknowledgments & References

### Open-Source Libraries

| Library | Version | Use |
|---------|---------|-----|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | ≥ 8.0 | Object detection model, TensorRT export |
| [OpenCV](https://opencv.org/) | ≥ 4.8 | GStreamer camera capture, MOG2 background subtraction, HUD rendering |
| [PyTorch](https://pytorch.org/) | 2.7 (JetPack) | Underlying ML runtime |
| [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers) | r36.4.0 | Base Docker image with JetPack 6.x, PyTorch, CUDA for Jetson |
| [Adafruit PCA9685](https://github.com/adafruit/Adafruit_Python_PCA9685) | ≥ 1.0 | PWM driver for servo actuation |
| [pyserial](https://pyserial.readthedocs.io/) | ≥ 3.5 | Serial UART communication with ESP32 |
| [Ruff](https://github.com/astral-sh/ruff) | ≥ 0.4 | Linting and code style enforcement |
| [pytest](https://pytest.org/) | ≥ 8.0 | Test framework |
| [bandit](https://bandit.readthedocs.io/) | ≥ 1.7 | Security static analysis |
| [pip-audit](https://github.com/pypa/pip-audit) | ≥ 2.7 | Dependency vulnerability scanning |

### Datasets

- Custom dataset: collected using `scripts/collect_dataset.py` on the Jetson Orin Nano with IMX219 CSI camera in the Datung University lab. All images and labels are original work.

### Course Materials Referenced

- `lab/LAB_12.md` — GHCR push workflow and self-hosted runner registration
- `homework/HOMEWORK_6.md` — 5-stage CI DAG, coverage gate configuration, rollback pattern
- Week 11 lecture slides — FastAPI + WebSocket MJPEG dashboard pattern

### Code Patterns Borrowed

- GStreamer pipeline string adapted from the Jetson Nano Developer Kit Getting Started Guide (NVIDIA, 2024) — modified for `nvarguscamerasrc` on JetPack 6.x
- `conftest.py` mock structure adapted from course lab examples; all test code is original

### Quick Verification

export MODEL_PATH=models/best_fp16.engine
python3 -m src.main

docker pull ghcr.io/phatdattran2k2/recycleright:latest
docker run --rm --runtime nvidia \
  -e MODEL_PATH=/models/best_fp16.engine \
  -v /home/jetson/recycleright/models:/models:ro \
  --entrypoint="" \
  ghcr.io/phatdattran2k2/recycleright:latest \
  bash -c "pip install --index-url https://pypi.org/simple pytest pytest-timeout -q && pytest tests/integration/ -v --timeout=120"
