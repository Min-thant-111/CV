# An IoT-Enabled Edge-CV Framework for Real-Time Traffic Density Estimation and Intelligent Signaling

A modular, edge-deployable computer-vision and IoT framework that:
- Ingests a traffic video (uploaded via web or CLI)
- Detects vehicles (car, motorcycle, bus, truck) using **YOLO**
- Tracks them across frames using **ByteTrack** (persistent IDs)
- Estimates traffic density using **PCU-weighted occupancy**
- Decides optimal signal timing (GREEN 30 / 50 / 70 s) using a rule-based engine
- Dispatches decisions to an IoT controller over **MQTT**
- Displays live results on a **web dashboard**

> **Academic context:** All density values are estimates derived from video frame analysis, not physical sensor measurements. Reported percentages represent a vehicle-count-to-road-capacity approximation (PCU units) and should be interpreted accordingly.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Web Dashboard                         │
│         (Upload · Progress · Density · Signal · Counts)      │
└────────────────────────┬─────────────────────────────────────┘
                         │  HTTP / Server-Sent Events
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     Flask Web Server                         │
│                      backend/app.py                          │
└────────────────────────┬─────────────────────────────────────┘
                         │  Python call (same process)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Traffic Pipeline                          │
│                   backend/pipeline.py                        │
│                                                              │
│  VideoProcessor → VehicleTracker → DensityEngine            │
│       → SignalDecisionEngine → MQTTPublisher                 │
└───────┬──────────────────────────────────────────────────────┘
        │  MQTT  (topic: traffic/intersection1/signal)
        ▼
┌──────────────────────────────────────────────────────────────┐
│               MQTT Broker (e.g., Mosquitto)                  │
└───────┬──────────────────────────────────────────────────────┘
        │  MQTT subscribe
        ▼
┌──────────────────────────────────────────────────────────────┐
│            Virtual IoT Traffic Controller                    │
│          backend/controller/virtual_controller.py            │
│                                                              │
│         🟢 GREEN → 🟡 YELLOW → 🔴 RED  (countdown)          │
└──────────────────────────────────────────────────────────────┘
```

### Layer Separation Rules (strictly enforced)
| Layer | Allowed to call | NOT allowed to call |
|-------|----------------|---------------------|
| Video | — | YOLO, MQTT, signal logic |
| Detection | Video frames | Tracker, density, MQTT |
| Tracking | Detector | Density, signal, MQTT |
| Density | Tracker output | Signal, MQTT |
| Signal | Density output | MQTT, controller |
| MQTT Publisher | Signal output | Video, detection, density |
| Virtual Controller | MQTT broker only | All CV modules |

---

## Project Structure

```
.
├── backend/
│   ├── app.py               # Flask web application (SSE streaming, upload API)
│   ├── main.py              # CLI entry point
│   ├── pipeline.py          # Full pipeline orchestrator
│   ├── pipeline_config.py   # Unified PipelineConfig dataclass
│   ├── __init__.py
│   │
│   ├── video/
│   │   ├── video_reader.py      # Frame-by-frame generator (OpenCV)
│   │   └── video_processor.py   # Metadata extraction, frame skip, resize
│   │
│   ├── detection/
│   │   └── yolo_detector.py     # YOLOv8 vehicle detection wrapper
│   │
│   ├── tracking/
│   │   └── tracker.py           # ByteTrack vehicle tracker (Ultralytics)
│   │
│   ├── road/
│   │   └── path_estimator.py    # Automatic multi-frame road-path detection
│   │
│   ├── density/
│   │   └── density_engine.py    # PCU-weighted density estimator + ROI support
│   │
│   ├── signaling/
│   │   └── signal_engine.py     # Rule-based signal decision engine (Strategy pattern)
│   │
│   ├── mqtt/
│   │   └── mqtt_publisher.py    # Paho-MQTT publisher with graceful offline handling
│   │
│   ├── controller/
│   │   └── virtual_controller.py  # IoT traffic light simulator (MQTT subscriber)
│   │
│   └── models/
│       ├── detection.py         # Detection dataclass
│       ├── tracking.py          # TrackedObject, FrameTracks dataclasses
│       ├── density.py           # DensityMetrics dataclass
│       ├── signaling.py         # SignalDecision dataclass
│       └── mqtt.py              # MQTTPayload dataclass
│
├── frontend/
│   ├── templates/
│   │   └── index.html           # Dashboard HTML (served by Flask)
│   └── static/
│       ├── css/styles.css       # Dark premium HUD stylesheet
│       └── js/app.js            # Upload, SSE stream, live UI updates
│
├── tests/
│   ├── test_video.py            # VideoReader and VideoProcessor tests
│   ├── test_detection.py        # YOLODetector tests (mocked inference)
│   ├── test_tracking.py         # ByteTrack tests (mocked tracking)
│   ├── test_road_paths.py       # Automatic road-path estimator tests
│   ├── test_density.py          # DensityEngine tests (LOW/MEDIUM/HIGH)
│   ├── test_signaling.py        # SignalDecisionEngine tests
│   ├── test_mqtt.py             # MQTTPublisher tests (mocked client)
│   ├── test_controller.py       # VirtualTrafficController tests
│   ├── test_pipeline.py         # End-to-end integration tests
│   └── test_app.py              # Flask API route tests
│
├── videos/                      # Input traffic videos (upload destination)
├── outputs/                     # Processing logs and outputs
├── models/                      # YOLO model weights
├── .env.example                 # Environment configuration template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [Mosquitto MQTT Broker](https://mosquitto.org/download/) *(optional — MQTT gracefully skips if unavailable)*
- A YOLOv8 model weight file (`yolov8n.pt` is auto-downloaded by Ultralytics on first run)

### 2. Clone and Setup

```bash
git clone <repository-url>
cd <project-folder>

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env to set MQTT broker, model path, and signal timings
```

Key `.env` variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Path to YOLO weights |
| `CONFIDENCE_THRESHOLD` | `0.35` | Detection confidence threshold |
| `MQTT_BROKER_HOST` | `localhost` | MQTT broker address |
| `MQTT_BROKER_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC_SIGNAL` | `traffic/intersection1/signal` | Publish topic |
| `GREEN_TIME_LOW` | `30` | Green duration (seconds) for LOW density |
| `GREEN_TIME_MED` | `50` | Green duration (seconds) for MEDIUM density |
| `GREEN_TIME_HIGH` | `70` | Green duration (seconds) for HIGH density |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Running the System

### Option A — Web Dashboard (recommended for demonstration)

```bash
python backend/app.py
```

Open **http://localhost:5000** in your browser, upload a traffic video, and watch the dashboard update in real time.

### Option B — CLI Pipeline

```bash
python -m backend.main --video videos/sample_traffic.mp4
```

CLI flags (all override `.env` values):

```
--video           Path to input video
--model           YOLO model path (default: yolov8n.pt)
--conf            Detection confidence (default: 0.35)
--frame-skip      Frames to skip between processing (default: 2)
--publish-interval MQTT/log publish every N processed frames (default: 30)
--mqtt-host       MQTT broker hostname
--mqtt-port       MQTT broker port
--mqtt-topic      MQTT topic
--no-mqtt         Disable MQTT entirely
```

### Option C — Virtual IoT Controller (separate terminal)

```bash
python -c "
from backend.controller.virtual_controller import VirtualTrafficController, ControllerConfig
ctrl = VirtualTrafficController(ControllerConfig(broker_host='localhost'))
ctrl.connect()
ctrl.run_forever()
"
```

The controller subscribes to `traffic/intersection1/signal` and displays the `GREEN → YELLOW → RED` countdown whenever the pipeline publishes a decision.

---

## Signal Decision Logic

The signal engine implements the **Strategy Pattern** and defaults to a rule-based strategy:

| Density Level | PCU Threshold | Base GREEN |
|---------------|--------------|------------|
| LOW | < 35% occupancy | 30 seconds |
| MEDIUM | 35% – 70% | 50 seconds |
| HIGH | ≥ 70% occupancy | 70 seconds |

The final green time is adaptive rather than fixed:

`base time + total-vehicle adjustment + vehicles-per-path adjustment`

With the default adjustments, 9 vehicles on one path receive 88 seconds,
while 9 vehicles across two paths receive 64 seconds. Total vehicle count is
also included separately, so two cases with the same density percentage do not
automatically receive the same green time. The configurable safety bounds are
still applied last.

### PCU Weights (Passenger Car Units)

| Vehicle Class | COCO Class ID | PCU Weight |
|---------------|---------------|-----------|
| Motorcycle | 3 | 0.5 |
| Car | 2 | 1.0 |
| Bus | 5 | 2.5 |
| Truck | 7 | 3.0 |

Density percentage = `(total PCU in frame / (capacity per path × path count)) × 100`

Default capacity is `10.0 PCU` per path (configurable in `DensityConfig`). Values
above 100% are retained to show demand beyond road capacity. For example, 30
cars on one path is 300%, while the same 30 cars on three paths is 100%.

The path count is detected automatically from multiple frames of each input
video by finding persistent perspective-aligned lane and road boundaries. When
the video does not contain enough reliable boundary evidence, the estimator
uses a conservative one-path fallback and reports low confidence.

---

## MQTT Message Format

The publisher sends the following JSON payload to `traffic/intersection1/signal`:

```json
{
  "intersection": 1,
  "signal": "GREEN",
  "duration": 70,
  "density": 78.5,
  "density_level": "HIGH",
  "vehicle_count": 8,
  "timestamp": 1786381591.8
}
```

The Virtual Controller expects `signal`, `duration`, and `density_level` as required keys.

---

## Running Tests

```bash
python -m unittest discover tests -v
```

Expected output:

```
Ran 73 tests in ~12s

OK
```

### Test Coverage by Module

| Test File | Module Tested | Tests |
|-----------|--------------|-------|
| `test_video.py` | VideoReader, VideoProcessor | 5 |
| `test_detection.py` | YOLODetector | 5 |
| `test_tracking.py` | VehicleTracker (ByteTrack) | 4 |
| `test_density.py` | DensityEngine | 5 |
| `test_signaling.py` | SignalDecisionEngine | 7 |
| `test_mqtt.py` | MQTTPublisher | 4 |
| `test_controller.py` | VirtualTrafficController | 10 |
| `test_pipeline.py` | Full Pipeline (E2E) | 5 |
| `test_app.py` | Flask API routes | 8 |
| **Total** | | **53** |

All YOLO, ByteTrack, and MQTT broker interactions are **mocked** — tests run without a GPU, without a real model file, and without a live broker.

---

## Design Decisions

### Why ByteTrack over SORT / DeepSORT?
ByteTrack retains low-confidence detections in a secondary buffer before discarding them, producing more stable track IDs across partial occlusions — essential for accurate vehicle counting in dense traffic.

### Why PCU weighting?
A single bus occupies significantly more road space than a motorcycle. Counting all vehicles equally would underestimate density for heavy traffic. PCU (Passenger Car Unit) is a standard traffic engineering metric that converts different vehicle types into a common unit proportional to their road space usage.

### Why MQTT?
MQTT's lightweight publish-subscribe architecture allows the CV system and the traffic controller to be completely decoupled. The controller can be a physical ESP32 / Raspberry Pi or a software simulator — the CV pipeline requires **zero changes** in either case.

### Why Server-Sent Events (SSE) over WebSockets?
SSE is a native browser API over standard HTTP — no additional libraries required, no connection upgrade handshake, and unidirectional (server → client) which matches the pipeline's data flow perfectly.

---

## Limitations and Future Work

- **Density model:** PCU occupancy is a simplified estimation, not a physical sensor reading.
- **Single intersection:** Currently hardcoded to `intersection 1`. Multiple intersections can be supported by parameterising the MQTT topic and intersection ID.
- **ROI:** The density engine supports an ROI polygon parameter, but the web dashboard does not yet expose a drawing interface for it.
- **Model:** Uses `yolov8n` (nano) by default for speed. Accuracy improves with `yolov8s`, `yolov8m`, or domain-specific fine-tuned weights.
- **Night / weather conditions:** Detection accuracy degrades in poor lighting without model fine-tuning.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8 | Video reading and frame processing |
| `numpy` | ≥ 1.24 | Array operations |
| `ultralytics` | ≥ 8.0 | YOLOv8 detection + ByteTrack |
| `paho-mqtt` | ≥ 2.0 | MQTT publish and subscribe |
| `python-dotenv` | ≥ 1.0 | `.env` configuration loading |
| `flask` | ≥ 3.0 | Web server and API |
| `flask-cors` | ≥ 5.0 | Cross-origin resource sharing |

---

## License

This project was developed as a Final Year Project. All rights reserved.
