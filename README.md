# An IoT-Enabled Edge-CV Framework for Real-Time Traffic Density Estimation and Intelligent Signaling

An edge computer vision and IoT framework that processes traffic camera feeds in real time, detects and tracks vehicles, estimates traffic density, dynamically determines optimal signal timing, and dispatches decisions to traffic controllers over MQTT.

---

## 🏗️ Architecture Flow

```
USER
 │
 │ Upload traffic video
 ▼
┌──────────────────┐
│   Web Interface  │
│   Video Upload   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Video Manager  │
│     OpenCV       │
└────────┬─────────┘
         │ frames
         ▼
┌──────────────────┐
│   YOLO Detector  │
│  Cars, Bikes,    │
│  Buses, Trucks   │
└────────┬─────────┘
         │ detected objects
         ▼
┌──────────────────┐
│    ByteTrack     │
│ Object Tracking  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Density Engine   │
│ Vehicle count    │
│ Occupancy        │
│ Density level    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Signal Decision  │
│     Engine       │
│ LOW  → 30 sec    │
│ MED  → 50 sec    │
│ HIGH → 70 sec    │
└────────┬─────────┘
         │ MQTT
         ▼
┌──────────────────┐
│  MQTT Broker     │
│   Mosquitto      │
└────────┬─────────┘
         │ MQTT
         ▼
┌──────────────────┐
│ Virtual Traffic  │
│   Controller     │
└────────┬─────────┘
         │
         ▼
 🚦 TRAFFIC LIGHT
 GREEN: 70 sec
```

---

## 📂 Project Structure

```
.
├── backend/
│   ├── controller/    # Virtual traffic controller module
│   ├── density/       # Density estimation engine & ROI calculation
│   ├── detection/     # Object detection wrapper (YOLO)
│   ├── models/        # Data models and schemas
│   ├── mqtt/          # MQTT telemetry & command publisher/subscriber
│   ├── signaling/     # Signal timing decision engine
│   ├── tracking/      # Multi-object tracking wrapper (ByteTrack)
│   ├── video/         # Video stream ingestion & frame manager
│   ├── __init__.py
│   └── main.py        # Application entrypoint
├── models/            # Model weights storage (e.g., YOLO weights)
├── outputs/           # Output logs, annotated videos, and telemetry metrics
├── tests/             # Unit and integration test suites
├── videos/            # Input video samples
├── .env.example       # Sample environment configuration
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
└── requirements.txt   # Project dependencies
```

---

## 🚀 Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Backend Entrypoint
From the root directory:
```bash
python -m backend.main
```
or
```bash
python backend/main.py
```
