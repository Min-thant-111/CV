"""
Flask Web Application for the Edge-CV Traffic Analysis Framework.

Serves the upload interface, processes video via the existing pipeline,
and streams real-time analysis results to the browser via SSE.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support running this file directly
ROOT_DIR = str(Path(__file__).parent.parent.resolve())
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Generator

# Ensure project root is in sys.path for backend imports
BASE_DIR = Path(__file__).parent.parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()  # Consume .env before any os.getenv call

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrafficApp")

# ── Flask App Setup ────────────────────────────────────────────────────────────
UPLOAD_DIR   = BASE_DIR / "videos"
OUTPUT_DIR   = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR   = BASE_DIR / "frontend" / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# ── In-memory job store ────────────────────────────────────────────────────────
# { job_id: { "status": str, "events": queue.Queue, "result": dict } }
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _sse_event(data: dict, event: str = "update") -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Accept video upload, create a processing job, return job_id."""
    if "video" not in request.files:
        return jsonify({"error": "No file field 'video' in request."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        ext_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed: {ext_list}"}), 415

    job_id   = str(uuid.uuid4())[:8]
    suffix   = Path(file.filename).suffix.lower()
    filename = f"{job_id}{suffix}"
    save_path = UPLOAD_DIR / filename

    file.save(str(save_path))
    logger.info(f"[{job_id}] Video saved: {save_path}")

    event_queue: queue.Queue = queue.Queue(maxsize=500)
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "filepath": str(save_path),
            "events": event_queue,
            "result": {},
        }

    # Start pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, str(save_path)),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "filename": file.filename}), 202


@app.route("/api/stream/<job_id>")
def stream_events(job_id: str):
    """Server-Sent Events endpoint streaming real-time analysis updates."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    def generate() -> Generator[str, None, None]:
        q: queue.Queue = job["events"]
        while True:
            try:
                event = q.get(timeout=30)
                yield event
                if '"type": "done"' in event or '"type": "error"' in event:
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield f"event: heartbeat\ndata: {{}}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/status/<job_id>")
def job_status(job_id: str):
    """Return current job status and latest result snapshot."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename", ""),
        "result": job.get("result", {}),
    })


# ── Pipeline Worker ────────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, video_path: str) -> None:
    """Run the traffic analysis pipeline in a background thread, pushing SSE events."""
    with _jobs_lock:
        job = _jobs[job_id]

    q: queue.Queue = job["events"]

    def push(data: dict, event: str = "update") -> None:
        try:
            q.put_nowait(_sse_event(data, event))
        except queue.Full:
            pass

    try:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))

        from backend.pipeline_config import PipelineConfig
        from backend.density.density_engine import DensityConfig
        from backend.signaling.signal_engine import SignalConfig
        from backend.mqtt.mqtt_publisher import MQTTConfig
        from backend.video.video_processor import VideoProcessor
        from backend.tracking.tracker import VehicleTracker
        from backend.detection.yolo_detector import YOLODetector
        from backend.density.density_engine import DensityEngine
        from backend.signaling.signal_engine import SignalDecisionEngine
        from backend.mqtt.mqtt_publisher import MQTTPublisher

        with _jobs_lock:
            _jobs[job_id]["status"] = "loading"

        push({"type": "status", "message": "Loading video metadata...", "step": "loading"})

        # ── Video metadata ──────────────────────────────────────────
        processor = VideoProcessor(video_path=video_path, frame_skip=2)
        meta = processor.metadata
        push({
            "type": "metadata",
            "filename":       job["filename"],
            "width":          meta.width,
            "height":         meta.height,
            "fps":            round(meta.fps, 2),
            "total_frames":   meta.total_frames,
            "duration":       round(meta.duration_seconds, 2),
        })

        # ── Initialise modules ──────────────────────────────────────
        push({"type": "status", "message": "Initialising YOLO model...", "step": "init"})
        with _jobs_lock:
            _jobs[job_id]["status"] = "initialising"

        detector      = YOLODetector(
            model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
        )
        tracker       = VehicleTracker(detector=detector)
        density_engine = DensityEngine(config=DensityConfig())
        signal_engine  = SignalDecisionEngine(config=SignalConfig(
            green_duration_low=int(os.getenv("GREEN_TIME_LOW", "30")),
            green_duration_medium=int(os.getenv("GREEN_TIME_MED", "50")),
            green_duration_high=int(os.getenv("GREEN_TIME_HIGH", "70")),
        ))
        mqtt_publisher = MQTTPublisher(config=MQTTConfig(
            broker_host=os.getenv("MQTT_BROKER_HOST", "localhost"),
            broker_port=int(os.getenv("MQTT_BROKER_PORT", "1883")),
            topic=os.getenv("MQTT_TOPIC_SIGNAL", "traffic/intersection1/signal"),
        ))
        mqtt_publisher.connect()

        push({"type": "status", "message": "Starting video analysis...", "step": "processing"})
        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"

        # ── Frame loop ──────────────────────────────────────────────
        PUBLISH_EVERY = 30   # MQTT + SSE update every N processed frames
        processed     = 0
        last_result   = {}

        for frame_index, timestamp, frame in processor.process_frames():
            frame_tracks  = tracker.track_frame(frame, frame_index, timestamp)
            density       = density_engine.compute_density(frame_tracks)
            decision      = signal_engine.evaluate(density)

            processed += 1

            if processed % PUBLISH_EVERY == 0 or processed == 1:
                mqtt_ok = mqtt_publisher.publish_signal(decision)

                result = {
                    "type":               "frame",
                    "frame_index":        frame_index,
                    "timestamp":          round(timestamp, 2),
                    "processed_frames":   processed,
                    "total_frames":       meta.total_frames,
                    "progress_pct":       round((frame_index / max(meta.total_frames, 1)) * 100, 1),
                    "vehicle_count":      density.total_vehicle_count,
                    "class_counts":       density.class_counts,
                    "density_percentage": round(density.density_percentage, 1),
                    "density_level":      density.density_level,
                    "signal":             decision.signal,
                    "green_duration":     decision.duration,
                    "reason":             decision.reason,
                    "mqtt_published":     mqtt_ok,
                }
                last_result = result
                push(result)

                with _jobs_lock:
                    _jobs[job_id]["result"] = result

        mqtt_publisher.disconnect()

        # ── Done ────────────────────────────────────────────────────
        final = {**last_result, "type": "done", "message": "Analysis complete."}
        push(final, event="done")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = final

        logger.info(f"[{job_id}] Pipeline complete. Frames processed: {processed}")

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error: {e}", exc_info=True)
        push({"type": "error", "message": str(e)}, event="error")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Flask development server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
