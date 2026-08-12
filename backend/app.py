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
import hashlib
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from typing import Generator
from urllib.parse import quote

# Ensure project root is in sys.path for backend imports
BASE_DIR = Path(__file__).parent.parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()  # Consume .env before any os.getenv call

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for
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
CACHE_DIR    = OUTPUT_DIR / ".browser_cache"
TEMPLATE_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR   = BASE_DIR / "frontend" / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

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
_transcode_lock = threading.Lock()
_analysis_lock = threading.Lock()
_detector_lock = threading.Lock()
_shared_detector = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _get_shared_detector():
    """Load YOLO once per server process and reuse its immutable weights."""
    global _shared_detector
    with _detector_lock:
        if _shared_detector is None:
            from backend.detection.yolo_detector import YOLODetector

            _shared_detector = YOLODetector(
                model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
                confidence_threshold=float(
                    os.getenv("CONFIDENCE_THRESHOLD", "0.20")
                ),
                device=os.getenv("YOLO_DEVICE", "").strip() or None,
                inference_size=int(os.getenv("DETECTION_IMGSZ", "640")),
                iou_threshold=float(os.getenv("DETECTION_IOU", "0.45")),
            )
        return _shared_detector


def _warm_shared_detector() -> None:
    """Warm model weights in the background before the first upload arrives."""
    started = time.monotonic()
    try:
        _get_shared_detector()
        logger.info(
            "YOLO model ready in %.1fs; Analyse jobs will reuse it.",
            time.monotonic() - started,
        )
    except Exception:
        # A job will retry and surface a user-visible error if loading still fails.
        logger.exception("Background YOLO model warm-up failed.")


def _sse_event(data: dict, event: str = "update") -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _media_items(directory: Path, collection: str) -> list:
    """Return browser-facing metadata for videos in a media directory."""
    items = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if (
            not path.is_file()
            or path.suffix.lower() not in ALLOWED_EXTENSIONS
            or any(part.startswith(".") for part in relative.parts)
        ):
            continue
        relative_path = relative.as_posix()
        stat = path.stat()
        items.append({
            "name": path.name,
            "path": relative_path,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "url": url_for(
                "serve_playable_media",
                collection=collection,
                filename=relative_path,
            ),
        })
    return sorted(items, key=lambda item: item["modified"], reverse=True)


def _browser_compatible_video(source: Path, collection: str, filename: str) -> Path:
    """Return an H.264/AAC MP4 copy suitable for HTML5 video playback."""
    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"{collection}\0{filename}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    ).hexdigest()[:24]
    cached = CACHE_DIR / f"{fingerprint}.mp4"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    ffmpeg_name = os.getenv("FFMPEG_PATH", "ffmpeg")
    ffmpeg_path = shutil.which(ffmpeg_name)
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is required to convert this video for browser playback."
        )

    with _transcode_lock:
        if cached.exists() and cached.stat().st_size > 0:
            return cached
        temporary = CACHE_DIR / f"{fingerprint}.tmp.mp4"
        command = [
            ffmpeg_path, "-y", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k", str(temporary),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if completed.returncode != 0 or not temporary.exists():
                detail = completed.stderr.strip() or "Unknown FFmpeg error."
                raise RuntimeError(f"Video conversion failed: {detail}")
            temporary.replace(cached)
        finally:
            if temporary.exists():
                temporary.unlink()
    return cached


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos")
def list_videos():
    """List uploaded source videos and generated/evaluation videos."""
    return jsonify({
        "uploads": _media_items(UPLOAD_DIR, "uploads"),
        "outputs": _media_items(OUTPUT_DIR, "outputs"),
    })


@app.route("/media/<collection>/<path:filename>")
def serve_media(collection: str, filename: str):
    """Serve playable media while restricting access to known media roots."""
    directories = {"uploads": UPLOAD_DIR, "outputs": OUTPUT_DIR}
    directory = directories.get(collection)
    if directory is None or not _allowed(filename):
        return jsonify({"error": "Video not found."}), 404
    return send_from_directory(str(directory), filename, conditional=True)


@app.route("/media/play/<collection>/<path:filename>")
def serve_playable_media(collection: str, filename: str):
    """Convert media to browser-safe MP4 once, cache it, and serve with ranges."""
    directories = {"uploads": UPLOAD_DIR, "outputs": OUTPUT_DIR}
    directory = directories.get(collection)
    if directory is None or not _allowed(filename):
        return jsonify({"error": "Video not found."}), 404

    root = directory.resolve()
    source = (root / filename).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        return jsonify({"error": "Video not found."}), 404

    try:
        playable = _browser_compatible_video(source, collection, filename)
    except RuntimeError as exc:
        logger.error("Unable to prepare browser video %s: %s", source, exc)
        return jsonify({"error": str(exc)}), 500
    return send_from_directory(
        str(CACHE_DIR), playable.name, mimetype="video/mp4", conditional=True
    )


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
            "source_url": url_for(
                "serve_playable_media", collection="uploads", filename=filename
            ),
            "events": event_queue,
            "result": {},
            "preview": None,
            "preview_version": 0,
            "preview_frame": -1,
        }

    # Start pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, str(save_path)),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "job_id": job_id,
        "filename": file.filename,
        "source_url": url_for(
            "serve_playable_media", collection="uploads", filename=filename
        ),
    }), 202


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
        "source_url": job.get("source_url", ""),
        "result": job.get("result", {}),
    })


@app.route("/api/preview/<job_id>")
def job_preview(job_id: str):
    """Return the newest annotated JPEG while the final MP4 is still open."""
    try:
        since = max(0, int(request.args.get("since", "0")))
    except ValueError:
        since = 0

    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404
        preview = job.get("preview")
        version = int(job.get("preview_version", 0))
        preview_frame = int(job.get("preview_frame", -1))

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "X-Preview-Version": str(version),
        "X-Preview-Frame": str(preview_frame),
    }
    if preview is None or version <= since:
        return Response(status=204, headers=headers)
    return Response(preview, mimetype="image/jpeg", headers=headers)


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

    writer = None
    mqtt_publisher = None
    analysis_lock_acquired = False
    output_path = OUTPUT_DIR / f"evaluated_{job_id}.mp4"
    staging_dir = OUTPUT_DIR / ".processing"
    staging_dir.mkdir(exist_ok=True)
    staging_path = staging_dir / output_path.name

    try:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))

        from backend.pipeline_config import PipelineConfig
        from backend.density.density_engine import DensityConfig
        from backend.signaling.signal_engine import SignalConfig
        from backend.mqtt.mqtt_publisher import MQTTConfig
        from backend.video.video_processor import VideoProcessor
        from backend.tracking.tracker import VehicleTracker
        from backend.density.density_engine import DensityEngine
        from backend.density.vehicle_count_estimator import estimate_visible_vehicles
        from backend.density.traffic_classifier import TrafficDensityClassifier
        from backend.signaling.signal_engine import SignalDecisionEngine
        from backend.mqtt.mqtt_publisher import MQTTPublisher
        from backend.road.path_estimator import RoadPathEstimator
        import cv2

        def store_preview(image, frame_number: int) -> None:
            preview_ok, preview_buffer = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if not preview_ok:
                return
            with _jobs_lock:
                _jobs[job_id]["preview"] = preview_buffer.tobytes()
                _jobs[job_id]["preview_version"] += 1
                _jobs[job_id]["preview_frame"] = frame_number

        with _jobs_lock:
            _jobs[job_id]["status"] = "loading"

        push({"type": "status", "message": "Loading video metadata...", "step": "loading"})

        # ── Video metadata ──────────────────────────────────────────
        frame_skip = max(0, int(os.getenv("FRAME_SKIP", "1")))
        processor = VideoProcessor(video_path=video_path, frame_skip=frame_skip)
        meta = processor.metadata
        push({"type": "status", "message": "Detecting road paths from video...", "step": "road_paths"})
        path_estimate = RoadPathEstimator().estimate_video(video_path)
        road_path_count = path_estimate.path_count
        logger.info(
            "[%s] Automatically detected %s road path(s), confidence %.1f%%",
            job_id,
            road_path_count,
            path_estimate.confidence * 100,
        )
        output_fps = max(meta.fps / (frame_skip + 1), 1.0)
        writer = cv2.VideoWriter(
            str(staging_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (meta.width, meta.height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the evaluated output video.")

        # Show an immediate first-frame preview while the detector initializes.
        preview_capture = cv2.VideoCapture(video_path)
        preview_read, initial_preview = preview_capture.read()
        preview_capture.release()
        if preview_read:
            cv2.rectangle(initial_preview, (0, 0), (meta.width, 28), (7, 13, 26), -1)
            cv2.putText(
                initial_preview, "Preparing vehicle detection...", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 242, 255), 1, cv2.LINE_AA,
            )
            store_preview(initial_preview, -1)
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

        if not _analysis_lock.acquire(blocking=False):
            push({
                "type": "status",
                "message": "Another analysis is running; this video is queued...",
                "step": "queued",
            })
            with _jobs_lock:
                _jobs[job_id]["status"] = "queued"
            _analysis_lock.acquire()
        analysis_lock_acquired = True

        detector = _get_shared_detector()
        # Ultralytics stores ByteTrack state on its predictor. Recreate only the
        # cheap predictor state for every video while retaining loaded weights.
        detector.model.predictor = None
        tracker       = VehicleTracker(
            detector=detector,
            high_recall=os.getenv("HIGH_RECALL_TILING", "true").lower()
            in {"1", "true", "yes", "on"},
            tile_inference_size=int(os.getenv("TILE_IMGSZ", "640")),
            tile_confidence_threshold=float(os.getenv("TILE_CONFIDENCE", "0.18")),
            tile_grid_size=int(os.getenv("TILE_GRID", "3")),
            tile_interval_frames=int(os.getenv("TILE_INTERVAL", "5")),
            far_field_recall=os.getenv("FAR_FIELD_RECALL", "true").lower()
            in {"1", "true", "yes", "on"},
            far_field_inference_size=int(os.getenv("FAR_FIELD_IMGSZ", "1280")),
            far_field_confidence_threshold=float(
                os.getenv("FAR_FIELD_CONFIDENCE", "0.05")
            ),
            detection_memory_frames=int(os.getenv("DETECTION_MEMORY", "2")),
            class_history_frames=int(os.getenv("CLASS_HISTORY_FRAMES", "12")),
            heavy_vehicle_min_confidence=float(
                os.getenv("HEAVY_VEHICLE_MIN_CONFIDENCE", "0.30")
            ),
            heavy_vehicle_min_observations=int(
                os.getenv("HEAVY_VEHICLE_MIN_OBSERVATIONS", "3")
            ),
            class_switch_margin=float(os.getenv("CLASS_SWITCH_MARGIN", "1.20")),
            suppress_camera_overlay=os.getenv(
                "SUPPRESS_CAMERA_OVERLAY", "true"
            ).lower() in {"1", "true", "yes", "on"},
            overlay_top_fraction=float(os.getenv("OVERLAY_TOP_FRACTION", "0.24")),
            overlay_left_fraction=float(os.getenv("OVERLAY_LEFT_FRACTION", "0.50")),
        )
        density_engine = DensityEngine(config=DensityConfig(
            road_path_count=road_path_count,
        ))
        traffic_classifier = None
        classifier_path = Path(os.getenv(
            "TRAFFIC_CLASSIFIER_MODEL",
            str(BASE_DIR / "models" / "traffic_density_cnn.pt"),
        ))
        if classifier_path.is_file():
            try:
                traffic_classifier = TrafficDensityClassifier(str(classifier_path))
                logger.info("[%s] Loaded traffic classifier: %s", job_id, classifier_path)
            except Exception as error:
                logger.warning(
                    "[%s] Traffic classifier unavailable; using geometric calibration: %s",
                    job_id,
                    error,
                )
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
        PUBLISH_EVERY = max(1, int(os.getenv("UI_UPDATE_EVERY", "5")))
        processed     = 0
        last_result   = {}
        last_mqtt_ok  = False
        processing_started = time.monotonic()
        box_font_scale = max(0.30, min(0.45, meta.width / 900))
        box_thickness = 1 if meta.width < 720 else 2
        peak_density = None
        peak_decision = None
        peak_vehicle_count = 0
        peak_class_counts = {}
        vehicle_observation_sum = 0
        class_observation_sums = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }
        traffic_level_scores = {"light": 0.0, "medium": 0.0, "heavy": 0.0}
        traffic_level = None
        traffic_level_confidence = 0.0

        for frame_index, timestamp, frame in processor.process_frames():
            if traffic_classifier is not None:
                predicted_level, predicted_confidence = traffic_classifier.predict(frame)
                traffic_level_scores[predicted_level] += predicted_confidence
                traffic_level = max(traffic_level_scores, key=traffic_level_scores.get)
                score_total = sum(traffic_level_scores.values())
                traffic_level_confidence = (
                    traffic_level_scores[traffic_level] / score_total
                    if score_total > 0 else 0.0
                )
            frame_tracks  = tracker.track_frame(frame, frame_index, timestamp)
            density       = density_engine.compute_density(frame_tracks)
            decision      = signal_engine.evaluate(density)

            # Adapt the useful statistic from the reference project: detections
            # accumulated across analysed frames divided by analysed frames.
            # This is an average concurrent count, not a unique vehicle total.
            vehicle_observation_sum += density.total_vehicle_count
            for class_name in class_observation_sums:
                class_observation_sums[class_name] += density.class_counts.get(
                    class_name, 0
                )
            analyzed_frames = processed + 1
            average_vehicle_count = round(
                vehicle_observation_sum / analyzed_frames, 2
            )
            average_class_counts = {
                class_name: round(total / analyzed_frames, 2)
                for class_name, total in class_observation_sums.items()
            }

            # The last frame is often blurred, occluded, or shows vehicles
            # leaving the image. Preserve the strongest stabilized whole-road
            # observation. This is a peak concurrent count, not a sum of track
            # IDs, because tracker ID churn would over-count the real traffic.
            if (
                peak_density is None
                or density.total_vehicle_count >= peak_vehicle_count
            ):
                peak_density = density
                peak_decision = decision
                peak_vehicle_count = density.total_vehicle_count
                peak_class_counts = dict(density.class_counts)

            count_estimate = estimate_visible_vehicles(
                peak_vehicle_count,
                peak_class_counts,
                meta.width,
                meta.height,
                road_path_count,
                traffic_level=traffic_level,
            )
            display_density = density_engine.compute_density_from_counts(
                count_estimate.class_counts,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            display_decision = signal_engine.evaluate(display_density)

            # Persist a viewable evaluated frame with detections and the decision.
            for track in frame_tracks.tracks:
                x1, y1, x2, y2 = (int(value) for value in track.bbox)
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 220, 192), box_thickness
                )
                label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, box_font_scale,
                    (0, 220, 192), box_thickness,
                    cv2.LINE_AA,
                )
            header_height = max(24, min(36, meta.height // 9))
            cv2.rectangle(frame, (0, 0), (meta.width, header_height), (7, 13, 26), -1)
            cv2.putText(
                frame,
                f"Detected: {peak_vehicle_count}  Estimated: {count_estimate.estimated_count}  Density: {display_density.density_level}",
                (8, header_height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.36, min(0.55, meta.width / 700)), (235, 242, 255), 1,
                cv2.LINE_AA,
            )
            writer.write(frame)

            store_preview(frame, frame_index)

            processed += 1

            should_publish = (
                processed == 1
                or processed % PUBLISH_EVERY == 0
                or frame_index >= meta.total_frames - 1
            )
            if should_publish:
                last_mqtt_ok = mqtt_publisher.publish_signal(display_decision)

            elapsed = max(0.001, time.monotonic() - processing_started)
            processing_fps = processed / elapsed
            remaining_source_frames = max(0, meta.total_frames - frame_index - 1)
            remaining_processed_frames = (
                remaining_source_frames + frame_skip
            ) // (frame_skip + 1)
            result = {
                "type":               "frame",
                "frame_index":        frame_index,
                "timestamp":          round(timestamp, 2),
                "processed_frames":   processed,
                "total_frames":       meta.total_frames,
                "progress_pct":       round(((frame_index + 1) / max(meta.total_frames, 1)) * 100, 1),
                "vehicle_count":      count_estimate.estimated_count,
                "estimated_vehicle_count": count_estimate.estimated_count,
                "count_correction_factor": count_estimate.correction_factor,
                "detected_peak_vehicle_count": peak_vehicle_count,
                "peak_vehicle_count": peak_vehicle_count,
                "active_vehicle_count": density.total_vehicle_count,
                "average_vehicle_count": average_vehicle_count,
                "class_counts":       count_estimate.class_counts,
                "detected_peak_class_counts": peak_class_counts,
                "active_class_counts": dict(density.class_counts),
                "average_class_counts": average_class_counts,
                "density_percentage": round(display_density.density_percentage, 1),
                "density_level":      display_density.density_level,
                "road_path_count":    road_path_count,
                "road_path_confidence": round(path_estimate.confidence, 3),
                "road_path_method":   path_estimate.method,
                "traffic_level":      traffic_level,
                "traffic_level_confidence": round(traffic_level_confidence, 3),
                "signal":             display_decision.signal,
                "green_duration":     display_decision.duration,
                "reason":             display_decision.reason,
                "mqtt_published":     last_mqtt_ok,
                "processing_fps":     round(processing_fps, 2),
                "eta_seconds":        round(
                    remaining_processed_frames / max(processing_fps, 0.001), 1
                ),
                "preview_url":        f"/api/preview/{job_id}",
            }
            last_result = result

            if should_publish:
                push(result)

            with _jobs_lock:
                _jobs[job_id]["result"] = result

        mqtt_publisher.disconnect()
        mqtt_publisher = None
        writer.release()
        writer = None
        staging_path.replace(output_path)

        # ── Done ────────────────────────────────────────────────────
        output_url = "/media/play/outputs/" + quote(
            output_path.relative_to(OUTPUT_DIR).as_posix()
        )
        final = {
            **last_result,
            "type": "done",
            "message": "Analysis complete.",
            "source_url": job["source_url"],
            "output_url": output_url,
        }
        push(final, event="done")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = final

        logger.info(f"[{job_id}] Pipeline complete. Frames processed: {processed}")

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error: {e}", exc_info=True)
        error_result = {"type": "error", "message": str(e)}
        push(error_result, event="error")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = error_result
    finally:
        if writer is not None:
            writer.release()
        if mqtt_publisher is not None:
            mqtt_publisher.disconnect()
        if staging_path.exists():
            staging_path.unlink()
        if analysis_lock_acquired:
            _analysis_lock.release()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Flask development server on http://localhost:5000")
    threading.Thread(
        target=_warm_shared_detector,
        name="yolo-warmup",
        daemon=True,
    ).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
