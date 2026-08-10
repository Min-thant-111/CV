"""
Main entry point for the Edge-CV Traffic Framework backend application.

Environment variables are loaded from a .env file (if present) before
any pipeline configuration is built. All configurable values can be
overridden via .env or CLI flags — CLI flags take precedence.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support running this file directly
ROOT_DIR = str(Path(__file__).parent.parent.resolve())
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging
import argparse
import os
from pathlib import Path

# Ensure project root is in sys.path for backend imports
_BASE_DIR = Path(__file__).parent.parent.resolve()
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from dotenv import load_dotenv

# Load .env before reading any os.getenv values
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrafficFramework")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="An IoT-Enabled Edge-CV Framework for Real-Time Traffic Density Estimation."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.getenv("VIDEO_SOURCE_PATH", ""),
        help="Path to input traffic video file. Env: VIDEO_SOURCE_PATH",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
        help="YOLO model path or identifier. Env: YOLO_MODEL_PATH",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
        help="YOLO confidence threshold (0.0-1.0). Env: CONFIDENCE_THRESHOLD",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Skip N frames between processing steps.",
    )
    parser.add_argument(
        "--publish-interval",
        type=int,
        default=30,
        help="Publish MQTT signal every N processed frames.",
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default=os.getenv("MQTT_BROKER_HOST", "localhost"),
        help="MQTT broker hostname. Env: MQTT_BROKER_HOST",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=int(os.getenv("MQTT_BROKER_PORT", "1883")),
        help="MQTT broker port. Env: MQTT_BROKER_PORT",
    )
    parser.add_argument(
        "--mqtt-topic",
        type=str,
        default=os.getenv("MQTT_TOPIC_SIGNAL", "traffic/intersection1/signal"),
        help="MQTT publish topic. Env: MQTT_TOPIC_SIGNAL",
    )
    parser.add_argument(
        "--no-mqtt",
        action="store_true",
        help="Disable MQTT publishing.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialise and run the Edge-CV Traffic Framework pipeline."""
    logger.info("=" * 66)
    logger.info("  IoT-Enabled Edge-CV Framework — Traffic Density Estimation")
    logger.info("=" * 66)

    args = parse_args()

    from backend.pipeline_config import PipelineConfig
    from backend.density.density_engine import DensityConfig
    from backend.signaling.signal_engine import SignalConfig
    from backend.mqtt.mqtt_publisher import MQTTConfig
    from backend.pipeline import TrafficPipeline

    config = PipelineConfig(
        video_path=args.video,
        model_path=args.model,
        confidence_threshold=args.conf,
        frame_skip=args.frame_skip,
        publish_interval_frames=args.publish_interval,
        log_interval_frames=args.publish_interval,
        mqtt_enabled=not args.no_mqtt,
        density=DensityConfig(
            low_threshold_pct=float(os.getenv("DENSITY_LOW_PCT", "35.0")),
            high_threshold_pct=float(os.getenv("DENSITY_HIGH_PCT", "70.0")),
        ),
        signal=SignalConfig(
            green_duration_low=int(os.getenv("GREEN_TIME_LOW", "30")),
            green_duration_medium=int(os.getenv("GREEN_TIME_MED", "50")),
            green_duration_high=int(os.getenv("GREEN_TIME_HIGH", "70")),
        ),
        mqtt=MQTTConfig(
            broker_host=args.mqtt_host,
            broker_port=args.mqtt_port,
            topic=args.mqtt_topic,
        ),
    )

    if not config.video_path:
        logger.error(
            "No video path provided. "
            "Use: python -m backend.main --video <path>  "
            "or set VIDEO_SOURCE_PATH in .env"
        )
        sys.exit(1)

    pipeline = TrafficPipeline(config=config)

    try:
        pipeline.run()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
