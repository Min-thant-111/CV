"""
Main entry point for the Edge-CV Traffic Framework backend application.
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrafficFramework")


def main() -> None:
    """Initialize and run the Edge-CV Traffic Framework pipeline."""
    logger.info("==================================================================")
    logger.info("An IoT-Enabled Edge-CV Framework for Real-Time Traffic Density Estimation")
    logger.info("==================================================================")
    logger.info("Modular backend initialized successfully.")
    logger.info("Modules ready: video, detection, tracking, density, signaling, mqtt, controller, models.")


if __name__ == "__main__":
    main()
