"""
MQTT Publisher module for sending traffic signal decisions and telemetry.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

try:
    import paho.mqtt.client as mqtt
    HAS_PAHO = True
except ImportError:
    mqtt = None
    HAS_PAHO = False


from backend.models.signaling import SignalDecision
from backend.models.mqtt import MQTTSignalPayload


logger = logging.getLogger("MQTTPublisher")


@dataclass
class MQTTConfig:
    """Configurable parameters for MQTT broker connection."""

    broker_host: str = "localhost"
    broker_port: int = 1883
    topic: str = "traffic/intersection1/signal"
    client_id: str = "edge_cv_traffic_publisher"
    qos: int = 1
    keepalive: int = 60


class MQTTPublisherError(Exception):
    """Base exception for MQTTPublisher errors."""

    pass


class MQTTPublisher:
    """Reusable MQTT Publisher with graceful error handling and auto-reconnect."""

    def __init__(self, config: Optional[MQTTConfig] = None):
        """Args:

        config: MQTTConfig instance holding broker connection parameters.
        """
        self.config = config or MQTTConfig()
        self._is_connected = False
        self.client: Optional[Any] = None

        if HAS_PAHO:
            self._init_client()
        else:
            logger.warning("paho-mqtt is not installed. MQTT publishing will be disabled.")

    def _init_client(self) -> None:
        """Initialize Paho MQTT client instance with callback handlers."""
        if not HAS_PAHO or mqtt is None:
            return

        try:
            # Check paho-mqtt version compatibility for API version enum
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    client_id=self.config.client_id,
                )
            else:
                self.client = mqtt.Client(client_id=self.config.client_id)

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

        except Exception as e:
            logger.error(f"Failed to initialize MQTT client: {e}")
            self.client = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback executed when client connects to broker."""
        if rc == 0:
            self._is_connected = True
            logger.info(
                f"Successfully connected to MQTT Broker at {self.config.broker_host}:{self.config.broker_port}"
            )
        else:
            self._is_connected = False
            logger.warning(f"MQTT Broker connection failed with return code {rc}")

    def _on_disconnect(self, client, userdata, flags_or_rc, rc_or_properties=None, properties=None):
        """Callback executed when client disconnects from broker."""
        self._is_connected = False
        logger.info("Disconnected from MQTT Broker.")

    @property
    def is_connected(self) -> bool:
        """Return current broker connection status."""
        return self._is_connected

    def connect(self) -> bool:
        """Attempt connection to the MQTT Broker gracefully without throwing exceptions.

        Returns:
            bool: True if connection process initiated, False otherwise.
        """
        if not HAS_PAHO:
            logger.warning("paho-mqtt library is not installed. Connection skipped.")
            return False

        if self.client is None:
            self._init_client()

        if self.client is None:
            logger.warning("Cannot connect: MQTT client initialization failed.")
            return False

        try:
            logger.info(
                f"Connecting asynchronously to MQTT Broker at "
                f"{self.config.broker_host}:{self.config.broker_port}..."
            )
            # Broker availability must never hold up CV inference. Paho's
            # network loop completes the connection (and retries) in parallel.
            self.client.connect_async(
                host=self.config.broker_host,
                port=self.config.broker_port,
                keepalive=self.config.keepalive,
            )
            self.client.loop_start()
            return True
        except Exception as e:
            self._is_connected = False
            logger.warning(
                f"Could not connect to MQTT Broker at {self.config.broker_host}:{self.config.broker_port}. "
                f"Error: {e}. (CV pipeline will continue operating without MQTT)"
            )
            return False

    def disconnect(self) -> None:
        """Safely disconnect from the MQTT Broker."""
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logger.warning(f"Error during MQTT disconnect: {e}")
            finally:
                self._is_connected = False

    def publish_signal(
        self,
        decision: Union[SignalDecision, Dict[str, Any]],
        intersection_id: int = 1,
        topic_override: Optional[str] = None,
    ) -> bool:
        """Publish a traffic signal decision payload to the MQTT broker.

        Args:
            decision: SignalDecision object or dictionary.
            intersection_id: ID of the traffic intersection.
            topic_override: Optional topic string overriding config.topic.

        Returns:
            bool: True if message was queued for publishing, False otherwise.
        """
        topic = topic_override or self.config.topic

        if isinstance(decision, SignalDecision):
            payload = MQTTSignalPayload(
                intersection_id=intersection_id,
                signal=decision.signal,
                duration=decision.duration,
                density=decision.density_percentage,
                density_level=decision.density_level,
                vehicle_count=decision.vehicle_count,
            )
        elif isinstance(decision, dict):
            payload = MQTTSignalPayload(
                intersection_id=intersection_id,
                signal=decision.get("signal", "GREEN"),
                duration=decision.get("duration", 30),
                density=decision.get("density_percentage", decision.get("density", 0.0)),
                density_level=decision.get("density_level", "LOW"),
                vehicle_count=decision.get("vehicle_count", 0),
            )
        else:
            logger.error("Invalid decision object provided to publish_signal.")
            return False

        payload_dict = payload.to_dict()
        payload_json = json.dumps(payload_dict)

        if not self._is_connected or self.client is None:
            logger.warning(
                f"MQTT Client is not connected. Skipping message publish to '{topic}': {payload_json}"
            )
            return False

        try:
            info = self.client.publish(topic, payload_json, qos=self.config.qos)
            success_code = mqtt.MQTT_ERR_SUCCESS if (HAS_PAHO and mqtt) else 0
            rc = info.rc if hasattr(info, "rc") else 0
            if rc == success_code:
                logger.info(f"Published signal decision to topic '{topic}': {payload_json}")
                return True
            else:
                logger.warning(f"Failed to publish to MQTT topic '{topic}' (code {rc})")
                return False
        except Exception as e:
            logger.warning(f"Exception during MQTT message publication: {e}")
            return False
