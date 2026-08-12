"""
Unit tests for MQTTSignalPayload model and MQTTPublisher using mocked Paho MQTT client.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from backend.models.mqtt import MQTTSignalPayload
from backend.models.signaling import SignalDecision
from backend.mqtt.mqtt_publisher import MQTTPublisher, MQTTConfig


class TestMQTTModule(unittest.TestCase):
    """Test suite for MQTT communication payload formatting and publisher logic."""

    def test_payload_to_dict(self):
        payload = MQTTSignalPayload(
            intersection_id=1,
            signal="GREEN",
            duration=70,
            density=78.5,
            density_level="HIGH",
            vehicle_count=8,
            timestamp=1000.0,
        )
        p_dict = payload.to_dict()
        self.assertEqual(p_dict["intersection"], 1)
        self.assertEqual(p_dict["signal"], "GREEN")
        self.assertEqual(p_dict["duration"], 70)
        self.assertEqual(p_dict["density"], 78.5)
        self.assertEqual(p_dict["density_level"], "HIGH")
        self.assertEqual(p_dict["vehicle_count"], 8)

    @patch("paho.mqtt.client.Client")
    def test_mqtt_publisher_connect_and_disconnect(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        publisher = MQTTPublisher(
            config=MQTTConfig(broker_host="localhost", broker_port=1883)
        )

        success = publisher.connect()
        self.assertTrue(success)
        mock_instance.connect_async.assert_called_once_with(
            host="localhost", port=1883, keepalive=60
        )
        mock_instance.loop_start.assert_called_once()

        publisher.disconnect()
        mock_instance.loop_stop.assert_called_once()
        mock_instance.disconnect.assert_called_once()

    @patch("paho.mqtt.client.Client")
    def test_publish_signal_when_connected(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        # Mock publish success return
        mock_info = MagicMock()
        mock_info.rc = 0
        mock_instance.publish.return_value = mock_info

        publisher = MQTTPublisher(
            config=MQTTConfig(topic="traffic/intersection1/signal", qos=1)
        )

        # Simulate on_connect callback setting connection state
        publisher._is_connected = True

        decision = SignalDecision(
            signal="GREEN",
            duration=50,
            density_level="MEDIUM",
            density_percentage=55.0,
            vehicle_count=5,
            reason="Moderate density",
        )

        res = publisher.publish_signal(decision, intersection_id=1)
        self.assertTrue(res)

        # Verify call to paho publish
        mock_instance.publish.assert_called_once()
        args, kwargs = mock_instance.publish.call_args
        topic_arg, payload_json = args[0], args[1]

        self.assertEqual(topic_arg, "traffic/intersection1/signal")
        self.assertEqual(kwargs.get("qos"), 1)

        sent_data = json.loads(payload_json)
        self.assertEqual(sent_data["intersection"], 1)
        self.assertEqual(sent_data["signal"], "GREEN")
        self.assertEqual(sent_data["duration"], 50)
        self.assertEqual(sent_data["density_level"], "MEDIUM")

    def test_publish_signal_graceful_handling_when_disconnected(self):
        """Test that publishing while disconnected returns False gracefully without crashing."""
        publisher = MQTTPublisher()
        publisher._is_connected = False

        decision = SignalDecision(
            signal="GREEN",
            duration=30,
            density_level="LOW",
            density_percentage=10.0,
            vehicle_count=1,
            reason="Low density",
        )

        # Must return False gracefully without raising exceptions
        res = publisher.publish_signal(decision)
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
