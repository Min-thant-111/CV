"""
Unit tests for VirtualTrafficController using mocked MQTT client.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from backend.controller.virtual_controller import (
    VirtualTrafficController,
    ControllerConfig,
)


class TestVirtualControllerPayloadValidation(unittest.TestCase):
    """Test suite for signal payload validation in VirtualTrafficController."""

    def setUp(self):
        with patch("paho.mqtt.client.Client"):
            self.controller = VirtualTrafficController(
                config=ControllerConfig(broker_host="localhost", broker_port=1883)
            )

    def test_valid_green_payload(self):
        payload = {"signal": "GREEN", "duration": 70, "density_level": "HIGH", "density": 78.0}
        self.assertTrue(self.controller._validate_payload(payload))

    def test_valid_red_payload(self):
        payload = {"signal": "RED", "duration": 10, "density_level": "LOW", "density": 5.0}
        self.assertTrue(self.controller._validate_payload(payload))

    def test_missing_required_key_fails(self):
        # Missing "duration"
        payload = {"signal": "GREEN", "density_level": "HIGH"}
        self.assertFalse(self.controller._validate_payload(payload))

    def test_invalid_signal_state_fails(self):
        payload = {"signal": "BLUE", "duration": 30, "density_level": "LOW"}
        self.assertFalse(self.controller._validate_payload(payload))

    def test_invalid_duration_zero_fails(self):
        payload = {"signal": "GREEN", "duration": 0, "density_level": "LOW"}
        self.assertFalse(self.controller._validate_payload(payload))

    def test_invalid_duration_negative_fails(self):
        payload = {"signal": "GREEN", "duration": -5, "density_level": "HIGH"}
        self.assertFalse(self.controller._validate_payload(payload))


class TestVirtualControllerMQTTMessage(unittest.TestCase):
    """Test suite for MQTT message parsing and handling."""

    def setUp(self):
        with patch("paho.mqtt.client.Client"):
            self.controller = VirtualTrafficController(
                config=ControllerConfig(broker_host="localhost", broker_port=1883)
            )
        # Prevent actual background cycle thread execution during tests
        self.controller._handle_signal_command = MagicMock()

    def test_valid_json_message_dispatched(self):
        """Valid JSON MQTT message should be parsed and dispatched."""
        payload = {
            "intersection": 1,
            "signal": "GREEN",
            "duration": 70,
            "density": 78.5,
            "density_level": "HIGH",
        }
        mock_msg = MagicMock()
        mock_msg.payload = json.dumps(payload).encode("utf-8")

        self.controller._on_message(None, None, mock_msg)
        self.controller._handle_signal_command.assert_called_once_with(payload)

    def test_invalid_json_message_ignored(self):
        """Malformed JSON should be caught gracefully without crashing."""
        mock_msg = MagicMock()
        mock_msg.payload = b"not-valid-json{{{"

        # Must not raise any exception
        try:
            self.controller._on_message(None, None, mock_msg)
        except Exception as e:
            self.fail(f"_on_message raised an exception on bad JSON: {e}")

        self.controller._handle_signal_command.assert_not_called()


class TestVirtualControllerConnection(unittest.TestCase):
    """Test suite for MQTT connection behavior."""

    @patch("paho.mqtt.client.Client")
    def test_connect_called_with_correct_params(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        config = ControllerConfig(broker_host="192.168.1.100", broker_port=1883)
        controller = VirtualTrafficController(config=config)

        success = controller.connect()
        self.assertTrue(success)
        mock_instance.connect.assert_called_once_with(
            host="192.168.1.100", port=1883, keepalive=60
        )
        mock_instance.loop_start.assert_called_once()

    @patch("paho.mqtt.client.Client")
    def test_disconnect_called_cleanly(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        controller = VirtualTrafficController()
        controller.connect()
        controller.disconnect()

        mock_instance.loop_stop.assert_called_once()
        mock_instance.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
