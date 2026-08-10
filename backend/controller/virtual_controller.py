"""
Virtual IoT Traffic Controller - Simulates an IoT traffic-light device.

Subscribes to MQTT broker for signal commands and executes
a countdown-based signal state machine in the terminal.

Note: This module intentionally does NOT import YOLO, OpenCV,
or the Signal Decision Engine. It communicates solely via MQTT,
simulating an independent IoT edge device (e.g., ESP32 / Raspberry Pi).
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Any

try:
    import paho.mqtt.client as mqtt
    HAS_PAHO = True
except ImportError:
    mqtt = None
    HAS_PAHO = False


logger = logging.getLogger("VirtualTrafficController")


# ANSI terminal colour codes for signal state display
COLOUR_GREEN  = "\033[92m"
COLOUR_YELLOW = "\033[93m"
COLOUR_RED    = "\033[91m"
COLOUR_RESET  = "\033[0m"
COLOUR_CYAN   = "\033[96m"
COLOUR_BOLD   = "\033[1m"

SIGNAL_COLOURS = {
    "GREEN":  COLOUR_GREEN,
    "YELLOW": COLOUR_YELLOW,
    "RED":    COLOUR_RED,
}

YELLOW_DURATION = 5   # seconds for yellow transition
RED_DURATION    = 10  # seconds for red phase before next green


@dataclass
class ControllerConfig:
    """Configurable parameters for the Virtual Traffic Controller."""

    broker_host: str  = "localhost"
    broker_port: int  = 1883
    topic:       str  = "traffic/intersection1/signal"
    client_id:   str  = "virtual_traffic_controller"
    qos:         int  = 1
    keepalive:   int  = 60


class VirtualTrafficController:
    """Virtual IoT Traffic Light Controller.

    Subscribes to MQTT signal commands and drives a software-simulated
    traffic light through GREEN → YELLOW → RED state transitions, with a
    terminal countdown display for each phase.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        """Args:

        config: ControllerConfig instance holding broker and topic parameters.
        """
        self.config = config or ControllerConfig()
        self.client: Optional[Any] = None
        self._is_connected = False
        self._current_signal: str = "RED"
        self._signal_lock = threading.Lock()
        self._cycle_thread: Optional[threading.Thread] = None
        self._stop_cycle = threading.Event()

        if HAS_PAHO:
            self._init_client()
        else:
            logger.error("paho-mqtt is not installed. Virtual controller cannot operate.")

    # ─────────────────────────── MQTT Setup ───────────────────────────

    def _init_client(self) -> None:
        """Initialise Paho MQTT client with required callbacks."""
        if not HAS_PAHO or mqtt is None:
            return

        try:
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    client_id=self.config.client_id,
                )
            else:
                self.client = mqtt.Client(client_id=self.config.client_id)

            self.client.on_connect    = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message    = self._on_message

        except Exception as e:
            logger.error(f"Failed to initialise MQTT client: {e}")
            self.client = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._is_connected = True
            logger.info(
                f"Controller connected to MQTT Broker at "
                f"{self.config.broker_host}:{self.config.broker_port}"
            )
            client.subscribe(self.config.topic, qos=self.config.qos)
            logger.info(f"Subscribed to topic: '{self.config.topic}'")
            self._print_standby()
        else:
            self._is_connected = False
            logger.warning(f"Controller connection failed (rc={rc})")

    def _on_disconnect(self, client, userdata, flags_or_rc, rc_or_props=None, props=None):
        self._is_connected = False
        logger.info("Controller disconnected from MQTT Broker.")

    def _on_message(self, client, userdata, message):
        """Handle incoming MQTT signal command messages."""
        try:
            raw = message.payload.decode("utf-8")
            payload = json.loads(raw)
            logger.info(f"Received signal command: {payload}")
            self._handle_signal_command(payload)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MQTT message: {e} — raw: {message.payload}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    # ─────────────────────────── Connection ───────────────────────────

    def connect(self) -> bool:
        """Connect to the MQTT broker and start the network loop.

        Returns:
            bool: True if connection was initiated successfully.
        """
        if not HAS_PAHO:
            return False

        if self.client is None:
            self._init_client()

        if self.client is None:
            return False

        try:
            logger.info(
                f"Connecting to MQTT Broker at "
                f"{self.config.broker_host}:{self.config.broker_port} ..."
            )
            self.client.connect(
                host=self.config.broker_host,
                port=self.config.broker_port,
                keepalive=self.config.keepalive,
            )
            self.client.loop_start()
            return True
        except Exception as e:
            logger.warning(
                f"Could not connect to MQTT Broker: {e}. "
                "Controller will remain offline."
            )
            return False

    def disconnect(self) -> None:
        """Stop the network loop and disconnect cleanly."""
        self._stop_cycle.set()
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._is_connected = False

    # ─────────────────────────── Signal Logic ───────────────────────────

    def _validate_payload(self, payload: dict) -> bool:
        """Validate required keys and types in the received signal command."""
        required = {"signal", "duration", "density_level"}
        if not required.issubset(payload.keys()):
            missing = required - payload.keys()
            logger.warning(f"Payload missing required fields: {missing}")
            return False

        if payload["signal"] not in ("GREEN", "YELLOW", "RED"):
            logger.warning(f"Unknown signal state: '{payload['signal']}'")
            return False

        if not isinstance(payload["duration"], (int, float)) or payload["duration"] <= 0:
            logger.warning(f"Invalid duration value: {payload['duration']}")
            return False

        return True

    def _handle_signal_command(self, payload: dict) -> None:
        """Parse validated signal command and trigger signal cycle in a background thread."""
        if not self._validate_payload(payload):
            return

        intersection_id = payload.get("intersection", 1)
        signal          = str(payload["signal"]).upper()
        duration        = int(payload["duration"])
        density_pct     = float(payload.get("density", 0.0))
        density_level   = str(payload.get("density_level", "UNKNOWN")).upper()

        # Stop any currently running cycle before starting new one
        if self._cycle_thread and self._cycle_thread.is_alive():
            self._stop_cycle.set()
            self._cycle_thread.join(timeout=3)
        self._stop_cycle.clear()

        self._cycle_thread = threading.Thread(
            target=self._run_signal_cycle,
            args=(intersection_id, signal, duration, density_pct, density_level),
            daemon=True,
        )
        self._cycle_thread.start()

    def _run_signal_cycle(
        self,
        intersection_id: int,
        signal: str,
        duration: int,
        density_pct: float,
        density_level: str,
    ) -> None:
        """Execute the full traffic signal phase cycle: GREEN → YELLOW → RED."""

        # ── GREEN Phase ──────────────────────────────────────────────
        with self._signal_lock:
            self._current_signal = "GREEN"
        self._print_signal_header(intersection_id, "GREEN", duration, density_level, density_pct)
        self._countdown(duration, "GREEN")

        if self._stop_cycle.is_set():
            return

        # ── YELLOW Phase ─────────────────────────────────────────────
        with self._signal_lock:
            self._current_signal = "YELLOW"
        self._print_signal_header(intersection_id, "YELLOW", YELLOW_DURATION, density_level, density_pct)
        self._countdown(YELLOW_DURATION, "YELLOW")

        if self._stop_cycle.is_set():
            return

        # ── RED Phase ────────────────────────────────────────────────
        with self._signal_lock:
            self._current_signal = "RED"
        self._print_signal_header(intersection_id, "RED", RED_DURATION, density_level, density_pct)
        self._countdown(RED_DURATION, "RED")

        if not self._stop_cycle.is_set():
            print(f"\n{COLOUR_CYAN}  ⏳ Awaiting next signal command from MQTT broker...{COLOUR_RESET}\n")

    # ─────────────────────────── Display ───────────────────────────

    def _print_standby(self) -> None:
        """Print initial standby banner on successful connection."""
        print(f"\n{COLOUR_BOLD}{COLOUR_CYAN}{'═' * 44}")
        print("   🚦  VIRTUAL TRAFFIC CONTROLLER ONLINE")
        print(f"{'═' * 44}{COLOUR_RESET}")
        print(f"  Broker  : {self.config.broker_host}:{self.config.broker_port}")
        print(f"  Topic   : {self.config.topic}")
        print(f"  Status  : {COLOUR_GREEN}CONNECTED ✓{COLOUR_RESET}")
        print(f"{COLOUR_BOLD}{COLOUR_CYAN}{'═' * 44}{COLOUR_RESET}\n")

    def _print_signal_header(
        self,
        intersection_id: int,
        signal: str,
        duration: int,
        density_level: str,
        density_pct: float,
    ) -> None:
        """Print formatted signal state display to terminal."""
        colour  = SIGNAL_COLOURS.get(signal, COLOUR_RESET)
        icon    = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(signal, "⚪")

        print(f"\n{COLOUR_BOLD}{'═' * 44}")
        print("   VIRTUAL TRAFFIC CONTROLLER")
        print(f"{'═' * 44}{COLOUR_RESET}")
        print(f"  Intersection : {intersection_id}")
        print(f"  Signal       : {colour}{COLOUR_BOLD}{icon}  {signal}{COLOUR_RESET}")
        print(f"  Duration     : {duration} seconds")
        print(f"  Density      : {density_level} ({density_pct:.1f}%)")
        print(f"{COLOUR_BOLD}{'═' * 44}{COLOUR_RESET}")

    def _countdown(self, duration: int, signal: str) -> None:
        """Display a live countdown timer for the active signal phase."""
        colour = SIGNAL_COLOURS.get(signal, COLOUR_RESET)
        for remaining in range(duration, 0, -1):
            if self._stop_cycle.is_set():
                break
            print(
                f"  {colour}⏱  {signal} — {remaining:>3}s remaining{COLOUR_RESET}",
                end="\r",
                flush=True,
            )
            time.sleep(1)
        print()  # Newline after countdown ends

    def run_forever(self) -> None:
        """Block the main thread while the MQTT network loop runs.

        Call after connect() to keep the controller alive waiting for messages.
        """
        print("  [Controller] Running. Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received — shutting down controller.")
            self.disconnect()
