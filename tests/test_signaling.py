"""
Unit tests for SignalDecision model and SignalDecisionEngine decision logic.
"""

import unittest
from backend.models.density import DensityMetrics
from backend.models.signaling import SignalDecision
from backend.signaling.signal_engine import (
    SignalDecisionEngine,
    SignalConfig,
    RuleBasedSignalStrategy,
    BaseSignalStrategy,
)


class TestSignalDecisionEngine(unittest.TestCase):
    """Test suite for intelligent traffic signal timing decisions across all density levels."""

    def setUp(self):
        # Configured defaults: LOW -> 30s, MEDIUM -> 50s, HIGH -> 70s
        self.engine = SignalDecisionEngine(
            config=SignalConfig(
                green_duration_low=30,
                green_duration_medium=50,
                green_duration_high=70,
            )
        )

    def test_signal_decision_to_dict(self):
        decision = SignalDecision(
            signal="GREEN",
            duration=70,
            density_level="HIGH",
            density_percentage=85.0,
            vehicle_count=8,
            reason="High traffic density detected",
        )
        d_dict = decision.to_dict()
        self.assertEqual(d_dict["signal"], "GREEN")
        self.assertEqual(d_dict["duration"], 70)
        self.assertEqual(d_dict["density_level"], "HIGH")
        self.assertEqual(d_dict["density_percentage"], 85.0)

    def test_low_density_decision(self):
        """Test that LOW density returns 30s GREEN timing."""
        decision = self.engine.evaluate(
            input_data="LOW", vehicle_count=2, density_percentage=15.0
        )
        self.assertEqual(decision.signal, "GREEN")
        self.assertEqual(decision.duration, 30)
        self.assertEqual(decision.density_level, "LOW")
        self.assertIn("30s", decision.reason)

    def test_medium_density_decision(self):
        """Test that MEDIUM density returns 50s GREEN timing."""
        decision = self.engine.evaluate(
            input_data="MEDIUM", vehicle_count=4, density_percentage=55.0
        )
        self.assertEqual(decision.signal, "GREEN")
        self.assertEqual(decision.duration, 50)
        self.assertEqual(decision.density_level, "MEDIUM")
        self.assertIn("50s", decision.reason)

    def test_high_density_decision(self):
        """Test that HIGH density returns 70s GREEN timing."""
        decision = self.engine.evaluate(
            input_data="HIGH", vehicle_count=8, density_percentage=85.0
        )
        self.assertEqual(decision.signal, "GREEN")
        self.assertEqual(decision.duration, 70)
        self.assertEqual(decision.density_level, "HIGH")
        self.assertIn("70s", decision.reason)

    def test_evaluation_with_density_metrics_object(self):
        """Test evaluate() when passed a DensityMetrics object directly."""
        metrics = DensityMetrics(
            frame_index=1,
            timestamp=0.033,
            total_vehicle_count=9,
            class_counts={"car": 5, "bus": 2},
            weighted_vehicle_units=10.0,
            capacity_units=10.0,
            density_percentage=100.0,
            density_level="HIGH",
            active_track_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        )

        decision = self.engine.evaluate(metrics)
        self.assertEqual(decision.duration, 70)
        self.assertEqual(decision.density_level, "HIGH")
        self.assertEqual(decision.vehicle_count, 9)

    def test_custom_config_overrides(self):
        """Test that custom SignalConfig timings are respected."""
        custom_config = SignalConfig(
            green_duration_low=20,
            green_duration_medium=40,
            green_duration_high=60,
        )
        custom_engine = SignalDecisionEngine(config=custom_config)

        res_low = custom_engine.evaluate("LOW")
        self.assertEqual(res_low.duration, 20)

        res_high = custom_engine.evaluate("HIGH")
        self.assertEqual(res_high.duration, 60)

    def test_strategy_swapping(self):
        """Test swapping to a custom strategy."""

        class CustomStrategy(BaseSignalStrategy):
            def decide(self, density_level, vehicle_count, density_percentage, config):
                return SignalDecision(
                    signal="GREEN",
                    duration=99,
                    density_level=density_level,
                    density_percentage=density_percentage,
                    vehicle_count=vehicle_count,
                    reason="Custom test strategy",
                )

        self.engine.set_strategy(CustomStrategy())
        res = self.engine.evaluate("LOW")
        self.assertEqual(res.duration, 99)
        self.assertEqual(res.reason, "Custom test strategy")


if __name__ == "__main__":
    unittest.main()
