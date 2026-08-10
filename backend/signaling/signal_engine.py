"""
Intelligent Traffic Signal Decision Engine module.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

from backend.models.density import DensityMetrics
from backend.models.signaling import SignalDecision


@dataclass
class SignalConfig:
    """Configurable timing parameters for traffic signal decision rules."""

    green_duration_low: int = 30     # Green duration for LOW density (seconds)
    green_duration_medium: int = 50  # Green duration for MEDIUM density (seconds)
    green_duration_high: int = 70    # Green duration for HIGH density (seconds)
    yellow_duration: int = 5         # Transition yellow buffer (seconds)
    min_green_duration: int = 15     # Absolute minimum safety green duration
    max_green_duration: int = 90     # Absolute maximum cap for green duration


class SignalEngineError(Exception):
    """Base exception for SignalEngine errors."""

    pass


class BaseSignalStrategy:
    """Strategy interface allowing future extensions (e.g. Adaptive PID or RL)."""

    def decide(
        self,
        density_level: str,
        vehicle_count: int,
        density_percentage: float,
        config: SignalConfig,
    ) -> SignalDecision:
        raise NotImplementedError


class RuleBasedSignalStrategy(BaseSignalStrategy):
    """Deterministic rule-based signal decision strategy mapping density levels:

    - LOW    -> GREEN 30 seconds
    - MEDIUM -> GREEN 50 seconds
    - HIGH   -> GREEN 70 seconds
    """

    def decide(
        self,
        density_level: str,
        vehicle_count: int,
        density_percentage: float,
        config: SignalConfig,
    ) -> SignalDecision:
        level_upper = str(density_level).upper().strip()

        if level_upper == "LOW":
            duration = config.green_duration_low
            reason = (
                f"Low traffic density detected ({density_percentage:.1f}% density, "
                f"{vehicle_count} vehicles). Allocated baseline GREEN signal of {duration}s."
            )
        elif level_upper in ("MED", "MEDIUM"):
            level_upper = "MEDIUM"
            duration = config.green_duration_medium
            reason = (
                f"Moderate traffic density detected ({density_percentage:.1f}% density, "
                f"{vehicle_count} vehicles). Allocated extended GREEN signal of {duration}s."
            )
        elif level_upper == "HIGH":
            duration = config.green_duration_high
            reason = (
                f"High traffic density detected ({density_percentage:.1f}% density, "
                f"{vehicle_count} vehicles). Allocated maximum GREEN signal of {duration}s to clear junction."
            )
        else:
            level_upper = "MEDIUM"
            duration = config.green_duration_medium
            reason = (
                f"Unrecognized density level '{density_level}'. "
                f"Defaulted to standard GREEN signal of {duration}s."
            )

        # Enforce configurable minimum and maximum safety bounds
        duration = max(
            config.min_green_duration, min(config.max_green_duration, duration)
        )

        return SignalDecision(
            signal="GREEN",
            duration=duration,
            density_level=level_upper,
            density_percentage=density_percentage,
            vehicle_count=vehicle_count,
            reason=reason,
        )


class SignalDecisionEngine:
    """Intelligent Traffic Signal Decision Engine orchestrating signal decision strategies."""

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        strategy: Optional[BaseSignalStrategy] = None,
    ):
        """Args:

        config: SignalConfig instance holding timing parameters.
        strategy: Decision strategy instance (defaults to RuleBasedSignalStrategy).
        """
        self.config = config or SignalConfig()
        self.strategy = strategy or RuleBasedSignalStrategy()

    def set_strategy(self, strategy: BaseSignalStrategy) -> None:
        """Swap or update the active decision strategy algorithm."""
        self.strategy = strategy

    def evaluate(
        self,
        input_data: Union[DensityMetrics, Dict[str, Any], str],
        vehicle_count: Optional[int] = None,
        density_percentage: Optional[float] = None,
    ) -> SignalDecision:
        """Evaluate traffic state and return dynamic green signal timing decision.

        Args:
            input_data: DensityMetrics object, dictionary, or density level string ("LOW"/"MEDIUM"/"HIGH").
            vehicle_count: Optional count if input_data is a string.
            density_percentage: Optional percentage if input_data is a string.

        Returns:
            SignalDecision object.
        """
        if isinstance(input_data, DensityMetrics):
            density_level = input_data.density_level
            v_count = input_data.total_vehicle_count
            d_pct = input_data.density_percentage
        elif isinstance(input_data, dict):
            density_level = input_data.get("density_level", "LOW")
            v_count = input_data.get("total_vehicle_count", 0)
            d_pct = input_data.get("density_percentage", 0.0)
        else:
            density_level = str(input_data)
            v_count = vehicle_count if vehicle_count is not None else 0
            d_pct = density_percentage if density_percentage is not None else 0.0

        return self.strategy.decide(
            density_level=density_level,
            vehicle_count=v_count,
            density_percentage=d_pct,
            config=self.config,
        )
