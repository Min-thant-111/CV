"""
Intelligent Traffic Signal Decision Engine module.
"""

from dataclasses import dataclass
import inspect
import math
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
    max_green_duration: int = 180    # Absolute maximum cap for green duration
    seconds_per_vehicle: float = 1.0
    seconds_per_vehicle_per_path: float = 1.0


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
        road_path_count: int = 1,
    ) -> SignalDecision:
        raise NotImplementedError


class RuleBasedSignalStrategy(BaseSignalStrategy):
    """Adaptive rule-based strategy using density, vehicles, and road paths.

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
        road_path_count: int = 1,
    ) -> SignalDecision:
        level_upper = str(density_level).upper().strip()
        vehicle_count = max(0, int(vehicle_count))
        road_path_count = max(1, int(road_path_count))

        if level_upper == "LOW":
            base_duration = config.green_duration_low
        elif level_upper in ("MED", "MEDIUM"):
            level_upper = "MEDIUM"
            base_duration = config.green_duration_medium
        elif level_upper == "HIGH":
            base_duration = config.green_duration_high
        else:
            level_upper = "MEDIUM"
            base_duration = config.green_duration_medium

        # Total count prevents equal-density scenarios with different demand
        # from receiving identical time. Vehicles per path adds queue pressure,
        # so the same nine vehicles get more time on one path than on two.
        count_adjustment = math.ceil(vehicle_count * config.seconds_per_vehicle)
        per_path_adjustment = math.ceil(
            (vehicle_count / road_path_count)
            * config.seconds_per_vehicle_per_path
        )
        uncapped_duration = base_duration + count_adjustment + per_path_adjustment
        duration = uncapped_duration

        # Enforce configurable minimum and maximum safety bounds
        duration = max(
            config.min_green_duration, min(config.max_green_duration, duration)
        )
        cap_note = (
            f" (capped from {uncapped_duration}s)"
            if duration != uncapped_duration else ""
        )
        reason = (
            f"{level_upper.title()} traffic density ({density_percentage:.1f}%) with "
            f"{vehicle_count} vehicles across {road_path_count} road "
            f"{'path' if road_path_count == 1 else 'paths'}. "
            f"GREEN time: {base_duration}s base + {count_adjustment}s vehicle demand "
            f"+ {per_path_adjustment}s per-path queue = {duration}s{cap_note}."
        )

        return SignalDecision(
            signal="GREEN",
            duration=duration,
            density_level=level_upper,
            density_percentage=density_percentage,
            vehicle_count=vehicle_count,
            reason=reason,
            road_path_count=road_path_count,
            base_duration=base_duration,
            vehicle_demand_duration=count_adjustment,
            per_path_queue_duration=per_path_adjustment,
            uncapped_duration=uncapped_duration,
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
        road_path_count: Optional[int] = None,
    ) -> SignalDecision:
        """Evaluate traffic state and return dynamic green signal timing decision.

        Args:
            input_data: DensityMetrics object, dictionary, or density level string ("LOW"/"MEDIUM"/"HIGH").
            vehicle_count: Optional count if input_data is a string.
            density_percentage: Optional percentage if input_data is a string.
            road_path_count: Optional path/way count if input_data is a string or dictionary.

        Returns:
            SignalDecision object.
        """
        if isinstance(input_data, DensityMetrics):
            density_level = input_data.density_level
            v_count = input_data.total_vehicle_count
            d_pct = input_data.density_percentage
            path_count = input_data.road_path_count
        elif isinstance(input_data, dict):
            density_level = input_data.get("density_level", "LOW")
            v_count = input_data.get("total_vehicle_count", 0)
            d_pct = input_data.get("density_percentage", 0.0)
            path_count = input_data.get("road_path_count", road_path_count or 1)
        else:
            density_level = str(input_data)
            v_count = vehicle_count if vehicle_count is not None else 0
            d_pct = density_percentage if density_percentage is not None else 0.0
            path_count = road_path_count if road_path_count is not None else 1

        decision_args = {
            "density_level": density_level,
            "vehicle_count": v_count,
            "density_percentage": d_pct,
            "config": self.config,
        }
        # Preserve compatibility with third-party strategies written before
        # road-path awareness was added.
        if "road_path_count" in inspect.signature(self.strategy.decide).parameters:
            decision_args["road_path_count"] = path_count
        return self.strategy.decide(**decision_args)
