"""
threshold-related operations
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from .data_models import ThresholdConfig, ThresholdCondition
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdDefaults:
    """Centralized default values."""
    RESPONSE_TIME_MS: int = 500
    ERROR_RATE_PCT: int = 5
    THROUGHPUT_REQ_PER_SEC: int = 10
    TARGET_PERCENTILE_FIELD: str = "p95_ms"

    RT_DEGRADATION_THRESHOLD_PCT: float = 10.0  # Allow 10% degradation
    ER_DEGRADATION_THRESHOLD_PCT: float = 2.0  # Allow 2% absolute increase
    TP_DEGRADATION_THRESHOLD_PCT: float = 5.0  # Allow 5% decrease


class ThresholdManager:
    """
    Modern threshold manager with clean architecture.
    """

    def __init__(self):
        self.defaults = ThresholdDefaults()
        logger.debug("ThresholdManager initialized")

    def resolve_thresholds(
            self,
            threshold_configs: List[ThresholdConfig],  # <-- CORRECT TYPE HINT: It receives a list
            user_args: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Resolves effective threshold values from a LIST of ThresholdConfig objects
        and user arguments, ensuring it handles the modern data structure correctly.
        """
        user_args = user_args or {}

        # Start with the system defaults.
        resolved = {
            "response_time": self.defaults.RESPONSE_TIME_MS,
            "error_rate": self.defaults.ERROR_RATE_PCT,
            "throughput": self.defaults.THROUGHPUT_REQ_PER_SEC,
        }

        if threshold_configs:
            for config in threshold_configs:
                if config.target in resolved:
                    resolved[config.target] = config.threshold_value
        user_overrides = {
            "response_time": user_args.get("rt_threshold", user_args.get("response_time_threshold")),
            "error_rate": user_args.get("er_threshold", user_args.get("error_rate_threshold")),
            "throughput": user_args.get("tp_threshold", user_args.get("throughput_threshold")),
            "p95_response_time": user_args.get("p95_threshold", user_args.get("p95_response_time_threshold"))
        }

        # Apply non-None user overrides
        for key, value in user_overrides.items():
            if value is not None:
                try:
                    resolved[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid threshold value for {key}: {value}, using resolved value: {resolved[key]}")

        logger.debug(f"Resolved thresholds: {resolved}")
        return resolved

    @classmethod
    def get_threshold_configs(cls, user_args):
        """
            DRY: Single method to create ThresholdConfig objects using clean names.
            Used by transformers for report generation.
            """
        effective_values = cls.get_effective_values(user_args)
        return [
            ThresholdConfig(
                target=threshold_key,
                threshold_value=float(value),
                condition=cls.THRESHOLD_DEFINITIONS[threshold_key]['condition']
            )
            for threshold_key, value in effective_values.items()
        ]

    @classmethod
    def get_effective_values(cls, user_args: Dict[str, Any]) -> Dict[str, float]:
        """
        DRY: Single method to resolve effective threshold values using clean keys.
        Used by ALL analyzers - PerformanceAnalyzer, ComparisonAnalyzer, etc.
        """
        return {
            threshold_key: user_args.get(
                definition['user_arg_key'],
                definition['default_value']
            )
            for threshold_key, definition in cls.THRESHOLD_DEFINITIONS.items()
        }

    @classmethod
    def get_legacy_threshold_dict(cls, user_args: Dict[str, Any]) -> Dict[str, float]:
        """
        BACKWARD COMPATIBILITY: Returns thresholds using legacy keys.
        Used by existing PerformanceAnalyzer methods.
        """
        effective_values = cls.get_effective_values(user_args)
        return {
            cls.THRESHOLD_DEFINITIONS[threshold_key]['legacy_key']: value
            for threshold_key, value in effective_values.items()
        }

    @classmethod
    def get_comparison_tolerances(cls, user_args: Dict[str, Any]) -> Dict[str, float]:
        """
        NEW: Get comparison tolerance thresholds for degradation analysis.
        Used by ComparisonAnalyzer to determine acceptable performance changes.
        """
        return {
            threshold_key: user_args.get(
                f"{threshold_key}_degradation_threshold",
                definition['comparison_tolerance_pct']
            )
            for threshold_key, definition in cls.THRESHOLD_DEFINITIONS.items()
        }

    @classmethod
    def get_threshold_value(cls, threshold_key: str, user_args: Dict[str, Any]) -> float:
        """
        DRY: Single method to get a specific threshold value by clean key.
        """
        return cls.get_effective_values(user_args)[threshold_key]

    @classmethod
    def get_percentile_field(cls) -> str:
        """DRY: Single method for percentile field access."""
        return ThresholdDefaults.TARGET_PERCENTILE_FIELD

    @classmethod
    def get_all_threshold_keys(cls) -> List[str]:
        """
        NEW: Get all available threshold keys.
        Useful for validation and iteration.
        """
        return list(cls.THRESHOLD_DEFINITIONS.keys())

    @classmethod
    def validate_user_args(cls, user_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        NEW: Validate and sanitize user arguments.
        Returns cleaned arguments with proper types.
        """
        cleaned_args = {}
        for threshold_key, definition in cls.THRESHOLD_DEFINITIONS.items():
            user_key = definition['user_arg_key']
            if user_key in user_args:
                try:
                    cleaned_args[user_key] = float(user_args[user_key])
                except (ValueError, TypeError):
                    # Use default if invalid value provided
                    cleaned_args[user_key] = definition['default_value']
        return cleaned_args

    THRESHOLD_DEFINITIONS = {
        'response_time': {
            'user_arg_key': 'response_time_threshold',
            'default_value': ThresholdDefaults.RESPONSE_TIME_MS,
            'condition': ThresholdCondition.LESS_THAN,
            'display_name': 'Response Time',
            'legacy_key': 'rt_threshold',
            'comparison_tolerance_pct': ThresholdDefaults.RT_DEGRADATION_THRESHOLD_PCT
        },
        'error_rate': {
            'user_arg_key': 'error_rate_threshold',
            'default_value': ThresholdDefaults.ERROR_RATE_PCT,
            'condition': ThresholdCondition.LESS_THAN,
            'display_name': 'Error Rate',
            'legacy_key': 'er_threshold',
            'comparison_tolerance_pct': ThresholdDefaults.ER_DEGRADATION_THRESHOLD_PCT
        },
        'throughput': {
            'user_arg_key': 'throughput_threshold',
            'default_value': ThresholdDefaults.THROUGHPUT_REQ_PER_SEC,
            'condition': ThresholdCondition.GREATER_THAN,
            'display_name': 'Throughput',
            'legacy_key': 'tp_threshold',
            'comparison_tolerance_pct': ThresholdDefaults.TP_DEGRADATION_THRESHOLD_PCT
        }
    }
