"""
Lighthouse-specific threshold configuration
Following the pattern from threshold_manager.py
"""

from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LighthouseMetric(Enum):
    """Lighthouse metric identifiers"""
    FCP = "first-contentful-paint"
    LCP = "largest-contentful-paint"
    CLS = "cumulative-layout-shift"
    TBT = "total-blocking-time"
    SI = "speed-index"
    TTI = "interactive"


@dataclass(frozen=True)
class LighthouseThresholdDefaults:
    """
    Centralized Lighthouse threshold defaults based on web.dev standards.
    No magic numbers - all thresholds defined here.
    """
    # First Contentful Paint (ms)
    FCP_GOOD: int = 1800
    FCP_POOR: int = 3000

    # Largest Contentful Paint (ms)
    LCP_GOOD: int = 2500
    LCP_POOR: int = 4000

    # Cumulative Layout Shift (score)
    CLS_GOOD: float = 0.1
    CLS_POOR: float = 0.25

    # Total Blocking Time (ms)
    TBT_GOOD: int = 200
    TBT_POOR: int = 600

    # Speed Index (ms)
    SI_GOOD: int = 3400
    SI_POOR: int = 5800

    # Time to Interactive (ms)
    TTI_GOOD: int = 3800
    TTI_POOR: int = 7300

    # Column width constraints
    MIN_COLUMN_WIDTH: int = 8
    MAX_COLUMN_WIDTH: int = 50
    METRIC_COLUMN_MIN_WIDTH: int = 30
    VALUE_COLUMN_MIN_WIDTH: int = 12
    STATUS_COLUMN_MIN_WIDTH: int = 20


class LighthouseThresholdManager:
    """
    Manages Lighthouse-specific thresholds following the backend pattern.
    """

    def __init__(self):
        self.defaults = LighthouseThresholdDefaults()
        self._threshold_map = self._build_threshold_map()
        logger.debug("LighthouseThresholdManager initialized")

    def _build_threshold_map(self) -> Dict[str, Dict[str, Any]]:
        """Build the threshold mapping configuration"""
        return {
            LighthouseMetric.FCP.value: {
                'good': self.defaults.FCP_GOOD,
                'poor': self.defaults.FCP_POOR,
                'display_name': 'First Contentful Paint',
                'unit': 'ms'
            },
            LighthouseMetric.LCP.value: {
                'good': self.defaults.LCP_GOOD,
                'poor': self.defaults.LCP_POOR,
                'display_name': 'Largest Contentful Paint',
                'unit': 'ms'
            },
            LighthouseMetric.CLS.value: {
                'good': self.defaults.CLS_GOOD,
                'poor': self.defaults.CLS_POOR,
                'display_name': 'Cumulative Layout Shift',
                'unit': 'score'
            },
            LighthouseMetric.TBT.value: {
                'good': self.defaults.TBT_GOOD,
                'poor': self.defaults.TBT_POOR,
                'display_name': 'Total Blocking Time',
                'unit': 'ms'
            },
            LighthouseMetric.SI.value: {
                'good': self.defaults.SI_GOOD,
                'poor': self.defaults.SI_POOR,
                'display_name': 'Speed Index',
                'unit': 'ms'
            },
            LighthouseMetric.TTI.value: {
                'good': self.defaults.TTI_GOOD,
                'poor': self.defaults.TTI_POOR,
                'display_name': 'Time to Interactive',
                'unit': 'ms'
            }
        }

    def get_thresholds_for_metric(self, metric_name: str) -> Dict[str, Any]:
        """Get threshold configuration for a specific metric"""
        # Check direct match first
        if metric_name in self._threshold_map:
            return self._threshold_map[metric_name]

        # Check if metric name contains any known metric
        metric_name_lower = metric_name.lower()
        for key, config in self._threshold_map.items():
            if key in metric_name_lower or config['display_name'].lower() in metric_name_lower:
                return config

        # Return empty dict if no match
        return {}

    def get_status(self, metric_name: str, value: float) -> str:
        """Determine status based on thresholds"""
        thresholds = self.get_thresholds_for_metric(metric_name)

        if not thresholds or value is None:
            return "Info"

        if value <= thresholds['good']:
            return "Good"
        elif value <= thresholds['poor']:
            return "Needs Improvement"
        else:
            return "Poor"
