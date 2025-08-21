"""
Core data models - data structures only.
Author: Karen Florykian
"""
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import logging

from pydantic import Field
from langchain_core.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)


# ================== ENUMS ==================
class PerformanceStatus(Enum):
    """Performance status enumeration."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"


class PerformanceDirection(Enum):
    """Performance change directions with visual indicators."""
    IMPROVED = "🟢 Improved"
    DEGRADED = "🔴 Degraded"
    STABLE = "🟡 Stable"
    NEW = "🔵 New"
    MISSING = "⚪ Missing"


class ThresholdCondition(str, Enum):
    """Enumeration for threshold comparison operators."""
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    EQUALS = "equals"


class ReportType(Enum):
    """Report type enumeration for type safety"""
    BASELINE = "baseline"
    COMPARISON = "comparison"
    UI_PERFORMANCE = "ui_performance"
    BACKEND = "backend"
    MOBILE = "mobile"
    GATLING = "GATLING"
    JMETER = "JMETER"


# ================== CONFIGURATION MODELS ==================
@dataclass
class ThresholdConfig:
    """Unified threshold configuration."""
    target: str
    threshold_value: float
    condition: ThresholdCondition = ThresholdCondition.LESS_THAN

    def is_violated(self, actual_value: float) -> bool:
        """Check if threshold is violated."""
        if self.condition == ThresholdCondition.LESS_THAN:
            return actual_value >= self.threshold_value
        elif self.condition == ThresholdCondition.GREATER_THAN:
            return actual_value <= self.threshold_value
        else:
            return abs(actual_value - self.threshold_value) > 0.001

# ================== ANALYSIS MODELS ==================
@dataclass
class Recommendation:
    """Performance recommendation with priority and details."""
    priority: str  # High, Medium, Low
    action: str
    details: str
    category: str = "Performance"


@dataclass
class AnalysisResult:
    """Results of performance analysis."""
    status: PerformanceStatus
    justification: str
    failed_metrics: List[str]
    recommendations: List[Recommendation]
    summary_stats: Dict[str, Any]
    confidence_score: float = 0.8


@dataclass(frozen=True)
class ComparisonResult:
    """Results of performance comparison analysis."""
    name: str
    metric_name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    percent_change: float
    direction: PerformanceDirection
    threshold_exceeded: bool = False

    @property
    def is_significant(self) -> bool:
        """Calculate significance based on percent change."""
        return abs(self.percent_change) >= 5.0

    @property
    def change_display(self) -> str:
        """Formatted change percentage for display."""
        if self.percent_change == 0:
            return "No Change"
        sign = "+" if self.percent_change > 0 else ""
        return f"{sign}{self.percent_change:.1f}%"


# ================== ETL MODELS ==================
@dataclass
class ETLContext:
    """Context object for ETL pipeline operations."""
    report_id: str
    api_wrapper: Any
    extraction_params: Dict[str, Any] = field(default_factory=dict)
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    loading_params: Dict[str, Any] = field(default_factory=dict)

    def get_param(self, stage: str, key: str, default: Any = None) -> Any:
        """Get parameter for specific stage."""
        params = getattr(self, f"{stage}_params", {})
        return params.get(key, default)


# ================== CORE METRICS ==================
@dataclass
class TransactionMetrics:
    """Core transaction performance metrics."""
    name: str
    samples: int
    ko: int
    avg: float
    min: float
    max: float
    pct90: float
    pct95: float
    pct99: float
    throughput: float
    received_kb_per_sec: float
    sent_kb_per_sec: float
    total: int

    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.samples < 0:
            raise ValueError(f"Samples cannot be negative: {self.samples}")
        if self.ko > self.samples:
            raise ValueError(f"KO count ({self.ko}) cannot exceed samples ({self.samples})")

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return (self.ko / self.samples * 100) if self.samples > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return 100.0 - self.error_rate


@dataclass(frozen=True)
class ReportSummary:
    """
    Immutable summary metrics for the entire performance test.
    Maintains legacy field structure for Excel compatibility.
    """
    max_user_count: int
    ramp_up_period: float
    error_rate: float
    date_start: str
    date_end: str
    throughput: float
    duration: float
    think_time: str

    baseline_throughput: Optional[float] = None
    throughput_improvement: Optional[float] = None

    def __post_init__(self):
        """Validate summary metrics after initialization."""
        if self.max_user_count < 0:
            raise ValueError(f"Max user count cannot be negative: {self.max_user_count}")

        if isinstance(self.ramp_up_period, timedelta):
            if self.ramp_up_period.total_seconds() < 0:
                raise ValueError(f"Ramp-up period cannot be negative: {self.ramp_up_period}")
        elif self.ramp_up_period < 0:
            raise ValueError(f"Ramp-up period cannot be negative: {self.ramp_up_period}")

        if not (0 <= self.error_rate <= 100):
            raise ValueError(f"Error rate must be 0-100: {self.error_rate}")

        if isinstance(self.throughput, timedelta):
            if self.throughput.total_seconds() < 0:
                raise ValueError(f"Throughput cannot be negative: {self.throughput}")
        elif self.throughput < 0:
            raise ValueError(f"Throughput cannot be negative: {self.throughput}")

        if isinstance(self.duration, timedelta):
            if self.duration.total_seconds() < 0:
                raise ValueError(f"Duration cannot be negative: {self.duration}")
        elif self.duration < 0:
            raise ValueError(f"Duration cannot be negative: {self.duration}")
        logger.debug(f"Validated ReportSummary with throughput: {self.throughput}")

    @property
    def report_id(self) -> str:
        """Generate report ID from date_start if not available."""
        return getattr(self, '_report_id', self.date_start.replace(' ', '_').replace(':', '-'))

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format for Excel compatibility."""
        return {
            'max_user_count': self.max_user_count,
            'ramp_up_period': self.ramp_up_period,
            'error_rate': self.error_rate,
            'date_start': self.date_start,
            'date_end': self.date_end,
            'throughput': self.throughput,
            'duration': self.duration,
            'think_time': self.think_time
        }

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> 'ReportSummary':
        """Create ReportSummary from legacy dictionary format."""
        return cls(
            max_user_count=int(data.get('max_user_count', 0)),
            ramp_up_period=float(data.get('ramp_up_period', 0.0)),
            error_rate=float(data.get('error_rate', 0.0)),
            date_start=str(data.get('date_start', '')),
            date_end=str(data.get('date_end', '')),
            throughput=float(data.get('throughput', 0.0)),
            duration=float(data.get('duration', 0.0)),
            think_time=str(data.get('think_time', 0.0))
        )

    @property
    def test_status(self) -> str:
        """Determine overall test status for Excel conditional formatting."""
        if self.error_rate > 10:
            return "FAILED"
        elif self.error_rate > 5:
            return "WARNING"
        elif self.throughput == 0:
            return "NO_TRAFFIC"
        else:
            return "PASSED"


class UIMetrics:
    """Data model for UI performance metrics."""

    def __init__(self, step_name: str, performance_score: float, audit: str, numeric_value: float):
        self.step_name = step_name
        self.performance_score = performance_score
        self.audit = audit
        self.numeric_value = numeric_value


# ================== PERFORMANCE REPORTS ==================
@dataclass
class PerformanceReport:
    """
    Complete performance report containing summary and transaction metrics.
    Provides both modern object-oriented interface and legacy dictionary compatibility.
    """
    summary: ReportSummary
    transactions: Dict[str, TransactionMetrics]
    build_status: PerformanceStatus = PerformanceStatus.WARNING
    analysis_summary: Optional[str] = None
    carrier_report_url: str = None
    thresholds: Optional[List[Any]] = None
    report_type: str = "GATLING"  # or "JMETER"
    test_name: str = "Unknown Test"
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the complete report structure."""
        if not self.transactions:
            raise ValueError("Report must contain at least one transaction")
        if 'Total' not in self.transactions:
            raise ValueError("Report must contain 'Total' transaction summary")
        logger.info(f"Validated PerformanceReport with {len(self.transactions)} transactions")


@dataclass(frozen=True)
class UIPerformanceReport:
    """
    Immutable summary for a UI performance test, following the backend report pattern.
    """
    report_id: str
    report_name: str
    test_status: str
    start_time: str
    end_time: str
    browser: str
    worksheets_data: List[UIMetrics]
    report_type: str
    carrier_report_url: Optional[str] = None
    build_status: PerformanceStatus = PerformanceStatus.WARNING

    def __post_init__(self):
        if not self.report_id:
            raise ValueError("report_id cannot be empty")
        if not self.report_name:
            raise ValueError("report_name cannot be empty")
        if not self.worksheets_data:
            raise ValueError("worksheets_data cannot be empty")


# ================== TICKET MODELS ==================
class TicketPayload(BaseModel):
    """Payload for performance test ticket creation."""
    test_name: str = Field(..., description="Name of the performance test")
    test_type: str = Field(..., description="Type of test (load, stress, spike, etc.)")
    duration: int = Field(..., description="Test duration in seconds")
    user_count: int = Field(..., description="Number of virtual users")
    ramp_up: int = Field(..., description="Ramp-up period in seconds")

    # Optional fields
    scenario: Optional[str] = Field(None, description="Test scenario description")
    environment: Optional[str] = Field("staging", description="Target environment")
    tags: Optional[List[str]] = Field(default_factory=list, description="Test tags")
    thresholds: Optional[List[ThresholdConfig]] = Field(default_factory=list, description="Performance thresholds")

    class Config:
        schema_extra = {
            "example": {
                "test_name": "API Load Test",
                "test_type": "load",
                "duration": 300,
                "user_count": 100,
                "ramp_up": 60,
                "scenario": "Standard user journey",
                "environment": "staging",
                "tags": ["api", "regression"],
                "thresholds": [
                    {
                        "target": "response_time_p95",
                        "threshold_value": 1000,
                        "condition": "less_than"
                    }
                ]
            }
        }


from langchain_core.pydantic_v1 import BaseModel, Field


class PerformanceAnalysisResult(BaseModel):
    """Structured result from LLM performance comparison analysis."""
    summary: str = Field(..., description="Executive summary of performance comparison")
    key_findings: List[str] = Field(..., description="List of key findings from the comparison")
    performance_trends: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                                         description="Identified performance trends")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Risk assessment if performance degraded")
    confidence_score: float = Field(default=0.8, description="Confidence in the analysis (0-1)")
