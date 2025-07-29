"""
Core data models - Pure data structures only.
No business logic, configuration, or analysis results.

Author: Karen Florykian
"""

from datetime import date, timedelta
from enum import Enum
from pydantic import Field
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging


# ================== ENUMS ==================
class PerformanceStatus(Enum):
    """Performance status enumeration."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


class PerformanceDirection(Enum):
    """Performance change directions with visual indicators."""
    IMPROVED = "🟢 Improved"
    DEGRADED = "🔴 Degraded"
    STABLE = "🟡 Stable"
    NEW = "🔵 New"
    MISSING = "⚪ Missing"


logger = logging.getLogger(__name__)


# ================== CORE METRICS ==================

@dataclass(frozen=True)
class TransactionMetrics:
    """
    Immutable performance metrics for a single transaction.
    Provides both legacy field names and modern field names for backward compatibility.

    This class maintains exact legacy field structure to prevent Excel report breakage.
    """
    request_name: str
    Total: int
    KO: int
    OK: int
    min: float
    max: float
    average: float
    median: float
    Error_pct: float = field(metadata={'legacy_name': 'Error%'})
    pct_90: float = field(metadata={'legacy_name': '90Pct'})
    pct_95: float = field(metadata={'legacy_name': '95Pct'})

    def __post_init__(self):
        """Validate metrics after initialization to prevent invalid data."""
        if not self.request_name or not self.request_name.strip():
            raise ValueError("Transaction name cannot be empty")

        if self.Total < 0:
            raise ValueError(f"Total requests cannot be negative: {self.Total}")

        if self.KO < 0:
            raise ValueError(f"KO count cannot be negative: {self.KO}")

        if self.OK < 0:
            raise ValueError(f"OK count cannot be negative: {self.OK}")

        if self.Total != (self.KO + self.OK):
            raise ValueError(f"Total ({self.Total}) must equal KO ({self.KO}) + OK ({self.OK})")

        if not (0 <= self.Error_pct <= 100):
            raise ValueError(f"Error percentage must be 0-100: {self.Error_pct}")

        if self.min < 0:
            raise ValueError(f"Min response time cannot be negative: {self.min}")

        if self.max < self.min:
            raise ValueError(f"Max ({self.max}) cannot be less than min ({self.min})")

        if self.average < 0:
            raise ValueError(f"Average response time cannot be negative: {self.average}")

        logger.debug(f"Validated TransactionMetrics for {self.request_name}")

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to exact legacy dictionary format for Excel compatibility.
        This prevents breaking existing Excel report generation code.
        """
        return {
            'request_name': self.request_name,
            'Total': self.Total,
            'KO': self.KO,
            'OK': self.OK,
            'min': self.min,
            'max': self.max,
            'average': self.average,
            'median': self.median,
            'Error%': self.Error_pct,  # Legacy field name
            '90Pct': self.pct_90,  # Legacy field name
            '95Pct': self.pct_95  # Legacy field name
        }

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> 'TransactionMetrics':
        """
        Create TransactionMetrics from legacy dictionary format.
        Handles both old and new field names for seamless migration.
        """
        return cls(
            request_name=data.get('request_name', 'Unknown'),
            Total=int(data.get('Total', 0)),
            KO=int(data.get('KO', 0)),
            OK=int(data.get('OK', 0)),
            min=float(data.get('min', 0.0)),
            max=float(data.get('max', 0.0)),
            average=float(data.get('average', 0.0)),
            median=float(data.get('median', 0.0)),
            Error_pct=float(data.get('Error%', 0.0)),  # Legacy field name
            pct_90=float(data.get('90Pct', 0.0)),  # Legacy field name
            pct_95=float(data.get('95Pct', 0.0))  # Legacy field name
        )

    @property
    def is_successful(self) -> bool:
        """Helper property to check if transaction has acceptable error rate."""
        return self.Error_pct < 5.0  # Configurable threshold

    @property
    def performance_grade(self) -> str:
        """
        Calculate performance grade for Excel conditional formatting.
        This enables color-coding in Excel reports.
        """
        if self.Error_pct > 10:
            return "CRITICAL"
        elif self.Error_pct > 5:
            return "WARNING"
        elif self.average > 2000:  # 2 seconds
            return "SLOW"
        elif self.average > 1000:  # 1 second
            return "MODERATE"
        else:
            return "EXCELLENT"


@dataclass(frozen=True)
class ReportSummary:
    """
    Immutable summary metrics for the entire performance test.
    Maintains legacy field structure for Excel compatibility.
    """

    # Core summary fields (legacy compatible)
    max_user_count: int
    ramp_up_period: float
    error_rate: float
    date_start: str
    date_end: str
    throughput: float
    duration: float
    think_time: str

    # Optional enhanced fields
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
    carrier_report_url: Optional[str] = None
    thresholds: Optional[List[Any]] = None
    report_type: str = "GATLING"  # or "JMETER"
    generated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate the complete report structure."""
        if not self.transactions:
            raise ValueError("Report must contain at least one transaction")
        if 'Total' not in self.transactions:
            raise ValueError("Report must contain 'Total' transaction summary")
        logger.info(f"Validated PerformanceReport with {len(self.transactions)} transactions")

    @property
    def excel_metadata(self) -> Dict[str, Any]:
        """Excel formatting metadata for conditional formatting."""
        return getattr(self, '_excel_metadata', {})

    @excel_metadata.setter
    def excel_metadata(self, value: Dict[str, Any]):
        """Set Excel formatting metadata."""
        object.__setattr__(self, '_excel_metadata', value)

    @property
    def comparison_metadata(self) -> Dict[str, Any]:
        """Comparison metadata for baseline reports."""
        return getattr(self, '_comparison_metadata', {})

    @comparison_metadata.setter
    def comparison_metadata(self, value: Dict[str, Any]):
        """Set comparison metadata."""
        object.__setattr__(self, '_comparison_metadata', value)

    @classmethod
    def create_from_raw_data(
            cls,
            transactions: Dict[str, TransactionMetrics],
            raw_summary_data: Dict[str, Any],
            total_transaction_key: str = "Total"
    ) -> "PerformanceReport":
        """
        Factory Method (SRP): Explicitly creates a complete PerformanceReport
        from raw data sources, handling the complex logic of summary generation.
        This separates data creation from data representation.
        """
        total_metrics = transactions.get(total_transaction_key)
        if not total_metrics:
            raise ValueError(f"Required '{total_transaction_key}' transaction not found in transaction data.")

        duration_seconds = raw_summary_data.get("duration_seconds", 0)
        if duration_seconds <= 0:
            raise ValueError("Test duration must be a positive number of seconds.")

        summary = ReportSummary(
            max_user_count=int(raw_summary_data.get("max_user_count", 0)),
            ramp_up_period=float(raw_summary_data.get("ramp_up_period", 0.0)),
            error_rate=float(raw_summary_data.get("error_rate", 0.0)),
            date_start=str(raw_summary_data.get("date_start", "")),
            date_end=str(raw_summary_data.get("date_end", "")),
            throughput=float(raw_summary_data.get("throughput", 0.0)),
            duration=float(raw_summary_data.get("duration", duration_seconds)),
            think_time=str(raw_summary_data.get("think_time", 0.0)),
            baseline_throughput=raw_summary_data.get("baseline_throughput"),
            throughput_improvement=raw_summary_data.get("throughput_improvement"),
        )

        return cls(
            summary=summary,
            transactions=transactions,
            build_status=PerformanceStatus(raw_summary_data.get("build_status", "WARNING")),
            carrier_report_url=raw_summary_data.get("carrier_report_url"),
            analysis_summary=raw_summary_data.get("analysis_summary"),
            thresholds=raw_summary_data.get("thresholds"),
            report_type=raw_summary_data.get("report_type", "GATLING"),
            generated_at=raw_summary_data.get("generated_at", datetime.now()),
        )

    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert entire report to legacy dictionary format.
        This ensures existing Excel generation code continues to work unchanged.
        """
        legacy_dict = self.summary.to_legacy_dict()
        legacy_dict['requests'] = {
            name: metrics.to_legacy_dict()
            for name, metrics in self.transactions.items()
        }
        return legacy_dict

    @classmethod
    def from_legacy_format(cls, data: Dict[str, Any]) -> 'PerformanceReport':
        """Create PerformanceReport from legacy dictionary format."""
        summary_data = {k: v for k, v in data.items() if k != 'requests'}
        summary = ReportSummary.from_legacy_dict(summary_data)
        transactions = {}
        if 'requests' in data:
            transactions = {
                name: TransactionMetrics.from_legacy_dict(metrics)
                for name, metrics in data['requests'].items()
            }
        return cls(summary=summary, transactions=transactions)

    def get_failed_transactions(self) -> List[TransactionMetrics]:
        """Get list of transactions with high error rates for Excel highlighting."""
        return [
            metrics for metrics in self.transactions.values()
            if not metrics.is_successful
        ]

    def get_slow_transactions(self, threshold_ms: float = 2000) -> List[TransactionMetrics]:
        """Get list of slow transactions for Excel conditional formatting."""
        return [
            metrics for metrics in self.transactions.values()
            if metrics.average > threshold_ms
        ]

    def calculate_throughput_improvement(self, baseline_report: 'PerformanceReport') -> float:
        """
        Calculate throughput improvement vs baseline for Excel comparison reports.
        Returns percentage improvement (positive = improvement, negative = degradation).
        """
        if baseline_report.summary.throughput == 0:
            return 0.0
        current_throughput = self.summary.throughput
        baseline_throughput = baseline_report.summary.throughput
        improvement = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
        return round(improvement, 2)


# ================== FACTORY FUNCTIONS ==================

def create_empty_transaction_metrics(name: str) -> TransactionMetrics:
    """
    Factory function to create empty transaction metrics.
    Used when no data is available for a transaction.
    """
    return TransactionMetrics(
        request_name=name,
        Total=0,
        KO=0,
        OK=0,
        min=0.0,
        max=0.0,
        average=0.0,
        median=0.0,
        Error_pct=0.0,
        pct_90=0.0,
        pct_95=0.0
    )


def create_transaction_metrics_from_stats(name: str, stats_dict: Dict[str, Any]) -> TransactionMetrics:
    """
    Factory function to create TransactionMetrics from statistics dictionary.
    Handles both legacy and modern field names.
    """
    try:
        return TransactionMetrics.from_legacy_dict({
            'request_name': name,
            **stats_dict
        })
    except Exception as e:
        logger.error(f"Failed to create TransactionMetrics for {name}: {e}")
        return create_empty_transaction_metrics(name)


# ================== VALIDATION UTILITIES ==================

def validate_performance_report(report: PerformanceReport) -> List[str]:
    """
    Comprehensive validation of performance report.
    Returns list of validation errors for debugging.
    """
    errors = []

    # Validate summary
    if report.summary.throughput == 0:
        errors.append("Zero throughput indicates no successful requests")

    if report.summary.error_rate > 20:
        errors.append(f"High error rate: {report.summary.error_rate}%")

    # Validate transactions
    if not report.transactions:
        errors.append("No transactions found in report")

    for name, metrics in report.transactions.items():
        if metrics.Total == 0:
            errors.append(f"Transaction '{name}' has no data")

        if metrics.Error_pct > 50:
            errors.append(f"Transaction '{name}' has high error rate: {metrics.Error_pct}%")

    return errors


@dataclass
class TransactionAnalysis:
    """Value object for transaction analysis results."""
    status: str
    severity: str
    notes: str
    issues: List[str]


# ================== ANALYSIS RESULTS ==================

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


@dataclass(frozen=True)
class ComparisonResult:
    """Results of performance comparison analysis."""
    transaction_name: str
    metric_name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    percent_change: float
    direction: PerformanceDirection
    is_significant: bool = field(init=False)
    threshold_exceeded: bool = False

    def __post_init__(self):
        # Calculate significance based on percent change
        object.__setattr__(self, 'is_significant', abs(self.percent_change) >= 5.0)

    @property
    def change_display(self) -> str:
        """Formatted change percentage for display."""
        if self.percent_change == 0:
            return "No Change"
        sign = "+" if self.percent_change > 0 else ""
        return f"{sign}{self.percent_change:.1f}%"


# ================== ETL INTEGRATION ==================

@dataclass
class ETLContext:
    """Context object for ETL pipeline operations."""
    report_id: str
    api_wrapper: Any
    extraction_params: Dict[str, Any] = field(default_factory=dict)
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    loading_params: Dict[str, Any] = field(default_factory=dict)


class ThresholdCondition(str, Enum):
    """Enumeration for threshold comparison operators."""
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"


@dataclass
class ThresholdConfig:
    """Verify this matches what we're using in the transformer."""
    target: str = Field(description="The metric this threshold applies to")
    threshold_value: float = Field(description="The numeric threshold value")
    condition: ThresholdCondition = Field(
        default=ThresholdCondition.LESS_THAN,
        description="The condition for success"
    )


# ================== EXCEL REPORT HELPERS ==================

@dataclass
class ExcelFormattingConfig:
    """
    Configuration for Excel conditional formatting.
    Used by Excel report generation to apply color-coding and highlighting.
    """

    # Performance thresholds
    error_rate_warning: float = 5.0
    error_rate_float = 10.0
    response_time_warning: float = 1000.0
    response_time_float = 2000.0
    error_rate_critical: float = 10.0
    response_time_critical: float = 2000.0
    throughput_warning_threshold: float = 10.0

    RED_COLOR = 'F7A9A9'
    GREEN_COLOR = 'AFF2C9'
    YELLOW_COLOR = 'F7F7A9'
    RED_COLOR_FONT = '00F90808'
    GREEN_COLOR_FONT = '002BBD4D'

    # Color schemes for Excel
    excellent_color: str = f"#{GREEN_COLOR_FONT}"  # Green
    good_color: str = f"#{GREEN_COLOR}"  # Light Green
    warning_color: str = f"#{YELLOW_COLOR}"  # Orange
    critical_color: str = f"#{RED_COLOR}"  # Red
    critical_color_font: str = f"#{RED_COLOR_FONT}"  # Red

    def get_color_for_error_rate(self, error_rate: float) -> str:
        """Get Excel color code based on error rate."""
        if error_rate >= self.error_rate_critical:
            return self.critical_color
        elif error_rate >= self.error_rate_warning:
            return self.warning_color
        else:
            return self.excellent_color

    def get_color_for_response_time(self, response_time: float) -> str:
        """Get Excel color code based on response time."""
        if response_time >= self.response_time_critical:
            return self.critical_color
        elif response_time >= self.response_time_warning:
            return self.warning_color
        else:
            return self.excellent_color


# ================== ANALYSIS RESULTS ==================

@dataclass
class AnalysisResult:
    """Results of performance analysis."""
    status: PerformanceStatus
    justification: str
    failed_metrics: List[str]
    recommendations: List[Recommendation]
    summary_stats: Dict[str, Any]


@dataclass(frozen=True)
class ComparisonResult:
    """Results of performance comparison analysis."""
    transaction_name: str
    metric_name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    percent_change: float
    direction: PerformanceDirection
    is_significant: bool = field(init=False)
    threshold_exceeded: bool = False

    def __post_init__(self):
        # Calculate significance based on percent change
        object.__setattr__(self, 'is_significant', abs(self.percent_change) >= 5.0)

    @property
    def change_display(self) -> str:
        """Formatted change percentage for display."""
        if self.percent_change == 0:
            return "No Change"
        sign = "+" if self.percent_change > 0 else ""
        return f"{sign}{self.percent_change:.1f}%"


# ================== ETL INTEGRATION ==================

@dataclass
class ETLContext:
    """Context object for ETL pipeline operations."""
    report_id: str
    api_wrapper: Any
    extraction_params: Dict[str, Any] = field(default_factory=dict)
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    loading_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    """Verify this matches what we're using in the transformer."""
    target: str = Field(description="The metric this threshold applies to")
    threshold_value: float = Field(description="The numeric threshold value")
    condition: ThresholdCondition = Field(
        default=ThresholdCondition.LESS_THAN,
        description="The condition for success"
    )


@dataclass()
class TicketPayload:
    """
    Defines the canonical structure for creating a ticket in an external system.
    This model belongs in data_models.py as it represents a core data entity.
    """
    title: str
    board_id: str
    description: str
    external_link: str
    engagement: str
    assignee: str
    start_date: date
    end_date: date
    severity: str = "Medium"
    type: str = "Performance Degradation"
    tags: List[str] = field(default_factory=list)


class ReportType(Enum):
    """Report type enumeration for type safety"""
    BASELINE = "baseline"
    COMPARISON = "comparison"
    UI_PERFORMANCE = "ui_performance"
    BACKEND = "backend"
    MOBILE = "mobile"
