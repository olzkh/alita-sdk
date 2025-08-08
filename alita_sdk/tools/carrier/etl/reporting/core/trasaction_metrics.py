import logging
from typing import Dict, Any, List

from alita_sdk.tools.carrier.etl.reporting.core.data_models import TransactionMetrics, PerformanceReport

logger = logging.getLogger(__name__)


def create_transaction_metrics_from_stats(name: str, stats: Dict[str, Any]) -> TransactionMetrics:
    """
    DRY: Centralized factory method to create TransactionMetrics from parser statistics.
    Handles both Gatling and JMeter stat formats.
    """
    try:
        # Extract values with safe defaults
        total = stats.get('Total', 0)
        ok = stats.get('OK', 0)
        ko = stats.get('KO', 0)

        return TransactionMetrics(
            name=name,
            samples=total,  # This is the required 'samples' field
            ko=ko,
            avg=float(stats.get('avg', 0)),
            min=float(stats.get('min', 0)),
            max=float(stats.get('max', 0)),
            pct90=float(stats.get('90Pct', 0)),
            pct95=float(stats.get('95Pct', 0)),
            pct99=float(stats.get('99Pct', stats.get('95Pct', 0) * 1.05)),  # Estimate if not available
            throughput=float(stats.get('throughput', 0)),
            received_kb_per_sec=float(stats.get('received_kb_per_sec', 0)),
            sent_kb_per_sec=float(stats.get('sent_kb_per_sec', 0)),
            total=total
        )
    except Exception as e:
        logger.error(f"Failed to create TransactionMetrics for {name}: {e}")
        logger.error(f"Stats provided: {stats}")
        raise


def create_empty_transaction_metrics(name: str) -> TransactionMetrics:
    """
    DRY: Creates an empty TransactionMetrics object with zero values.
    Used when no data is available for a transaction.
    """
    return TransactionMetrics(
        name=name,
        samples=0,
        ko=0,
        avg=0.0,
        min=0.0,
        max=0.0,
        pct90=0.0,
        pct95=0.0,
        pct99=0.0,
        throughput=0.0,
        received_kb_per_sec=0.0,
        sent_kb_per_sec=0.0,
        total=0
    )


def validate_performance_report(report: PerformanceReport) -> List[str]:
    """
    DRY: Centralized validation for PerformanceReport objects.
    Returns a list of validation errors (empty if valid).
    """
    errors = []

    # Validate summary
    if not report.summary:
        errors.append("Report missing summary")
    else:
        # Validate summary fields
        if report.summary.throughput < 0:
            errors.append(f"Invalid throughput: {report.summary.throughput}")
        if report.summary.error_rate < 0 or report.summary.error_rate > 100:
            errors.append(f"Invalid error rate: {report.summary.error_rate}")

    # Validate transactions
    if not report.transactions:
        errors.append("Report has no transactions")
    else:
        # Check for required 'Total' transaction
        if 'Total' not in report.transactions:
            errors.append("Report missing 'Total' transaction")

        # Validate each transaction
        for name, metrics in report.transactions.items():
            if not isinstance(metrics, TransactionMetrics):
                errors.append(f"Transaction '{name}' is not a TransactionMetrics object")
                continue

            # Validate metrics consistency
            if metrics.samples < 0:
                errors.append(f"Transaction '{name}' has negative samples: {metrics.samples}")
            if metrics.ko > metrics.samples:
                errors.append(f"Transaction '{name}' has more errors than samples: {metrics.ko} > {metrics.samples}")
            if metrics.min > metrics.max:
                errors.append(f"Transaction '{name}' has min > max: {metrics.min} > {metrics.max}")
            if metrics.avg < metrics.min or metrics.avg > metrics.max:
                errors.append(f"Transaction '{name}' has avg outside min/max range")

    return errors
