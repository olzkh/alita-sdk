"""
Performance Analysis Engine
Author: Karen Florykian
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from .data_models import (
    TransactionMetrics, AnalysisResult, ComparisonResult, Recommendation,
    PerformanceStatus, PerformanceDirection, ThresholdConfig, PerformanceReport
)
from .threshold_manager import ThresholdManager
from alita_sdk.tools.carrier.reporting.formatting.formatting import TextFormatter
from alita_sdk.tools.carrier.utils.utils import AnalysisRulesConfig

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Production-ready performance analyzer with proper attribute mapping and enhanced logging.
    Fixed all attribute access issues and added comprehensive error handling.
    """

    def __init__(self, rules_config: AnalysisRulesConfig = None, formatter: TextFormatter = None):
        """Initialize analyzer with proper dependency injection and logging."""
        self.rules = rules_config or AnalysisRulesConfig()
        self.formatter = formatter or TextFormatter()
        self.threshold_manager = ThresholdManager()

        logger.info("PerformanceAnalyzer initialized with production configuration")
        logger.debug(f"Rules config: {type(self.rules).__name__}")
        logger.debug(f"Formatter: {type(self.formatter).__name__}")

    def _map_transaction_attributes(self, metric: TransactionMetrics) -> Dict[str, float]:
        """
        Map TransactionMetrics attributes to expected analyzer format.
        CRITICAL: This handles the attribute mismatch causing test failures.
        """
        try:
            # Calculate throughput if not available (requests per second)
            throughput = 0.0
            if hasattr(metric, 'Total') and metric.Total > 0:
                # Assuming test duration - this should come from context or be calculated
                # For now, using a default calculation
                throughput = metric.Total / 60.0  # Assume 60 second test duration

            mapped_attrs = {
                'transaction_name': metric.request_name,
                'average_response_time': metric.average,
                'p95_response_time': metric.pct_95,
                'error_rate': metric.Error_pct,
                'throughput': throughput,
                'sample_count': metric.Total,
                'min_response_time': metric.min,
                'max_response_time': metric.max,
                'median_response_time': metric.median,
                'p90_response_time': metric.pct_90
            }

            logger.debug(f"Mapped attributes for {metric.request_name}: {mapped_attrs}")
            return mapped_attrs

        except AttributeError as e:
            logger.error(f"Attribute mapping failed for metric {getattr(metric, 'request_name', 'unknown')}: {e}")
            raise ValueError(f"Invalid TransactionMetrics object: {e}")

    def _analyze_response_times(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze response times with proper attribute mapping and enhanced logging.
        Uses correct attribute names from TransactionMetrics.
        """
        failed_transactions = []
        justifications = []

        logger.info(f"Analyzing response times for {len(metrics)} transactions")
        logger.debug(f"Response time threshold: {thresholds.get('response_time', 'not set')}")

        try:
            # Support both list and dict input for metrics
            if isinstance(metrics, dict):
                metrics_iter = metrics.values()
            else:
                metrics_iter = metrics

            rt_threshold = thresholds['response_time']
            p95_threshold = thresholds.get('p95_response_time')

            # Check average response times with proper attribute mapping
            avg_failures = []
            for metric in metrics_iter:
                mapped_attrs = self._map_transaction_attributes(metric)
                if mapped_attrs['average_response_time'] > rt_threshold:
                    avg_failures.append(mapped_attrs['transaction_name'])
                    logger.warning(
                        f"Transaction {mapped_attrs['transaction_name']} failed RT threshold: "
                        f"{mapped_attrs['average_response_time']:.2f}ms > {rt_threshold}ms"
                    )

            # Re-iterate for P95 check if needed
            if isinstance(metrics, dict):
                metrics_iter = metrics.values()
            else:
                metrics_iter = metrics

            # Check P95 response times if threshold is set
            p95_failures = []
            if p95_threshold:
                for metric in metrics_iter:
                    mapped_attrs = self._map_transaction_attributes(metric)
                    if mapped_attrs['p95_response_time'] > p95_threshold:
                        p95_failures.append(mapped_attrs['transaction_name'])
                        logger.warning(
                            f"Transaction {mapped_attrs['transaction_name']} failed P95 threshold: "
                            f"{mapped_attrs['p95_response_time']:.2f}ms > {p95_threshold}ms"
                        )

            # Build results
            if avg_failures:
                failed_transactions.extend(avg_failures)
                justifications.append(
                    f"Average response time exceeded {rt_threshold}ms for "
                    f"{len(avg_failures)} transaction(s): {', '.join(avg_failures[:3])}"
                    f"{'...' if len(avg_failures) > 3 else ''}"
                )

            if p95_failures:
                failed_transactions.extend(p95_failures)
                justifications.append(
                    f"P95 response time exceeded {p95_threshold}ms for "
                    f"{len(p95_failures)} transaction(s): {', '.join(p95_failures[:3])}"
                    f"{'...' if len(p95_failures) > 3 else ''}"
                )

            logger.info(f"Response time analysis complete: {len(failed_transactions)} failures")
            return failed_transactions, justifications

        except Exception as e:
            logger.error(f"Response time analysis failed: {e}")
            raise

    def _analyze_error_pcts(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze error rates with proper attribute mapping and enhanced logging.
        Uses Error_pct attribute from TransactionMetrics.
        """
        logger.info(f"Analyzing error rates for {len(metrics)} transactions")

        try:
            er_threshold = thresholds['error_rate']
            logger.debug(f"Error rate threshold: {er_threshold}%")

            failed_transactions = []
            for metric in metrics:
                mapped_attrs = self._map_transaction_attributes(metric)
                if mapped_attrs['error_rate'] > er_threshold:
                    failed_transactions.append(mapped_attrs['transaction_name'])
                    logger.warning(
                        f"Transaction {mapped_attrs['transaction_name']} failed error rate threshold: "
                        f"{mapped_attrs['error_rate']:.2f}% > {er_threshold}%"
                    )

            if not failed_transactions:
                logger.info("All transactions passed error rate threshold")
                return [], []

            justification = (
                f"Error rate exceeded {er_threshold}% for "
                f"{len(failed_transactions)} transaction(s): {', '.join(failed_transactions[:3])}"
                f"{'...' if len(failed_transactions) > 3 else ''}"
            )

            logger.info(f"Error rate analysis complete: {len(failed_transactions)} failures")
            return failed_transactions, [justification]

        except Exception as e:
            logger.error(f"Error rate analysis failed: {e}")
            raise

    def _analyze_throughput(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze throughput with proper calculation and enhanced logging.
        Calculates throughput from Total requests.
        """
        logger.info(f"Analyzing throughput for {len(metrics)} transactions")

        try:
            tp_threshold = thresholds['throughput']
            logger.debug(f"Throughput threshold: {tp_threshold} req/s")

            failed_transactions = []
            for metric in metrics:
                mapped_attrs = self._map_transaction_attributes(metric)
                if mapped_attrs['throughput'] < tp_threshold:
                    failed_transactions.append(mapped_attrs['transaction_name'])
                    logger.warning(
                        f"Transaction {mapped_attrs['transaction_name']} failed throughput threshold: "
                        f"{mapped_attrs['throughput']:.2f} req/s < {tp_threshold} req/s"
                    )

            if not failed_transactions:
                logger.info("All transactions passed throughput threshold")
                return [], []

            justification = (
                f"Throughput below {tp_threshold} req/s for "
                f"{len(failed_transactions)} transaction(s): {', '.join(failed_transactions[:3])}"
                f"{'...' if len(failed_transactions) > 3 else ''}"
            )

            logger.info(f"Throughput analysis complete: {len(failed_transactions)} failures")
            return failed_transactions, [justification]

        except Exception as e:
            logger.error(f"Throughput analysis failed: {e}")
            raise

    def analyze(
            self,
            report: PerformanceReport,
            thresholds: ThresholdConfig,
            user_args: Dict[str, Any] = None
    ) -> AnalysisResult:
        """
        Main analysis method with comprehensive logging and error handling.
        Accepts a full PerformanceReport object and extracts metrics internally.
        """
        user_args = user_args or {}

        logger.info("=" * 50)
        # Use the transactions from the report object. We need the dictionary's values.
        metrics = list(report.transactions.values())
        logger.info(f"Starting performance analysis for {len(metrics)} transactions")
        logger.info(f"User args: {user_args}")
        logger.info("=" * 50)

        try:
            # Validate inputs
            if not metrics:
                logger.warning("No metrics provided for analysis")
                return AnalysisResult(
                    status=PerformanceStatus.PASSED,
                    justification="No metrics to analyze",
                    failed_metrics=[],
                    recommendations=[],
                    summary_stats={}
                )

            # Get effective threshold values
            threshold_values = self.threshold_manager.resolve_thresholds(thresholds, user_args)
            logger.info(f"Resolved thresholds: {threshold_values}")

            # Perform analysis checks with enhanced logging
            logger.info("Performing response time analysis...")
            failed_rt, rt_justifications = self._analyze_response_times(metrics, threshold_values)

            logger.info("Performing error rate analysis...")
            failed_er, er_justifications = self._analyze_error_pcts(metrics, threshold_values)

            logger.info("Performing throughput analysis...")
            failed_tp, tp_justifications = self._analyze_throughput(metrics, threshold_values)

            # Aggregate results
            all_failures = list(set(failed_rt + failed_er + failed_tp))  # Remove duplicates
            status = PerformanceStatus.FAILED if all_failures else PerformanceStatus.PASSED

            logger.info(f"Analysis status: {status.value}")
            logger.info(f"Failed transactions: {len(all_failures)}")
            if all_failures:
                logger.info(f"Failed transaction names: {', '.join(all_failures)}")

            # Build justification
            all_justifications = rt_justifications + er_justifications + tp_justifications
            justification = self._build_justification(all_justifications)

            # Generate recommendations
            logger.info("Generating performance recommendations...")
            recommendations = self._generate_recommendations(metrics, threshold_values, all_failures)
            logger.info(f"Generated {len(recommendations)} recommendations")

            # Calculate summary statistics
            logger.info("Calculating summary statistics...")
            summary_stats = self._calculate_summary_stats(metrics, threshold_values)

            result = AnalysisResult(
                status=status,
                justification=justification,
                failed_metrics=all_failures,
                recommendations=recommendations,
                summary_stats=summary_stats
            )

            logger.info("=" * 50)
            logger.info(f"Analysis complete: {status.value}")
            logger.info(f"Total recommendations: {len(recommendations)}")
            logger.info(f"Failed metrics: {len(all_failures)}")
            logger.info("=" * 50)

            return result

        except Exception as e:
            logger.error(f"Analysis failed with error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

    def _build_justification(self, justifications: List[str]) -> str:
        """Build comprehensive justification with proper formatting."""
        if not justifications:
            return "All performance metrics meet defined thresholds"

        # Join with proper separators and formatting
        formatted_justifications = []
        for i, justification in enumerate(justifications, 1):
            formatted_justifications.append(f"{i}. {justification}")

        return " | ".join(formatted_justifications)

    def _generate_recommendations(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float],
            failed_metrics: List[str]
    ) -> List[Recommendation]:
        """
        Generate structured recommendations with proper attribute mapping.
        Uses mapped attributes for all calculations.
        """
        logger.info("Generating performance recommendations...")
        recommendations = []

        try:
            # Map all metrics for easier processing
            mapped_metrics = []
            for metric in metrics:
                mapped_attrs = self._map_transaction_attributes(metric)
                mapped_metrics.append(mapped_attrs)

            # Critical response time issues
            recommendations.extend(self._rec_critical_response_times(mapped_metrics, thresholds))

            # High error rate issues
            recommendations.extend(self._rec_high_error_rates(mapped_metrics, thresholds))

            # Throughput gaps
            recommendations.extend(self._rec_throughput_gaps(mapped_metrics, thresholds))

            # System reliability
            recommendations.extend(self._rec_system_reliability(mapped_metrics))

            # Performance imbalance
            recommendations.extend(self._rec_performance_imbalance(mapped_metrics))

            # Sort by priority and limit
            priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
            recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))

            final_recommendations = recommendations[:self.rules.MAX_RECOMMENDATIONS]
            logger.info(f"Generated {len(final_recommendations)} final recommendations")

            return final_recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []

    def _rec_critical_response_times(
            self,
            mapped_metrics: List[Dict[str, Any]],
            thresholds: Dict[str, float]
    ) -> List[Recommendation]:
        """Generate critical response time recommendations using mapped attributes."""
        rt_threshold = thresholds['response_time']
        critical_multiplier = self.rules.RT_CRITICAL_MULTIPLIER

        critical_transactions = [
            (m['transaction_name'], m['p95_response_time'])
            for m in mapped_metrics
            if m['p95_response_time'] > rt_threshold * critical_multiplier
        ]

        if not critical_transactions:
            return []

        worst = max(critical_transactions, key=lambda x: x[1])
        logger.warning(f"Critical response time detected: {worst[0]} at {worst[1]:.0f}ms")

        return [Recommendation(
            priority='Critical',
            action='Critical Response Time Investigation',
            details=(
                f'Found {len(critical_transactions)} transaction(s) with P95 response times '
                f'exceeding {critical_multiplier}x threshold ({rt_threshold * critical_multiplier:.0f}ms). '
                f'Worst performer: {self.formatter.format_transaction_name(worst[0])} '
                f'at {worst[1]:.0f}ms.'
            ),
            category='Response Time'
        )]

    def _rec_high_error_rates(
            self,
            mapped_metrics: List[Dict[str, Any]],
            thresholds: Dict[str, float]
    ) -> List[Recommendation]:
        """Generate error rate recommendations using mapped attributes."""
        er_threshold = thresholds['error_rate']
        high_multiplier = self.rules.ER_HIGH_MULTIPLIER

        high_error_transactions = [
            (m['transaction_name'], m['error_rate'])
            for m in mapped_metrics
            if m['error_rate'] > er_threshold
        ]

        if not high_error_transactions:
            return []

        worst = max(high_error_transactions, key=lambda x: x[1])
        avg_error_rate = sum(t[1] for t in high_error_transactions) / len(high_error_transactions)
        is_critical = any(t[1] > er_threshold * high_multiplier for t in high_error_transactions)

        logger.warning(f"High error rate detected: {worst[0]} at {worst[1]:.1f}%")

        return [Recommendation(
            priority='Critical' if is_critical else 'High',
            action='Error Rate Analysis',
            details=(
                f'Detected {len(high_error_transactions)} transaction(s) exceeding '
                f'{er_threshold}% error threshold. Highest error rate: '
                f'{self.formatter.format_transaction_name(worst[0])} with {worst[1]:.1f}% failures. '
                f'Average error rate across affected transactions: {avg_error_rate:.1f}%.'
            ),
            category='Error Rate'
        )]

    def _rec_throughput_gaps(
            self,
            mapped_metrics: List[Dict[str, Any]],
            thresholds: Dict[str, float]
    ) -> List[Recommendation]:
        """Generate throughput recommendations using mapped attributes."""
        tp_threshold = thresholds['throughput']

        low_throughput_transactions = [
            (m['transaction_name'], m['throughput'])
            for m in mapped_metrics
            if m['throughput'] < tp_threshold
        ]

        if not low_throughput_transactions:
            return []

        # Calculate overall throughput deficit
        total_actual = sum(m['throughput'] for m in mapped_metrics)
        total_expected = tp_threshold * len(mapped_metrics)
        deficit_pct = ((total_expected - total_actual) / total_expected) * 100 if total_expected > 0 else 0

        is_critical = deficit_pct > self.rules.TP_HIGH_SEVERITY_DEFICIT_PCT

        logger.warning(f"Throughput gap detected: {deficit_pct:.1f}% below target")

        return [Recommendation(
            priority='Critical' if is_critical else 'High',
            action='Throughput Performance Gap',
            details=(
                f'System throughput is {deficit_pct:.1f}% below target. '
                f'{len(low_throughput_transactions)} transaction(s) not meeting '
                f'{tp_threshold} req/s threshold.'
            ),
            category='Throughput'
        )]

    def _rec_system_reliability(self, mapped_metrics: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate system reliability recommendations using mapped attributes."""
        if not mapped_metrics:
            return []

        # Calculate overall success rate
        total_requests = sum(m['sample_count'] for m in mapped_metrics if m['sample_count'] > 0)
        if total_requests == 0:
            return []

        # Calculate weighted error rate
        total_errors = sum(
            (m['error_rate'] / 100) * m['sample_count']
            for m in mapped_metrics if m['sample_count'] > 0
        )
        overall_error_rate = (total_errors / total_requests) * 100
        success_rate = 100 - overall_error_rate

        if success_rate >= self.rules.RELIABILITY_STANDARD_SUCCESS_RATE:
            return []

        is_critical = success_rate < self.rules.RELIABILITY_HIGH_PRIORITY_THRESHOLD

        logger.warning(f"System reliability issue: {success_rate:.1f}% success rate")

        return [Recommendation(
            priority='Critical' if is_critical else 'High',
            action='System Reliability Below Standard',
            details=(
                f'Overall success rate is {success_rate:.1f}%. Industry standard '
                f'typically requires {self.rules.RELIABILITY_STANDARD_SUCCESS_RATE}%+. '
                f'Total error rate: {overall_error_rate:.2f}% across {total_requests} requests.'
            ),
            category='Reliability'
        )]

    def _rec_performance_imbalance(self, mapped_metrics: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate performance imbalance recommendations using mapped attributes."""
        if len(mapped_metrics) < self.rules.IMBALANCE_MIN_TRANSACTIONS:
            return []

        # Sort by average response time
        sorted_metrics = sorted(mapped_metrics, key=lambda x: x['average_response_time'])
        fastest = sorted_metrics[0]
        slowest = sorted_metrics[-1]

        if fastest['average_response_time'] > 0:
            imbalance_ratio = slowest['average_response_time'] / fastest['average_response_time']

            if imbalance_ratio > self.rules.IMBALANCE_FACTOR:
                logger.warning(f"Performance imbalance detected: {imbalance_ratio:.1f}x difference")

                return [Recommendation(
                    priority='Medium',
                    action='Performance Imbalance Detected',
                    details=(
                        f'The slowest transaction ({self.formatter.format_transaction_name(slowest["transaction_name"])} '
                        f'@ {slowest["average_response_time"]:.0f}ms) is {imbalance_ratio:.1f}x slower than '
                        f'the fastest ({self.formatter.format_transaction_name(fastest["transaction_name"])} '
                        f'@ {fastest["average_response_time"]:.0f}ms), indicating uneven performance distribution.'
                    ),
                    category='Performance Balance'
                )]

        return []

    def _calculate_summary_stats(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate summary statistics using mapped attributes."""
        if not metrics:
            return {}

        try:
            # Map all metrics for calculations
            mapped_metrics = [self._map_transaction_attributes(m) for m in metrics]

            summary_stats = {
                "total_transactions": len(mapped_metrics),
                "avg_response_time": sum(m['average_response_time'] for m in mapped_metrics) / len(mapped_metrics),
                "max_response_time": max(m['average_response_time'] for m in mapped_metrics),
                "min_response_time": min(m['average_response_time'] for m in mapped_metrics),
                "avg_error_rate": sum(m['error_rate'] for m in mapped_metrics) / len(mapped_metrics),
                "total_throughput": sum(m['throughput'] for m in mapped_metrics),
                "total_requests": sum(m['sample_count'] for m in mapped_metrics),
                "thresholds_used": thresholds,
                "transactions_over_rt_threshold": len([
                    m for m in mapped_metrics
                    if m['average_response_time'] > thresholds['response_time']
                ]),
                "transactions_over_er_threshold": len([
                    m for m in mapped_metrics
                    if m['error_rate'] > thresholds['error_rate']
                ]),
                "transactions_under_tp_threshold": len([
                    m for m in mapped_metrics
                    if m['throughput'] < thresholds['throughput']
                ])
            }

            logger.info(f"Summary stats calculated: {summary_stats['total_transactions']} transactions")
            return summary_stats

        except Exception as e:
            logger.error(f"Summary stats calculation failed: {e}")
            return {"error": f"Failed to calculate summary stats: {e}"}


# Enhanced ComparisonAnalyzer with proper attribute mapping
class ComparisonAnalyzer:
    """
    Production-ready comparison analyzer with proper attribute mapping.
    All attribute access issues resolved.
    """

    def __init__(self, formatter: TextFormatter = None):
        self.formatter = formatter or TextFormatter()
        self.threshold_manager = ThresholdManager()
        logger.info("ComparisonAnalyzer initialized with production configuration")

    def _map_transaction_attributes(self, metric: TransactionMetrics) -> Dict[str, float]:
        """Map TransactionMetrics attributes for comparison analysis."""
        try:
            # Calculate throughput if not available
            throughput = 0.0
            if hasattr(metric, 'Total') and metric.Total > 0:
                throughput = metric.Total / 60.0  # Assume 60 second duration

            return {
                'transaction_name': metric.request_name,
                'average_response_time': metric.average,
                'p95_response_time': metric.pct_95,
                'error_rate': metric.Error_pct,
                'throughput': throughput,
                'sample_count': metric.Total
            }
        except AttributeError as e:
            logger.error(f"Attribute mapping failed: {e}")
            raise ValueError(f"Invalid TransactionMetrics object: {e}")

    def compare_metrics(
            self,
            baseline_metrics: List[TransactionMetrics],
            current_metrics: List[TransactionMetrics],
            thresholds: ThresholdConfig = None
    ) -> List[ComparisonResult]:
        """
        Compare metrics with proper attribute mapping and enhanced logging.
        Uses mapped attributes for all comparisons.
        """
        logger.info(f"Comparing {len(baseline_metrics)} baseline vs {len(current_metrics)} current metrics")

        try:
            results = []

            # Map all metrics for easier processing
            baseline_mapped = {
                self._map_transaction_attributes(m)['transaction_name']:
                    self._map_transaction_attributes(m)
                for m in baseline_metrics
            }

            current_mapped = {
                self._map_transaction_attributes(m)['transaction_name']:
                    self._map_transaction_attributes(m)
                for m in current_metrics
            }

            # Get all unique transaction names
            all_transactions = set(baseline_mapped.keys()) | set(current_mapped.keys())
            logger.info(f"Comparing {len(all_transactions)} unique transactions")

            # Compare each transaction across all metrics
            for tx_name in sorted(all_transactions):
                baseline_tx = baseline_mapped.get(tx_name)
                current_tx = current_mapped.get(tx_name)

                results.extend(self._compare_transaction_metrics(tx_name, baseline_tx, current_tx))

            logger.info(f"Generated {len(results)} comparison results")
            return results

        except Exception as e:
            logger.error(f"Metrics comparison failed: {e}")
            raise

    def _compare_transaction_metrics(
            self,
            tx_name: str,
            baseline: Optional[Dict[str, Any]] = None,
            current: Optional[Dict[str, Any]] = None
    ) -> List[ComparisonResult]:
        """Compare all metrics for a single transaction using mapped attributes."""

        # Define metrics to compare
        metrics_to_compare = [
            ("average_response_time", "Average Response Time"),
            ("p95_response_time", "P95 Response Time"),
            ("error_rate", "Error Rate"),
            ("throughput", "Throughput")
        ]

        comparison_results = []

        for attr_name, display_name in metrics_to_compare:
            baseline_value = baseline.get(attr_name, 0.0) if baseline else 0.0
            current_value = current.get(attr_name, 0.0) if current else 0.0

            result = self._create_comparison_result(
                tx_name, display_name, baseline_value, current_value
            )
            comparison_results.append(result)

        return comparison_results

    def _create_comparison_result(
            self,
            tx_name: str,
            metric_name: str,
            baseline_value: float,
            current_value: float) -> ComparisonResult:
        """Create comparison result with proper change direction calculation."""

        # Calculate absolute and percentage changes
        absolute_change = current_value - baseline_value

        # Handle percentage change calculation
        if baseline_value == 0 and current_value > 0:
            percent_change = 100.0
            direction = PerformanceDirection.NEW
        elif baseline_value > 0 and current_value == 0:
            percent_change = -100.0
            direction = PerformanceDirection.MISSING
        elif baseline_value == 0 and current_value == 0:
            percent_change = 0.0
            direction = PerformanceDirection.STABLE
        else:
            percent_change = (absolute_change / baseline_value) * 100
            direction = self._determine_change_direction(metric_name, percent_change)

        # Determine if change is significant
        is_significant: bool = abs(percent_change) >= 5.0  # 5% threshold for significance

        return ComparisonResult(
            transaction_name=tx_name,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            absolute_change=absolute_change,
            percent_change=percent_change,
            direction=direction,
            is_significant=is_significant
        )

    def _determine_change_direction(self, metric_name: str, percent_change: float) -> PerformanceDirection:
        """Determine if change is improvement or degradation based on metric type."""

        # Define metrics where lower values are better
        lower_is_better = {"Error Rate", "Average Response Time", "P95 Response Time"}

        # Stability threshold
        if abs(percent_change) < 5.0:
            return PerformanceDirection.STABLE

        # For metrics where lower is better, negative change is improvement
        if metric_name in lower_is_better:
            return PerformanceDirection.IMPROVED if percent_change < 0 else PerformanceDirection.DEGRADED
        else:
            # For metrics like throughput, positive change is improvement
            return PerformanceDirection.IMPROVED if percent_change > 0 else PerformanceDirection.DEGRADED


class TrendAnalyzer:
    """
    NEW: Advanced trend analysis for multiple report comparisons.
    Clean implementation with no legacy dependencies.
    """

    def __init__(self):
        self.comparison_analyzer = ComparisonAnalyzer()
        logger.debug("TrendAnalyzer initialized")

    def analyze_trend(
            self,
            metrics_series: List[List[TransactionMetrics]],
            time_labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance trends across multiple time periods.

        Args:
            metrics_series: List of metric lists for each time period
            time_labels: Optional labels for each time period

        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Analyzing trends across {len(metrics_series)} time periods")

        if len(metrics_series) < 2:
            raise ValueError("Need at least 2 time periods for trend analysis")

        time_labels = time_labels or [f"Period {i + 1}" for i in range(len(metrics_series))]

        # Calculate period-to-period comparisons
        trend_results = []
        for i in range(1, len(metrics_series)):
            comparison = self.comparison_analyzer.compare_metrics(
                baseline_metrics=metrics_series[i - 1],
                current_metrics=metrics_series[i]
            )
            trend_results.append({
                "from_period": time_labels[i - 1],
                "to_period": time_labels[i],
                "comparisons": comparison
            })

        # Analyze overall trends
        overall_trends = self._calculate_overall_trends(trend_results)

        # Identify concerning patterns
        concerning_patterns = self._identify_concerning_patterns(trend_results)

        return {
            "period_comparisons": trend_results,
            "overall_trends": overall_trends,
            "concerning_patterns": concerning_patterns,
            "summary": self._generate_trend_summary(overall_trends, concerning_patterns)
        }

    def _calculate_overall_trends(self, trend_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall trend directions across all periods."""

        # Aggregate trends by transaction and metric
        trend_aggregates = {}

        for period_result in trend_results:
            for comparison in period_result["comparisons"]:
                key = f"{comparison.transaction_name}_{comparison.metric_name}"

                if key not in trend_aggregates:
                    trend_aggregates[key] = {
                        "transaction": comparison.transaction_name,
                        "metric": comparison.metric_name,
                        "changes": [],
                        "directions": []
                    }

                trend_aggregates[key]["changes"].append(comparison.percent_change)
                trend_aggregates[key]["directions"].append(comparison.direction)

        # Calculate trend statistics
        overall_trends = {}
        for key, data in trend_aggregates.items():
            avg_change = sum(data["changes"]) / len(data["changes"])
            consistent_direction = all(d == data["directions"][0] for d in data["directions"])

            overall_trends[key] = {
                "transaction": data["transaction"],
                "metric": data["metric"],
                "average_change_pct": avg_change,
                "total_periods": len(data["changes"]),
                "consistent_direction": consistent_direction,
                "trend_stability": self._calculate_trend_stability(data["changes"])
            }

        return overall_trends

    def _calculate_trend_stability(self, changes: List[float]) -> str:
        """Calculate stability of trend changes."""
        if not changes:
            return "Unknown"

        # Calculate coefficient of variation
        mean_change = sum(changes) / len(changes)
        if mean_change == 0:
            return "Stable"

        variance = sum((x - mean_change) ** 2 for x in changes) / len(changes)
        std_dev = variance ** 0.5
        coefficient_of_variation = abs(std_dev / mean_change)

        if coefficient_of_variation < 0.1:
            return "Very Stable"
        elif coefficient_of_variation < 0.3:
            return "Stable"
        elif coefficient_of_variation < 0.6:
            return "Moderately Volatile"
        else:
            return "Highly Volatile"

    def _identify_concerning_patterns(self, trend_results: List[Dict]) -> List[Dict[str, Any]]:
        """Identify concerning performance patterns across trends."""
        concerning_patterns = []

        # Pattern 1: Consistent degradation
        degradation_counts = {}
        for period_result in trend_results:
            for comparison in period_result["comparisons"]:
                if comparison.direction == PerformanceDirection.DEGRADED:
                    key = f"{comparison.transaction_name}_{comparison.metric_name}"
                    degradation_counts[key] = degradation_counts.get(key, 0) + 1

        # Identify consistently degrading metrics
        total_periods = len(trend_results)
        for key, count in degradation_counts.items():
            if count >= total_periods * 0.7:  # 70% of periods showing degradation
                tx_name, metric_name = key.rsplit('_', 1)
                concerning_patterns.append({
                    "pattern_type": "Consistent Degradation",
                    "transaction": tx_name,
                    "metric": metric_name.replace('_', ' ').title(),
                    "severity": "High",
                    "description": f"Shows degradation in {count}/{total_periods} periods",
                    "recommendation": f"Immediate investigation required for {tx_name} {metric_name}"
                })

        # Pattern 2: High volatility
        volatility_patterns = self._detect_volatility_patterns(trend_results)
        concerning_patterns.extend(volatility_patterns)

        # Pattern 3: Performance cliff (sudden major degradation)
        cliff_patterns = self._detect_performance_cliffs(trend_results)
        concerning_patterns.extend(cliff_patterns)

        return concerning_patterns

    def _detect_volatility_patterns(self, trend_results: List[Dict]) -> List[Dict[str, Any]]:
        """Detect highly volatile performance patterns."""
        volatility_patterns = []

        # Track variance for each metric
        metric_changes = {}
        for period_result in trend_results:
            for comparison in period_result["comparisons"]:
                key = f"{comparison.transaction_name}_{comparison.metric_name}"
                if key not in metric_changes:
                    metric_changes[key] = []
                metric_changes[key].append(comparison.percent_change)

        # Identify high volatility metrics
        for key, changes in metric_changes.items():
            if len(changes) >= 3:  # Need at least 3 data points
                variance = sum((x - sum(changes) / len(changes)) ** 2 for x in changes) / len(changes)
                std_dev = variance ** 0.5

                if std_dev > 20.0:  # High volatility threshold
                    tx_name, metric_name = key.rsplit('_', 1)
                    volatility_patterns.append({
                        "pattern_type": "High Volatility",
                        "transaction": tx_name,
                        "metric": metric_name.replace('_', ' ').title(),
                        "severity": "Medium",
                        "description": f"Standard deviation of {std_dev:.1f}% indicates unstable performance",
                        "recommendation": f"Investigate root cause of {tx_name} performance instability"
                    })

        return volatility_patterns

    def _detect_performance_cliffs(self, trend_results: List[Dict]) -> List[Dict[str, Any]]:
        """Detect sudden major performance degradations."""
        cliff_patterns = []

        for period_result in trend_results:
            for comparison in period_result["comparisons"]:
                # Look for sudden large degradations (>30% in single period)
                if (comparison.direction == PerformanceDirection.DEGRADED and
                        abs(comparison.percent_change) > 30.0):
                    cliff_patterns.append({
                        "pattern_type": "Performance Cliff",
                        "transaction": comparison.transaction_name,
                        "metric": comparison.metric_name,
                        "severity": "Critical",
                        "description": (
                            f"Sudden {abs(comparison.percent_change):.1f}% degradation "
                            f"from {period_result['from_period']} to {period_result['to_period']}"
                        ),
                        "recommendation": f"Critical: Immediate rollback consideration for {comparison.transaction_name}"
                    })

        return cliff_patterns

    def _generate_trend_summary(
            self,
            overall_trends: Dict[str, Any],
            concerning_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate executive summary of trend analysis."""

        total_metrics = len(overall_trends)
        improving_metrics = len([
            t for t in overall_trends.values()
            if t["average_change_pct"] < -5.0  # 5% improvement threshold
        ])
        degrading_metrics = len([
            t for t in overall_trends.values()
            if t["average_change_pct"] > 5.0  # 5% degradation threshold
        ])

        critical_issues = len([p for p in concerning_patterns if p["severity"] == "Critical"])
        high_issues = len([p for p in concerning_patterns if p["severity"] == "High"])

        # Overall health score (0-100)
        # Overall health score (0-100)
        health_score = max(0, 100 - (critical_issues * 30 + high_issues * 15 + degrading_metrics * 5))

        # Determine overall trend direction
        if improving_metrics > degrading_metrics * 1.5:
            overall_direction = "Improving"
        elif degrading_metrics > improving_metrics * 1.5:
            overall_direction = "Degrading"
        else:
            overall_direction = "Stable"

        return {
            "health_score": health_score,
            "overall_direction": overall_direction,
            "total_metrics_analyzed": total_metrics,
            "improving_metrics": improving_metrics,
            "degrading_metrics": degrading_metrics,
            "stable_metrics": total_metrics - improving_metrics - degrading_metrics,
            "critical_issues": critical_issues,
            "high_priority_issues": high_issues,
            "total_concerning_patterns": len(concerning_patterns),
            "key_insights": self._generate_key_insights(overall_trends, concerning_patterns)
        }

    def _generate_key_insights(
            self,
            overall_trends: Dict[str, Any],
            concerning_patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate key insights from trend analysis."""
        insights = []

        # Most improved metric
        if overall_trends:
            most_improved = min(overall_trends.values(), key=lambda x: x["average_change_pct"])
            if most_improved["average_change_pct"] < -10.0:
                insights.append(
                    f"🟢 Best improvement: {most_improved['transaction']} "
                    f"{most_improved['metric']} improved by {abs(most_improved['average_change_pct']):.1f}%"
                )

        # Most degraded metric
        if overall_trends:
            most_degraded = max(overall_trends.values(), key=lambda x: x["average_change_pct"])
            if most_degraded["average_change_pct"] > 10.0:
                insights.append(
                    f"🔴 Biggest concern: {most_degraded['transaction']} "
                    f"{most_degraded['metric']} degraded by {most_degraded['average_change_pct']:.1f}%"
                )

        # Critical patterns
        critical_patterns = [p for p in concerning_patterns if p["severity"] == "Critical"]
        if critical_patterns:
            insights.append(
                f"⚠️ {len(critical_patterns)} critical performance cliff(s) detected requiring immediate attention"
            )

        # Stability insights
        stable_metrics = [t for t in overall_trends.values() if t["trend_stability"] in ["Very Stable", "Stable"]]
        if len(stable_metrics) > len(overall_trends) * 0.8:
            insights.append("✅ Majority of metrics show stable performance trends")

        return insights[:5]
