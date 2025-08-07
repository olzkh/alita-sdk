import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import statistics

from .data_models import (
    TransactionMetrics, AnalysisResult, Recommendation,
    PerformanceStatus, ThresholdConfig, PerformanceReport, ReportSummary)
from .threshold_manager import ThresholdManager

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRules:
    """Configuration for analysis rules and thresholds."""
    RT_HIGH_SEVERITY_MULTIPLIER: float = 2.0
    ER_HIGH_SEVERITY_THRESHOLD: float = 10.0
    TP_HIGH_SEVERITY_DEFICIT_PCT: float = 20.0
    CV_IMBALANCE_THRESHOLD: float = 0.5
    MAX_RECOMMENDATIONS: int = 5


class PerformanceAnalyzer:
    """
    Analyzes performance metrics against thresholds and generates recommendations.
    This is the brain of individual report analysis.
    """

    def __init__(self, rules: AnalysisRules = None):
        self.rules = rules or AnalysisRules()
        self.threshold_manager = ThresholdManager()
        logger.info("PerformanceAnalyzer initialized")

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
        # Use the transactions from the report object
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

            # Perform analysis checks
            logger.info("Performing response time analysis...")
            failed_rt, rt_justifications = self._analyze_response_times(metrics, threshold_values)

            logger.info("Performing error rate analysis...")
            failed_er, er_justifications = self._analyze_error_pcts(metrics, threshold_values)

            logger.info("Performing throughput analysis...")
            # Pass the summary for throughput analysis
            failed_tp, tp_justifications = self._analyze_throughput(report.summary, threshold_values)

            # Aggregate results
            all_failures = list(set(failed_rt + failed_er + failed_tp))
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
            recommendations = self._generate_recommendations(
                metrics, threshold_values, all_failures, report.summary
            )
            logger.info(f"Generated {len(recommendations)} recommendations")

            # Calculate summary statistics
            logger.info("Calculating summary statistics...")
            summary_stats = self._calculate_summary_stats(metrics, threshold_values, report.summary)

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

    def _analyze_response_times(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze response times for each transaction.
        Returns tuple of (failed_transaction_names, justifications).
        """
        logger.info(f"Analyzing response times for {len(metrics)} transactions")
        failed_transactions = []
        justifications = []

        rt_threshold = thresholds['response_time']
        logger.debug(f"Response time threshold: {rt_threshold} ms")

        for metric in metrics:
            logger.debug(f"Checking {metric.name}: {metric.avg} ms vs threshold {rt_threshold} ms")
            if metric.avg > rt_threshold:
                failed_transactions.append(metric.name)
                justification = (
                    f"{metric.name}: avg response time ({metric.avg:.2f} ms) "
                    f"exceeds threshold of {rt_threshold} ms"
                )
                justifications.append(justification)
                logger.warning(justification)

        logger.info(f"Response time analysis complete: {len(failed_transactions)} failures")
        return failed_transactions, justifications

    def _analyze_error_pcts(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze error percentages for each transaction.
        Returns tuple of (failed_transaction_names, justifications).
        """
        logger.info(f"Analyzing error rates for {len(metrics)} transactions")
        failed_transactions = []
        justifications = []

        er_threshold = thresholds['error_rate']
        logger.debug(f"Error rate threshold: {er_threshold}%")

        for metric in metrics:
            logger.debug(f"Checking {metric.name}: {metric.error_rate}% vs threshold {er_threshold}%")
            if metric.error_rate > er_threshold:
                failed_transactions.append(metric.name)
                justification = (
                    f"{metric.name}: error rate ({metric.error_rate:.2f}%) "
                    f"exceeds threshold of {er_threshold}%"
                )
                justifications.append(justification)
                logger.warning(justification)

        logger.info(f"Error rate analysis complete: {len(failed_transactions)} failures")
        return failed_transactions, justifications

    def _analyze_throughput(
            self,
            summary: ReportSummary,
            thresholds: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze throughput at the summary level only.
        Throughput comparison should be done for overall test, not individual transactions.
        """
        logger.info("Analyzing overall throughput")

        try:
            tp_threshold = thresholds['throughput']
            logger.debug(f"Throughput threshold: {tp_threshold} req/s")

            # Get throughput from summary
            actual_throughput = summary.throughput
            logger.info(f"Summary throughput: {actual_throughput} req/s")

            if actual_throughput < tp_threshold:
                logger.warning(
                    f"Overall throughput failed threshold: "
                    f"{actual_throughput:.2f} req/s < {tp_threshold} req/s"
                )
                justification = (
                    f"Overall throughput ({actual_throughput:.2f} req/s) "
                    f"is below threshold of {tp_threshold} req/s"
                )
                return ["Overall Throughput"], [justification]

            logger.info("Overall throughput passed threshold")
            return [], []

        except Exception as e:
            logger.error(f"Throughput analysis failed: {e}")
            raise

    def _build_justification(self, justifications: List[str]) -> str:
        """Build a comprehensive justification message."""
        if not justifications:
            return "All performance metrics passed their thresholds"

        return "Performance test failed due to the following issues:\n" + "\n".join(
            f"- {j}" for j in justifications
        )

    def _generate_recommendations(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float],
            failed_metrics: List[str],
            summary: ReportSummary = None
    ) -> List[Recommendation]:
        """
        Generate structured recommendations with proper attribute mapping.
        Uses mapped attributes for all calculations.
        """
        logger.info("Generating performance recommendations...")
        recommendations = []

        recommendations.extend(self._rec_critical_response_times(metrics, thresholds))
        recommendations.extend(self._rec_high_error_rates(metrics, thresholds))

        # Pass summary for throughput recommendations
        if summary:
            recommendations.extend(self._rec_throughput_gaps(summary, thresholds))

        recommendations.extend(self._rec_system_reliability(metrics))
        recommendations.extend(self._rec_performance_imbalance(metrics))

        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))

        # Limit recommendations
        final_recommendations = recommendations[:self.rules.MAX_RECOMMENDATIONS]
        logger.info(f"Generated {len(final_recommendations)} final recommendations")

        return final_recommendations

    def _rec_critical_response_times(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> List[Recommendation]:
        """Generate recommendations for critical response time issues."""
        rt_threshold = thresholds['response_time']
        critical_threshold = rt_threshold * self.rules.RT_HIGH_SEVERITY_MULTIPLIER

        critical_transactions = [
            m for m in metrics if m.avg > critical_threshold
        ]

        if not critical_transactions:
            return []

        worst_transaction = max(critical_transactions, key=lambda m: m.avg)

        return [Recommendation(
            priority='Critical',
            action='Critical Response Time Issue',
            details=(
                f'Transaction "{worst_transaction.name}" has critically high response time '
                f'({worst_transaction.avg:.2f} ms), which is {worst_transaction.avg / rt_threshold:.1f}x '
                f'the threshold. This severely impacts user experience and requires immediate attention.'
            ),
            category='Response Time'
        )]

    def _rec_high_error_rates(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float]
    ) -> List[Recommendation]:
        """Generate recommendations for high error rates."""
        er_threshold = thresholds['error_rate']

        high_error_transactions = [
            m for m in metrics
            if m.error_rate > self.rules.ER_HIGH_SEVERITY_THRESHOLD
        ]

        if not high_error_transactions:
            return []

        worst_transaction = max(high_error_transactions, key=lambda m: m.error_rate)

        return [Recommendation(
            priority='Critical',
            action='High Error Rate Detected',
            details=(
                f'Transaction "{worst_transaction.name}" has a critical error rate of '
                f'{worst_transaction.error_rate:.2f}%, indicating severe reliability issues. '
                f'This requires immediate investigation of application logs and infrastructure.'
            ),
            category='Error Rate'
        )]

    def _rec_throughput_gaps(
            self,
            summary: ReportSummary,
            thresholds: Dict[str, float]
    ) -> List[Recommendation]:
        """Generate throughput recommendations based on summary throughput."""
        tp_threshold = thresholds['throughput']

        actual_throughput = summary.throughput

        if actual_throughput >= tp_threshold:
            return []

        deficit_pct = ((tp_threshold - actual_throughput) / tp_threshold) * 100
        is_critical = deficit_pct > self.rules.TP_HIGH_SEVERITY_DEFICIT_PCT

        logger.warning(f"Throughput gap detected: {deficit_pct:.1f}% below target")

        return [Recommendation(
            priority='Critical' if is_critical else 'High',
            action='Throughput Performance Gap',
            details=(
                f'System throughput ({actual_throughput:.2f} req/s) is {deficit_pct:.1f}% '
                f'below target of {tp_threshold} req/s. This may indicate capacity constraints '
                f'or performance bottlenecks affecting overall system performance.'
            ),
            category='Throughput'
        )]

    def _rec_system_reliability(
            self,
            metrics: List[TransactionMetrics]
    ) -> List[Recommendation]:
        """Generate recommendations for overall system reliability."""
        # Calculate overall error rate
        total_samples = sum(m.samples for m in metrics)
        total_errors = sum(m.samples * m.error_rate / 100 for m in metrics)
        overall_error_rate = (total_errors / total_samples * 100) if total_samples > 0 else 0

        if overall_error_rate <= 5.0:  # Could be made configurable
            return []

        error_transactions = [m for m in metrics if m.error_rate > 0]
        error_details = ', '.join(
            f'{m.name} ({m.error_rate:.1f}%)'
            for m in sorted(error_transactions, key=lambda x: x.error_rate, reverse=True)[:3]
        )

        return [Recommendation(
            priority='High',
            action='System Reliability Concerns',
            details=(
                f'Overall system error rate is {overall_error_rate:.2f}%, indicating reliability issues. '
                f'Top failing transactions: {error_details}. '
                f'Review application logs, database connections, and third-party service integrations.'
            ),
            category='Reliability'
        )]

    def _rec_performance_imbalance(
            self,
            metrics: List[TransactionMetrics]
    ) -> List[Recommendation]:
        """Generate recommendations for performance imbalances between transactions."""
        if len(metrics) < 3:
            return []

        response_times = [m.avg for m in metrics]
        mean_rt = statistics.mean(response_times)

        if mean_rt == 0:
            return []

        cv = statistics.stdev(response_times) / mean_rt

        if cv <= self.rules.CV_IMBALANCE_THRESHOLD:
            return []

        # Find outliers
        outliers = [
            m for m in metrics
            if abs(m.avg - mean_rt) > 2 * statistics.stdev(response_times)
        ]

        if not outliers:
            return []

        outlier_names = ', '.join(m.name for m in sorted(outliers, key=lambda x: x.avg, reverse=True)[:3])

        return [Recommendation(
            priority='Medium',
            action='Performance Imbalance Detected',
            details=(
                f'Significant performance variance detected (CV={cv:.2f}). '
                f'Transactions with outlier response times: {outlier_names}. '
                f'This may indicate inefficient code paths or resource contention.'
            ),
            category='Performance Balance'
        )]

    def _calculate_summary_stats(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float],
            summary: ReportSummary = None
    ) -> Dict[str, Any]:
        """Calculate summary statistics directly from TransactionMetrics objects."""
        if not metrics:
            return {}

        try:
            summary_stats = {
                "total_transactions": len(metrics),
                "avg_response_time": sum(m.avg for m in metrics) / len(metrics),
                "max_response_time": max(m.avg for m in metrics),
                "min_response_time": min(m.avg for m in metrics),
                "avg_error_rate": sum(m.error_rate for m in metrics) / len(metrics),
                "total_requests": sum(m.samples for m in metrics),
                "thresholds_used": thresholds,
                "transactions_over_rt_threshold": len([
                    m for m in metrics if m.avg > thresholds['response_time']
                ]),
                "transactions_over_er_threshold": len([
                    m for m in metrics if m.error_rate > thresholds['error_rate']
                ])
            }

            # Add summary-level stats if available
            if summary:
                summary_stats["overall_throughput"] = summary.throughput
                summary_stats["throughput_below_threshold"] = summary.throughput < thresholds['throughput']
                summary_stats["overall_error_rate"] = summary.error_rate
                summary_stats["total_duration"] = summary.duration

            # Add percentile information if available
            if metrics:
                all_response_times = []
                for m in metrics:
                    # If we have percentile data, use it; otherwise just use avg
                    if hasattr(m, 'percentiles') and m.percentiles:
                        all_response_times.extend([m.avg] * m.samples)
                    else:
                        all_response_times.extend([m.avg] * m.samples)

                if all_response_times:
                    sorted_times = sorted(all_response_times)
                    summary_stats["p50_response_time"] = sorted_times[len(sorted_times) // 2]
                    summary_stats["p90_response_time"] = sorted_times[int(len(sorted_times) * 0.9)]
                    summary_stats["p95_response_time"] = sorted_times[int(len(sorted_times) * 0.95)]
                    summary_stats["p99_response_time"] = sorted_times[int(len(sorted_times) * 0.99)]

            # Performance health indicators
            summary_stats["performance_health"] = self._calculate_health_score(metrics, thresholds, summary)

            logger.info(f"Summary stats calculated for {summary_stats['total_transactions']} transactions.")
            return summary_stats

        except Exception as e:
            logger.error(f"Summary stats calculation failed: {e}")
            return {"error": f"Failed to calculate summary stats: {e}"}

    def _calculate_health_score(
            self,
            metrics: List[TransactionMetrics],
            thresholds: Dict[str, float],
            summary: ReportSummary = None
    ) -> Dict[str, Any]:
        """Calculate overall health score and indicators."""
        health = {
            "score": 100.0,
            "indicators": []
        }

        # Response time health (40% weight)
        rt_failures = len([m for m in metrics if m.avg > thresholds['response_time']])
        rt_penalty = (rt_failures / len(metrics)) * 40 if metrics else 0
        health["score"] -= rt_penalty
        if rt_penalty > 0:
            health["indicators"].append(f"{rt_failures} transactions exceed response time threshold")

        # Error rate health (40% weight)
        er_failures = len([m for m in metrics if m.error_rate > thresholds['error_rate']])
        er_penalty = (er_failures / len(metrics)) * 40 if metrics else 0
        health["score"] -= er_penalty
        if er_penalty > 0:
            health["indicators"].append(f"{er_failures} transactions exceed error rate threshold")

        # Throughput health (20% weight)
        if summary and summary.throughput < thresholds['throughput']:
            tp_deficit = ((thresholds['throughput'] - summary.throughput) / thresholds['throughput']) * 20
            health["score"] -= tp_deficit
            health["indicators"].append("System throughput below threshold")

        # Ensure score is between 0 and 100
        health["score"] = max(0, min(100, health["score"]))

        # Add health status
        if health["score"] >= 90:
            health["status"] = "Excellent"
        elif health["score"] >= 75:
            health["status"] = "Good"
        elif health["score"] >= 60:
            health["status"] = "Fair"
        elif health["score"] >= 40:
            health["status"] = "Poor"
        else:
            health["status"] = "Critical"

        return health
