from .data_models import PerformanceReport
from .threshold_manager import ThresholdManager
from typing import Dict, Any, List

import logging

logger = logging.getLogger(__name__)

class ComparisonResult:
    """Simple comparison result container replacing ReportSummarizer."""
    def __init__(self, status: str, response_time_analysis: Dict, error_rate_analysis: Dict, throughput_analysis: Dict):
        self.status = status
        self.response_time_analysis = response_time_analysis
        self.error_rate_analysis = error_rate_analysis
        self.throughput_analysis = throughput_analysis

class ComparisonAnalyzer:
    """
    REFACTORED: Now includes all ReportSummarizer functionality following DRY principles.
    Single responsibility for all comparison and summarization logic.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_reports(self, baseline_report: PerformanceReport, current_report: PerformanceReport,
                        user_args: dict) -> ComparisonResult:
        """
        DRY: Uses centralized threshold management for comparison analysis.
        """
        self.logger.info("Starting report comparison analysis.")

        # DRY: Get thresholds from centralized manager
        thresholds = ThresholdManager.get_effective_values(user_args)

        # Perform comparisons using centralized threshold values
        response_time_comparison = self._compare_response_times(baseline_report, current_report, thresholds)
        error_rate_comparison = self._compare_error_rates(baseline_report, current_report, thresholds)
        throughput_comparison = self._compare_throughput(baseline_report, current_report, thresholds)

        # Combine results
        overall_status = self._determine_comparison_status(
            response_time_comparison, error_rate_comparison, throughput_comparison
        )

        return ComparisonResult(
            status=overall_status,
            response_time_analysis=response_time_comparison,
            error_rate_analysis=error_rate_comparison,
            throughput_analysis=throughput_comparison
        )

    def _compare_response_times(self, baseline: PerformanceReport, current: PerformanceReport,
                                thresholds: Dict) -> Dict:
        """
        DRY: Use centralized threshold for response time comparison.
        """
        # DRY: Get threshold using standardized key
        rt_threshold = thresholds['response_time']
        percentile_field = ThresholdManager.get_percentile_field()

        comparison_results = {}

        for transaction_name in baseline.transactions.keys():
            if transaction_name in current.transactions:
                baseline_rt = getattr(baseline.transactions[transaction_name], percentile_field, 0)
                current_rt = getattr(current.transactions[transaction_name], percentile_field, 0)

                # Calculate percentage change
                percentage_change = ((current_rt - baseline_rt) / baseline_rt * 100) if baseline_rt > 0 else 0

                # Check against threshold
                exceeds_threshold = current_rt > rt_threshold

                comparison_results[transaction_name] = {
                    'baseline_value': baseline_rt,
                    'current_value': current_rt,
                    'percentage_change': percentage_change,
                    'exceeds_threshold': exceeds_threshold,
                    'threshold_value': rt_threshold
                }

        return comparison_results

    def _compare_error_rates(self, baseline: PerformanceReport, current: PerformanceReport, thresholds: Dict) -> Dict:
        """
        DRY: Use centralized threshold for error rate comparison.
        """
        # DRY: Get threshold using standardized key
        er_threshold = thresholds['error_rate']

        comparison_results = {}

        for transaction_name in baseline.transactions.keys():
            if transaction_name in current.transactions:
                # In the comparison analyzer where the error occurs
                logger.debug(f"TransactionMetrics attributes: {dir(baseline.transactions[transaction_name])}")
                logger.debug(f"TransactionMetrics dict: {vars(baseline.transactions[transaction_name])}")
                baseline_er = baseline.transactions[transaction_name].Error_pct
                current_er = current.transactions[transaction_name].Error_pct

                # Calculate absolute change (for error rates, percentage change can be misleading)
                absolute_change = current_er - baseline_er

                # Check against threshold
                exceeds_threshold = current_er > er_threshold

                comparison_results[transaction_name] = {
                    'baseline_value': baseline_er,
                    'current_value': current_er,
                    'absolute_change': absolute_change,
                    'exceeds_threshold': exceeds_threshold,
                    'threshold_value': er_threshold
                }

        return comparison_results

    def _compare_throughput(self, baseline: PerformanceReport, current: PerformanceReport, thresholds: Dict) -> Dict:
        """
        DRY: Use centralized threshold for throughput comparison.
        """
        # DRY: Get threshold using standardized key
        tp_threshold = thresholds['throughput']

        # Compare overall throughput
        baseline_throughput = getattr(baseline.summary, 'throughput', 0)
        current_throughput = getattr(current.summary, 'throughput', 0)

        # Calculate percentage change
        percentage_change = ((
                                         current_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0

        # Check against threshold
        below_threshold = current_throughput < tp_threshold

        return {
            'baseline_value': baseline_throughput,
            'current_value': current_throughput,
            'percentage_change': percentage_change,
            'below_threshold': below_threshold,
            'threshold_value': tp_threshold
        }

    def _determine_comparison_status(self, rt_comparison: Dict, er_comparison: Dict, tp_comparison: Dict) -> str:
        """
        Determine overall comparison status based on threshold violations.
        """
        # Check if any transactions exceed thresholds
        rt_violations = any(result.get('exceeds_threshold', False) for result in rt_comparison.values())
        er_violations = any(result.get('exceeds_threshold', False) for result in er_comparison.values())
        tp_violation = tp_comparison.get('below_threshold', False)

        if rt_violations or er_violations or tp_violation:
            return "DEGRADED"
        else:
            return "IMPROVED"

    def generate_consolidated_summary(self, reports: List[PerformanceReport],
                                    output_format: str = "excel") -> Dict[str, Any]:
        """
        SOLID: Single method to generate consolidated summaries for multiple reports.
        DRY: Replaces ReportSummarizer with centralized comparison logic.

        Args:
            reports: List of PerformanceReport objects to compare
            output_format: "excel", "markdown", or "both"

        Returns:
            Dictionary with summary data and formatted outputs
        """
        self.logger.info(f"Generating consolidated summary for {len(reports)} reports")

        if len(reports) < 2:
            raise ValueError("Need at least 2 reports for comparison analysis")

        # Use the most recent report as baseline
        baseline_report = reports[-1]
        comparison_results = []

        # Compare each report against baseline
        for i, current_report in enumerate(reports[:-1]):
            comparison = self.compare_reports(baseline_report, current_report, {})
            comparison_results.append({
                "report_index": i,
                "report_id": getattr(current_report, 'report_id', f"Report_{i}"),
                "comparison": comparison,
                "summary": self._generate_comparison_summary(baseline_report, current_report)
            })

        # Generate consolidated insights
        consolidated_insights = self._generate_consolidated_insights(comparison_results)

        # Generate outputs based on format
        outputs = {}
        if output_format in ["excel", "both"]:
            outputs["excel_data"] = self._format_for_excel(comparison_results, consolidated_insights)

        if output_format in ["markdown", "both"]:
            outputs["markdown"] = self._format_as_markdown(comparison_results, consolidated_insights)

        return {
            "summary": consolidated_insights,
            "comparison_results": comparison_results,
            "outputs": outputs
        }

    def _generate_comparison_summary(self, baseline: PerformanceReport,
                                   current: PerformanceReport) -> Dict[str, Any]:
        """
        DRY: Centralized summary generation replacing ReportSummarizer logic.
        """
        summary = {
            "total_transactions": len(baseline.transactions),
            "degraded_count": 0,
            "improved_count": 0,
            "stable_count": 0,
            "baseline_error_rate": baseline.summary.error_rate,
            "current_error_rate": current.summary.error_rate,
            "details": []
        }

        baseline_tx = baseline.transactions or {}
        current_tx = current.transactions or {}

        for name, b_metric in baseline_tx.items():
            if name == "Total" or name not in current_tx:
                continue

            c_metric = current_tx[name]
            b_p95 = getattr(b_metric, 'pct_95', 0)
            c_p95 = getattr(c_metric, 'pct_95', 0)

            pct_change = ((c_p95 - b_p95) / b_p95) * 100.0 if b_p95 > 0 else 0.0

            if pct_change > 10.0:
                status = "degraded"
                summary["degraded_count"] += 1
            elif pct_change < -10.0:
                status = "improved"
                summary["improved_count"] += 1
            else:
                status = "stable"
                summary["stable_count"] += 1

            summary["details"].append({
                "transaction": name,
                "baseline_p95": b_p95,
                "current_p95": c_p95,
                "percent_change": pct_change,
                "status": status
            })

        return summary

    def _generate_consolidated_insights(self, comparison_results: List[Dict]) -> Dict[str, Any]:
        """Generate high-level insights across all comparisons."""
        total_reports = len(comparison_results)

        # Aggregate metrics
        all_degraded = []
        all_improved = []
        consistent_issues = {}

        for result in comparison_results:
            summary = result["summary"]

            # Track degraded transactions
            for detail in summary["details"]:
                if detail["status"] == "degraded":
                    tx_name = detail["transaction"]
                    all_degraded.append(tx_name)
                    consistent_issues[tx_name] = consistent_issues.get(tx_name, 0) + 1
                elif detail["status"] == "improved":
                    all_improved.append(detail["transaction"])

        # Find consistently problematic transactions
        consistent_problems = [
            tx for tx, count in consistent_issues.items()
            if count >= total_reports * 0.6  # 60% of reports
        ]

        return {
            "total_reports_analyzed": total_reports,
            "unique_degraded_transactions": len(set(all_degraded)),
            "unique_improved_transactions": len(set(all_improved)),
            "consistent_problem_transactions": consistent_problems,
            "overall_trend": self._determine_overall_trend(comparison_results),
            "key_recommendations": self._generate_recommendations(consistent_problems, comparison_results)
        }

    def _format_for_excel(self, comparison_results: List[Dict], insights: Dict) -> Dict[str, Any]:
        """Format data structure suitable for Excel generation."""
        return {
            "summary_sheet": {
                "insights": insights,
                "comparison_matrix": self._build_comparison_matrix(comparison_results)
            },
            "details_sheet": {
                "transaction_details": self._build_transaction_details(comparison_results)
            }
        }

    def _format_as_markdown(self, comparison_results: List[Dict], insights: Dict) -> str:
        """
        DRY: Single method for Markdown generation replacing ReportSummarizer.
        """
        lines = []
        lines.append("# Performance Comparison Analysis")
        lines.append("")
        lines.append(f"**Reports Analyzed:** {insights['total_reports_analyzed']}")
        lines.append(f"**Overall Trend:** {insights['overall_trend']}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append(f"- **Unique Degraded Transactions:** {insights['unique_degraded_transactions']}")
        lines.append(f"- **Unique Improved Transactions:** {insights['unique_improved_transactions']}")
        lines.append("")

        # Consistent issues
        if insights['consistent_problem_transactions']:
            lines.append("## Consistent Performance Issues")
            for tx in insights['consistent_problem_transactions']:
                lines.append(f"- **{tx}**: Degraded across multiple test runs")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        for rec in insights['key_recommendations']:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def _determine_overall_trend(self, comparison_results: List[Dict]) -> str:
        """Determine overall performance trend."""
        total_degraded = sum(r["summary"]["degraded_count"] for r in comparison_results)
        total_improved = sum(r["summary"]["improved_count"] for r in comparison_results)

        if total_improved > total_degraded * 1.5:
            return "Improving"
        elif total_degraded > total_improved * 1.5:
            return "Degrading"
        else:
            return "Stable"

    def _generate_recommendations(self, consistent_problems: List[str],
                                comparison_results: List[Dict]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if consistent_problems:
            recommendations.append(
                f"🔍 Investigate {len(consistent_problems)} consistently degraded transaction(s): "
                f"{', '.join(consistent_problems[:3])}"
            )

        # Check for error rate trends
        error_rates = [r["summary"]["current_error_rate"] for r in comparison_results]
        avg_error_rate = sum(error_rates) / len(error_rates)

        if avg_error_rate > 5.0:
            recommendations.append(f"⚠️ High average error rate: {avg_error_rate:.1f}% - investigate failed requests")

        if not recommendations:
            recommendations.append("✅ No significant performance issues detected across test runs")

        return recommendations

    def _build_comparison_matrix(self, comparison_results: List[Dict]) -> List[List]:
        """Build matrix data for Excel comparison sheet."""
        # Implementation for Excel matrix format
        return []

    def _build_transaction_details(self, comparison_results: List[Dict]) -> List[Dict]:
        """Build detailed transaction data for Excel."""
        # Implementation for Excel details format
        return []
