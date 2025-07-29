import logging
import io

from typing import List, Dict

from .etl_pipeline import BaseTransformer
from ..reporting.core.analyzers import PerformanceAnalyzer
from ..reporting.core.threshold_manager import ThresholdManager
from ..parsers import parsers
from ..utils.utils import GatlingConfig
from ..reporting.core.data_models import (PerformanceReport, ComparisonResult, PerformanceStatus, PerformanceDirection)

logger = logging.getLogger(__name__)


class CarrierExcelTransformer(BaseTransformer):
    """
    SOLID Transformer for Performance Reports.
    Its single responsibility is to orchestrate the transformation of raw
    performance data into a fully parsed and analyzed PerformanceReport object.
    It delegates parsing and analysis to specialized, reusable components.
    """

    def __init__(self):
        super().__init__()
        self.performance_analyzer = PerformanceAnalyzer()

    def transform(self, extracted_data: bytes, context: dict) -> PerformanceReport:
        """
        Orchestrates the transformation by delegating the complete creation of a
        PerformanceReport object to a specialized parser.
        """
        logger.info("Starting transformation phase.")
        report_object = self._parse_and_build_report(extracted_data, context)
        enriched_report = self._enrich_with_threshold_analysis(report_object, context.get("user_args", {}), context)

        logger.info("Transformation phase completed successfully.")
        return enriched_report

    def _parse_and_build_report(self, extracted_data: bytes, context: dict) -> PerformanceReport:
        """
        Selects the correct parser and calls its single `.parse()` method,
        expecting a fully validated PerformanceReport object in return.
        """
        log_stream = extracted_data["log_content_stream"]
        report_metadata = extracted_data["report_metadata"]
        user_args = context.get("user_args", {})

        parser = self._get_parser_for(report_metadata.get("lg_type"), log_stream, user_args)

        # This is the new, robust data contract. The parser handles everything.
        report = parser.parse()

        logger.info(f"Successfully received PerformanceReport object from '{parser.__class__.__name__}'.")
        return report

    def _get_parser_for(self, parser_type: str, stream: io.StringIO, args: dict) -> parsers.BaseReportParser:
        """Private factory method to select and instantiate the correct parser."""
        logger.debug(f"Selecting parser for type: '{parser_type}'.")
        parser_map = {
            "gatling": parsers.GatlingReportParser,
            "jmeter": parsers.JMeterReportParser,
        }
        ParserClass = parser_map.get(parser_type)

        if not ParserClass:
            raise ValueError(f"Unsupported performance parser type: '{parser_type}'")

        parser_kwargs = {}
        if parser_type == "gatling":
            # Use the central config for the default value
            gatling_config = GatlingConfig()
            parser_kwargs["think_times"] = args.get("think_time", gatling_config.DEFAULT_THINK_TIME)

        return ParserClass(stream, **parser_kwargs)

    def _enrich_with_threshold_analysis(
            self, report: PerformanceReport, user_args: dict, context: dict
    ) -> PerformanceReport:
        """
        Enriches the report by performing analysis and applying the results.
        """
        logger.info("Enriching report with threshold analysis.")

        # Perform analysis using existing analyzer
        # We need to pass the thresholds from user_args
        thresholds = ThresholdManager.get_threshold_configs(user_args)
        analysis_result = self.performance_analyzer.analyze(report, thresholds, user_args)
        report.build_status = analysis_result.status
        report.analysis_summary = analysis_result.justification
        report.thresholds = ThresholdManager.get_threshold_configs(user_args)

        # Keep existing metadata logic
        self._add_contextual_metadata(report, context)

        logger.info(f"Enrichment complete. Final build status: {report.build_status}")
        return report

    def _enrich_with_comparison(
            self, current_report: PerformanceReport, baseline_report: PerformanceReport, context: dict
    ) -> PerformanceReport:
        """Enriches the report by comparing it to a baseline report."""
        logger.info(f"Enriching report with baseline comparison against report ID: {baseline_report.summary.report_id}")
        comparison_results = self.comparison_analyzer.compare(baseline_report, current_report)

        context['comparison_results'] = comparison_results
        current_report.analysis_summary = self._summarize_comparison(comparison_results)
        current_report.build_status = PerformanceStatus.WARNING

        self._add_contextual_metadata(current_report, context)
        logger.info("Baseline comparison enrichment complete.")
        return current_report

    def _add_contextual_metadata(self, report: PerformanceReport, context: dict) -> None:
        """DRY Helper: Adds metadata common to all enrichment types."""
        api_wrapper = context.get("api_wrapper")

        report_id = context.get("report_id") or context.get("source_report_id")

        if api_wrapper and report_id:
            report.carrier_report_url = f"{api_wrapper.url.strip('/')}/-/performance/backend/results?result_id={report_id}"
            logger.info(f"Successfully set carrier_report_url for report ID {report_id}")
        else:
            logger.warning("Could not set carrier_report_url: api_wrapper or report_id missing from context.")

    def _summarize_comparison(self, comparison_results: List[ComparisonResult]) -> str:
        """Creates a high-level summary string from comparison results."""
        degradations = [r for r in comparison_results if r.direction == PerformanceDirection.DEGRADED]
        improvements = [r for r in comparison_results if r.direction == PerformanceDirection.IMPROVED]

        summary = f"Comparison found {len(degradations)} degradations and {len(improvements)} improvements. "
        if degradations:
            try:
                # Find the degradation with the largest positive percent change
                worst_degradation = max(degradations, key=lambda r: r.percent_change)
                summary += (f"Worst degradation: '{worst_degradation.transaction_name}' "
                            f"({worst_degradation.metric_name}) changed by "
                            f"{worst_degradation.percent_change:+.1f}%.")
            except (ValueError, TypeError):
                # Handle case where degradations list is empty after filtering somehow
                pass
        return summary
