import logging
import io
import os
import tempfile
import zipfile
from typing import List, Dict, Any, Optional

from langchain_core.tools import ToolException

from ..etl_pipeline import BaseTransformer
from alita_sdk.tools.carrier.api_wrapper import CarrierAPIWrapper
from ..extractors import CarrierArtifactExtractor
from ..reporting.core.analyzers import PerformanceAnalyzer
from ..reporting.core.threshold_manager import ThresholdManager
from ..parsers import parsers
from alita_sdk.tools.carrier.utils.utils import get_latest_log_file
from alita_sdk.tools.carrier.utils.utils import GatlingConfig
from ..reporting.core.data_models import PerformanceReport
from ..reporting.core.data_models import PerformanceAnalysisResult

logger = logging.getLogger(__name__)


class CarrierExcelTransformer(BaseTransformer):
    """
    SOLID Transformer for Performance Reports.
    Its single responsibility is to orchestrate the transformation of raw
    performance data into a fully parsed and analyzed PerformanceReport object.
    It now handles the downloading and merging of artifacts before parsing.
    """

    def __init__(self):
        super().__init__()
        self.performance_analyzer = PerformanceAnalyzer()

    def transform(self, extracted_data: Dict[str, Any], context: Dict[str, Any]) -> PerformanceReport:
        """
        Orchestrates the transformation by first processing artifacts into a single
        log stream, then delegating parsing and analysis.
        """
        logger.info("Starting transformation phase.")

        # Step 1: Process artifacts (download, unzip, merge) to get a log stream.
        log_stream = self._process_artifacts_to_stream(extracted_data, context)

        # Step 2: Parse the stream and enrich the resulting report object.
        report_object = self._parse_and_build_report(log_stream, extracted_data, context)
        enriched_report = self._enrich_with_threshold_analysis(report_object, context)

        logger.info("Transformation phase completed successfully.")
        return enriched_report

    def _process_artifacts_to_stream(self, extracted_data: Dict[str, Any], context: Dict[str, Any]) -> io.StringIO:
        """
        Handles the download, unzip, and merge logic for report artifacts.
        Returns a single, in-memory stream of the merged log file.
        """
        api_wrapper: CarrierAPIWrapper = context["api_wrapper"]
        artifact_filenames = extracted_data["artifact_filenames"]
        bucket_name = extracted_data["bucket_name"]
        lg_type = extracted_data["report_metadata"].get("lg_type", "gatling")

        extracted_dirs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory for artifact processing: {temp_dir}")
            for filename in artifact_filenames:
                try:
                    logger.info(f"Downloading artifact '{filename}' from bucket '{bucket_name}'.")
                    zip_path = api_wrapper.download_artifact(bucket_name, filename, temp_dir)
                    extract_path = self._unzip_file(zip_path, temp_dir)
                    extracted_dirs.append(extract_path)
                except Exception as e:
                    logger.error(f"Failed to process artifact {filename}: {e}", exc_info=True)
                    # Continue processing other files if one fails
                    continue

            if not extracted_dirs:
                raise RuntimeError("Failed to download or extract any report artifacts.")

            return self._merge_logs_to_stream(extracted_dirs, lg_type)

    def _unzip_file(self, zip_path: str, extract_to_dir: str) -> str:
        """Unzips a file and returns the path to the extraction directory."""
        extract_dir = os.path.join(extract_to_dir, os.path.basename(zip_path).replace('.zip', ''))
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Unzipped '{zip_path}' to '{extract_dir}'.")
        return extract_dir

    def _merge_logs_to_stream(self, extracted_dirs: List[str], lg_type: str) -> io.StringIO:
        """Merges multiple log files into a single in-memory StringIO stream."""
        log_stream = io.StringIO()
        is_first_file = True

        for log_dir in extracted_dirs:
            try:
                if lg_type == "jmeter":
                    report_file = os.path.join(log_dir, "jmeter.jtl")
                else:
                    report_file = get_latest_log_file(log_dir, "simulation.log")

                if not os.path.exists(report_file):
                    logger.warning(f"Log file not found in expected location: {report_file}")
                    continue

                with open(report_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if is_first_file:
                        log_stream.writelines(lines)
                        is_first_file = False
                    else:
                        # Skip header for subsequent files
                        log_stream.writelines(lines[1:])
            except Exception as e:
                logger.error(f"Failed to read or merge log file from {log_dir}: {e}")

        log_stream.seek(0)  # Rewind stream to the beginning for reading
        logger.info(f"Successfully merged {len(extracted_dirs)} log(s) into an in-memory stream.")
        return log_stream

    def _parse_and_build_report(self, log_stream: io.StringIO, extracted_data: Dict,
                                context: Dict) -> PerformanceReport:
        """
        Selects the correct parser and calls its `.parse()` method,
        expecting a fully validated PerformanceReport object in return.
        """
        report_metadata = extracted_data["report_metadata"]
        user_args = context.get("user_args", {})

        parser = self._get_parser_for(report_metadata.get("lg_type"), log_stream, user_args)
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
        ParserClass = parser_map.get(str(parser_type).lower())

        if not ParserClass:
            raise ValueError(f"Unsupported performance parser type: '{parser_type}'")

        parser_kwargs = {}
        if parser_type == "gatling":
            gatling_config = GatlingConfig()
            parser_kwargs["think_times"] = args.get("think_time", gatling_config.DEFAULT_THINK_TIME)
        return ParserClass(stream)

    def _enrich_with_threshold_analysis(self, report: PerformanceReport, context: dict) -> PerformanceReport:
        """
        Enriches the report by performing analysis and applying the results.
        """
        logger.info("Enriching report with threshold analysis.")
        user_args = context.get("user_args", {})
        thresholds = ThresholdManager.get_threshold_configs(user_args)
        analysis_result = self.performance_analyzer.analyze(report, thresholds, user_args)

        report.build_status = analysis_result.status
        report.analysis_summary = analysis_result.justification
        report.thresholds = thresholds

        self._add_contextual_metadata(report, context)
        logger.info(f"Enrichment complete. Final build status: {report.build_status}")
        return report

    def _add_contextual_metadata(self, report: PerformanceReport, context: dict) -> None:
        """DRY Helper: Adds metadata common to all enrichment types."""
        api_wrapper = context.get("api_wrapper")
        report_id = context.get("report_id") or context.get("source_report_id")

        if api_wrapper and report_id:
            report.carrier_report_url = f"{api_wrapper.url.strip('/')}/-/performance/backend/results?result_id={report_id}"
            logger.info(f"Successfully set carrier_report_url for report ID {report_id}")
        else:
            logger.warning("Could not set carrier_report_url: api_wrapper or report_id missing from context.")


class ComparisonExcelTransformer(BaseTransformer):
    """
    Transforms a list of reports into a single comparison workbook.
    It orchestrates the processing of each individual report before generating
    the final comparison data and AI-powered analysis.
    """

    def __init__(self):
        super().__init__()
        self.single_report_transformer = CarrierExcelTransformer()
        logger.info("ComparisonExcelTransformer initialized")

    def transform(self, extracted_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares comparison data for the stateful loader. It processes individual reports
        and generates AI analysis, but does NOT create an Excel workbook.
        """
        logger.info("Preparing comparison data for the stateful loader...")
        reports_meta = extracted_data["reports_meta"]

        performance_reports = self._process_individual_reports(reports_meta, context)

        if len(performance_reports) < 2:
            raise ToolException(f"Could only process {len(performance_reports)} reports successfully. Need at least 2.")

        comparison_analysis = None
        if context.get("enable_ai_analysis"):
            try:
                logger.info("Attempting AI-powered comparison analysis...")
                comparison_analysis = self._perform_llm_analysis(performance_reports, context)
                if comparison_analysis:
                    logger.info("Successfully generated AI comparison analysis.")
            except Exception as e:
                logger.warning(f"AI analysis step failed but proceeding. Error: {e}", exc_info=True)

        return {
            "all_reports": performance_reports,
            "comparison_analysis": comparison_analysis
        }

    def _process_individual_reports(self, reports_meta: List[Dict], global_context: Dict) -> List[PerformanceReport]:
        """
        Processes a list of report metadata into a list of PerformanceReport objects.
        """
        processed_reports = []
        test_name = global_context.get("test_name", "Unknown Test")

        for report_meta in reports_meta:
            try:
                report_id = report_meta["id"]
                logger.info(f"Processing individual report ID: {report_id}")

                single_report_context = global_context.copy()
                single_report_context["report_id"] = report_id

                extractor = CarrierArtifactExtractor()
                extracted_data = extractor.extract(single_report_context)

                performance_report = self.single_report_transformer.transform(extracted_data, single_report_context)
                performance_report.test_name = test_name

                processed_reports.append(performance_report)

            except Exception as e:
                logger.error(f"Failed to process report ID {report_meta.get('id', 'N/A')}: {e}", exc_info=True)
                # Continue to the next report if one fails
                continue

        return processed_reports

    def _determine_report_type(self, report: PerformanceReport) -> str:
        return getattr(report, 'report_type', 'GATLING').upper()

    def _perform_llm_analysis(self, reports: List[PerformanceReport], context: Dict) -> Optional[
        PerformanceAnalysisResult]:
        """
        Perform LLM analysis by calling the new, direct structured data extraction method.
        """
        llm = context.get("llm")
        if not llm:
            logger.warning("LLM not available in context - skipping analysis")
            return None

        try:
            from alita_sdk.tools.carrier.utils.intent_utils import CarrierIntentExtractor
            from alita_sdk.tools.carrier.etl.reporting.core.data_models import PerformanceAnalysisResult
            from alita_sdk.tools.carrier.utils.prompts import build_enhanced_comparison_prompt

            # --- THE FINAL FIX IS HERE ---
            # 1. Prepare the data for the prompt builder.
            reports_data_for_prompt = [
                {"description": f"Report from {r.summary.date_start}", "content": r.summary.to_legacy_dict()}
                for r in reports
            ]

            # 2. Call the centralized prompt builder.
            analysis_request = build_enhanced_comparison_prompt(reports_data_for_prompt, context)

            # 3. Use the new, direct data extraction method.
            intent_extractor = CarrierIntentExtractor(llm=llm)
            logger.info("Extracting performance comparison analysis using direct structured data method.")

            analysis_result = intent_extractor.extract_structured_data(
                user_message=analysis_request,
                tool_schema=PerformanceAnalysisResult,
                context={"task": "internal_structured_analysis"}
            )

            return analysis_result

        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}", exc_info=True)
            return None
