"""
ETL Component Factory - Enhanced Version

This module provides a factory for creating properly configured ETL components.
Following the Factory pattern, it centralizes the creation and configuration
of Extract, Transform, and Load components.

Author: Karen Florykian

# Currently implemented:
- gatling_to_excel: Gatling logs → Excel report

Backend Performance Pipelines:
- jmeter_to_excel: JMeter JTL → Excel report
- gatling_comparison: Compare multiple Gatling reports
- jmeter_comparison: Compare multiple JMeter reports
- performance_baseline: Create/update performance baselines
- threshold_validation: Validate results against thresholds
- error_analysis: Extract and analyze error patterns
- performance_trends: Historical trend analysis

Web UI Performance Pipelines:
- sitespeed_to_excel: Sitespeed results → Excel report
- lighthouse_to_excel: Lighthouse results → Excel report
- ui_comparison: Compare UI test results
- ui_baseline: Create/update UI test baselines
- ui_trends: Historical UI performance trends
- har_analysis: Clean UP HAR file analysis and extraction

DEV Pipelines:
- har_to_jmeter: Convert HAR files → JMeter scripts
- har_to_gatling: Convert HAR files → Gatling scripts
- har_change_detection: Detect API changes from HAR files
- script_update_recommendations: Analyze HAR diffs → Script update suggestions
- api_contract_validation: Compare HAR against API specs
- user_flow_extraction: Extract user journeys from HAR sequences

# Regression and validation
- regression_test_selector: AI-based test selection for changes
- test_impact_analysis: Analyze which tests need re-running
- baseline_comparison: Compare against established baselines
- fix_validation: Validate performance fixes automatically
- test_optimization: Identify redundant/unnecessary tests

Support Pipelines:
- request_bottleneck_analysis: Identify slowest requests with context
- response_time_breakdown: Decompose request timing (DNS, connect, TTFB, etc.)
- error_correlation: Correlate errors with performance degradation
- resource_impact_analysis: Analyze static resource impact on performance
- api_dependency_mapping: Map API call chains and dependencies
- comparative_slowness: Compare request performance across builds/environments

Inception:
- usage_pattern_extraction: Extract common user patterns from logs
- scenario_template_generation: Generate test scenarios from patterns
- load_model_creation: Create realistic load models from production data
- peak_usage_analysis: Identify and model peak usage patterns
- scenario_validation: Validate scenarios against production behavior
- test_data_generation: Generate test data based on scenarios

Ticketing and Reporting Pipelines:
- ticket_summary: Extract and summarize tickets
- weekly_progress: Generate weekly progress reports
- board_analytics: Board statistics and analytics
- engagement_report: Full engagement status report

Insights and Analytics Pipelines:
- ai_performance_insights: LLM-based performance analysis
- scri: Predictive system behavior
- thansaction_breakdown: Detect performance anomalies
- root_cause_analysis: AI-powered RCA

# Missing pipelines:
- config_audit: Configuration audit trail
- integration_validation: Validate integrations
- secret_rotation: Secret rotation reports
"""
from .etl_pipeline import ETLPipeline
from .extractors import CarrierArtifactExtractor, CarrierUIReportExtractor
from .loaders.backend_loaders import CarrierExcelLoader
from .loaders.ui_loaders import CarrierUIExcelLoader
from . import transformers
import logging

logger = logging.getLogger(__name__)


class ETLComponentFactory:
    """
    Assembles and returns a configured ETL pipeline based on a registered key.
    This dynamic factory uses a registration pattern for easy extension.
    """
    _pipelines = {}

    @classmethod
    def _register_pipelines(cls):
        """Internal method to define all available pipelines. Cleanly separated."""
        if cls._pipelines:
            return

        # Backend Performance Pipelines
        excel_transformer = transformers.CarrierExcelTransformer()
        excel_pipeline = ETLPipeline(
            extractor=CarrierArtifactExtractor(),
            transformer=excel_transformer,
            loader=CarrierExcelLoader()
        )
        cls._pipelines["gatling_to_excel"] = excel_pipeline
        cls._pipelines["jmeter_to_excel"] = excel_pipeline

        # UI Performance Pipelines - UPDATED to use UI loader
        ui_transformer = transformers.CarrierUIExcelTransformer()
        ui_excel_pipeline = ETLPipeline(
            extractor=CarrierUIReportExtractor(),
            transformer=ui_transformer,
            loader=CarrierUIExcelLoader()
        )
        cls._pipelines["lighthouse_to_excel"] = ui_excel_pipeline
        cls._pipelines["sitespeed_to_excel"] = ui_excel_pipeline

        logger.info(f"ETL Factory initialized with {len(cls._pipelines)} pipelines: {list(cls._pipelines.keys())}")

    @staticmethod
    def get_pipeline(pipeline_type: str) -> ETLPipeline:
        """
        Returns a pre-configured ETLPipeline for a given type.
        Raises a ValueError if the pipeline type is not registered.
        """
        ETLComponentFactory._register_pipelines()  # Ensure definitions are loaded

        pipeline = ETLComponentFactory._pipelines.get(pipeline_type)
        if not pipeline:
            logger.error(f"Unknown pipeline type requested: '{pipeline_type}'")
            raise ValueError(
                f"Unknown pipeline type: '{pipeline_type}'. Available types: {list(ETLComponentFactory._pipelines.keys())}")

        logger.info(f"Returning configured pipeline for '{pipeline_type}'")
        return pipeline