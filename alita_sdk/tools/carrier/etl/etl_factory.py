"""
ETL Component Factory

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
from .extractors import GatlingLogExtractor
from .loaders import CarrierExcelLoader
from . import transformers


class ETLComponentFactory:
    """Assembles and returns a configured ETL pipeline based on the task type."""

    @staticmethod
    def get_pipeline(pipeline_type: str) -> ETLPipeline | None:
        """
        Returns a pre-configured ETLPipeline for a given type.
        This is the central point for extending the toolkit with new capabilities.
        """

        if pipeline_type == "excel_report":
            # Example for your existing CreateExcelReportTool
            extractor = GatlingLogExtractor()
            transformer = transformers.CarrierExcelTransformer()
            loader = CarrierExcelLoader()
            return ETLPipeline(extractor, transformer, loader)

        elif pipeline_type == "ui_to_excel":
            # Example for your existing CreateExcelReportTool
            extractor = GatlingLogExtractor()
            transformer = transformers.CarrierExcelTransformer()
            loader = CarrierExcelLoader()
            return ETLPipeline(extractor, transformer, loader)

        elif pipeline_type == "pptx_update":
            # Example for the new PPT functionality
            # extractor = PPTXExtractor() # Extracts template structure and transcript
            # transformer = PPTContentTransformer() # Uses ETLLLMIntegrator to generate content
            # loader = CarrierPPTXLoader() # Uploads the new PPTX
            # return ETLPipeline(extractor, transformer, loader)
            pass  # Placeholder

        elif pipeline_type == "excel_comparison_update":
            # Example for the CompareExcelReportsTool
            # extractor = MasterComparisonExtractor() # Extracts BOTH master and new result sheets
            # transformer = ComparisonSheetTransformer() # Adds new sheet to master workbook
            # loader = CarrierExcelLoader() # Uploads the updated master workbook
            # return ETLPipeline(extractor, transformer, loader)
            pass  # Placeholder


        else:
            raise ValueError(f"Unknown pipeline type: '{pipeline_type}'. Cannot build ETL pipeline.")
