"""
ETL Component Factory - Enhanced Version

This module provides a factory for creating properly configured ETL components.
Following the Factory pattern, it centralizes the creation and configuration
of Extract, Transform, and Load components.

Author: Karen Florykian
"""
from .etl_pipeline import ETLPipeline
from .extractors import CarrierArtifactExtractor, CarrierUIReportExtractor, ComparisonExtractor

from .loaders.backend_loaders import CarrierExcelLoader, ComparisonExcelLoader
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
    _initialized = False

    @classmethod
    def _register_pipelines(cls):
        """Internal method to define all available pipelines. Cleanly separated."""
        if cls._initialized:
            return

        cls._initialized = True

        # Backend Performance Pipelines - Single Report
        excel_transformer = transformers.CarrierExcelTransformer()
        excel_pipeline = ETLPipeline(
            extractor=CarrierArtifactExtractor(),
            transformer=excel_transformer,
            loader=CarrierExcelLoader()
        )
        cls._pipelines["gatling_to_excel"] = excel_pipeline
        cls._pipelines["jmeter_to_excel"] = excel_pipeline

        # Backend Performance Pipelines - Comparison Reports
        comparison_pipeline = ETLPipeline(
            extractor=ComparisonExtractor(),
            transformer=transformers.ComparisonExcelTransformer(),
            loader=ComparisonExcelLoader()
        )
        # Register all comparison aliases
        cls._pipelines["gatling_comparison_between_the_tests"] = comparison_pipeline
        cls._pipelines["jmeter_comparison_between_the_tests"] = comparison_pipeline
        cls._pipelines["report_comparison_with_baseline"] = comparison_pipeline
        # Add shorter aliases for convenience
        cls._pipelines["gatling_comparison"] = comparison_pipeline
        cls._pipelines["jmeter_comparison"] = comparison_pipeline

        # UI Performance Pipelines
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

    @classmethod
    def register_pipeline(cls, name: str, pipeline: ETLPipeline):
        """
        Register a custom pipeline. Useful for extensions.

        Args:
            name: Pipeline identifier
            pipeline: Configured ETLPipeline instance
        """
        cls._register_pipelines()  # Ensure base pipelines are loaded
        cls._pipelines[name] = pipeline
        logger.info(f"Registered custom pipeline: '{name}'")

    @classmethod
    def list_available_pipelines(cls) -> list:
        """Return list of all available pipeline types."""
        cls._register_pipelines()
        return list(cls._pipelines.keys())