"""
ETL Component Factory

This module provides a factory for creating properly configured ETL components.
Following the Factory pattern, it centralizes the creation and configuration
of Extract, Transform, and Load components.

Author: Karen Florykian
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

        if pipeline_type == "gatling_to_excel":
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
