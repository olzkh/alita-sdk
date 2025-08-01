"""
UI Excel Transformer - Following Backend Pattern

Author: Karen Florykian
"""
import logging
import io
from typing import Dict, Any, List
from dataclasses import replace

from ..etl_pipeline import BaseTransformer
from langchain_core.tools import ToolException
from ..parsers.parsers import LighthouseJsonParser
from ..reporting.core.data_models import UIPerformanceReport

logger = logging.getLogger(__name__)


class CarrierUIExcelTransformer(BaseTransformer):
    """
    Transforms UI extraction data into UIPerformanceReport objects for Excel generation.
    Processes ALL UI reports from the extraction phase.
    """

    def transform(self, extracted_data: Dict[str, Any], context: Dict[str, Any]) -> 'UITransformResult':
        """
        Transform extracted UI data into UITransformResult with all UI reports.

        Returns:
            UITransformResult: Contains dictionary of UIPerformanceReport objects
        """
        logger.info("Starting UI transformation phase.")

        if not extracted_data:
            raise ToolException("Extracted data is required")

        api_wrapper = context.get("api_wrapper")
        if not api_wrapper:
            raise ToolException("API wrapper missing from context")

        json_links_info = extracted_data.get("json_links", [])
        report_metadata = extracted_data.get("report_metadata", {})

        if not json_links_info:
            raise ToolException("No JSON artifacts to process")

        # Process ALL JSON files into UIPerformanceReport objects
        worksheets_data = {}

        for idx, link_info in enumerate(json_links_info):
            worksheet_name = link_info.get("worksheet_name", f"UI_Report_{idx + 1}")
            json_url = link_info.get("json_url")

            if not json_url:
                logger.error(f"No JSON URL found for worksheet: {worksheet_name}")
                raise ToolException(f"Missing JSON URL for worksheet: {worksheet_name}")

            logger.info(f"Processing UI report {idx + 1}/{len(json_links_info)}: {worksheet_name}")

            # Download and parse JSON
            json_content = api_wrapper.download_raw_from_url(json_url)
            if not json_content:
                raise ToolException(f"Empty JSON content from {json_url}")

            # Parse JSON into UIPerformanceReport
            parser = LighthouseJsonParser(io.StringIO(json_content))
            ui_performance_report = parser.parse()

            # Update report metadata if needed
            updates = {}

            if report_metadata.get("name"):
                updates["report_name"] = f"{report_metadata.get('name')}_{worksheet_name}"

            if report_metadata.get("finalDisplayedUrl"):
                updates["carrier_report_url"] = report_metadata.get("finalDisplayedUrl")

            # Apply updates if any
            if updates:
                ui_performance_report = replace(ui_performance_report, **updates)

            worksheets_data[worksheet_name] = ui_performance_report
            logger.info(f"Successfully processed UI report: {worksheet_name}")

        if not worksheets_data:
            raise ToolException("No UI reports could be processed")

        logger.info(f"Successfully transformed {len(worksheets_data)} UI reports")

        # Return UITransformResult object that the loader expects
        return UITransformResult(
            worksheets_data=worksheets_data,
            report_id=str(context.get("report_id", "unknown")),
            report_name=report_metadata.get("name", "ui_report"),
            carrier_report_url=report_metadata.get("finalDisplayedUrl"),
            total_worksheets=len(worksheets_data)
        )


class UITransformResult:
    """
    Result object for UI transformation that loaders can work with.
    Handles multiple UIPerformanceReport objects.
    """

    def __init__(self, worksheets_data: Dict[str, UIPerformanceReport], report_id: str,
                 report_name: str, carrier_report_url: str = None, total_worksheets: int = 0):
        if not worksheets_data:
            raise ValueError("Worksheets data is required")
        if not report_id:
            raise ValueError("Report ID is required")
        if not report_name:
            raise ValueError("Report name is required")

        self.worksheets_data = worksheets_data
        self.report_id = report_id
        self.report_name = report_name
        self.carrier_report_url = carrier_report_url
        self.total_worksheets = total_worksheets or len(worksheets_data)
