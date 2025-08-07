"""
UI Excel Loader

Author: Karen Florykian
"""
import io
import os
import logging
from typing import Dict, Any
from datetime import datetime

from ..etl_pipeline import BaseLoader
from ..reporting.ui_excel_reporter import UIExcelReporter
from langchain_core.tools import ToolException
from alita_sdk.tools.carrier.utils.utils import CarrierArtifactUploader

logger = logging.getLogger(__name__)


def _generate_download_link(api_wrapper, bucket_name: str, file_name: str) -> str:
    """
    Generate download URL using same pattern as Backend loader.
    """
    project_id = api_wrapper._client.credentials.project_id
    base_url = api_wrapper._client.credentials.url.rstrip('/')

    download_url = f"{base_url}/api/v1/artifacts/artifact/default/{project_id}/{bucket_name}/{file_name}?integration_id=1&is_local=False"
    logger.info(f"Generated UI ZIP download link: {download_url}")
    return download_url


class CarrierUIExcelLoader(BaseLoader):
    """
    UI Excel loader that works with the new UITransformResult objects.
    Simplified and follows Backend loader pattern exactly.
    """

    def __init__(self):
        super().__init__()

    def load(self, transformed_data, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and upload UI Excel report."""
        logger.info("Starting UI loader phase...")

        if not transformed_data:
            raise ToolException("Transformed data is required")

        api_wrapper = context.get("api_wrapper")
        if not api_wrapper:
            raise ToolException("API wrapper missing from context")

        # Generate Excel workbook
        logger.info("Generating UI Excel workbook...")
        ui_reporter = UIExcelReporter()
        workbook = ui_reporter.generate_workbook(transformed_data)

        # Convert to bytes
        excel_buffer = io.BytesIO()
        workbook.save(excel_buffer)
        excel_bytes = excel_buffer.getvalue()

        if not excel_bytes:
            raise ToolException("Failed to generate Excel bytes")

        logger.info(f"Generated UI Excel report ({len(excel_bytes)} bytes)")

        # Upload following Backend pattern
        bucket_name, file_name = self._get_upload_details(transformed_data)
        uploader = CarrierArtifactUploader(api_wrapper)

        logger.info(f"Uploading '{file_name}' to bucket '{bucket_name}'...")
        uploader.upload_leg(excel_bytes, bucket_name, file_name)

        # Generate download link
        zip_file_name = f"{os.path.splitext(file_name)[0]}.zip"
        download_url = _generate_download_link(api_wrapper, bucket_name, zip_file_name)

        logger.info("UI loading completed successfully")

        return {
            "status": "success",
            "message": "UI Excel report generated successfully",
            "file_name": zip_file_name,
            "excel_file_name": file_name,
            "bucket_name": bucket_name,
            "download_url": download_url,
            "file_size_bytes": len(excel_bytes),
            "metadata": {
                "report_id": transformed_data.report_id,
                "report_name": transformed_data.report_name,
                "timestamp": datetime.now().isoformat(),
                "worksheets_count": len(transformed_data.worksheets_data),
                "loader_type": "CarrierUIExcelLoader"
            }
        }

    @staticmethod
    def _get_upload_details(transformed_data) -> tuple[str, str]:
        """Extract upload details from transformed data."""
        if not transformed_data.report_name:
            raise ToolException("Report name is required for upload")
        if not transformed_data.report_id:
            raise ToolException("Report ID is required for upload")

        report_name = transformed_data.report_name
        bucket_name = report_name.replace("_", "").replace(" ", "").lower()
        file_name = f"ui_excel_report_{transformed_data.report_id}.xlsx"
        return bucket_name, file_name
