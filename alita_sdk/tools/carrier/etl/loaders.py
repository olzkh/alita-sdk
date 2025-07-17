import io
import logging
from typing import Dict, Any

from .etl_pipeline import BaseLoader
from ..reporting.core.data_models import PerformanceReport
from ..reporting.excel_reporter import ExcelReporter
from langchain_core.tools import ToolException


class CarrierExcelLoader(BaseLoader):
    """
    Takes a final PerformanceReport data object, renders it as an Excel file
    in memory using a dedicated reporter, and uploads it to a Carrier bucket.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: PerformanceReport, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the rendering and uploading of the final report.
        """
        self.logger = logging.getLogger(__name__)  # Ensure logger is available
        self.logger.info("Initiating Loader phase: Rendering and uploading Excel report.")

        api_wrapper = context.get("api_wrapper")
        report_metadata = context.get("report_metadata")
        if not api_wrapper or not report_metadata:
            raise ToolException("Loader failed: API wrapper or report metadata missing from context.")

        try:
            # --- Step 1: Instantiate the SOLID ExcelReporter and call its public method ---
            self.logger.debug("Instantiating ExcelReporter to render the report object.")
            excel_reporter = ExcelReporter()
            print(transformed_data)
            report_wb = excel_reporter.generate_workbook(transformed_data)

            # --- Step 2: Save the workbook to an in-memory byte buffer ---
            final_report_buffer = io.BytesIO()
            report_wb.save(final_report_buffer)
            self._log_report_generation(final_report_buffer.tell())

            # --- Step 3: Prepare upload parameters from metadata ---
            bucket_name, excel_report_name = self._get_upload_details(report_metadata)
            self._log_upload_attempt(bucket_name, excel_report_name)

            # --- Step 4: Upload the bytes to Carrier ---
            success = api_wrapper.upload_report_from_bytes(
                file_bytes=final_report_buffer.getvalue(),
                bucket_name=bucket_name,
                remote_file_name=excel_report_name,
            )
            if not success:
                self.logger.error("Upload to Carrier FAILED.")
                raise ToolException("Loader failed: The final report was generated but failed to upload.")

            self.logger.info("Upload to Carrier SUCCEEDED.")

            # --- Step 5: Return the final result dictionary ---
            report_url = self._get_artifact_url(api_wrapper, bucket_name, excel_report_name)
            return {
                "status": "Success",
                "message": f"Successfully generated and uploaded performance report for report ID {report_metadata.get('id')}.",
                "report_url": report_url
            }
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in the Loader phase: {e}", exc_info=True)
            raise ToolException(f"Loader failed with an unexpected error: {e}")

    def _get_upload_details(self, metadata: dict) -> tuple[str, str]:
        """Extracts and formats upload details from metadata."""
        bucket_name = metadata.get("name", "default_bucket").replace("_", "").replace(" ", "").lower()
        build_id = metadata.get("build_id", "unknown_build")
        return bucket_name, f"performance_report_{build_id}.xlsx"

    def _log_report_generation(self, file_size: int):
        """Logs the result of the in-memory report generation."""
        self.logger.info("Successfully rendered Excel report to an in-memory buffer.")
        self.logger.debug(f"Generated report size: {file_size} bytes.")

    def _log_upload_attempt(self, bucket: str, filename: str):
        """Logs the parameters of the upload attempt."""
        self.logger.info(f"Uploading report to Carrier: bucket='{bucket}', filename='{filename}'.")

    def _get_artifact_url(self, api, bucket, filename):
        """Constructs the final URL for the uploaded artifact."""
        return (f"{api.url.strip('/')}/api/v1/artifacts/artifact/"
                f"{api.project_id}/{bucket}/{filename}")


class CarrierPPTXLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("CarrierPPTXLoader is not yet implemented.")
        raise NotImplementedError


class CarrierDocxLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("CarrierDocxLoader is not yet implemented.")
        raise NotImplementedError
