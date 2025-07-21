import io
import os
import logging
from typing import Dict, Any
from datetime import datetime

from .etl_pipeline import BaseLoader
from ..reporting.core.data_models import PerformanceReport
from ..reporting.excel_reporter import ExcelReporter
from langchain_core.tools import ToolException
from ..utils.utils import CarrierArtifactUploader


class CarrierExcelLoader(BaseLoader):
    """
    🎯 Production-ready Excel loader that generates reports and provides download links.
    Uses existing API wrapper methods following DRY principles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: PerformanceReport, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        💾 Generates Excel report and uploads as ZIP using legacy pattern.
        """
        self.logger.info("Initiating Loader phase...")

        try:
            # Validate context
            api_wrapper = context.get("api_wrapper")
            report_metadata = context.get("report_metadata", {})

            if not api_wrapper:
                raise ToolException("API wrapper missing from context.")

            # Step 1: Generate Excel report in memory
            self.logger.info("📊 Generating Excel report...")
            excel_reporter = ExcelReporter()
            report_workbook = excel_reporter.generate_workbook(transformed_data)

            # Convert to bytes
            excel_buffer = io.BytesIO()
            report_workbook.save(excel_buffer)
            excel_bytes = excel_buffer.getvalue()
            excel_size = len(excel_bytes)


            self.logger.info(f"✅ Successfully rendered Excel report ({excel_size} bytes).")

            # Step 2: Prepare upload details
            bucket_name, file_name = self._get_upload_details(report_metadata)
            temp_file_path = f"/tmp/karen/{file_name}"
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(excel_bytes)

            # Step 3: ✅ FIX: Use legacy ZIP upload method
            uploader = CarrierArtifactUploader(api_wrapper)

            self.logger.info(f"📤 Uploading '{file_name}' as ZIP to bucket '{bucket_name}'...")

            upload_success = uploader.upload_leg(excel_bytes, bucket_name, file_name)

            if not upload_success:
                raise ToolException("Report was generated but upload failed.")

            # Step 4: ✅ FIX: Generate correct ZIP download link
            zip_file_name = f"{os.path.splitext(file_name)[0]}.zip"  # Change .xlsx to .zip
            download_url = self._generate_download_link(api_wrapper, bucket_name, zip_file_name)

            # Step 5: Return comprehensive result
            self.logger.info("✅ Loading phase completed successfully.")

            return {
                "status": "success",
                "message": "Excel report generated and uploaded as ZIP successfully",
                "file_name": zip_file_name,  # ✅ Return ZIP filename
                "excel_file_name": file_name,  # Original Excel filename inside ZIP
                "bucket_name": bucket_name,
                "file_size_bytes": excel_size,
                "download_url": download_url,
                "upload_success": upload_success,
                "metadata": {
                    "report_id": report_metadata.get("id"),
                    "report_name": report_metadata.get("name"),
                    "build_status": getattr(transformed_data, 'build_status', 'Unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "loader_type": "CarrierExcelLoader",
                    "archive_type": "ZIP"  # ✅ Indicate it's zipped
                }
            }

        except Exception as e:
            self.logger.error(f"💥 Loading phase failed: {e}", exc_info=True)
            raise ToolException(f"Excel loading failed: {str(e)}")

    def _get_upload_details(self, metadata: dict) -> tuple[str, str]:
        """
        📋 Extract upload details from metadata using existing patterns.
        """
        # Use same bucket naming as existing code
        bucket_name = metadata.get("name", "default_bucket").replace("_", "").replace(" ", "").lower()

        # Generate filename using existing pattern
        build_id = metadata.get("build_id")
        if not build_id:
            # Fallback to report ID if no build_id
            report_id = metadata.get("id", "unknown")
            build_id = f"report_{report_id}"

        file_name = f"performance_report_{build_id}.xlsx"

        self.logger.debug(f"📁 Upload details: bucket='{bucket_name}', file='{file_name}'")
        return bucket_name, file_name

    def _generate_download_link(self, api_wrapper, bucket_name: str, file_name: str) -> str:
        """
        🔗 Generate download URL using legacy pattern from CarrierClient.
        """
        try:
            if hasattr(api_wrapper, 'carrier_client'):
                project_id = api_wrapper.carrier_client.credentials.project_id
                base_url = api_wrapper.carrier_client.credentials.url.rstrip('/')
            elif hasattr(api_wrapper, '_client'):
                project_id = api_wrapper._client.credentials.project_id
                base_url = api_wrapper._client.credentials.url.rstrip('/')
            else:
                # Fallback to inspect api_wrapper
                self.logger.debug(f"API Wrapper attributes: {dir(api_wrapper)}")
                raise AttributeError("Cannot find client credentials in api_wrapper")

            # ✅ Use exact legacy download URL pattern
            download_url = f"{base_url}/api/v1/artifacts/artifact/default/{project_id}/{bucket_name}/{file_name}?integration_id=1&is_local=False"

            self.logger.info(f"🔗 Generated ZIP download link: {download_url}")
            return download_url

        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate download link: {e}")
            return f"✅ ZIP archive uploaded to Carrier artifacts - Bucket: {bucket_name}, File: {file_name}"


class CarrierPPTXLoader(BaseLoader):
    """
    📊 PowerPoint loader - placeholder for future implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("📊 CarrierPPTXLoader is not yet implemented.")
        raise NotImplementedError("PPTX loading not implemented yet")


class CarrierDocxLoader(BaseLoader):
    """
    📝 Word document loader - placeholder for future implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("📝 CarrierDocxLoader is not yet implemented.")
        raise NotImplementedError("DOCX loading not implemented yet")
