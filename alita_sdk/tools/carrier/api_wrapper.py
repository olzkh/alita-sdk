import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator, SecretStr

from .sdk.client import CarrierClient, CarrierCredentials, CarrierAPIError
from .etl.reporting.core.data_models import TicketPayload
from .sdk.data_models import ReportRequest, ReportType

logger = logging.getLogger(__name__)


class CarrierAPIWrapper(BaseModel):
    url: str = Field(..., description="Carrier API Base URL")
    organization: str = Field(..., description="Organization identifier")
    private_token: SecretStr = Field(..., description="API authentication token")
    project_id: str = Field(..., description="Carrier Project ID")

    _client: Optional[CarrierClient] = None

    @model_validator(mode='after')
    def initialize_client(self):
        try:
            credentials = CarrierCredentials(
                url=self.url,
                token=self.private_token.get_secret_value(),
                organization=self.organization,
                project_id=self.project_id
            )
            self._client = CarrierClient(credentials=credentials)
            logger.info("Carrier client initialized successfully.")
        except Exception as e:
            logger.exception("Carrier client initialization failed.")
            raise e
        return self

    def _api_call(self, method_name: str, *args, **kwargs):
        """Generalized method to handle API calls with unified logging and error handling."""
        method = getattr(self._client, method_name)
        logger.info(f"API args '{args}'.")
        try:
            response = method(*args, **kwargs)
            logger.info(f"API call '{method_name}' succeeded.")
            return response
        except CarrierAPIError as e:
            logger.error(f"Carrier API error during '{method_name}': {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during '{method_name}'.")
            raise

    # API wrapper methods (delegated to CarrierClient)
    def fetch_tickets(self, board_id: str) -> List[Dict[str, Any]]:
        return self._api_call('fetch_tickets', board_id)

    def create_ticket(self, ticket_payload: TicketPayload) -> Dict[str, Any]:
        return self._api_call('create_ticket', ticket_payload)

    def fetch_test_data(self, start_time: str) -> List[Dict[str, Any]]:
        return self._api_call('fetch_test_data', start_time)

    def get_reports_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_reports_list')

    def get_tests_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_tests_list')

    def create_test(self, data: dict) -> Dict[str, Any]:
        return self._api_call('create_test', data)

    def get_integrations(self, name: str) -> List[Dict[str, Any]]:
        return self._api_call('get_integrations', name)

    def get_available_locations(self) -> Dict[str, Any]:
        return self._api_call('get_available_locations')

    def run_test(self, test_id: str, json_body: dict) -> Dict[str, Any]:
        return self._api_call('run_test', test_id, json_body)

    def run_ui_test(self, test_id: str, json_body: dict) -> Dict[str, Any]:
        return self._api_call('run_ui_test', test_id, json_body)

    def add_tag_to_report(self, report_id: str, tag_name: str) -> Dict[str, Any]:
        """Add a tag to a backend performance report."""
        return self._api_call('add_tag_to_report', report_id, tag_name)

    def get_engagements_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_engagements_list')

    def get_report_file_log(self, bucket: str, file_name: str) -> str:
        return self._api_call('get_report_file_log', bucket, file_name)

    def upload_file(self, bucket_name: str, file_name: str) -> bool:
        return self._api_call('upload_file', bucket_name, file_name)

    def get_ui_report_links(self, report_uid: str) -> List[str]:
        """Gets the list of artifact links for a specific UI report UID."""
        return self._api_call('get_ui_report_links', report_uid)

    def download_raw_from_url(self, url: str) -> str:
        """Downloads raw content from a provided URL."""
        return self._api_call('download_raw_from_url', url)

    def get_ui_reports_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_ui_reports_list')

    def get_ui_tests_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_ui_tests_list')

    def upload_report_from_bytes(self, file_bytes: bytes, bucket_name: str,
                                 remote_file_name: str) -> bool:
        """
        Upload file from bytes to Carrier storage.
        This is a wrapper that uses the existing upload_file method.
        """
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, remote_file_name)

        try:
            with open(temp_file_path, 'wb') as f:
                f.write(file_bytes)
            success = self.upload_file(
                bucket_name=bucket_name,
                file_name=temp_file_path
            )
            return success

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def get_locations(self) -> Dict[str, Any]:
        """Get list of available locations/cloud settings from the Carrier platform."""
        # Delegate to the correct client method name on CarrierClient
        return self._api_call('get_available_locations')

    def update_ui_test(self, test_id: str, json_body: dict) -> Dict[str, Any]:
        return self._api_call('update_ui_test', test_id, json_body)

    def get_ui_test_details(self, test_id: str) -> Dict[str, Any]:
        return self._api_call('get_ui_test_details', test_id)

    def create_ui_test(self, json_body: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_call('create_ui_test', json_body)

    def cancel_ui_test(self, test_id: str) -> Dict[str, Any]:
        return self._api_call('cancel_ui_test', test_id)

    def validate_report_request(self, request: ReportRequest) -> str:
        reports = (self.get_reports_list() if request.report_type == ReportType.backend else self.get_ui_reports_list())
        report_ids = {str(report["id"]) for report in reports}
        if str(request.report_id) not in report_ids:
            alternative_reports = (
                self.get_ui_reports_list() if request.report_type == ReportType.backend else self.get_reports_list())
            alternative_ids = {str(report["id"]) for report in alternative_reports}
            if str(request.report_id) in alternative_ids:
                return (f"⚠️ ID '{request.report_id}' found as {request.report_type.value} report, "
                        f"but you requested {request.report_type.value}. Please confirm your intent.")
            return f"❌ Report ID '{request.report_id}' not found. Please verify the ID."
        return f"✅ Report ID '{request.report_id}' verified as {request.report_type.value} report. Proceeding..."

    def list_artifacts(self, bucket_name: str) -> Dict[str, Any]:
        """Lists all artifacts (files) within a specified storage bucket."""
        return self._api_call('list_artifacts', bucket_name)

    def get_report_metadata(self, report_id: str) -> Dict[str, Any]:
        """
        Provides a clean, direct interface for fetching a single report's metadata.
        This delegates to the get_report_info method on the client.
        """
        logger.info(f"Wrapper: Fetching metadata for report_id: {report_id}")
        return self._api_call('get_report_info', report_id)

    def process_report_artifacts(self, report_id: str, extract_to: str = "/tmp") -> Dict[str, Any]:
        """
        Orchestrates the downloading, unzipping, and merging of report artifacts.
        This method calls the complex logic in the client but exposes it as a single,
        clear action.
        Returns a dictionary with the report metadata and local paths to the processed logs.
        """
        logger.info(f"Wrapper: Processing artifacts for report_id: {report_id}")
        try:
            report_info, test_log_path, errors_log_path = self._api_call(
                'get_report_file_name', report_id, extract_to
            )

            return {
                "metadata": report_info,
                "test_log_path": test_log_path,
                "error_log_path": errors_log_path
            }
        except Exception as e:
            logger.error(f"Wrapper: Failed to process artifacts for report {report_id}: {e}", exc_info=True)
            raise

    def download_artifact(self, bucket_name: str, file_name: str, extract_to: str) -> str:
        """
        Downloads a single artifact to a specified local directory.
        Returns the full path to the downloaded file.
        """
        return self._api_call('download_artifact_to_file', bucket_name, file_name, extract_to)

    @staticmethod
    def _clean_html_name(file_name: str) -> str:
        match = re.match(r"(.+?\.html)", file_name)
        return match.group(1) if match else file_name

    # =============================
    # Backend metadata & thresholds
    # =============================
    def get_backend_environments(self, test_name: str) -> List[str]:
        return self._api_call('get_backend_environments', test_name)

    def get_backend_requests(self, test_name: str, environment: str) -> List[str]:
        return self._api_call('get_backend_requests', test_name, environment)

    def create_backend_threshold(self, threshold_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_call('create_backend_threshold', threshold_data)

    def get_backend_thresholds(self) -> Dict[str, Any]:
        return self._api_call('get_backend_thresholds')

    def delete_backend_threshold(self, threshold_id: str) -> Dict[str, Any]:
        return self._api_call('delete_backend_threshold', threshold_id)

    def update_backend_threshold(self, threshold_id: str, threshold_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._api_call('update_backend_threshold', threshold_id, threshold_data)
