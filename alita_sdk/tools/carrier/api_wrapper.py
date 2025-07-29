import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator, SecretStr

from .sdk.client import CarrierClient, CarrierCredentials, CarrierAPIError
from .reporting.core.data_models import TicketPayload
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

    def get_engagements_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_engagements_list')

    def get_report_file_name(self, report_id: str, extract_to: str = "/tmp") -> str:
        return self._api_call('get_report_file_name', report_id, extract_to)

    def get_report_file_log(self, bucket: str, file_name: str) -> str:
        return self._api_call('get_report_file_log', bucket, file_name)

    def upload_file(self, bucket_name: str, file_name: str) -> bool:
        return self._api_call('upload_file', bucket_name, file_name)

    def get_ui_reports_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_ui_reports_list')

    def get_ui_tests_list(self) -> List[Dict[str, Any]]:
        return self._api_call('get_ui_tests_list')

    def get_locations(self) -> Dict[str, Any]:
        return self._api_call('get_locations')

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

    @staticmethod
    def _clean_html_name(file_name: str) -> str:
        match = re.match(r"(.+?\.html)", file_name)
        return match.group(1) if match else file_name
