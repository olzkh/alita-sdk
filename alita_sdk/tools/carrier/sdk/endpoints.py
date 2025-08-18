from typing import Dict
import logging

logger = logging.getLogger("carrier_sdk.endpoints")


class _EndpointManager:
    """
    A private helper class to manage and construct API endpoints.
    This is the single source of truth for all API paths, ensuring
    maintainability and consistency (DRY). Its responsibility is

    strictly limited to URL construction.
    """

    def __init__(self, project_id: str):
        self._project_id = project_id
        self._endpoint_templates: Dict[str, str] = {
            # === Backend Performance ===
            'list_reports': f'api/v1/backend_performance/reports/{self._project_id}',
            'get_report': f'api/v1/backend_performance/reports/{self._project_id}',  # Often used with query params
            'add_tag_to_report': f'api/v1/backend_performance/tags/{self._project_id}/{{report_id}}',
            'list_tests': f'api/v1/backend_performance/tests/{self._project_id}',
            'create_test': f'api/v1/backend_performance/tests/{self._project_id}',
            'run_test': f'api/v1/backend_performance/test/{self._project_id}/{{test_id}}',
            # Backend metadata & thresholds
            'list_backend_environments': f'api/v1/backend_performance/environments/{self._project_id}',
            'list_backend_requests': f'api/v1/backend_performance/requests/{self._project_id}',
            'backend_thresholds': f'api/v1/backend_performance/thresholds/{self._project_id}',
            'backend_threshold_by_id': f'api/v1/backend_performance/thresholds/{self._project_id}/{{threshold_id}}',

            # === UI Performance ===
            'list_ui_tests': f'api/v1/ui_performance/tests/{self._project_id}',
            'create_ui_test': f'api/v1/ui_performance/tests/{self._project_id}',
            'ui_test_details': f'api/v1/ui_performance/test/{self._project_id}/{{test_id}}',
            'update_ui_test': f'api/v1/ui_performance/test/{self._project_id}/{{test_id}}',
            'run_ui_test': f'api/v1/ui_performance/test/{self._project_id}/{{test_id}}',
            'cancel_ui_test': f'api/v1/ui_performance/report_status/{self._project_id}/{{report_id}}',
            'list_ui_reports': f'api/v1/ui_performance/reports/{self._project_id}',

            # === Issues (Tickets) ===
            'create_ticket': f'api/v1/issues/issues/{self._project_id}',
            'list_tickets': f'api/v1/issues/issues/{self._project_id}',

            # === Shared Resources & Artifacts ===
            'list_locations': f'api/v1/shared/locations/{self._project_id}',
            'list_integrations': f'api/v1/integrations/integrations/{self._project_id}',
            'list_engagements': f'api/v1/engagements/engagements/{self._project_id}',
            'list_artifacts': f'api/v1/artifacts/artifacts/default/{self._project_id}/{{bucket_name}}',
            'list_artifacts_ui': f'api/v1/ui_performance/results/{self._project_id}/{{uid}}?sort=loop&order=asc',
            'download_artifact': f'api/v1/artifacts/artifact/default/{self._project_id}/{{bucket_name}}/{{file_name}}',
            'upload_artifact': f'api/v1/artifacts/artifacts/{self._project_id}/{{bucket_name}}'
        }

    def build_endpoint(self, key: str, **kwargs) -> str:
        """Builds a full endpoint path from a key and format arguments."""
        if key not in self._endpoint_templates:
            logger.error(f"Developer Error: Endpoint key '{key}' not found in EndpointManager.")
            raise KeyError(f"Endpoint key '{key}' not found in EndpointManager.")

        template = self._endpoint_templates[key]
        return template.format(**kwargs)
