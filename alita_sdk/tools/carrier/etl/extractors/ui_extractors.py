"""
UI Report Extractors

Author: Karen Florykian
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..etl_pipeline import BaseExtractor
from langchain_core.tools import ToolException

logger = logging.getLogger(__name__)


class CarrierUIReportExtractor(BaseExtractor):
    """
    Extracts UI report metadata and JSON report links following the same pattern
    as CarrierArtifactExtractor for consistency.
    """

    def __init__(self):
        super().__init__()
        logger.info("CarrierUIReportExtractor initialized")

    def extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts UI report metadata and JSON report links from Carrier
        using the provided API wrapper - following Backend extractor pattern.
        """
        logger.info("Starting UI report metadata extraction...")

        # Step 1: Validate context and get required components
        api_wrapper = context.get("api_wrapper")
        if not api_wrapper:
            raise ToolException("API wrapper is required in context")

        report_id = context.get("report_id")
        if not report_id:
            raise ToolException("Report ID is required in context")

        report_id = str(report_id)
        logger.info(f"Extracting UI metadata for report_id: {report_id}")

        # Step 2: Get UI report metadata
        ui_reports = api_wrapper.get_ui_reports_list()
        target_report = self._find_report_by_id(ui_reports, report_id)

        if not target_report:
            raise ToolException(f"UI report with ID {report_id} not found")

        # Step 3: Validate report status and type
        self._validate_report_compatibility(target_report, report_id)

        # Step 4: Get report links using UID
        uid = target_report.get("uid")
        if not uid:
            raise ToolException(f"No UID found for UI report {report_id}")

        report_links = api_wrapper.get_ui_report_links(uid)

        # Step 5: Enhanced validation with specific error messages
        if not report_links:
            self._handle_missing_artifacts(target_report, report_id)

        # Step 6: Convert HTML links to JSON links
        json_links = self._convert_html_to_json_links(report_links)

        if not json_links:
            raise ToolException(f"No valid JSON files found for UI report {report_id}. Found HTML links but conversion failed.")

        logger.info(f"Found {len(json_links)} JSON report files to process")

        # Step 7: Prepare structured output following Backend pattern
        return self._prepare_extraction_result(target_report, json_links, report_id)

    def _validate_report_compatibility(self, report: Dict[str, Any], report_id: str):
        """Validate if the report is compatible with Excel generation."""
        test_status = report.get("test_status", {})
        status = test_status.get("status", "Unknown")
        test_type = report.get("test_type", "")

        # Check if test failed
        if status in ["Failed", "Canceled"]:
            raise ToolException(
                f"UI report {report_id} cannot be processed because test status is '{status}'. "
                f"Only successfully completed tests can generate Excel reports."
            )

        # Check if test is still running
        if status not in ["Finished"]:
            raise ToolException(
                f"UI report {report_id} is not ready for processing. Current status: '{status}'. "
                f"Please wait for the test to complete before generating Excel report."
            )

    def _handle_missing_artifacts(self, report: Dict[str, Any], report_id: str):
        """Handle missing artifacts with detailed error message based on test type."""
        test_status = report.get("test_status", {})
        status = test_status.get("status", "Unknown")
        test_type = report.get("test_type", "Unknown")
        browser = report.get("browser", "Unknown")

        error_details = [
            f"UI report {report_id} has no artifacts available for Excel generation.",
            f"Report Details:",
            f"- Status: {status}",
            f"- Test Type: {test_type}",
            f"- Browser: {browser}",
            f"- UID: {report.get('uid', 'N/A')}"
        ]

        # Provide specific guidance based on test type
        if test_type.lower() in ["lighthouse", "performance"]:
            error_details.extend([
                "",
                "Possible causes for missing Lighthouse artifacts:",
                "1. Test configuration may not include Lighthouse analysis",
                "2. Lighthouse analysis failed during test execution",
                "3. Artifacts were not generated due to test environment issues",
                "4. Report artifacts may have been cleaned up or moved"
            ])
        else:
            error_details.extend([
                "",
                f"Test type '{test_type}' may not generate Lighthouse artifacts.",
                "Only Lighthouse/Performance UI tests can generate Excel reports with performance metrics."
            ])

        error_details.extend([
            "",
            "Recommendations:",
            "1. Verify test configuration includes Lighthouse/Performance analysis",
            "2. Check test logs for artifact generation errors",
            "3. Re-run the test if artifacts should have been generated",
            "4. Contact support if this is a recurring issue"
        ])

        raise ToolException("\n".join(error_details))

    def _find_report_by_id(self, reports: List[Dict], report_id: str) -> Dict[str, Any]:
        """Find report by ID in the reports list."""
        for report in reports:
            if str(report.get("id")) == str(report_id):
                return report
        return None

    def _convert_html_to_json_links(self, html_links: List[str]) -> List[Dict[str, str]]:
        """Convert HTML report links to JSON links with bucket/file info."""
        json_links = []
        for html_url in html_links:
            if not html_url:
                continue

            # Convert HTML URL to JSON URL
            json_url = html_url.replace('.html', '.json')

            # Extract bucket and file name from URL structure
            url_parts = json_url.split('/')
            if len(url_parts) >= 2:
                bucket = url_parts[-2]  # "reports"
                file_name = url_parts[-1]  # actual file name
            else:
                bucket = "reports"
                file_name = json_url.split('/')[-1]

            json_links.append({
                "json_url": json_url,
                "bucket": bucket,
                "file_name": file_name,
                "worksheet_name": self._create_worksheet_name(file_name)
            })

        return json_links

    def _create_worksheet_name(self, json_file_name: str) -> str:
        """Create Excel worksheet name from JSON file name."""
        if not json_file_name:
            raise ValueError("JSON file name cannot be empty")

        # Remove .json extension
        name = json_file_name.replace('.json', '')

        # Replace special characters
        name = name.replace(':', '_').replace('/', '_').replace('\\', '_')
        name = name.replace('[', '_').replace(']', '_').replace('*', '_').replace('?', '_')

        # Extract timestamp part (first 4 underscore-separated parts)
        parts = name.split('_')
        if len(parts) >= 4:
            name = '_'.join(parts[:4])

        # Excel worksheet name limit
        return name[:31] if len(name) > 31 else name

    def _prepare_extraction_result(self, report_metadata: Dict, json_links: List[Dict], report_id: str) -> Dict[str, Any]:
        """
        Prepare structured extraction result following Backend pattern.
        """
        return {
            "report_metadata": report_metadata,
            "json_links": json_links,
            "report_type": "ui_lighthouse",
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "extractor_type": "CarrierUIReportExtractor",
                "json_files_found": len(json_links),
                "report_id": report_id,
                "success": True
            }
        }
