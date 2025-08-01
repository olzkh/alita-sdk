"""
UI Excel Report Tool
Author: Karen Florykian
"""
import logging
import traceback
from typing import Type
from langchain_core.tools import BaseTool, ToolException
from pydantic.fields import Field
from pydantic import create_model, BaseModel
from .api_wrapper import CarrierAPIWrapper
from .etl.etl_factory import ETLComponentFactory

logger = logging.getLogger(__name__)


class CreateUIExcelReportTool(BaseTool):
    """
    UI Excel Report Tool using ETL Framework.

    1. Uses ETL Factory to get configured pipeline
    2. Delegates all processing to ETL components
    3. Maintains consistent error handling and logging
    4. Provides same user experience with better maintainability
    """

    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "create_excel_report_ui"
    description: str = "Create Excel report from UI test results JSON files from the Carrier platform using ETL framework."
    args_schema: Type[BaseModel] = create_model(
        "CreateUIExcelReportInput",
        report_id=(str, Field(default="", description="UI Report ID to generate Excel report for")),
    )

    def _run(self, report_id: str = ""):
        """
        Main execution method now using ETL framework following Backend pattern.
        """
        try:
            # Step 1: Validate input (same as before)
            if not report_id or report_id.strip() == "":
                return self._missing_input_response()

            # Step 2: Check if report exists (same validation as before)
            if not self._validate_report_exists(report_id):
                return self._show_available_reports_message()

            # Step 3: Use ETL Framework (NEW - following Backend pattern)
            return self._process_using_etl_framework(report_id)

        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error creating UI Excel report: {stacktrace}")
            raise ToolException(stacktrace)

    def _validate_report_exists(self, report_id: str) -> bool:
        """
        Validate that the report exists before processing.
        Separated for clarity and reusability.
        """
        try:
            ui_reports = self.api_wrapper.get_ui_reports_list()
            return any(str(report.get("id")) == str(report_id) for report in ui_reports)
        except Exception as e:
            logger.error(f"Failed to validate report existence: {e}")
            return False

    def _process_using_etl_framework(self, report_id: str) -> str:
        """
        Process UI report using ETL framework - following Backend pattern exactly.
        """
        try:
            # Step 1: Get ETL pipeline from factory
            logger.info(f"Getting UI ETL pipeline for report {report_id}")
            etl_pipeline = ETLComponentFactory.get_pipeline("lighthouse_to_excel")

            # Step 2: Prepare context (same as Backend tools)
            context = {
                "api_wrapper": self.api_wrapper,
                "report_id": report_id,
                "user_args": {},  # Can be extended later for user preferences
            }

            # Step 3: Execute ETL pipeline
            logger.info("Executing UI ETL pipeline...")
            result = etl_pipeline.run(context)

            # Step 4: Format success response (following Backend pattern)
            return self._format_success_response(result)

        except ToolException as e:
            # Handle ETL-specific errors with detailed user guidance
            return self._format_detailed_error_response(str(e), report_id)
        except Exception as e:
            logger.error(f"ETL pipeline execution failed: {e}", exc_info=True)
            return self._format_generic_error_response(str(e), report_id)

    def _format_success_response(self, result: dict) -> str:
        """
        Format success response following Backend tool pattern.
        """
        return f"""# ✅ UI Excel Report Generated Successfully!

## Report Information:
- **Report ID:** `{result.get('metadata', {}).get('report_id', 'N/A')}`
- **Report Name:** `{result.get('metadata', {}).get('report_name', 'N/A')}`
- **Worksheets Created:** `{result.get('metadata', {}).get('worksheets_count', 'N/A')}`
- **Excel File:** `{result.get('excel_file_name', 'N/A')}`
- **Bucket:** `{result.get('bucket_name', 'N/A')}`
- **File Size:** `{result.get('file_size_bytes', 0)} bytes`

## 📥 Download Link:
[Download UI Excel Report]({result.get('download_url', 'N/A')})

## 🎯 What's included:
- Multiple worksheets for each JSON report file
- Lighthouse performance metrics formatted for analysis
- Conditional formatting for easy identification of performance issues
- Color-coded thresholds (Green: Good, Yellow: Warning, Red: Poor)

## 📊 Performance Metrics:
- First Contentful Paint, Speed Index, Interactive Time
- Total Blocking Time, Largest Contentful Paint
- Cumulative Layout Shift, Network Requests
- JavaScript Execution Time, Time to First Byte"""

    def _format_detailed_error_response(self, error_message: str, report_id: str) -> str:
        """
        Format detailed error response for ToolException errors.
        """
        # Check if this is an artifact-related error
        if "no artifacts available" in error_message.lower() or "no artifact links" in error_message.lower():
            return f"""# ❌ UI Excel Report Generation Failed - No Artifacts

## Problem:
Report **{report_id}** does not have the required Lighthouse artifacts to generate an Excel report.

## Error Details:
```
{error_message}
```

## 🔍 Why This Happens:
1. **Test Configuration:** Test may not be configured to generate Lighthouse reports
2. **Test Type:** Only Lighthouse/Performance UI tests generate analyzable artifacts  
3. **Test Failure:** Lighthouse analysis may have failed during test execution
4. **Artifact Issues:** Generated artifacts may be missing or corrupted

## 💡 Solutions:

### Immediate Actions:
1. **Verify Test Type:** Ensure this is a Lighthouse/Performance UI test
2. **Check Test Configuration:** Confirm Lighthouse analysis is enabled
3. **Review Test Logs:** Look for artifact generation errors during test execution

### Alternative Reports:
{self._get_compatible_reports_list()}

### Re-run Test:
If this should generate artifacts, consider re-running the test with Lighthouse enabled.

## 📞 Need Help?
Contact support if you believe this test should generate Excel reports."""

        # Handle other extraction errors
        return f"""# ❌ UI Excel Report Generation Failed

## Problem:
Unable to process report **{report_id}** for Excel generation.

## Error Details:
```
{error_message}
```

## 🔍 Troubleshooting Steps:
1. Verify the report ID is correct: `{report_id}`
2. Check if the test has completed successfully
3. Ensure the test type supports Excel report generation
4. Try again in a few minutes if the test just completed

## 📋 Available Reports:
{self._get_recent_reports_list()}"""

    def _format_generic_error_response(self, error_message: str, report_id: str) -> str:
        """
        Format generic error response for unexpected errors.
        """
        return f"""# ❌ UI Excel Report Generation Failed - System Error

## Problem:
An unexpected error occurred while processing report **{report_id}**.

## Error Details:
```
{error_message}
```

## 🔍 Immediate Actions:
1. Try the request again in a few minutes
2. Verify the report ID: `{report_id}`
3. Check system status and connectivity

## 📋 Recent Reports:
{self._get_recent_reports_list()}

## 📞 Support:
If this error persists, please contact support with:
- Report ID: `{report_id}`
- Timestamp: {self._get_timestamp()}
- Error details above"""

    def _get_compatible_reports_list(self) -> str:
        """
        Get list of reports that are likely to be compatible with Excel generation.
        """
        try:
            ui_reports = self.api_wrapper.get_ui_reports_list()
            compatible_reports = []

            for report in ui_reports[:10]:
                test_type = report.get("test_type", "").lower()
                status = report.get("test_status", {}).get("status", "")

                # Look for Lighthouse/Performance tests that finished successfully
                if ("lighthouse" in test_type or "performance" in test_type) and status == "Finished":
                    compatible_reports.append({
                        "id": report.get("id"),
                        "name": report.get("name", "Unnamed"),
                        "test_type": report.get("test_type", "Unknown"),
                        "start_time": report.get("start_time", "")
                    })

            if not compatible_reports:
                return "No compatible Lighthouse/Performance reports found."

            report_list = []
            for report in compatible_reports:
                report_list.append(
                    f"- **ID: {report['id']}** - {report['name']} ({report['test_type']}) - {report['start_time']}"
                )

            return "\n".join(report_list)

        except Exception:
            return "Could not retrieve compatible reports list."

    def _get_recent_reports_list(self) -> str:
        """
        Get list of recent reports for troubleshooting.
        """
        try:
            ui_reports = self.api_wrapper.get_ui_reports_list()
            if not ui_reports:
                return "No UI reports found."

            report_list = []
            for report in ui_reports[:5]:  # Show first 5
                report_id = report.get("id")
                report_name = report.get("name", "Unnamed Report")
                test_status = report.get("test_status", {}).get("status", "Unknown")
                test_type = report.get("test_type", "Unknown")
                start_time = report.get("start_time", "")
                report_list.append(f"- **ID: {report_id}** - {report_name} ({test_type}, {test_status}) - {start_time}")

            return "\n".join(report_list)

        except Exception:
            return "Could not retrieve report list."

    def _get_timestamp(self) -> str:
        """Get current timestamp for error reporting."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _missing_input_response(self):
        """Response when report_id is missing - same as before."""
        return "Please provide me test id for generating excel report from your UI test"

    def _show_available_reports_message(self):
        """Show available reports when no matching report_id found - same as before."""
        try:
            ui_reports = self.api_wrapper.get_ui_reports_list()

            if not ui_reports:
                return "❌ **No UI test reports found.**"

            message = ["# ❌ No report found for the specified report ID\n"]
            message.append("## Available Report IDs:")

            for report in ui_reports[:10]:  # Show first 10 reports
                report_id = report.get("id")
                report_name = report.get("name", "Unnamed Report")
                test_status = report.get("test_status", {}).get("status", "Unknown")
                test_type = report.get("test_type", "Unknown")
                start_time = report.get("start_time", "")

                message.append(f"- **Report ID: {report_id}** - {report_name} ({test_type}, {test_status}) - {start_time}")

            if len(ui_reports) > 10:
                message.append(f"... and {len(ui_reports) - 10} more reports")

            message.append("\n## 💡 Example:")
            message.append("```")
            message.append(f"report_id: {ui_reports[0].get('id') if ui_reports else 'YOUR_REPORT_ID'}")
            message.append("```")
            return "\n".join(message)

        except Exception:
            return "❌ **Error retrieving available report IDs. Please check your report_id and try again.**"
