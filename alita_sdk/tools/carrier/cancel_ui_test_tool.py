import logging
import traceback
from typing import Type, Optional
from langchain_core.tools import BaseTool, ToolException
from pydantic.fields import Field
from pydantic import BaseModel
from .api_wrapper import CarrierAPIWrapper

logger = logging.getLogger(__name__)


class CancelUITestInput(BaseModel):
    """Input model for CancelUITestTool."""
    report_id: Optional[str] = Field(default=None, description="The report ID of the UI test to cancel")


class CancelUITestTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "cancel_ui_test"
    description: str = "Cancel a UI test or show available tests to cancel in the Carrier platform."
    args_schema: Type[BaseModel] = CancelUITestInput

    def _run(self, report_id: Optional[str] = None):
        try:
            if report_id:
                # User provided a specific report ID to cancel
                return self._cancel_specific_test(report_id)
            else:
                # User didn't provide ID, show available tests to cancel
                return self._show_cancelable_tests()
                
        except Exception as e:
            logger.error(f"Error in cancel UI test: {e}")
            raise ToolException(f"Failed to process cancel UI test request: {str(e)}")

    def _show_cancelable_tests(self) -> str:
        """Show list of tests that can be canceled."""
        try:
            # Get all UI reports/tests
            reports = self.api_wrapper.get_ui_reports_list()
            
            if not reports:
                return "❌ **No UI tests found.**"
            
            # Filter tests that can be canceled (not in final states)
            final_states = {"Canceled", "Finished", "Failed"}
            cancelable_tests = []
            
            for report in reports:
                test_status = report.get("test_status", {})
                status = test_status.get("status", "Unknown")
                
                # Check if status is not in final states
                if status not in final_states:
                    cancelable_tests.append({
                        "id": report.get("id"),
                        "name": report.get("name", "Unknown"),
                        "status": status,
                        "percentage": test_status.get("percentage", 0),
                        "description": test_status.get("description", "")
                    })
            
            if not cancelable_tests:
                return """# ℹ️ No Tests Available for Cancellation

All UI tests are already in final states (Canceled, Finished, or Failed).

## 🔍 To cancel a specific test:
Use the command: `Cancel UI test <report_id>`

Example: `Cancel UI test 12345`"""
            
            # Build the response message
            response = """# 🚫 UI Tests Available for Cancellation

The following tests are currently running and can be canceled:

## 📋 Active Tests:
"""
            
            for test in cancelable_tests:
                response += f"""
### 🔸 Report ID: `{test['id']}`
- **Name:** `{test['name']}`
- **Status:** `{test['status']}`
- **Progress:** {test['percentage']}%
- **Description:** {test['description']}
"""
            
            response += """
## 🚫 To cancel a specific test:
Use the command: `Cancel UI test <report_id>`

Example: `Cancel UI test 12345`"""
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching cancelable tests: {e}")
            return f"❌ **Error fetching tests:** {str(e)}"

    def _cancel_specific_test(self, report_id: str) -> str:
        """Cancel a specific UI test by report ID."""
        try:
            # First, get the current status of the test
            reports = self.api_wrapper.get_ui_reports_list()
            target_test = None
            
            for report in reports:
                if str(report.get("id")) == report_id:
                    target_test = report
                    break
            
            if not target_test:
                return f"❌ **Test with report ID `{report_id}` not found.**"
            
            # Check if test can be canceled
            test_status = target_test.get("test_status", {})
            current_status = test_status.get("status", "Unknown")
            final_states = {"Canceled", "Finished", "Failed"}
            
            if current_status in final_states:
                return f"""# ❌ Cannot Cancel Test

## Test Information:
- **Report ID:** `{report_id}`
- **Name:** `{target_test.get('name', 'Unknown')}`
- **Current Status:** `{current_status}`

## 🚫 Reason:
This test cannot be canceled because it is already in a final state (`{current_status}`).

Only tests with status **not** in `Canceled`, `Finished`, or `Failed` can be canceled."""
            
            # Attempt to cancel the test
            try:
                cancel_response = self.api_wrapper.cancel_ui_test(report_id)
                
                return f"""# ✅ UI Test Canceled Successfully!

## Test Information:
- **Report ID:** `{report_id}`
- **Name:** `{target_test.get('name', 'Unknown')}`
- **Previous Status:** `{current_status}`
- **New Status:** `Canceled`

## 🎯 Result:
The test has been successfully canceled and will stop executing."""
                
            except Exception as cancel_error:
                logger.error(f"Error canceling test {report_id}: {cancel_error}")
                return f"""# ❌ Failed to Cancel Test

## Test Information:
- **Report ID:** `{report_id}`
- **Name:** `{target_test.get('name', 'Unknown')}`
- **Current Status:** `{current_status}`

## 🚫 Error:
{str(cancel_error)}

Please check the report ID and try again."""
                
        except Exception as e:
            logger.error(f"Error canceling specific test {report_id}: {e}")
            return f"❌ **Error processing cancellation for test `{report_id}`: {str(e)}**"
