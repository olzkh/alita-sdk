import logging
import json
import traceback
import re
from typing import Type
from langchain_core.tools import BaseTool, ToolException
from pydantic.fields import Field
from pydantic import create_model, BaseModel
from .api_wrapper import CarrierAPIWrapper


logger = logging.getLogger(__name__)


class UpdateUITestScheduleTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "update_ui_test_schedule"
    description: str = ("Add new or update existing UI test schedule on the Carrier platform. "
                        "Provide test_id or test_name, schedule_name, and cron_timer to add new schedule. "
                        "To update existing schedule, use the same schedule_name and optionally provide new_schedule_name, or leave empty to see available tests.")
    args_schema: Type[BaseModel] = create_model(
        "UpdateUITestScheduleInput",
        test_id=(str, Field(default="", description="Test ID to update schedule for")),
        test_name=(str, Field(default="", description="Test name to find and update schedule for (alternative to test_id)")),
        schedule_name=(str, Field(default="", description="Name for the schedule (for new) or existing schedule name to update")),
        new_schedule_name=(str, Field(default="", description="New name for the schedule when updating (optional)")),
        cron_timer=(str, Field(default="", description="Cron expression for schedule timing (e.g., '0 2 * * *')")),
        crontab=(str, Field(default="", description="Alternative name for cron expression (same as cron_timer)")),
        crontab_expression=(str, Field(default="", description="Alternative name for cron expression (same as cron_timer)")),
        active=(bool, Field(default=True, description="Whether the schedule should be active (default: True)")),
    )

    def _run(self, test_id: str = "", test_name: str = "", schedule_name: str = "", new_schedule_name: str = "", cron_timer: str = "", crontab: str = "", crontab_expression: str = "", active: bool = True):
        try:
            # Parameter normalization - handle different parameter names from orchestration engine
            # If test_name is provided but test_id is not, we need to find the test_id by name
            if test_name and not test_id:
                # Find test by name
                ui_tests = self.api_wrapper.get_ui_tests_list()
                for test in ui_tests:
                    if test.get("name", "").strip().lower() == test_name.strip().lower():
                        test_id = str(test.get("id", ""))
                        break
                
                if not test_id:
                    available_tests = []
                    for test in ui_tests:
                        available_tests.append(f"ID: {test.get('id')}, Name: {test.get('name')}")
                    
                    return f"❌ **Test not found for name: {test_name}**\n\n**Available UI tests:**\n" + "\n".join([f"- {test}" for test in available_tests])
            
            # Normalize cron timer parameter (orchestration engine might send different parameter names)
            if not cron_timer:
                if crontab:
                    cron_timer = crontab
                elif crontab_expression:
                    cron_timer = crontab_expression
            
            # Check if no parameters provided - show available tests
            if (not test_id or test_id.strip() == "") and (not schedule_name or schedule_name.strip() == "") and (not cron_timer or cron_timer.strip() == ""):
                return self._show_available_tests_and_instructions()
            
            # Check if test_id is missing but other params provided
            if (not test_id or test_id.strip() == ""):
                return self._show_missing_test_id_message()
            
            # Check if schedule_name or cron_timer is missing
            if (not schedule_name or schedule_name.strip() == "") or (not cron_timer or cron_timer.strip() == ""):
                return self._show_missing_parameters_message(test_id, schedule_name, cron_timer)
            
            # Validate cron timer format
            if not self._validate_cron_timer(cron_timer):
                return self._show_invalid_cron_message(cron_timer)
            
            # Get UI tests list to verify test exists
            ui_tests = self.api_wrapper.get_ui_tests_list()
            test_data = None
            test_id_int = None
            
            # Try to find test by ID
            if test_id.isdigit():
                test_id_int = int(test_id)
                for test in ui_tests:
                    if test.get("id") == test_id_int:
                        test_data = test
                        break
            
            if not test_data:
                available_tests = []
                for test in ui_tests:
                    available_tests.append(f"ID: {test.get('id')}, Name: {test.get('name')}")
                
                return f"❌ **Test not found for ID: {test_id}**\n\n**Available UI tests:**\n" + "\n".join([f"- {test}" for test in available_tests])
            
            # Get detailed test configuration
            test_details = self.api_wrapper.get_ui_test_details(str(test_id_int))
            
            if not test_details:
                return f"❌ **Could not retrieve test details for test ID {test_id_int}.**"
            
            # Check if we're updating an existing schedule by name
            existing_schedule_id = self._find_schedule_by_name(test_details, schedule_name)
            is_update = existing_schedule_id is not None
            
            # Parse and update the test configuration
            updated_config = self._parse_and_update_test_data(test_details, schedule_name, new_schedule_name, cron_timer, existing_schedule_id, active, is_update)
            
            # Execute the PUT request to update the test
            result = self.api_wrapper.update_ui_test(str(test_id_int), updated_config)
            
            operation_name = new_schedule_name if new_schedule_name else schedule_name
            return self._format_success_message(test_data.get('name', 'Unknown'), test_id_int, operation_name, cron_timer, is_update, existing_schedule_id, active)
            
        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error updating UI test schedule: {stacktrace}")
            raise ToolException(stacktrace)
    
    def _find_schedule_by_name(self, test_details: dict, schedule_name: str) -> str:
        """Find schedule ID by schedule name. Returns None if not found."""
        schedules = test_details.get("schedules", [])
        for schedule in schedules:
            if schedule.get("name", "").strip().lower() == schedule_name.strip().lower():
                return str(schedule.get("id", ""))
        return None
    
    def _show_available_tests_and_instructions(self):
        """Show available tests and instructions when no parameters provided."""
        try:
            ui_tests = self.api_wrapper.get_ui_tests_list()
            
            if not ui_tests:
                return "❌ **No UI tests found.**"
            
            message = ["# 📋 Update UI Test Schedule\n"]
            message.append("## Available UI Tests:")
            
            for test in ui_tests:
                test_id = test.get('id')
                test_name = test.get('name')
                
                # Get schedules for this test
                try:
                    test_details = self.api_wrapper.get_ui_test_details(str(test_id))
                    schedules = test_details.get("schedules", []) if test_details else []
                    
                    message.append(f"- **ID: {test_id}**, Name: `{test_name}`, Runner: `{test.get('runner')}`")
                    
                    if schedules:
                        message.append("  **Existing Schedules:**")
                        for schedule in schedules:
                            schedule_id = schedule.get('id', 'N/A')
                            schedule_name = schedule.get('name', 'Unnamed')
                            cron = schedule.get('cron', 'N/A')
                            active = schedule.get('active', False)
                            status = "✅ Active" if active else "❌ Inactive"
                            message.append(f"    - Schedule ID: `{schedule_id}`, Name: `{schedule_name}`, Cron: `{cron}`, Status: {status}")
                    else:
                        message.append("  **No existing schedules**")
                    message.append("")
                except Exception as e:
                    logger.warning(f"Could not fetch schedules for test {test_id}: {e}")
                    message.append(f"- **ID: {test_id}**, Name: `{test_name}`, Runner: `{test.get('runner')}`")
            
            message.append("## 📝 Instructions:")
            message.append("**To ADD a new schedule:**")
            message.append("- **`test_id`** or **`test_name`** - The ID or name of the test")
            message.append("- **`schedule_name`** - A name for your new schedule")
            message.append("- **`cron_timer`** - Cron expression for timing (e.g., `0 2 * * *` for daily at 2 AM)")
            message.append("- **`active`** - Whether schedule should be active (optional, default: true)")
            
            message.append("\n**To UPDATE an existing schedule:**")
            message.append("- **`test_id`** or **`test_name`** - The ID or name of the test")
            message.append("- **`schedule_name`** - The name of the existing schedule to update")
            message.append("- **`new_schedule_name`** - New name for the schedule (optional)")
            message.append("- **`cron_timer`** - New cron expression (optional)")
            message.append("- **`active`** - Whether schedule should be active (optional)")
            
            message.append("\n## 💡 Examples:")
            message.append("**Add new schedule:**")
            message.append("```")
            message.append("test_id: 42")
            message.append("schedule_name: Daily Morning Test")
            message.append("cron_timer: 0 2 * * *")
            message.append("```")
            
            message.append("**Update existing schedule (using test name):**")
            message.append("```")
            message.append("test_name: My_UI_Test")
            message.append("schedule_name: Daily Morning Test")
            message.append("new_schedule_name: Updated Morning Test")
            message.append("cron_timer: 0 14 * * *")
            message.append("active: false")
            message.append("```")
            
            return "\n".join(message)
            
        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error fetching UI tests list: {stacktrace}")
            raise ToolException(stacktrace)
    
    def _show_missing_test_id_message(self):
        """Show message when test_id is missing."""
        return """# ❌ Missing Test ID

**To add new schedule, please provide:**
- **`test_id`** or **`test_name`** - The ID or name of the test you want to add schedule to  
- **`schedule_name`** - A name for your new schedule
- **`cron_timer`** - Cron expression for timing

**To update existing schedule, please provide:**
- **`test_id`** or **`test_name`** - The ID or name of the test
- **`schedule_name`** - The name of the existing schedule to update
- **`new_schedule_name`** - New name (optional)
- **`cron_timer`** - New cron expression (optional)
- **`active`** - Active status (optional)

Use the tool without parameters to see available tests and their schedules."""
    
    def _show_missing_parameters_message(self, test_id: str, schedule_name: str, cron_timer: str):
        """Show message when some parameters are missing."""
        missing = []
        if not schedule_name or schedule_name.strip() == "":
            missing.append("**`schedule_name`**")
        if not cron_timer or cron_timer.strip() == "":
            missing.append("**`cron_timer`**")
        
        message = [f"# ❌ Missing Parameters for Test ID: {test_id}\n"]
        message.append("**Missing parameters:**")
        for param in missing:
            message.append(f"- {param}")
        
        message.append("\n**For adding new schedule, please provide:**")
        message.append("- **`test_id`** or **`test_name`** ✅ (provided)")
        message.append("- **`schedule_name`** - A name for your new schedule")
        message.append("- **`cron_timer`** - Cron expression for timing (e.g., `0 2 * * *`)")
        
        message.append("\n**For updating existing schedule, please provide:**")
        message.append("- **`test_id`** or **`test_name`** ✅ (provided)")
        message.append("- **`schedule_name`** - Name of the existing schedule to update")
        message.append("- **`new_schedule_name`** - New name (optional)")
        message.append("- **`cron_timer`** - New cron expression (optional)")
        message.append("- **`active`** - Active status (optional)")
        
        return "\n".join(message)
    
    def _validate_cron_timer(self, cron_timer: str) -> bool:
        """Validate cron timer format."""
        # Basic cron validation - should have 5 parts separated by spaces
        parts = cron_timer.strip().split()
        if len(parts) != 5:
            return False
        
        # Each part should contain only digits, *, /, -, or ,
        cron_pattern = re.compile(r'^[0-9*,/-]+$')
        return all(cron_pattern.match(part) for part in parts)
    
    def _show_invalid_cron_message(self, cron_timer: str):
        """Show message for invalid cron timer."""
        return f"""# ❌ Invalid Cron Timer Format

**Provided:** `{cron_timer}`

**Cron format should be:** `minute hour day month weekday`

## Valid Examples:
- `0 2 * * *` - Daily at 2:00 AM
- `30 14 * * 1` - Every Monday at 2:30 PM  
- `0 */6 * * *` - Every 6 hours
- `15 10 1 * *` - First day of every month at 10:15 AM
- `0 9 * * 1-5` - Weekdays at 9:00 AM

## Format Rules:
- **Minute:** 0-59
- **Hour:** 0-23
- **Day:** 1-31
- **Month:** 1-12
- **Weekday:** 0-7 (0 and 7 are Sunday)
- Use **`*`** for "any value"
- Use **`,`** for multiple values
- Use **`-`** for ranges
- Use **`/`** for step values"""
    
    def _parse_and_update_test_data(self, get_data: dict, schedule_name: str, new_schedule_name: str, cron_timer: str, schedule_id: str = "", active: bool = True, is_update: bool = False) -> dict:
        """Parse GET response data and transform it into the required format for PUT request."""
        
        # Extract environment and test type from test parameters
        env_type = ""
        test_type = ""
        for param in get_data.get("test_parameters", []):
            if param.get("name") == "env_type":
                env_type = param.get("default", "")
            elif param.get("name") == "test_type":
                test_type = param.get("default", "")
        
        # Construct common_params from GET data
        common_params = {
            "aggregation": get_data.get("aggregation", "max"),
            "cc_env_vars": get_data.get("cc_env_vars", {}),
            "entrypoint": get_data.get("entrypoint", ""),
            "env_type": env_type,
            "env_vars": get_data.get("env_vars", {}),
            "location": get_data.get("location", ""),
            "loops": get_data.get("loops", 1),
            "name": get_data.get("name", ""),
            "parallel_runners": get_data.get("parallel_runners", 1),
            "runner": get_data.get("runner", ""),
            "source": get_data.get("source", {}),
            "test_type": test_type
        }
        
        # Extract only required integrations (reporters and system)
        integrations = {
            "reporters": get_data.get("integrations", {}).get("reporters", {}),
            "system": get_data.get("integrations", {}).get("system", {})
        }
        
        # Process schedules
        schedules = []
        updated_existing = False
        
        # Keep existing schedules and update if needed
        for schedule in get_data.get("schedules", []):
            existing_schedule = {
                "active": schedule.get("active", False),
                "cron": schedule.get("cron", ""),
                "cron_radio": "custom",
                "errors": {},
                "id": schedule.get("id"),
                "name": schedule.get("name", ""),
                "project_id": schedule.get("project_id"),
                "rpc_kwargs": schedule.get("rpc_kwargs"),
                "test_id": schedule.get("test_id"),
                "test_params": schedule.get("test_params", [])
            }
            
            # If we're updating and this is the target schedule
            if is_update and str(schedule.get("id")) == schedule_id:
                # Update with new values
                if new_schedule_name:
                    existing_schedule["name"] = new_schedule_name
                elif schedule_name:
                    existing_schedule["name"] = schedule_name
                if cron_timer:
                    existing_schedule["cron"] = cron_timer
                existing_schedule["active"] = active
                updated_existing = True
            
            schedules.append(existing_schedule)
        
        # If updating but schedule not found
        if is_update and not updated_existing:
            raise ToolException(f"Schedule with name '{schedule_name}' not found in test schedules")
        
        # If adding new schedule (not updating)
        if not is_update:
            new_schedule = {
                "active": active,
                "cron": cron_timer,
                "cron_radio": "custom",
                "errors": {},
                "id": None,  # New schedule, no ID yet
                "name": schedule_name,
                "test_params": []
            }
            schedules.append(new_schedule)
        
        # Assemble the final PUT request data
        put_data = {
            "common_params": common_params,
            "integrations": integrations,
            "run_test": False,
            "schedules": schedules,
            "test_parameters": []  # Empty as required in PUT request
        }
        
        return put_data

    def _format_success_message(self, test_name: str, test_id: int, schedule_name: str, cron_timer: str, is_update: bool = False, schedule_id: str = "", active: bool = True) -> str:
        """Format success message in markdown."""
        operation = "Updated" if is_update else "Added"
        action_desc = "Schedule Updated" if is_update else "New Schedule Added"
        
        message = [f"# ✅ UI Test Schedule {operation} Successfully!"]
        message.append("")
        message.append("## Test Information:")
        message.append(f"- **Test Name:** `{test_name}`")
        message.append(f"- **Test ID:** `{test_id}`")
        message.append("")
        message.append(f"## {action_desc}:")
        if schedule_name:
            message.append(f"- **Schedule Name:** `{schedule_name}`")
        if cron_timer:
            message.append(f"- **Cron Timer:** `{cron_timer}`")
        message.append(f"- **Status:** {'✅ Active' if active else '❌ Inactive'}")
        message.append("")
        message.append("## 🎯 What happens next:")
        if active:
            message.append("The test will now run automatically according to the specified schedule. You can view and manage schedules in the Carrier platform UI.")
            message.append("")
            if cron_timer:
                message.append(f"**Schedule will execute:** Based on cron expression `{cron_timer}`")
        else:
            message.append("The schedule has been deactivated and will not run automatically.")
        
        return "\n".join(message)
