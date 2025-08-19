import logging
import json
from typing import Type, Optional, List, Dict, Union

from datetime import datetime
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from .api_wrapper import CarrierAPIWrapper
from .backend_reports_tool import BaseCarrierTool
from .utils.utils import tool_logger

logger = logging.getLogger(__name__)


# ====================================================================================
# TOOL: GetBackendTestsTool (Restored User-Friendly Output)
# ====================================================================================
class GetTestsInput(BaseModel):
    """Input model for GetBackendTestsTool. Takes no parameters."""
    pass


class GetBackendTestsTool(BaseTool):
    """A tool to fetch a summarized, user-friendly list of all available backend performance tests."""
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "get_backend_tests"
    description: str = "Get a summarized list of all available backend performance tests."
    args_schema: Type[BaseModel] = GetTestsInput

    @tool_logger
    def _run(self):
        """
        This version provides the full, user-friendly list as requested.
        """
        try:
            tests = self.api_wrapper.get_tests_list()
            test_count = len(tests)

            if test_count == 0:
                return "✅ I found no backend tests in this project."

            # Start with the summary header
            summary = f"✅ I found {test_count} backend tests. Here is a preview:\n"

            # List the first 5 tests
            for test in tests[:5]:
                summary += f"- ID: {test.get('id')}, Name: \"{test.get('name')}\", Runner: {test.get('runner')}\n"

            # Add the user-friendly "and X more" message if applicable
            if test_count > 5:
                summary += f"\n💡 ...and {test_count - 5} more. You can ask for a specific test by its ID or run one directly."

            return summary
        except Exception as e:
            logger.error(f"Failed to fetch backend tests: {e}")
            raise ToolException("An error occurred while trying to fetch the list of backend tests.")

    @tool_logger
    def _run_legacy(self):
        try:
            import traceback
            tests = self.api_wrapper.get_tests_list()

            # Fields to keep in each test
            base_fields = {
                "id", "name", "entrypoint", "runner", "location", "job_type", "source"
            }

            trimmed_tests = []
            for test in tests:
                # Keep only desired base fields
                trimmed = {k: test[k] for k in base_fields if k in test}

                # Simplify test_parameters from test_config
                trimmed["test_parameters"] = [
                    {"name": param["name"], "default": param["default"]}
                    for param in test.get("test_parameters", [])
                ]

                trimmed_tests.append(trimmed)

            return json.dumps(trimmed_tests)
        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error getting tests: {stacktrace}")
            raise ToolException(stacktrace)

    @tool_logger  
    def get_tests_with_environments(self) -> str:
        """
        Extended method to get backend tests with their environments for thresholds configuration.
        This replaces the need for a separate ShowBackendTestsAndEnvsTool.
        """
        try:
            tests = self.api_wrapper.get_tests_list()
            if not tests:
                return "❌ No backend tests found."

            lines: List[str] = [
                "🎯 Available backend tests and environments:",
                ""
            ]
            for test in tests:
                name = test.get("name")
                if not name:
                    continue
                try:
                    envs = self.api_wrapper.get_backend_environments(name)
                    envs_str = ", ".join(envs) if envs else "No environments found"
                except Exception as e:
                    logger.warning(f"Failed to fetch envs for {name}: {e}")
                    envs_str = "Failed to fetch"
                lines.append(f"  • {name}: (envs: {envs_str})")

            lines += [
                "",
                "Next: use get_backend_requests to list request names for a test/environment,",
                "or use create_backend_threshold to create a threshold."
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.exception("Failed to list tests and envs")
            raise ToolException(str(e))


# ====================================================================================
# TOOL: GetTestByIDTool (No changes needed, but included for completeness)
# ====================================================================================
class GetTestByIdInput(BaseModel):
    test_id: str = Field(description="The ID of the test to retrieve.")


class GetTestByIDTool(BaseTool):
    """A tool to get the full data for a single backend test from the Carrier platform."""
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "get_test_by_id"
    description: str = "Get detailed test data for a specific backend test from the Carrier platform."
    args_schema: Type[BaseModel] = GetTestByIdInput

    @tool_logger
    def _run(self, test_id: str):
        try:
            tests = self.api_wrapper.get_tests_list()
            test_data = next((test for test in tests if str(test.get("id")) == test_id), None)
            if not test_data:
                raise ToolException(f"Test with ID {test_id} not found.")
            return json.dumps(test_data, indent=2)
        except Exception as e:
            logger.error(f"Error finding test {test_id}: {e}")
            raise ToolException(f"Could not retrieve test {test_id}.")


# ====================================================================================
# TOOL: RunTestByIDTool (Corrected Parameter Overriding Logic)
# ====================================================================================
class RunTestInput(BaseModel):
    """Defines the complete set of arguments for running a test."""
    test_id: int = Field(description="The numeric ID of the test to run.")
    duration: Optional[int] = Field(default=None, description="Optional. Test duration in seconds.")
    users: Optional[int] = Field(default=None, description="Optional. Number of virtual users.")
    location: Optional[str] = Field(default=None, description="Optional. Location to run the test.")


class RunTestByIDTool(BaseTool):
    """
    Runs a backend test by its ID, correctly overriding default
    parameters with any user-provided values.
    """
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "run_test"
    description: str = "Execute a backend performance test plan from the Carrier platform."
    args_schema: Type[BaseModel] = RunTestInput

    @tool_logger
    def _run(self, test_id: int, **kwargs):
        """
        This method now correctly receives all resolved parameters (including overrides)
        and applies them before executing the test.
        """
        try:
            logger.info(f"Attempting to run test {test_id} with provided overrides: {kwargs}")

            # 1. Fetch the complete test data from the API
            tests = self.api_wrapper.get_tests_list()
            test_data = next((t for t in tests if str(t.get("id")) == str(test_id)), None)
            if not test_data:
                raise ToolException(f"Test with id {test_id} not found.")

            # 2. Create a dictionary of the test's default parameters
            default_params_list = test_data.get("test_parameters", [])
            final_params = {p['name']: p['default'] for p in default_params_list}
            logger.debug(f"Default parameters loaded: {final_params}")

            for key, value in kwargs.items():
                if value is not None:
                    logger.info(f"Overriding parameter '{key}': from '{final_params.get(key)}' to '{value}'")
                    final_params[key] = value

            logger.info(f"Final parameters after override: {final_params}")

            # 4. Convert the final parameters back to the list-of-dicts format the API expects
            api_test_parameters = [{"name": k, "default": str(v)} for k, v in final_params.items()]

            # 5. Build the 'common_params' dictionary for the API request body
            loc_ = kwargs.get("location", test_data.get("location", "default"))
            common_params = {
                "name": test_data.get("name"),
                "entrypoint": test_data.get("entrypoint"),
                "runner": test_data.get("runner"),
                "source": test_data.get("source"),
                "test_type": final_params.get("test_type", test_data.get("test_type")),
                "env_type": final_params.get("env_type", test_data.get("env_type")),
                "parallel_runners": test_data.get("parallel_runners"),
                "location": loc_,
                "env_vars": test_data.get("env_vars", {}),
            }
            if "cloud_settings" in kwargs and kwargs["cloud_settings"] is not None:
                common_params["env_vars"]["cloud_settings"] = kwargs["cloud_settings"]

            # 6. Construct the final JSON body for the API call
            json_body = {
                "common_params": common_params,
                "test_parameters": api_test_parameters,
                "integrations": test_data.get("integrations", {})
            }

            # 7. Execute the test via the API wrapper
            report_id = self.api_wrapper.run_test(str(test_id), json_body)
            # Build the correct report URL
            base_url = self.api_wrapper.url.rstrip('/')
            report_url = f"{base_url}/-/performance/backend/results?result_id={report_id}"
            return json.dumps({
                "message": "✅ Test execution started successfully",
                "report_id": report_id,
                "report_url": report_url,
                "test_name": test_data.get("name"),
                "execution_details": {
                    "location": loc_,
                    "parameters_applied": json_body,
                    "estimated_duration": test_data.get("estimated_duration", "unknown")
                }
            }, indent=2)

        except Exception as e:
            logger.exception(f"Critical failure in RunTestByIDTool for test {test_id}")
            raise ToolException(f"Failed to run test {test_id}. Error: {e}")


# ====================================================================================
# TOOL: CreateBackendTestTool (Included for completeness)
# ====================================================================================
class CreateBackendTestInput(BaseModel):
    test_name: str = Field(..., description="Test name")
    test_type: str = Field(..., description="Test type")
    env_type: str = Field(..., description="Env type")
    entrypoint: str = Field(..., description="Entrypoint for the test (JMeter script path or Gatling simulation path)")
    custom_cmd: str = Field(..., description="Custom command line to execute the test")
    runner: str = Field(..., description="Test runner (Gatling or JMeter)")
    source: Optional[Dict[str, Optional[str]]] = Field(None, description="Test source configuration (Git repo)")
    test_parameters: Optional[List[Dict[str, str]]] = Field(None, description="Test parameters")
    email_integration: Optional[Dict[str, Optional[Union[int, List[str]]]]] = Field(None,
                                                                                    description="Email integration configuration")

class CreateBackendTestTool(BaseCarrierTool):
    """⚗️ Creates new performance tests with comprehensive validation."""

    name: str = "create_backend_test"
    description: str = "⚗️ Create a new performance test configuration"
    args_schema: Type[BaseModel] = CreateBackendTestInput

    def _run(self, test_name: str, entrypoint: str, runner: str,
             source: Dict, test_parameters: List[Dict] = None) -> str:
        operation = f"creating test {test_name}"
        start_time = datetime.now()

        self.log_operation_start(
            operation,
            test_name=test_name,
            runner=runner,
            entrypoint=entrypoint,
            param_count=len(test_parameters or [])
        )

        try:
            # Validate runner format
            available_runners = {
                "JMeter_v5.6.3": "v5.6.3",
                "JMeter_v5.5": "v5.5",
                "Gatling_v3.7": "v3.7",
                "Gatling_maven": "maven",
            }

            # Normalize runner
            runner_value = available_runners.get(runner, runner)
            if runner_value not in available_runners.values():
                raise ToolException(f"🔧 Invalid runner '{runner}'. Available: {list(available_runners.keys())}")

            # Build test configuration
            test_config = {
                "common_params": {
                    "name": test_name,
                    "test_type": "default",
                    "env_type": "default",
                    "entrypoint": entrypoint,
                    "runner": runner_value,
                    "source": source,
                    "env_vars": {
                        "cpu_quota": 1,
                        "memory_quota": 4,
                        "cloud_settings": {},
                        "custom_cmd": ""
                    },
                    "parallel_runners": 1,
                    "cc_env_vars": {},
                    "customization": {},
                    "location": "default"
                },
                "test_parameters": test_parameters or [],
                "integrations": {},
                "scheduling": [],
                "run_test": False
            }

            # Create test
            response = self.api_wrapper.create_test(test_config)
            test_info = response.json() if hasattr(response, 'json') else {"id": "created"}

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(operation, duration, test_id=test_info.get('id'))

            return json.dumps({
                "message": f"✅ Test '{test_name}' created successfully",
                "test_id": test_info.get('id'),
                "test_name": test_info.get('name', test_name),
                "runner": runner_value,
                "creation_timestamp": datetime.now().isoformat()
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)

