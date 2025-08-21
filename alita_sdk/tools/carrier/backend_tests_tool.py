import logging
import json
from typing import Type, Optional, List, Dict, Union

from datetime import datetime
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, validator

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
    test_id: Optional[str] = Field(default=None, description="The ID of the test to retrieve.")
    test_name: Optional[str] = Field(default=None, description="The name of the test to retrieve.")

    @validator('test_id', pre=True, always=True)
    def validate_test_identifier(cls, v, values):
        """Ensure either test_id or test_name is provided."""
        test_name = values.get('test_name')
        if not v and not test_name:
            raise ValueError("Either test_id or test_name must be provided")
        return v


class GetTestByIDTool(BaseTool):
    """A tool to get the full data for a single backend test from the Carrier platform."""
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "get_test_by_id"
    description: str = "Get detailed test data for a specific backend test from the Carrier platform."
    args_schema: Type[BaseModel] = GetTestByIdInput

    @tool_logger
    def _run(self, test_id: Optional[str] = None, test_name: Optional[str] = None):
        try:
            tests = self.api_wrapper.get_tests_list()
            
            # Find test by ID or name
            test_data = None
            if test_id is not None:
                test_data = next((test for test in tests if str(test.get("id")) == test_id), None)
                if not test_data:
                    raise ToolException(f"Test with ID {test_id} not found.")
            elif test_name is not None:
                test_data = next((test for test in tests if test.get("name") == test_name), None)
                if not test_data:
                    raise ToolException(f"Test with name '{test_name}' not found.")
            else:
                raise ToolException("Either test_id or test_name must be provided.")
            
            return json.dumps(test_data, indent=2)
        except Exception as e:
            logger.error(f"Error finding test: {e}")
            raise ToolException(f"Could not retrieve test.")


# ====================================================================================
# TOOL: RunTestByIDTool (Corrected Parameter Overriding Logic)
# ====================================================================================
class RunTestInput(BaseModel):
    """Defines the complete set of arguments for running a test."""
    test_id: Optional[int] = Field(default=None, description="The numeric ID of the test to run.")
    test_name: Optional[str] = Field(default=None, description="The name of the test to run.")
    duration: Optional[int] = Field(default=None, description="Optional. Test duration in seconds.")
    users: Optional[int] = Field(default=None, description="Optional. Number of virtual users.")
    location: Optional[str] = Field(default=None, description="Optional. Location to run the test.")

    @validator('test_id', pre=True, always=True)
    def validate_test_identifier(cls, v, values):
        """Ensure either test_id or test_name is provided."""
        test_name = values.get('test_name')
        if not v and not test_name:
            raise ValueError("Either test_id or test_name must be provided")
        return v


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
    def _run(self, test_id: Optional[int] = None, test_name: Optional[str] = None, **kwargs):
        """
        This method now correctly receives all resolved parameters (including overrides)
        and applies them before executing the test. Supports both test_id and test_name.
        """
        try:
            # 1. Fetch the complete test data from the API
            tests = self.api_wrapper.get_tests_list()
            
            # 2. Find test by ID or name
            test_data = None
            if test_id is not None:
                logger.info(f"Attempting to run test by ID: {test_id}")
                test_data = next((t for t in tests if str(t.get("id")) == str(test_id)), None)
                if not test_data:
                    raise ToolException(f"Test with ID {test_id} not found.")
            elif test_name is not None:
                logger.info(f"Attempting to run test by name: {test_name}")
                test_data = next((t for t in tests if t.get("name") == test_name), None)
                if not test_data:
                    raise ToolException(f"Test with name '{test_name}' not found.")
            else:
                raise ToolException("Either test_id or test_name must be provided.")
            
            # Log the test being executed
            actual_test_id = test_data.get('id')
            logger.info(f"Found test: ID={actual_test_id}, Name='{test_data.get('name')}'")
            logger.info(f"Provided overrides: {kwargs}")

            # 3. Create a dictionary of the test's default parameters
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

            # 7. Execute the test via the API wrapper - use the actual test ID from test_data
            report_id = self.api_wrapper.run_test(str(actual_test_id), json_body)
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
            logger.exception(f"Critical failure in RunTestByIDTool for test {test_id or test_name}")
            raise ToolException(f"Failed to run test {test_id or test_name}. Error: {e}")


# ====================================================================================
# TOOL: CreateBackendTestTool (Included for completeness)
# ====================================================================================
class CreateBackendTestInput(BaseModel):
    """Flexible input schema that can handle both individual parameters and full JSON configurations."""
    
    # Core parameters for simple usage
    test_name: Optional[str] = Field(None, description="Test name")
    name: Optional[str] = Field(None, description="Test name (alternative field)")
    entrypoint: Optional[str] = Field(None, description="Entrypoint for the test")
    runner: Optional[str] = Field(None, description="Test runner (Gatling or JMeter)")
    source: Optional[Dict] = Field(None, description="Test source configuration")
    test_parameters: Optional[List[Dict]] = Field(None, description="Test parameters")
    
    # Full configuration structure - allows passing complete JSON
    common_params: Optional[Dict] = Field(None, description="Full common_params structure")
    integrations: Optional[Dict] = Field(None, description="Integration settings")
    scheduling: Optional[List] = Field(None, description="Scheduling configuration")
    run_test: Optional[bool] = Field(None, description="Whether to run test immediately")
    
    # Additional individual parameters
    test_type: Optional[str] = Field(None, description="Test type")
    env_type: Optional[str] = Field(None, description="Environment type")
    env_vars: Optional[Dict] = Field(None, description="Environment variables")
    parallel_runners: Optional[int] = Field(None, description="Number of parallel runners")
    location: Optional[str] = Field(None, description="Test location")
    
    class Config:
        extra = "allow"  # Allow additional fields for maximum flexibility

class CreateBackendTestTool(BaseCarrierTool):
    """⚗️ Creates new performance tests with comprehensive validation."""

    name: str = "create_backend_test"
    description: str = "⚗️ Create a new performance test configuration"
    args_schema: Type[BaseModel] = CreateBackendTestInput

    def _run(self, **kwargs) -> str:
        """
        Create a backend test with flexible input handling.
        Can process both individual parameters and full JSON configurations.
        """
        try:
            # Check if this is a full JSON configuration (has common_params)
            if 'common_params' in kwargs and kwargs['common_params']:
                # Use the provided configuration directly
                test_config = {
                    "common_params": kwargs['common_params'],
                    "test_parameters": kwargs.get('test_parameters', []),
                    "integrations": kwargs.get('integrations', {}),
                    "scheduling": kwargs.get('scheduling', []),
                    "run_test": kwargs.get('run_test', False)
                }
                test_name = kwargs['common_params'].get('name', 'Unknown')
            else:
                # Extract individual parameters (handle both 'name' and 'test_name')
                test_name = kwargs.get('test_name') or kwargs.get('name')
                if not test_name:
                    raise ToolException("test_name (or name) is required")
                    
                entrypoint = kwargs.get('entrypoint')
                if not entrypoint:
                    raise ToolException("entrypoint is required")
                    
                runner = kwargs.get('runner') or kwargs.get('test_runner')
                if not runner:
                    raise ToolException("runner is required")
                
                # Handle source parameter mapping (multiple possible names)
                source = kwargs.get('source') or kwargs.get('source_repo_info') or kwargs.get('source_repo') or {}
                
                # Transform source if it has different field names
                if source:
                    if 'repository' in source:
                        # Convert from intent extraction format to API format
                        source = {
                            "name": source.get('type', 'git_https'),
                            "repo": source.get('repository'),
                            "branch": source.get('branch', 'main'),
                            "username": source.get('username', ''),
                            "password": source.get('password', '')
                        }
                    elif 'url' in source:
                        # Convert from alternate intent format to API format
                        source = {
                            "name": "git_https",
                            "repo": source.get('url'),
                            "branch": source.get('branch', 'main'),
                            "username": source.get('username', ''),
                            "password": source.get('password', '')
                        }
                
                # Validate and normalize runner
                available_runners = {
                    "JMeter_v5.6.3": "v5.6.3",
                    "JMeter_v5.5": "v5.5", 
                    "Gatling_v3.7": "v3.7",
                    "Gatling_maven": "maven",
                }
                runner_value = available_runners.get(runner, runner)
                if runner_value not in available_runners.values():
                    raise ToolException(f"🔧 Invalid runner '{runner}'. Available: {list(available_runners.keys())}")
                
                # Handle test_parameters format conversion
                test_parameters = kwargs.get('test_parameters', [])
                
                # Convert different formats to API format
                if isinstance(test_parameters, dict):
                    # Convert from dict format: {'vUsers': 1, 'rampUp': 1} 
                    # to list format: [{'name': 'vUsers', 'default': '1'}, ...]
                    test_parameters = [
                        {
                            'name': key,
                            'default': str(value),
                            'type': 'string',
                            'description': '',
                            'action': ''
                        }
                        for key, value in test_parameters.items()
                    ]
                elif isinstance(test_parameters, list) and test_parameters:
                    # Handle list format - convert 'value' to 'default' if needed
                    for param in test_parameters:
                        if isinstance(param, dict):
                            if 'value' in param and 'default' not in param:
                                param['default'] = str(param.pop('value'))
                            # Ensure required fields exist
                            param.setdefault('type', 'string')
                            param.setdefault('description', '')
                            param.setdefault('action', '')
                
                # Handle environment variables and custom command
                env_vars = kwargs.get('env_vars', {})
                custom_cmd = kwargs.get('custom_cmd', '')
                
                # Extract resource allocation if provided
                resource_allocation = kwargs.get('resource_allocation', {})
                
                # Build env_vars structure
                if not env_vars:
                    env_vars = {
                        "cpu_quota": kwargs.get('cpu_quota') or resource_allocation.get('cpu_quota', 1),
                        "memory_quota": kwargs.get('memory_quota') or resource_allocation.get('memory_quota', 4),
                        "cloud_settings": {},
                        "custom_cmd": custom_cmd
                    }
                elif custom_cmd and 'custom_cmd' not in env_vars:
                    env_vars['custom_cmd'] = custom_cmd
                
                # Build configuration from individual parameters
                test_config = {
                    "common_params": {
                        "name": test_name,
                        "test_type": kwargs.get('test_type', 'default'),
                        "env_type": kwargs.get('env_type', kwargs.get('environment', 'default')),
                        "entrypoint": entrypoint,
                        "runner": runner_value,
                        "source": source,
                        "env_vars": env_vars,
                        "parallel_runners": kwargs.get('parallel_runners') or kwargs.get('number_of_parallel_runners') or resource_allocation.get('parallel_runners', 1),
                        "cc_env_vars": {},
                        "customization": {},
                        "location": kwargs.get('location', 'default')
                    },
                    "test_parameters": test_parameters,
                    "integrations": kwargs.get('integrations', kwargs.get('email_integration', {})),
                    "scheduling": kwargs.get('scheduling', []),
                    "run_test": kwargs.get('run_test', False)
                }

            operation = f"creating test {test_name}"
            start_time = datetime.now()

            self.log_operation_start(
                operation,
                test_name=test_name,
                runner=test_config['common_params'].get('runner', 'unknown'),
                entrypoint=test_config['common_params'].get('entrypoint', 'unknown'),
                param_count=len(test_config.get('test_parameters', []))
            )

            # Create test using the API wrapper
            response = self.api_wrapper.create_test(test_config)
            test_info = response.json() if hasattr(response, 'json') else {"id": "created"}

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(operation, duration, test_id=test_info.get('id'))

            return json.dumps({
                "message": f"✅ Test '{test_name}' created successfully",
                "test_id": test_info.get('id'),
                "test_name": test_info.get('name', test_name),
                "runner": test_config['common_params'].get('runner'),
                "creation_timestamp": datetime.now().isoformat()
            }, indent=2)

        except Exception as e:
            operation = f"creating test"
            self.handle_api_error(operation, e)

