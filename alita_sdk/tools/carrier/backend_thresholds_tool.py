import logging
import json
import traceback
from typing import Type, Optional, List, Dict, Union
from langchain_core.tools import BaseTool, ToolException
from pydantic.fields import Field
from pydantic import create_model, BaseModel
from .api_wrapper import CarrierAPIWrapper


logger = logging.getLogger(__name__)


class SetBackendThresholdsTool(BaseTool):
    """
    Interactive tool for setting backend performance thresholds in Carrier platform.
    
    This tool automatically displays available tests with their environments and 
    guides the user through creating thresholds.
    """
    
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "set_backend_thresholds"
    description: str = (
        "Set backend performance thresholds for tests in the Carrier platform. "
        "This tool will automatically show available tests with their environments and guide you through creating thresholds."
    )
    args_schema: Type[BaseModel] = create_model(
        "SetBackendThresholdsInput",
        step=(str, Field(default="show_tests_and_envs", description="Current step: 'show_tests_and_envs', 'get_requests', or 'create_threshold'")),
        test_name=(Optional[str], Field(default=None, description="Selected test name")),
        environment=(Optional[str], Field(default=None, description="Selected environment")),
        threshold_config=(Optional[Dict], Field(default=None, description="Threshold configuration object with all required fields"))
    )

    def _run(self, step: str = "show_tests_and_envs", test_name: Optional[str] = None, 
             environment: Optional[str] = None, threshold_config: Optional[Dict] = None):
        try:
            if step == "show_tests_and_envs":
                return self._show_tests_and_environments()
            
            elif step == "get_requests":
                if not test_name or not environment:
                    return "Error: both test_name and environment are required for getting request names"
                return self._get_request_names(test_name, environment)
            
            elif step == "create_threshold":
                if not threshold_config:
                    return self._get_threshold_template()
                return self._create_threshold(threshold_config, test_name, environment)
            
            else:
                return "Error: Invalid step. Valid steps are: 'show_tests_and_envs', 'get_requests', 'create_threshold'"
                
        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error in SetBackendThresholdsTool: {stacktrace}")
            raise ToolException(stacktrace)

    def _show_tests_and_environments(self) -> str:
        """Get all available tests and their environments automatically."""
        try:
            # Get all tests
            tests = self.api_wrapper.get_tests_list()
            
            if not tests:
                return "No backend tests found in this project."
            
            # For each test, get its environments
            tests_with_envs = []
            for test in tests:
                test_name = test.get("name")
                if test_name:
                    try:
                        environments = self.api_wrapper.get_backend_environments(test_name)
                        envs_str = ", ".join(environments) if environments else "No environments found"
                        tests_with_envs.append(f"    {test_name}: (envs: {envs_str})")
                    except Exception as e:
                        logger.warning(f"Failed to get environments for test {test_name}: {e}")
                        tests_with_envs.append(f"    {test_name}: (envs: Failed to fetch)")
            
            # Build the response
            result_parts = [
                "To create a new backend performance threshold, here are the available options:",
                "",
                "1. **Test Names**: Available test names and environments"
            ]
            result_parts.extend(tests_with_envs)
            result_parts.extend([
                "",
                "2. **Threshold Configuration**: This includes:",
                "   - **scope**: 'all', 'every', or specific request name",
                "   - **target**: 'response_time', 'throughput', or 'error_rate'", 
                "   - **aggregation**: 'max', 'min', 'avg', 'pct95', or 'pct50'",
                "   - **comparison**: See comparison types below",
                "   - **value**: Threshold value (numeric)",
                "",
                "**Step: Choose a Comparison Type**",
                "Select how you want to compare values. You can choose from:",
                "",
                "- `>` — greater than",
                "- `>=` — greater than or equal to",
                "- `<` — less than",
                "- `<=` — less than or equal to",
                "- `==` — equal to",
                "",
                "**Next steps:**",
                "- To see available request names for a specific test and environment, use:",
                "  step='get_requests' with test_name='<TEST_NAME>' and environment='<ENV_NAME>'",
                "- To create a threshold directly, use:",
                "  step='create_threshold' with a complete threshold_config object"
            ])
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error getting tests and environments: {e}")
            return f"Error retrieving tests and environments: {str(e)}"

    def _get_request_names(self, test_name: str, environment: str) -> str:
        """Get available request names for a specific test and environment."""
        try:
            requests = self.api_wrapper.get_backend_requests(test_name, environment)
            
            result = {
                "message": f"Available request names for test '{test_name}' in environment '{environment}':",
                "requests": requests if requests else ["No specific requests found - you can use 'all' or 'every' scope"],
                "next_step": "Now you can create a threshold using step='create_threshold' with a threshold_config object",
                "threshold_template": {
                    "test": test_name,
                    "environment": environment,
                    "scope": "all",  # Can be "all", "every", or specific request name
                    "target": "response_time",  # Can be "response_time", "throughput", "error_rate"
                    "aggregation": "pct95",  # Can be "max", "min", "avg", "pct95", "pct50"
                    "comparison": "gte",  # Can be "gt", "gte", "lt", "lte", "eq", "neq"
                    "value": 1000  # Threshold value
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error getting request names: {e}")
            return f"Error retrieving request names for test '{test_name}' in environment '{environment}': {str(e)}"

    def _get_threshold_template(self) -> str:
        """Return threshold configuration template with available tests and environments."""
        try:
            # Get all tests and their environments
            tests = self.api_wrapper.get_tests_list()
            tests_with_envs = []
            
            if tests:
                for test in tests:
                    test_name = test.get("name")
                    if test_name:
                        try:
                            environments = self.api_wrapper.get_backend_environments(test_name)
                            envs_str = ", ".join(environments) if environments else "No environments found"
                            tests_with_envs.append(f"    {test_name}: (envs: {envs_str})")
                        except Exception as e:
                            logger.warning(f"Failed to get environments for test {test_name}: {e}")
                            tests_with_envs.append(f"    {test_name}: (envs: Failed to fetch)")
            
            # Build the template response
            template_parts = [
                "Please provide a threshold_config object with the following structure:",
                "",
                "**Available Tests and Environments:**"
            ]
            
            if tests_with_envs:
                template_parts.extend(tests_with_envs)
            else:
                template_parts.append("    No tests found in this project")
            
            template_parts.extend([
                "",
                "**Threshold Configuration Template:**",
                "```json",
                "{",
                '  "test": "<TEST_NAME>",           // Use one of the test names listed above',
                '  "environment": "<ENV_NAME>",     // Use one of the environments for your selected test',
                '  "scope": "all",                  // Can be "all", "every", or specific request name',
                '  "target": "response_time",       // Can be "response_time", "throughput", "error_rate"',
                '  "aggregation": "pct95",          // Can be "max", "min", "avg", "pct95", "pct50"',
                '  "comparison": ">=",              // See comparison types below',
                '  "value": 1000                    // Threshold value (numeric)',
                "}",
                "```",
                "",
                "**Step: Choose a Comparison Type**",
                "Select how you want to compare values. You can choose from:",
                "",
                "- `>` — greater than",
                "- `>=` — greater than or equal to", 
                "- `<` — less than",
                "- `<=` — less than or equal to",
                "- `==` — equal to",
                "",
                "**Valid Values:**",
                "- **scope**: ['all', 'every', '<specific_request_name>']",
                "- **target**: ['response_time', 'throughput', 'error_rate']", 
                "- **aggregation**: ['max', 'min', 'avg', 'pct95', 'pct50']",
                "",
                "**Alternative Usage:**",
                "You can also provide test_name and environment as separate parameters:",
                "step='create_threshold', test_name='<TEST_NAME>', environment='<ENV_NAME>', threshold_config={...}"
            ])
            
            return "\n".join(template_parts)
            
        except Exception as e:
            logger.error(f"Error getting threshold template: {e}")
            # Fallback template without dynamic data
            fallback_template = {
                "message": "Please provide a threshold_config object with the following structure:",
                "template": {
                    "test": "<TEST_NAME>",
                    "environment": "<ENV_NAME>",
                    "scope": "all",
                    "target": "response_time",
                    "aggregation": "pct95",
                    "comparison": ">=",
                    "value": 1000
                },
                "comparison_types": {
                    "title": "Step: Choose a Comparison Type",
                    "description": "Select how you want to compare values. You can choose from:",
                    "options": [
                        "> — greater than",
                        ">= — greater than or equal to", 
                        "< — less than",
                        "<= — less than or equal to",
                        "== — equal to"
                    ]
                },
                "valid_values": {
                    "scope": ["all", "every", "<specific_request_name>"],
                    "target": ["response_time", "throughput", "error_rate"],
                    "aggregation": ["max", "min", "avg", "pct95", "pct50"]
                },
                "error": f"Could not fetch available tests and environments: {str(e)}"
            }
            
            return json.dumps(fallback_template, indent=2)

    def _create_threshold(self, threshold_config: Dict, test_name: Optional[str] = None, environment: Optional[str] = None) -> str:
        """Create a new backend threshold."""
        try:
            # If test_name and environment are provided as separate parameters, add them to threshold_config
            if test_name and 'test' not in threshold_config:
                threshold_config['test'] = test_name
            if environment and 'environment' not in threshold_config:
                threshold_config['environment'] = environment
            
            # Validate required fields
            required_fields = ["test", "environment", "scope", "target", "aggregation", "comparison", "value"]
            missing_fields = [field for field in required_fields if field not in threshold_config]
            
            if missing_fields:
                return f"Error: Missing required fields: {missing_fields}"
            
            # Validate field values
            valid_targets = ["response_time", "throughput", "error_rate"]
            valid_aggregations = ["max", "min", "avg", "pct95", "pct50"]
            valid_comparisons_user = [">", ">=", "<", "<=", "=="]  # User-friendly format
            valid_comparisons_api = ["gt", "gte", "lt", "lte", "eq"]  # API format
            
            # Map user-friendly comparisons to API format
            comparison_mapping = {
                ">": "gt",
                ">=": "gte", 
                "<": "lt",
                "<=": "lte",
                "==": "eq"
            }
            
            if threshold_config["target"] not in valid_targets:
                return f"Error: Invalid target '{threshold_config['target']}'. Valid values: {valid_targets}"
            
            if threshold_config["aggregation"] not in valid_aggregations:
                return f"Error: Invalid aggregation '{threshold_config['aggregation']}'. Valid values: {valid_aggregations}"
            
            # Check if comparison is in user-friendly format and convert it
            comparison_value = threshold_config["comparison"]
            if comparison_value in valid_comparisons_user:
                # Convert user-friendly to API format
                threshold_config["comparison"] = comparison_mapping[comparison_value]
            elif comparison_value not in valid_comparisons_api:
                return f"Error: Invalid comparison '{comparison_value}'. Valid values: {valid_comparisons_user}"
            
            # Ensure value is numeric
            try:
                threshold_config["value"] = float(threshold_config["value"])
            except (ValueError, TypeError):
                return "Error: 'value' must be a numeric value"
            
            # Create the threshold
            response = self.api_wrapper.create_backend_threshold(threshold_config)
            
            result = {
                "message": "Backend threshold created successfully!",
                "threshold_config": threshold_config,
                "response": response
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error creating threshold: {e}")
            return f"Error creating threshold: {str(e)}"


class GetBackendThresholdsTool(BaseTool):
    """Tool to get existing backend thresholds from Carrier platform."""
    
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "get_backend_thresholds"
    description: str = "Get existing backend performance thresholds from the Carrier platform."
    args_schema: Type[BaseModel] = create_model(
        "GetBackendThresholdsInput",
    )

    def _run(self):
        try:
            # This would require implementing the GET thresholds endpoint in the SDK
            # For now, we'll return a placeholder message
            return "Getting existing thresholds is not yet implemented. Use set_backend_thresholds to create new ones."
            
        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error in GetBackendThresholdsTool: {stacktrace}")
            raise ToolException(stacktrace)
