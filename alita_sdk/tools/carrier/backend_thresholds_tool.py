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
        "This tool will automatically show available tests with their environments and guide you through creating thresholds. Also providing a small examples about Thresholds logic."
        "As an output uses well-structured user-friendly format with a litle bit emojis."
    )
    args_schema: Type[BaseModel] = create_model(
        "SetBackendThresholdsInput",
        step=(str, Field(default="show_tests_and_envs", description="Current step: 'show_tests_and_envs', 'get_requests', or 'create_threshold'")),
        test_name=(str, Field(default=None, description="Selected test name")),
        environment=(str, Field(default=None, description="Selected environment")),
        examples=(str, Field(default=None, description="Examples of Threshold Logic. How to properly configure thresholds")),
        threshold_config=(str, Field(default=None, description="Threshold configuration object with all required fields"))
    )
    
    def _get_threshold_configuration_guide(self, include_template: bool = True, test_name: Optional[str] = None, environment: Optional[str] = None) -> List[str]:
        """
        Reusable method that generates threshold configuration guidance.
        
        Args:
            include_template: Whether to include JSON template
            test_name: Optional specific test name for template
            environment: Optional specific environment for template
        
        Returns:
            List of strings that can be joined to form the guidance message
        """
        guide_parts = []
        
        if include_template:
            guide_parts.extend([
                "**🛠️ Threshold Configuration Template:**",
                "```json",
                "{",
                f'  "test": "{test_name or "<TEST_NAME>"}",           // Use one of the test names listed above',
                f'  "environment": "{environment or "<ENV_NAME>"}",     // Use one of the environments for your selected test',
                '  "scope": "all",                  // Can be "all", "every", or specific request name',
                '  "target": "response_time",       // Can be "response_time", "throughput", "error_rate"',
                '  "aggregation": "pct95",          // Can be "max", "min", "avg", "pct95", "pct50"',
                '  "comparison": ">=",              // See comparison types below',
                '  "value": 1000                    // Threshold value (numeric)',
                "}",
                "```",
                ""
            ])
        
        guide_parts.extend([
            "**🔍 Comparison Types:**",
            "Select how you want to compare values. You can choose from:",
            "",
            "- `>` — greater than",
            "- `>=` — greater than or equal to",
            "- `<` — less than", 
            "- `<=` — less than or equal to",
            "- `==` — equal to",
            "",
            "**💡 Examples of Threshold Logic:**",
            "",
            "**🚨 All Error rate threshold > 10 (%)**",
            "The total error rate should be less than 10%. If it exceeds 10% build will fail.",
            "",
            "**⚡ All Throughput < 3 requests per second**",
            "The total throughput should be more than 3 requests per second. If it is less than 3 requests the build will fail.",
            "",
            "**⏱️ Every Response Time > 3000 milliseconds (3 seconds)**",
            "Every request's response time should be less than 3000 milliseconds. If it exceeds 3000 milliseconds the build will fail.",
            "*⚠️ Note: The option \"Every\" in Scope is applicable only for the \"Response Time\" threshold.*",
            ""
        ])
        
        return guide_parts
    
    def _get_quick_examples(self) -> List[str]:
        """
        Reusable method that generates quick configuration examples.
        
        Returns:
            List of strings with quick example configurations
        """
        return [
            "💡 **Quick Examples:**",
            "",
            "- 🚨 **Error rate**: {\"scope\": \"all\", \"target\": \"error_rate\", \"comparison\": \">\", \"value\": 10}",
            "  *The total error rate should be less than 10%. If it exceeds 10 percent the build will fail.*",
            "",
            "- ⚡ **Throughput**: {\"scope\": \"all\", \"target\": \"throughput\", \"comparison\": \"<\", \"value\": 3}",
            "  *The total throughput should be more than 3 requests per second. If it is less than 3 requests the build will fail.*",
            "",
            "- ⏱️ **Response time**: {\"scope\": \"every\", \"target\": \"response_time\", \"comparison\": \">\", \"value\": 3000}",
            "  *Note: Every request's response time should be less than 3000 milliseconds. If it exceeds 3000 milliseconds the build will fail.*",
            "",
            "⚠️ **Important**: The option \"Every\" in Scope is applicable only for the \"Response Time\" threshold."
        ]
    
    def _get_valid_values_info(self) -> List[str]:
        """
        Reusable method that generates valid values information.
        
        Returns:
            List of strings with valid values for each field
        """
        return [
            "**✅ Valid Values:**",
            "- **🎯 scope**: ['all', 'every', '<specific_request_name>']",
            "- **📊 target**: ['response_time', 'throughput', 'error_rate']", 
            "- **📈 aggregation**: ['max', 'min', 'avg', 'pct95', 'pct50']"
        ]

    def _run(self, step: str = "show_tests_and_envs", test_name: Optional[str] = None, 
             environment: Optional[str] = None, threshold_config: Optional[Dict] = None):
        try:
            if step == "show_tests_and_envs":
                return self._show_tests_and_environments()
            
            elif step == "get_requests":
                if not test_name or not environment:
                    return "❌ Error: both test_name and environment are required for getting request names"
                return self._get_request_names(test_name, environment)
            
            elif step == "create_threshold":
                if not threshold_config:
                    return self._get_threshold_template()
                return self._create_threshold(threshold_config, test_name, environment)
            
            else:
                return "❌ Error: Invalid step. Valid steps are: 'show_tests_and_envs', 'get_requests', 'create_threshold'"
                
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
                return "❌ No backend tests found in this project."
            
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
                "🎯 To create a new backend performance threshold, here are the available options:",
                "",
                "1. **📋 Test Names**: Available test names and environments"
            ]
            result_parts.extend(tests_with_envs)
            result_parts.extend([
                "",
                "2. **⚙️ Threshold Configuration**: This includes:",
                "   - **🎯 scope**: 'all', 'every', or specific request name",
                "   - **📊 target**: 'response_time', 'throughput', or 'error_rate'", 
                "   - **📈 aggregation**: 'max', 'min', 'avg', 'pct95', or 'pct50'",
                "   - **🔍 comparison**: See comparison types below",
                "   - **🔢 value**: Threshold value (numeric)",
                ""
            ])
            
            # Add the reusable threshold configuration guide
            result_parts.extend(self._get_threshold_configuration_guide(include_template=False))
            
            result_parts.extend([
                "**🚀 Next steps:**",
                "- To see available request names for a specific test and environment, use:",
                "  step='get_requests' with test_name='<TEST_NAME>' and environment='<ENV_NAME>'",
                "- To create a threshold directly, use:",
                "  step='create_threshold' with a complete threshold_config object"
            ])
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error getting tests and environments: {e}")
            return f"❌ Error retrieving tests and environments: {str(e)}"

    def _get_request_names(self, test_name: str, environment: str) -> str:
        """Get available request names for a specific test and environment."""
        try:
            requests = self.api_wrapper.get_backend_requests(test_name, environment)
            
            result_parts = [
                f"📋 Available request names for test '{test_name}' in environment '{environment}':",
                ""
            ]
            
            if requests:
                result_parts.extend([f"  - {req}" for req in requests])
            else:
                result_parts.append("  - No specific requests found - you can use 'all' or 'every' scope")
            
            result_parts.extend([
                ""
            ])
            
            # Add the reusable threshold configuration guide with specific test and environment
            result_parts.extend(self._get_threshold_configuration_guide(include_template=True, test_name=test_name, environment=environment))
            
            # Add quick examples
            result_parts.extend(self._get_quick_examples())
            
            result_parts.extend([
                "",
                "✅ **Next step:** Use step='create_threshold' with your threshold_config object"
            ])
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error getting request names: {e}")
            return f"❌ Error retrieving request names for test '{test_name}' in environment '{environment}': {str(e)}"

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
                "📋 Please provide a threshold_config object with the following structure:",
                "",
                "**🎯 Available Tests and Environments:**"
            ]
            
            if tests_with_envs:
                template_parts.extend(tests_with_envs)
            else:
                template_parts.append("    No tests found in this project")
            
            template_parts.extend([
                ""
            ])
            
            # Add the reusable threshold configuration guide
            template_parts.extend(self._get_threshold_configuration_guide(include_template=True))
            
            # Add valid values info
            template_parts.extend(self._get_valid_values_info())
            
            template_parts.extend([
                "",
                "**🔄 Alternative Usage:**",
                "You can also provide test_name and environment as separate parameters:",
                "step='create_threshold', test_name='<TEST_NAME>', environment='<ENV_NAME>', threshold_config={...}"
            ])
            
            return "\n".join(template_parts)
            
        except Exception as e:
            logger.error(f"Error getting threshold template: {e}")
            # Fallback template using the reusable guide
            fallback_parts = [
                f"❌ Error: Could not fetch available tests and environments: {str(e)}",
                "",
                "📋 Please provide a threshold_config object with the following structure:",
                ""
            ]
            
            # Add the reusable threshold configuration guide
            fallback_parts.extend(self._get_threshold_configuration_guide(include_template=True))
            fallback_parts.extend(self._get_valid_values_info())
            
            return "\n".join(fallback_parts)

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
                return f"❌ Error: Missing required fields: {missing_fields}"
            
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
                return f"❌ Error: Invalid target '{threshold_config['target']}'. Valid values: {valid_targets}"
            
            if threshold_config["aggregation"] not in valid_aggregations:
                return f"❌ Error: Invalid aggregation '{threshold_config['aggregation']}'. Valid values: {valid_aggregations}"
            
            # Check if comparison is in user-friendly format and convert it
            comparison_value = threshold_config["comparison"]
            if comparison_value in valid_comparisons_user:
                # Convert user-friendly to API format
                threshold_config["comparison"] = comparison_mapping[comparison_value]
            elif comparison_value not in valid_comparisons_api:
                return f"❌ Error: Invalid comparison '{comparison_value}'. Valid values: {valid_comparisons_user}"
            
            # Ensure value is numeric
            try:
                threshold_config["value"] = float(threshold_config["value"])
            except (ValueError, TypeError):
                return "❌ Error: 'value' must be a numeric value"
            
            # Create the threshold
            response = self.api_wrapper.create_backend_threshold(threshold_config)
            
            # Convert comparison back to user-friendly format for display
            display_comparison = comparison_value if comparison_value in valid_comparisons_user else {
                "gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=="
            }.get(threshold_config["comparison"], threshold_config["comparison"])
            
            result_parts = [
                "✅ **Backend threshold created successfully!**",
                "",
                "🎯 **Threshold Details:**",
                f"  - **Test**: {threshold_config['test']}",
                f"  - **Environment**: {threshold_config['environment']}",
                f"  - **Scope**: {threshold_config['scope']}",
                f"  - **Target**: {threshold_config['target']}",
                f"  - **Aggregation**: {threshold_config['aggregation']}",
                f"  - **Comparison**: {display_comparison}",
                f"  - **Value**: {threshold_config['value']}",
                "",
                f"📋 **Summary**: {threshold_config['scope']} {threshold_config['target']} {display_comparison} {threshold_config['value']} for test '{threshold_config['test']}' in '{threshold_config['environment']}' environment",
                "",
                "🔧 **API Response**:",
                f"{response}"
            ]
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error creating threshold: {e}")
            return f"❌ Error creating threshold: {str(e)}"


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
            # Get all backend thresholds from the API
            thresholds = self.api_wrapper.get_backend_thresholds()
            
            if not thresholds:
                return "No backend performance thresholds found in this project."
            
            # Return the raw response with minimal formatting
            return f"Found {len(thresholds)} backend threshold(s):\n\n{json.dumps(thresholds, indent=2)}"
            
        except Exception as e:
            logger.error(f"Error in GetBackendThresholdsTool: {e}")
            return f"Error retrieving backend thresholds: {str(e)}"


class DeleteBackendThresholdsTool(BaseTool):
    """Tool to delete backend thresholds from Carrier platform."""
    
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "delete_backend_thresholds"
    description: str = "Delete backend performance thresholds from the Carrier platform. If no threshold_id is provided, it will show available thresholds."
    args_schema: Type[BaseModel] = create_model(
        "DeleteBackendThresholdsInput",
        threshold_id=(Optional[str], Field(default=None, description="ID of the threshold to delete"))
    )

    def _run(self, threshold_id: Optional[str] = None):
        try:
            # If no threshold_id provided, show available thresholds
            if not threshold_id:
                thresholds_response = self.api_wrapper.get_backend_thresholds()
                
                if not thresholds_response or not thresholds_response.get("rows"):
                    return "No backend performance thresholds found in this project."
                
                thresholds = thresholds_response.get("rows", [])
                
                result_parts = [
                    f"Found {len(thresholds)} backend threshold(s). To delete a threshold, provide the threshold_id:",
                    ""
                ]
                
                for threshold in thresholds:
                    result_parts.append(
                        f"ID: {threshold.get('id')} - Test: {threshold.get('test')} - "
                        f"Environment: {threshold.get('environment')} - "
                        f"Target: {threshold.get('target')} {threshold.get('comparison')} {threshold.get('value')}"
                    )
                
                result_parts.extend([
                    "",
                    "Example: To delete threshold 909, use threshold_id='909'"
                ])
                
                return "\n".join(result_parts)
            
            # Delete the specified threshold
            response = self.api_wrapper.delete_backend_threshold(threshold_id)
            return f"✅ Successfully deleted backend threshold with ID: {threshold_id}"
            
        except Exception as e:
            logger.error(f"Error in DeleteBackendThresholdsTool: {e}")
            return f"❌ Failed to delete threshold {threshold_id}: {str(e)}"
