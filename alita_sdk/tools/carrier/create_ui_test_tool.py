import logging
import json
import traceback
from typing import Type
from langchain_core.tools import BaseTool, ToolException
from pydantic.fields import Field
from pydantic import create_model, BaseModel
from .api_wrapper import CarrierAPIWrapper

logger = logging.getLogger(__name__)


class CreateUITestTool(BaseTool):
    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API Wrapper instance")
    name: str = "create_ui_test"
    description: str = "Create a new UI test in the Carrier platform."
    args_schema: Type[BaseModel] = create_model(
        "CreateUITestInput",
        message=(str, Field(description="User request message for creating UI test")),
        name=(str, Field(description="Test name (e.g., 'My UI Test')")),
        test_type=(str, Field(description="Test type (e.g., 'performance')")),
        env_type=(str, Field(description="Environment type (e.g., 'staging')")),
        entrypoint=(str, Field(description="Entry point file (e.g., 'my_test.js')")),
        runner=(str, Field(
            description="Test runner type. Available runners: Lighthouse-NPM_V12, Lighthouse-Nodejs, Lighthouse-NPM, Lighthouse-NPM_V11, Sitespeed (Browsertime), Sitespeed (New Entrypoint BETA), Sitespeed (New Version BETA), Sitespeed V36")),
        repo=(str, Field(description="Git repository URL (e.g., 'https://github.com/user/repo.git')")),
        branch=(str, Field(description="Git branch name (e.g., 'main')")),
        username=(str, Field(description="Git username")),
        password=(str, Field(description="Git password")),
        cpu_quota=(int, Field(description="CPU quota in cores (e.g., 2)")),
        memory_quota=(int, Field(description="Memory quota in GB (e.g., 5)")),
        parallel_runners=(int, Field(description="Number of parallel runners (e.g., 1)")),
        loops=(int, Field(description="Number of loops (e.g., 1)")),
        **{"custom_cmd": (str, Field(default="", description="Optional custom command (e.g., '--login=\"qwerty\"')"))}
    )

    def _run(self, **kwargs):
        try:
            # Create the UI test with provided parameters
            return self._create_ui_test(kwargs)

        except Exception:
            stacktrace = traceback.format_exc()
            logger.error(f"Error creating UI test: {stacktrace}")
            raise ToolException(stacktrace)

    def _parse_validation_error(self, error_message: str) -> str:
        """Parse validation error message and format it for user display."""
        try:
            # Try to extract JSON validation errors from the message
            import re
            json_match = re.search(r'\[.*\]', error_message)
            if json_match:
                json_str = json_match.group(0)
                try:
                    validation_errors = json.loads(json_str)
                    if isinstance(validation_errors, list):
                        formatted_errors = []
                        for error in validation_errors:
                            if isinstance(error, dict):
                                field = error.get("loc", ["unknown"])[0] if error.get("loc") else "unknown"
                                message = error.get("msg", "Invalid value")
                                formatted_errors.append(f"- **{field}**: {message}")

                        if formatted_errors:
                            return "\n".join(formatted_errors)
                except json.JSONDecodeError:
                    pass
            return error_message

        except Exception:
            return error_message

    def _get_param_from_list(self, param_list: list, param_name: str) -> str | None:
        """
        Finds a parameter in a list of parameter dictionaries and returns its 'default' value.
        """
        for param in param_list:
            if param.get("name") == param_name:
                return param.get("default")
        return None

    def _get_test_param_from_list(self, param_list: list, param_name: str) -> str | None:
        """
        Finds a parameter in a list of parameter dictionaries and returns its 'default' value.
        """
        for param in param_list:
            if param.get("name") == param_name:
                if "auto-generated from test type" in param.get("description", ""):
                    logger.info(f"Skipping auto-generated parameter: {param_name}")
                    continue
                return param.get("default")
        return None

    def _duplicate_ui_test(self, source_name: str, new_name: str) -> str:
        """
        Duplicates a UI test by fetching the source configuration, correctly parsing it,
        validating required fields, and creating a new test.
        """
        logger.info(f"Executing duplication: from '{source_name}' to '{new_name}'.")
        try:
            # 1. Find and fetch the source test configuration
            all_tests = self.api_wrapper.get_ui_tests_list()
            source_test = next((test for test in all_tests if test.get("name") == source_name), None)
            if not source_test:
                raise ToolException(f"Source test '{source_name}' not found.")
            source_test_id = source_test["id"]

            source_config = self.api_wrapper.get_ui_test_details(source_test_id)
            logger.debug(f"Source test configuration fetched: {source_config}")

            # 2. Parse test_type and env_type from test_parameters list
            test_params_list = source_config.get("test_parameters", [])
            test_type = self._get_param_from_list(test_params_list, "test_type")
            env_type = self._get_param_from_list(test_params_list, "env_type")

            # 3. Get env_vars from the correct location
            env_vars = source_config.get("env_vars", {})

            # 4. Get source details
            source_details = source_config.get("source", {})

            # 5. Validate all required fields
            if not test_type:
                raise ValueError(f"Source test '{source_name}' is invalid: missing test_type in test_parameters")
            if not env_type:
                raise ValueError(f"Source test '{source_name}' is invalid: missing env_type in test_parameters")

            # 6. Build the POST body with correct structure
            post_body = {
                "common_params": {
                    "name": new_name,
                    "test_type": test_type,
                    "env_type": env_type,
                    "entrypoint": source_config.get("entrypoint"),
                    "runner": source_config.get("runner"),
                    "source": {
                        "name": source_details.get("name"),
                        "repo": source_details.get("repo"),
                        "branch": source_details.get("branch"),
                        # Handle both git_ssh (private_key) and git_https (username/password)
                        "private_key": source_details.get("private_key"),
                        "username": source_details.get("username"),
                        "password": source_details.get("password")
                    },
                    "env_vars": {
                        "cpu_quota": env_vars.get("cpu_quota"),
                        "memory_quota": env_vars.get("memory_quota"),
                        "cloud_settings": env_vars.get("cloud_settings", {}),
                        "custom_cmd": env_vars.get("custom_cmd", "")
                    },
                    "parallel_runners": source_config.get("parallel_runners"),
                    "cc_env_vars": source_config.get("cc_env_vars", {}),
                    "location": source_config.get("location"),
                    "loops": source_config.get("loops"),
                    "aggregation": source_config.get("aggregation")
                },
                "test_parameters": [],  # Empty for create API
                "integrations": source_config.get("integrations", {}),
                "schedules": [],
                "run_test": False
            }

            # Remove None values from source to avoid API validation errors
            source_data = post_body["common_params"]["source"]
            post_body["common_params"]["source"] = {k: v for k, v in source_data.items() if v is not None}

            logger.debug(f"Constructed post_body for duplication: {post_body}")

            # 7. Create the new test
            response = self.api_wrapper.create_ui_test(post_body)
            new_test_id = response.get("id", "Unknown")

            logger.info(f"Successfully duplicated test. New test ID: {new_test_id}")
            return f"✅ UI Test duplicated successfully! The new test '{new_name}' has been created with ID: {new_test_id}."

        except Exception:
            logger.error("Duplication failed. Propagating raw exception for a full stack trace.")
            raise

    def _create_ui_test(self, params: dict):
        """
        Dispatches UI test creation.
        - If source_test_name is provided, duplicates an existing test.
        - Otherwise, creates a new test from scratch.
        """
        # If the intent is to duplicate, call the dedicated method
        if "source_test_name" in params and "test_name" in params:
            return self._duplicate_ui_test(
                source_name=params["source_test_name"],
                new_name=params["test_name"]
            )

        # Otherwise, proceed with creating a new test from scratch
        try:
            test_name = params.get("test_name") or params.get("name")
            if not test_name:
                raise ValueError("Missing required parameter: 'test_name' or 'name'.")

            # Construct the POST body for a new test
            post_body = {
                "common_params": {
                    "name": test_name,
                    "test_type": params["test_type"],
                    "env_type": params["env_type"],
                    "entrypoint": params["entrypoint"],
                    "runner": params["runner"],
                    "source": {
                        "name": "git_httpss",
                        "repo": params["repo"],
                        "branch": params["branch"],
                        "username": params["username"],
                        "password": params["password"]
                    },
                    "env_vars": {
                        "cpu_quota": params["cpu_quota"],
                        "memory_quota": params["memory_quota"],
                        "cloud_settings": {}
                    },
                    "parallel_runners": params["parallel_runners"],
                    "cc_env_vars": {},
                    "location": "default",
                    "loops": params["loops"],
                    "aggregation": "max"
                },
                "test_parameters": [],
                "integrations": {},
                "schedules": [],
                "run_test": False
            }

            if params.get("custom_cmd"):
                post_body["common_params"]["env_vars"]["custom_cmd"] = params["custom_cmd"]

            response = self.api_wrapper.create_ui_test(post_body)

            if response:
                test_id = response.get("id", "Unknown")
                return f"""# ✅ UI Test Created Successfully!

                        ## Test Information:
                        - **Test ID:** `{test_id}`
                        - **Name:** `{test_name}`
                        - **Type:** `{params.get('test_type')}`
                        - **Environment:** `{params.get('env_type')}`
                        - **Runner:** `{params.get('runner')}`
                        - **Repository:** `{params.get('repo')}`
                        - **Branch:** `{params.get('branch')}`
                        - **Entry Point:** `{params.get('entrypoint')}`

                        ## Configuration:
                        - **CPU Quota:** {params.get('cpu_quota')} cores
                        - **Memory Quota:** {params.get('memory_quota')} GB
                        - **Parallel Runners:** {params.get('parallel_runners')}
                        - **Loops:** {params.get('loops')}
                        - **Aggregation:** max
                        {f"- **Custom Command:** `{params.get('custom_cmd')}`" if params.get('custom_cmd') else ""}

                        ## 🎯 Next Steps:
                        - Your UI test has been created and is ready to run
                        - You can execute it using the UI test runner tools
                        - Configure schedules and integrations as needed"""
            else:
                raise ToolException("Failed to create UI test: received an empty response from the API.")

        except KeyError as e:
            raise ValueError(f"Missing required parameter to create a new test: {e}")
        except Exception:
            logger.error("Creation of a new test failed. Propagating raw exception.")
            raise
