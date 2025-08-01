import json
import os
from typing import Type, List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, BaseModel
from .api_wrapper import CarrierAPIWrapper

from .etl.etl_factory import ETLComponentFactory

logger = logging.getLogger(__name__)


# =========================================================================
#  PRODUCTION-READY INPUT SCHEMAS WITH ENHANCED VALIDATION
# =========================================================================

class BaseToolInput(BaseModel):
    """🔧 Base input schema with common validation patterns."""

    class Config:
        extra = "forbid"  # Prevent unexpected parameters
        validate_assignment = True


class GetTestsInput(BaseToolInput):
    """Input schema for listing available tests."""
    pass


class GetTestByIdInput(BaseToolInput):
    """Input schema for retrieving specific test details."""
    test_id: str = Field(
        ...,
        min_length=1,
        description="🔢 Unique identifier for the test to retrieve"
    )


class RunTestByIdInput(BaseToolInput):
    """Input schema for executing performance tests."""
    test_id: str = Field(..., min_length=1, description="🔢 Test identifier to execute")
    location: str = Field(
        default="default",
        description="🌍 Execution location (e.g., 'aws-us-east-1', 'default')"
    )
    test_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="⚙️ Test parameter overrides (e.g., {'vUsers': 50, 'duration': '300'})"
    )
    cloud_settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="☁️ Cloud-specific configuration settings"
    )


class CreateBackendTestInput(BaseToolInput):
    """Input schema for creating new performance tests."""
    test_name: str = Field(..., min_length=1, max_length=100, description="📝 Unique test name")
    entrypoint: str = Field(..., min_length=1, description="🎯 Script entry point path")
    runner: str = Field(..., description="🏃‍♂️ Test runner (JMeter_v5.5, Gatling_maven)")
    source: Dict[str, str] = Field(..., description="📦 Git repository configuration")
    test_parameters: List[Dict[str, str]] = Field(
        default_factory=list,
        description="⚙️ Test parameter definitions"
    )


class ProcessReportInput(BaseToolInput):
    """Input schema for report processing with enhanced validation."""
    report_id: str = Field(
        ...,
        min_length=1,
        description="🔢 Carrier report identifier"
    )
    pipeline_type: str = Field(
        default="gatling_to_excel",
        pattern=r"^(gatling_to_excel|jmeter_to_excel)$",
        description="🔧 Processing pipeline type"
    )
    pct: str = Field(
        default="95Pct",
        pattern=r"^(50|75|90|95|99|99\.9)Pct$",
        description="📊 Performance percentile for analysis"
    )
    tp_threshold: int = Field(
        default=10,
        ge=0,
        le=10000,
        description="⚡ Throughput threshold (req/sec)"
    )
    rt_threshold: int = Field(
        default=500,
        ge=0,
        le=60000,
        description="⏱️ Response time threshold (ms)"
    )
    er_threshold: int = Field(
        default=5,
        ge=0,
        le=100,
        description="❌ Error rate threshold (%)"
    )


class GetReportsInput(BaseToolInput):
    """
    Input schema for retrieving reports with advanced filtering and sorting.
    """
    limit: int = Field(
        default=5,  # Changed default to 5 as per your "latest 5" request
        ge=1,
        le=100,
        description="📊 Maximum number of reports to return."
    )
    test_name: Optional[str] = Field(
        default=None,
        description="🧪 Filter by a partial or full test name (case-insensitive)."
    )
    entrypoint: Optional[str] = Field(
        default=None,
        description="🎯 Filter by the exact script entrypoint (e.g., 'tests/BasicEcommerceWithTransaction.jmx')."
    )
    environment: Optional[str] = Field(
        default=None,
        description="🌍 Filter by environment name."
    )
    status: Optional[str] = Field(
        default=None,
        description="🚦 Filter by status (e.g., 'finished', 'failed', 'success')."
    )
    runner_type: Optional[str] = Field(
        default=None,
        description="🏃 Filter by runner type (e.g., 'jmeter', 'gatling')."
    )
    tag_name: Optional[str] = Field(
        default=None,
        description="🏷️ Filter by an assigned tag."
    )
    sort_by: str = Field(
        default="start_time",
        pattern="^(start_time|name|duration|vusers)$",
        description=" Filter by report attribute ('start_time', 'name', 'duration', 'vusers')."
    )
    sort_order: str = Field(
        default="desc",
        pattern="^(asc|desc)$",
        description="📉 Sort order: 'asc' for ascending, 'desc' for descending."
    )


class GetReportByIdInput(BaseToolInput):
    """Input schema for detailed report retrieval."""
    report_id: str = Field(..., min_length=1, description="🔢 Report identifier")
    include_errors: bool = Field(default=True, description="📋 Include error analysis")
    error_limit: int = Field(default=100, ge=1, le=1000, description="🔢 Error entry limit")


class FormattingMixin:
    """🎨 Reusable formatting methods with conditional styling."""

    @staticmethod
    def format_test_summary(test: Dict) -> Dict:
        """Format test data for user-friendly display."""
        return {
            "id": test.get("id"),
            "name": test.get("name"),
            "entrypoint": test.get("entrypoint"),
            "runner": test.get("runner", "unknown"),
            "status": f"✅ {test.get('status', 'active')}" if test.get("status") != "inactive" else "⏸️ inactive",
            "test_parameters": [
                {"name": param["name"], "default": param["default"]}
                for param in test.get("test_parameters", [])
            ]
        }

    @staticmethod
    def format_report_summary(report: Dict) -> Dict:
        """Format report data with conditional status indicators."""
        logger.info(f"🚀 Formatting summary for report_id: {report.get('id')}, build_id: {report.get('build_id')}")
        status_info = report.get("test_status", "unknown")
        if isinstance(status_info, dict):
            status = str(status_info.get("status", "unknown")).lower()
            status_percentage = status_info.get("percentage")
            status_description = status_info.get("description", "")
        else:
            status = str(status_info).lower()
            status_percentage = None
            status_description = ""

        status_indicators = {
            "finished": "✅",
            "failed": "❌",
            "running": "🔄",
            "stopped": "⏹️"
        }

        lg_type = report.get("lg_type", "").lower()
        pipeline_recommendations = {
            "gatling": {"type": "gatling_to_excel", "emoji": "🎯"},
            "jmeter": {"type": "jmeter_to_excel", "emoji": "⚡"},
        }

        recommendation = pipeline_recommendations.get(lg_type, {"type": "check_type", "emoji": "🔍"})

        return {
            "id": report.get("id"),
            "build_id": report.get("build_id"),
            "name": report.get("name"),
            "environment": report.get("environment"),
            "status": f"{status_indicators.get(status, '❓')} {status}",
            "status_percentage": status_percentage,
            "status_description": status_description,
            "vusers": report.get("vusers"),
            "duration": report.get("duration"),
            "start_time": report.get("start_time"),
            "recommended_pipeline": f"{recommendation['emoji']} {recommendation['type']}",
            "tags": [tag.get("title", "") for tag in report.get("tags", []) if isinstance(tag, dict)]
        }


class LoggingMixin:
    """📝 Enhanced logging with structured context."""

    def log_operation_start(self, operation: str, **context):
        """Log operation start with context."""
        logger.info(f"🚀 Starting {operation}")
        # for key, value in context.items():
        #     logger.info(f"   📋 {key}: {value}")

    def log_operation_success(self, operation: str, duration: float = None, **context):
        """Log successful operation completion."""
        duration_str = f" in {duration:.2f}s" if duration else ""
        logger.info(f"✅ Completed {operation}{duration_str}")
        # for key, value in context.items():
        #     logger.info(f"   📊 {key}: {value}")

    def log_operation_error(self, operation: str, error: Exception, **context):
        """Log operation failure with context."""
        logger.error(f"💥 Failed {operation}: {str(error)}")
        # for key, value in context.items():
        #     logger.error(f"   🔍 {key}: {value}")


# =========================================================================
#  REFACTORED TOOL CLASSES WITH DRY AND SOLID PRINCIPLES
# =========================================================================

class BaseCarrierTool(BaseTool, FormattingMixin, LoggingMixin):
    """🏗️ Base class for all Carrier tools implementing common patterns."""

    api_wrapper: CarrierAPIWrapper = Field(..., description="Carrier API client")

    def handle_api_error(self, operation: str, error: Exception) -> None:
        """Centralized API error handling."""
        self.log_operation_error(operation, error)
        if "404" in str(error) or "not found" in str(error).lower():
            raise ToolException(f"🔍 Resource not found during {operation}")
        elif "403" in str(error) or "unauthorized" in str(error).lower():
            raise ToolException(f"🔑 Access denied during {operation}")
        elif "timeout" in str(error).lower():
            raise ToolException(f"⏱️ Timeout occurred during {operation}")
        else:
            raise ToolException(f"❌ {operation} failed: {str(error)}")


class GetTestsTool(BaseCarrierTool):
    """📋 Retrieves available performance tests with enhanced formatting."""

    name: str = "get_tests"
    description: str = "🔍 Get formatted list of all available performance tests"
    args_schema: Type[BaseModel] = GetTestsInput

    def _run(self) -> str:
        operation = "fetching tests list"
        start_time = datetime.now()

        self.log_operation_start(operation)

        try:
            tests = self.api_wrapper.get_tests_list()
            formatted_tests = [self.format_test_summary(test) for test in tests]

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(operation, duration, count=len(formatted_tests))

            return json.dumps({
                "message": f"📋 Found {len(formatted_tests)} available tests",
                "tests": formatted_tests
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)


class GetTestByIDTool(BaseCarrierTool):
    """🔍 Retrieves detailed test configuration with validation."""

    name: str = "get_test_by_id"
    description: str = "🔍 Get detailed configuration for a specific test"
    args_schema: Type[BaseModel] = GetTestByIdInput

    def _run(self, test_id: str) -> str:
        operation = f"fetching test {test_id}"
        start_time = datetime.now()

        self.log_operation_start(operation, test_id=test_id)

        try:
            # Use API wrapper method to get specific test
            tests = self.api_wrapper.get_tests_list()
            test_data = next(
                (test for test in tests if str(test["id"]) == test_id),
                None
            )

            if not test_data:
                raise ToolException(f"🔍 Test '{test_id}' not found in Carrier platform")

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(operation, duration)

            return json.dumps({
                "message": f"✅ Test details retrieved for {test_id}",
                "test": test_data
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)


class RunTestByIDTool(BaseCarrierTool):
    """🚀 Executes performance tests with comprehensive parameter handling."""

    name: str = "run_test_by_id"
    description: str = "🚀 Execute a performance test with specified parameters"
    args_schema: Type[BaseModel] = RunTestByIdInput

    def _run(self, test_id: str, location: str = "default",
             test_params: Dict = None, cloud_settings: Dict = None) -> str:
        operation = f"executing test {test_id}"
        start_time = datetime.now()

        self.log_operation_start(
            operation,
            test_id=test_id,
            location=location,
            param_count=len(test_params or {}),
            has_cloud_settings=bool(cloud_settings)
        )

        try:
            # Get test configuration
            tests = self.api_wrapper.get_tests_list()
            test_data = next(
                (test for test in tests if str(test["id"]) == test_id),
                None
            )

            if not test_data:
                raise ToolException(f"🔍 Test '{test_id}' not found in Carrier platform")

            # Build execution payload
            execution_payload = self._build_execution_payload(
                test_data, location, test_params or {}, cloud_settings or {}
            )

            # Execute test
            report_id = self.api_wrapper.run_test(test_id=test_id, json_body=execution_payload)
            report_url = f"{self.api_wrapper.url.rstrip('/')}/-/performance/backend/results?result_id={report_id}"

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(operation, duration, report_id=report_id)

            return json.dumps({
                "message": "✅ Test execution started successfully",
                "report_id": report_id,
                "report_url": report_url,
                "test_name": test_data.get("name"),
                "execution_details": {
                    "location": location,
                    "parameters_applied": len(test_params or {}),
                    "estimated_duration": test_data.get("estimated_duration", "unknown")
                }
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)

    def _build_execution_payload(self, test_data: Dict, location: str,
                                 test_params: Dict, cloud_settings: Dict) -> Dict:
        """Build execution payload following DRY principles."""
        # Get default parameters and apply overrides
        default_params = {p['name']: p for p in test_data.get("test_parameters", [])}

        for name, value in test_params.items():
            if name in default_params:
                default_params[name]['default'] = value
                logger.info(f"🔧 Override parameter {name}: {value}")

        # Build common_params from test data
        common_params = {
            param["name"]: param
            for param in test_data.get("test_parameters", [])
            if param["name"] in {"test_name", "test_type", "env_type"}
        }

        # Add essential execution parameters
        common_params.update({
            "env_vars": test_data.get("env_vars", {}),
            "parallel_runners": test_data.get("parallel_runners", 1),
            "location": location
        })

        # Add cloud settings if provided
        if cloud_settings:
            common_params["env_vars"]["cloud_settings"] = cloud_settings

        return {
            "common_params": common_params,
            "test_parameters": list(default_params.values()),
            "integrations": test_data.get("integrations", {})
        }


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


class CreateBackendReportTool(BaseTool):
    """🚀 Generate Excel reports using ETL pipeline and factory pattern."""

    name: str = "process_and_generate_report"
    description: str = """Generate Excel reports from performance test results.

    Parameters:
    - report_id (required): The ID of the report to process
    """

    def __init__(self, api_wrapper: 'CarrierAPIWrapper'):
        super().__init__()
        self._api_wrapper = api_wrapper

        logger.info("🚀 ProcessAndGenerateReportTool initialized")

    def _run(self, report_id: str):
        """
        🎯 Process report using ETL factory - Simple and DRY.
        """
        try:
            logger.info(f"🚀 Processing report {report_id} using ETL factory")
            report = self._api_wrapper.get_report_metadata(report_id)
            if report['lg_type'] == 'jmeter':
                pipeline = ETLComponentFactory.get_pipeline("jmeter_to_excel")
            elif report['lg_type'] == 'gatling':
                pipeline = ETLComponentFactory.get_pipeline("gatling_to_excel")
            else:
                raise Exception
            logger.info("✅ ETL pipeline obtained from factory")

            # Step 2: Prepare context with default parameters
            context = {
                "api_wrapper": self._api_wrapper,
                "report_id": str(report_id),
                "think_time": "2,0-5,0",
                "pct": "95Pct",
                "tp_threshold": 10,
                "rt_threshold": 500,
                "er_threshold": 5
            }

            result = pipeline.run(context)
            logger.info(f"🔍 Loader result type: {type(result)}")
            logger.info(f"🔍 Loader result content: {result}")

            # ✅ Extract key information for response
            if isinstance(result, dict) and result.get("status") == "success":
                download_url = result.get("download_url", "No download link available")
                file_name = result.get("file_name", "Unknown file")

                return f"✅ Excel report generated successfully! File: {file_name}, Download: {download_url}"
            else:
                return f"⚠️ Processing completed with issues: {result}"

            # Step 4: Format result for user
        # return self._format_success_result(result)
        except Exception as e:
            logger.error(f"💥 Processing failed for report {report_id}: {str(e)}")
            return f"❌ Failed to process report {report_id}: {str(e)}"

    def _format_success_result(self, result: dict) -> str:
        """📝 Format successful result for user."""
        if result.get("status") == "Success":
            filename = result.get("excel_filename", "report.xlsx")
            download = result.get("download_link", "")

            return (f"✅ Excel report generated successfully!\n"
                    f"📊 File: {filename}\n"
                    f"🔗 Download: {download}")
        else:
            return f"❌ Processing failed: {result.get('error', 'Unknown error')}"


class GetReportsTool(BaseCarrierTool):
    """📋 Retrieves and filters performance reports with advanced sorting."""

    name: str = "get_reports"
    description: str = "📋 Get a filtered and sorted list of performance reports."
    args_schema: Type[BaseModel] = GetReportsInput

    def _run(self, limit: int = 5, tag_name: Optional[str] = None, environment: Optional[str] = None,
             test_name: Optional[str] = None, entrypoint: Optional[str] = None, status: Optional[str] = None,
             runner_type: Optional[str] = None, sort_by: str = "start_time", sort_order: str = "desc") -> str:
        operation = "fetching reports"
        start_time = datetime.now()

        active_filters = {
            "tag": tag_name, "environment": environment, "test_name": test_name,
            "entrypoint": entrypoint, "status": status, "runner_type": runner_type,
            "sort_by": sort_by, "sort_order": sort_order, "limit": limit
        }
        self.log_operation_start(operation, **{k: v for k, v in active_filters.items() if v is not None})

        try:
            raw_reports = self.api_wrapper.get_reports_list()

            # Filter, sort, and then limit the reports
            processed_reports = self._filter_and_sort_reports(
                raw_reports, limit, sort_by, sort_order,
                tag_name=tag_name, environment=environment, test_name=test_name,
                entrypoint=entrypoint, status=status, runner_type=runner_type
            )

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(
                operation, duration,
                total_fetched=len(raw_reports),
                returned_count=len(processed_reports)
            )

            return json.dumps({
                "message": f"📋 Found {len(processed_reports)} reports matching criteria.",
                "filtering_applied": {k: v for k, v in active_filters.items() if v is not None},
                "reports": [self.format_report_summary(report) for report in processed_reports]
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)

    def _filter_and_sort_reports(self, reports: List[Dict], limit: int, sort_by: str, sort_order: str, **filters) -> \
            List[Dict]:
        """Apply filters, sort the results, and then apply the limit."""

        # This mapping defines how filter names map to keys in the report data.
        # To add a new filter, only add a line here and to the Pydantic model.
        filter_mapping = {
            'test_name': 'name',
            'entrypoint': 'test_config.entrypoint',
            'environment': 'environment',
            'runner_type': 'lg_type',
            'status': 'test_status.status'
        }

        # --- Filtering Stage ---
        filtered_list = []
        for report in reports:
            match = True
            for filter_key, filter_value in filters.items():
                if filter_value is None:
                    continue

                report_value = self._get_nested_value(report, filter_mapping.get(filter_key))

                if filter_key == 'tag_name':
                    tags = report.get("tags", [])
                    tag_titles = [str(tag.get("title", "")).lower() for tag in tags if isinstance(tag, dict)]
                    if filter_value.lower() not in tag_titles:
                        match = False;
                        break
                elif report_value is None or filter_value.lower() not in str(report_value).lower():
                    match = False;
                    break

            if match:
                filtered_list.append(report)

        # --- Sorting Stage ---
        is_reverse = sort_order == 'desc'

        def sort_key_func(report):
            # Handle sorting by start_time by parsing the string to a datetime object
            if sort_by == 'start_time':
                time_str = self._get_nested_value(report, sort_by)
                return datetime.fromisoformat(time_str.replace('Z', '+00:00')) if time_str else datetime.min
            # For other fields, use the raw value, defaulting to 0 for numeric types
            return self._get_nested_value(report, sort_by) or 0

        try:
            sorted_list = sorted(filtered_list, key=sort_key_func, reverse=is_reverse)
        except (TypeError, ValueError) as e:
            self.log_operation_error("sorting reports", e, sort_by=sort_by)
            raise ToolException(f"Failed to sort reports by '{sort_by}'. Ensure the field is valid.")

        return sorted_list[:limit]

    def _get_nested_value(self, data: Dict, key_path: Optional[str]) -> Any:
        """Safely retrieve a value from a nested dictionary using a dot-separated path."""
        if not key_path:
            return None
        keys = key_path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


# Snippet to REPLACE the entire GetReportByIDTool class

class GetReportByIDTool(BaseCarrierTool):
    """🔍 Retrieves detailed report information with error analysis."""

    name: str = "get_report_by_id"
    description: str = "🔍 Get detailed report information including error analysis"
    args_schema: Type[BaseModel] = GetReportByIdInput

    def _run(self, report_id: str, include_errors: bool = True, error_limit: int = 100) -> str:
        operation = f"analyzing report {report_id}"
        start_time = datetime.now()

        self.log_operation_start(
            operation,
            report_id=report_id,
            include_errors=include_errors,
            error_limit=error_limit
        )

        try:
            # Step 1: Use the new, clean wrapper method to process artifacts.
            # This call handles the download/unzip/merge logic and returns the final paths.
            artifact_data = self.api_wrapper.process_report_artifacts(report_id)
            report_metadata = artifact_data["metadata"]
            error_log_path = artifact_data["error_log_path"]

            # Step 2: Process errors if requested, using the path from the artifact data.
            error_analysis = {"errors_included": False, "error_count": 0}
            if include_errors:
                error_analysis = self._analyze_errors(error_log_path, error_limit)

            # Step 3: Enhance the metadata with recommendations.
            enhanced_report = self._enhance_with_recommendations(report_metadata, error_analysis)

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(operation, duration)

            return json.dumps({
                "message": "📋 Detailed report analysis completed",
                "report_id": report_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "report_details": enhanced_report,
                "error_analysis": error_analysis,
                "processing_time_seconds": round(duration, 2)
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)

    def _analyze_errors(self, error_log_path: str, limit: int) -> Dict:
        """Analyze error logs with conditional processing."""
        # This method no longer needs the 'report' dict passed to it.
        if not error_log_path or not os.path.exists(error_log_path):
            return {
                "errors_included": False,
                "message": "📂 No error log file found or created. The test likely completed without errors."
            }

        try:
            with open(error_log_path, 'r', encoding='utf-8') as f:
                error_lines = [line.strip() for line in f.readlines()[:limit] if line.strip()]

            return {
                "errors_included": True,
                "error_count": len(error_lines),
                "errors_preview": error_lines[:10],  # Show first 10 errors
                "error_analysis": self._categorize_errors(error_lines),
                "recommendations": self._get_error_recommendations(error_lines)
            }
        except Exception as e:
            logger.warning(f"⚠️ Error processing log file {error_log_path}: {e}")
            return {
                "errors_included": False,
                "error": f"Failed to process error logs: {str(e)}"
            }

    def _categorize_errors(self, error_lines: List[str]) -> Dict:
        """Categorize errors for better analysis."""
        categories = {
            "timeout_errors": 0,
            "connection_errors": 0,
            "http_errors": 0,
            "other_errors": 0
        }

        for line in error_lines:
            line_lower = line.lower()
            if "timeout" in line_lower or "timed out" in line_lower:
                categories["timeout_errors"] += 1
            elif "connection" in line_lower or "connect" in line_lower:
                categories["connection_errors"] += 1
            elif any(code in line for code in ["4xx", "5xx", "404", "500", "502", "503"]):
                categories["http_errors"] += 1
            else:
                categories["other_errors"] += 1

        return categories

    def _get_error_recommendations(self, error_lines: List[str]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        error_text = " ".join(error_lines).lower()

        if "timeout" in error_text:
            recommendations.append("🕐 Consider increasing timeout values")
        if "connection" in error_text:
            recommendations.append("🌐 Check network connectivity and target system capacity")
        if "502" in error_text or "503" in error_text:
            recommendations.append("⚡ Target system may be overloaded - reduce load or check system health")
        if len(error_lines) > 50:
            recommendations.append("🔍 High error count detected - investigate system stability before retesting")

        return recommendations or ["✅ Error patterns look manageable - safe to proceed with analysis"]

    def _enhance_with_recommendations(self, report: Dict, error_analysis: Dict) -> Dict:
        """Enhance report with processing recommendations."""
        enhanced = dict(report)  # Copy original report

        status_raw = report.get("test_status", "unknown")
        if isinstance(status_raw, dict):
            status = str(status_raw.get("status", "unknown")).lower()
        else:
            status = str(status_raw).lower()

        lg_type_raw = report.get("lg_type", "")
        if isinstance(lg_type_raw, dict):
            lg_type = str(lg_type_raw.get("type", "")).lower()
        else:
            lg_type = str(lg_type_raw).lower()

        enhanced["processing_recommendations"] = {
            "status_assessment": self._assess_status(status),
            "recommended_pipeline": f"{lg_type}_to_excel" if lg_type in ["gatling", "jmeter"] else "check_type",
            "readiness_score": self._calculate_readiness_score(status, error_analysis),
            "next_steps": self._generate_next_steps(status, lg_type, error_analysis)
        }

        logger.info(f"✅ Enhanced report with recommendations for status: {status}, lg_type: {lg_type}")
        return enhanced

    def _assess_status(self, status: str) -> Dict:
        """Assess test status for processing readiness."""
        assessments = {
            "finished": {"ready": True, "message": "✅ Ready for processing"},
            "failed": {"ready": False, "message": "❌ Review errors before processing"},
            "running": {"ready": False, "message": "🔄 Wait for completion"},
            "stopped": {"ready": True, "message": "⚠️ Partial data available"}
        }
        return assessments.get(status, {"ready": False, "message": f"❓ Unknown status: {status}"})

    def _calculate_readiness_score(self, status: str, error_analysis: Dict) -> int:
        """Calculate readiness score from 0-100."""
        score = 0
        if status == "finished":
            score += 60
        elif status == "stopped":
            score += 40
        elif status == "failed":
            score += 20

        error_count = error_analysis.get("error_count", 0)
        if error_count == 0:
            score += 40
        elif error_count < 10:
            score += 30
        elif error_count < 50:
            score += 20
        else:
            score += 10
        return min(score, 100)

    def _generate_next_steps(self, status: str, lg_type: str, error_analysis: Dict) -> List[Dict]:
        """Generate actionable next steps with conditional logic and priority assessment."""
        steps = []
        readiness_score = self._calculate_readiness_score(status, error_analysis)
        error_count = error_analysis.get("error_count", 0)

        if status == "finished" and error_count < 10 and readiness_score >= 80:
            steps.append(
                {"action": "🚀 Generate Excel Report", "tool": "process_and_generate_report", "priority": "high",
                 "description": "Ready for immediate processing - optimal conditions detected"})
        if error_count > 0:
            severity = "critical" if error_count > 100 else "moderate" if error_count > 20 else "minor"
            steps.append({"action": f"🔍 Review {severity.title()} Error Patterns",
                          "priority": "medium" if severity != "critical" else "high",
                          "description": f"Analyze {error_count} errors for performance insights"})
        if lg_type in ["gatling", "jmeter"]:
            pipeline_type = f"{lg_type}_to_excel"
            steps.append(
                {"action": f"⚙️ Process with {lg_type.title()} Pipeline", "tool": "process_and_generate_report",
                 "priority": "medium" if readiness_score >= 60 else "low",
                 "description": f"Use {pipeline_type} for optimized report generation"})
        if status == "failed":
            steps.append({"action": "🔧 Investigate Test Failure", "priority": "high",
                          "description": "Review test configuration and system availability before retry"})
        elif status == "running":
            steps.append({"action": "⏳ Monitor Test Progress", "priority": "low",
                          "description": "Wait for completion before processing"})
        elif status == "stopped":
            steps.append({"action": "📊 Partial Data Analysis", "priority": "medium",
                          "description": "Extract insights from incomplete test data"})
        if not steps:
            steps.append({"action": "🔍 Manual Review Required", "priority": "medium",
                          "description": "Report status requires manual assessment"})

        priority_order = {"high": 3, "medium": 2, "low": 1}
        steps.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        return steps
