import json
from typing import Type, List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, BaseModel
from .api_wrapper import CarrierAPIWrapper

from .etl.etl_factory import ETLComponentFactory

logger = logging.getLogger(__name__)


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


class GetReportByIDTool(BaseCarrierTool):
    """Tool for retrieving performance test report by ID."""

    name: str = "get_report_by_id"
    description: str = "Get performance test report details by report ID"

    def _run(self, report_id: str) -> str:
        """Execute report retrieval."""
        operation = f"retrieving report {report_id}"

        try:
            logger.info(f"🚀 Starting {operation}")

            # Get report metadata
            report_info = self.api_wrapper.get_report_metadata(report_id)

            if not report_info:
                return f"❌ Report {report_id} not found"

            # Format and return the report summary
            return self._format_report_summary(report_info, report_id)

        except Exception as e:
            error_msg = f"💥 Failed {operation}: {str(e)}"
            logger.error(error_msg)
            raise ToolException(error_msg)

    def _format_report_summary(self, report_info: Dict[str, Any], report_id: str) -> str:
        """Format report information into a readable summary."""
        try:
            # Extract key information
            test_name = report_info.get('name', 'Unknown')
            status = report_info.get('status', 'Unknown')
            start_time = report_info.get('start_time', 'N/A')
            duration = report_info.get('duration', 'N/A')

            # Build summary
            summary = f"""
📊 **Performance Test Report #{report_id}**

**Test Name:** {test_name}
**Status:** {status}
**Start Time:** {start_time}
**Duration:** {duration}

**Key Metrics:**
"""

            # Add performance metrics if available
            if 'metrics' in report_info:
                metrics = report_info['metrics']
                summary += f"""
- **Total Requests:** {metrics.get('total_requests', 'N/A')}
- **Success Rate:** {metrics.get('success_rate', 'N/A')}%
- **Average Response Time:** {metrics.get('avg_response_time', 'N/A')} ms
- **95th Percentile:** {metrics.get('pct95', 'N/A')} ms
- **Throughput:** {metrics.get('throughput', 'N/A')} req/s
"""

            # Add error information if any
            if 'errors' in report_info and report_info['errors']:
                summary += "\n**Errors:**\n"
                for error in report_info['errors'][:5]:  # Show first 5 errors
                    summary += f"- {error}\n"

            # Add link to full report if available
            if hasattr(self.api_wrapper, 'url'):
                report_url = f"{self.api_wrapper.url.strip('/')}/-/performance/backend/results?result_id={report_id}"
                summary += f"\n🔗 [View Full Report]({report_url})"

            return summary

        except Exception as e:
            logger.error(f"Error formatting report summary: {e}")
            # Return basic info if formatting fails
            return f"Report #{report_id}: {report_info}"


class CreateBackendExcelReportInput(BaseModel):
    """Input schema for CreateBackendExcelReportTool."""
    report_id: Optional[str] = Field(None, description="Report ID for single report generation")
    test_name: Optional[str] = Field(None, description="Test name for comparison report generation")
    run_count: int = Field(5, ge=1, le=20, description="Number of recent runs to compare (default: 5)")
    enable_ai_analysis: bool = Field(True, description="Enable AI-powered analysis (default: True)")
    output_format: str = Field(default="excel", description="📋 Output format")


class CreateBackendExcelReportTool(BaseCarrierTool):
    """
    Tool for creating backend performance Excel reports.
    Supports single-report generation and multi-report comparisons.
    """

    name: str = "create_backend_excel_report"
    description: str = "Create an Excel report from backend performance test results."
    args_schema: Type[BaseModel] = CreateBackendExcelReportInput
    llm: Optional[Any] = Field(default=None, description="Language model for AI analysis")

    def __init__(self, api_wrapper: CarrierAPIWrapper, llm: Optional[Any] = None, **kwargs):
        super().__init__(api_wrapper=api_wrapper, **kwargs)
        self.llm = llm
        logger.info("CreateBackendExcelReportTool initialized.")

    def _run(self, **kwargs) -> str:
        """Validates input and routes to the correct report creation method."""
        try:
            # Pydantic model validation handles type conversion and constraints
            params = CreateBackendExcelReportInput(**kwargs)
            llm = kwargs.get('llm', self.llm)

            if params.report_id and params.test_name:
                return self._error_response("Provide either 'report_id' or 'test_name', not both.")

            if params.report_id:
                return self._create_single_report(params.report_id, llm)

            if params.test_name:
                return self._create_comparison_report(params.test_name, params.run_count, params.enable_ai_analysis,
                                                      llm)

            return self._error_response("Either 'report_id' or 'test_name' must be provided.")

        except Exception as e:
            logger.error(f"Report creation failed: {e}", exc_info=True)
            return self._error_response(f"An unexpected error occurred: {e}")

    def _create_single_report(self, report_id: str, llm: Optional[Any]) -> str:
        """Generates an Excel report for a single test run."""
        logger.info(f"Creating single report for ID: {report_id}")
        try:
            report_meta = self.api_wrapper.get_report_metadata(report_id)
            lg_type = report_meta.get('lg_type', 'jmeter').lower()
            pipeline = ETLComponentFactory.get_pipeline(f"{lg_type}_to_excel")

            context = {"api_wrapper": self.api_wrapper, "report_id": report_id, "llm": llm, "user_args": {}}
            result = pipeline.run(context)

            if result.get("status") == "success":
                return self._success_response({
                    "message": "Single report created successfully.",
                    "download_url": result.get("download_url", "N/A")
                })
            return self._error_response(f"Pipeline execution failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Single report creation for ID {report_id} failed: {e}", exc_info=True)
            return self._error_response(str(e))

    def _create_comparison_report(self, test_name: str, run_count: int, enable_ai: bool, llm: Optional[Any]) -> str:
        """Generates a comparison report across multiple test runs."""
        logger.info(f"Creating comparison report for test: '{test_name}' using last {run_count} runs.")
        try:
            reports_meta = self._get_recent_test_reports(test_name, run_count)
            if len(reports_meta) < 2:
                return self._error_response(f"Found only {len(reports_meta)} finished reports. Need at least 2.")

            lg_type = reports_meta[0].get('lg_type', 'jmeter').lower()
            pipeline = ETLComponentFactory.get_pipeline(f"{lg_type}_comparison_between_the_tests")

            context = {
                "api_wrapper": self.api_wrapper,
                "llm": llm,
                "reports_meta": reports_meta,
                "enable_ai_analysis": enable_ai,
                "test_name": test_name,
                "user_args": {}
            }
            result = pipeline.run(context)

            if result.get("status") == "success":
                return self._success_response({
                    "message": "Consolidated comparison report created successfully.",
                    "reports_analyzed": len(reports_meta),
                    "download_url": result.get("download_url", "N/A")
                })

            error_message = result.get('error', 'Unknown error during pipeline execution')
            return self._error_response(f"Comparison pipeline failed: {error_message}")

        except Exception as e:
            logger.error(f"Comparison report for '{test_name}' failed: {e}", exc_info=True)
            return self._error_response(str(e))

    def _get_recent_test_reports(self, test_name: str, run_count: int) -> List[Dict]:
        """Fetches metadata for recent, finished test runs."""
        all_reports = self.api_wrapper.get_reports_list()

        finished_reports = [
            r for r in all_reports
            if r.get("name") == test_name and self._is_finished_status(r)
        ]

        return sorted(
            finished_reports,
            key=lambda r: r.get("start_time", ""),
            reverse=True
        )[:run_count]

    def _is_finished_status(self, report: Dict) -> bool:
        """Checks if a report's status is 'finished'."""
        status_info = report.get("test_status", {})
        if isinstance(status_info, dict):
            return status_info.get("status", "").lower() == "finished"
        return str(status_info).lower() == "finished"

    def _success_response(self, data: Dict) -> str:
        """Formats a standard success JSON response."""
        return json.dumps({"status": "success", **data}, indent=2)

    def _error_response(self, message: str) -> str:
        """Formats a standard error JSON response."""
        return json.dumps({"status": "error", "message": f"❌ {message}"}, indent=2)
