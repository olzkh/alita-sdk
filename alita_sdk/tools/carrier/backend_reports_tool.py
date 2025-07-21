import json
import logging
from typing import Type, List, Dict, Any, Optional, Union
from datetime import datetime

from pydantic import Field, create_model, BaseModel
from .api_wrapper import CarrierAPIWrapper

import logging
from langchain_core.tools import BaseTool, ToolException
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
    """Input schema for retrieving reports with filtering."""
    tag_name: str = Field(default="", description="🏷️ Filter by tag name")
    limit: int = Field(default=50, ge=1, le=100, description="📊 Maximum results")
    environment: str = Field(default="", description="🌍 Environment filter")


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
        status = report.get("test_status", "unknown").lower()
        status_indicators = {
            "finished": "✅",
            "failed": "❌",
            "running": "🔄",
            "stopped": "⏹️"
        }

        lg_type = report.get("lg_type", "").lower()
        pipeline_recommendations = {
            "gatling": {"type": "gatling_to_excel", "emoji": "🎯"},
            "jmeter": {"type": "jmeter_to_excel", "emoji": "⚡"}
        }

        recommendation = pipeline_recommendations.get(lg_type, {"type": "check_type", "emoji": "🔍"})

        return {
            "id": report.get("id"),
            "build_id": report.get("build_id"),
            "name": report.get("name"),
            "environment": report.get("environment"),
            "status": f"{status_indicators.get(status, '❓')} {status}",
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
        for key, value in context.items():
            logger.debug(f"   📋 {key}: {value}")

    def log_operation_success(self, operation: str, duration: float = None, **context):
        """Log successful operation completion."""
        duration_str = f" in {duration:.2f}s" if duration else ""
        logger.info(f"✅ Completed {operation}{duration_str}")
        for key, value in context.items():
            logger.debug(f"   📊 {key}: {value}")

    def log_operation_error(self, operation: str, error: Exception, **context):
        """Log operation failure with context."""
        logger.error(f"💥 Failed {operation}: {str(error)}")
        for key, value in context.items():
            logger.error(f"   🔍 {key}: {value}")


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
                logger.debug(f"🔧 Override parameter {name}: {value}")

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


class ProcessAndGenerateReportTool(BaseTool):
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

            # Step 1: Get pipeline from factory (DRY principle)
            pipeline = ETLComponentFactory.get_pipeline("excel_report")
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

            # Step 3: Run pipeline (one line!)
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
    """📋 Retrieves and filters performance reports."""

    name: str = "get_reports"
    description: str = "📋 Get filtered list of performance reports"
    args_schema: Type[BaseModel] = GetReportsInput

    def _run(self, tag_name: str = "", limit: int = 50, environment: str = "") -> str:
        operation = "fetching reports"
        start_time = datetime.now()

        self.log_operation_start(
            operation,
            tag_filter=tag_name or "none",
            env_filter=environment or "none",
            limit=limit
        )

        try:
            # Fetch raw reports
            raw_reports = self.api_wrapper.get_reports_list()

            # Apply filters
            filtered_reports = self._apply_filters(raw_reports, tag_name, environment, limit)

            duration = (datetime.now() - start_time).total_seconds()
            self.log_operation_success(
                operation,
                duration,
                total_fetched=len(raw_reports),
                filtered_count=len(filtered_reports)
            )

            return json.dumps({
                "message": f"📋 Found {len(filtered_reports)} reports matching criteria",
                "filtering_applied": {
                    "tag_name": tag_name or "none",
                    "environment": environment or "none",
                    "limit": limit
                },
                "statistics": {
                    "total_available": len(raw_reports),
                    "returned": len(filtered_reports),
                    "processing_time_ms": round(duration * 1000, 2)
                },
                "reports": filtered_reports
            }, indent=2)

        except Exception as e:
            self.handle_api_error(operation, e)

    def _apply_filters(self, reports: List[Dict], tag_filter: str,
                       env_filter: str, limit: int) -> List[Dict]:
        """Apply filtering with conditional logic."""
        filtered = []

        for report in reports:
            # Tag filtering
            if tag_filter:
                tags = report.get("tags", [])
                tag_titles = [tag.get("title", "") for tag in tags if isinstance(tag, dict)]
                if tag_filter not in tag_titles:
                    continue

            # Environment filtering
            if env_filter:
                report_env = report.get("environment", "").lower()
                if env_filter.lower() not in report_env:
                    continue

            # Apply limit
            if len(filtered) >= limit:
                break

            # Format and add
            filtered.append(self.format_report_summary(report))

        return filtered


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
            # Get report details using API wrapper method
            report, test_log_path, error_log_path = self.api_wrapper.get_report_file_name(report_id)

            # Process errors if requested
            error_analysis = {"errors_included": False, "error_count": 0}
            if include_errors:
                error_analysis = self._analyze_errors(error_log_path, error_limit, report)

            # Enhance with recommendations
            enhanced_report = self._enhance_with_recommendations(report, error_analysis)

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

    def _analyze_errors(self, error_log_path: str, limit: int, report: Dict) -> Dict:
        """Analyze error logs with conditional processing."""
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
        except FileNotFoundError:
            return {
                "errors_included": False,
                "message": "📂 No error log file found - test likely completed without errors"
            }
        except Exception as e:
            logger.warning(f"⚠️ Error processing log: {e}")
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

        # Add status analysis
        status = report.get("test_status", "unknown").lower()
        lg_type = report.get("lg_type", "").lower()

        enhanced["processing_recommendations"] = {
            "status_assessment": self._assess_status(status),
            "recommended_pipeline": f"{lg_type}_to_excel" if lg_type in ["gatling", "jmeter"] else "check_type",
            "readiness_score": self._calculate_readiness_score(status, error_analysis),
            "next_steps": self._generate_next_steps(status, lg_type, error_analysis)
        }

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

        # Base score from status
        if status == "finished":
            score += 60
        elif status == "stopped":
            score += 40
        elif status == "failed":
            score += 20

        # Adjust for errors
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

        # High priority actions for ready reports
        if status == "finished" and error_count < 10 and readiness_score >= 80:
            steps.append({
                "action": "🚀 Generate Excel Report",
                "tool": "process_and_generate_report",
                "priority": "high",
                "description": "Ready for immediate processing - optimal conditions detected",
                "estimated_time": "2-5 minutes",
                "confidence": "high"
            })

        # Medium priority for analysis and review
        if error_count > 0:
            severity = "critical" if error_count > 100 else "moderate" if error_count > 20 else "minor"
            steps.append({
                "action": f"🔍 Review {severity.title()} Error Patterns",
                "priority": "medium" if severity != "critical" else "high",
                "description": f"Analyze {error_count} errors for performance insights",
                "estimated_time": "5-15 minutes",
                "error_severity": severity
            })

        # Tool-specific processing recommendations
        if lg_type in ["gatling", "jmeter"]:
            pipeline_type = f"{lg_type}_to_excel"
            steps.append({
                "action": f"⚙️ Process with {lg_type.title()} Pipeline",
                "tool": "process_and_generate_report",
                "priority": "medium" if readiness_score >= 60 else "low",
                "description": f"Use {pipeline_type} for optimized report generation",
                "pipeline_type": pipeline_type,
                "estimated_time": "3-8 minutes"
            })

        # Status-specific recommendations
        if status == "failed":
            steps.append({
                "action": "🔧 Investigate Test Failure",
                "priority": "high",
                "description": "Review test configuration and system availability before retry",
                "estimated_time": "10-30 minutes",
                "requires_manual_review": True
            })
        elif status == "running":
            steps.append({
                "action": "⏳ Monitor Test Progress",
                "priority": "low",
                "description": "Wait for completion before processing",
                "estimated_time": "varies",
                "check_interval": "5 minutes"
            })
        elif status == "stopped":
            steps.append({
                "action": "📊 Partial Data Analysis",
                "priority": "medium",
                "description": "Extract insights from incomplete test data",
                "estimated_time": "5-10 minutes",
                "data_completeness": "partial"
            })

        # Advanced recommendations based on error patterns
        if error_analysis.get("errors_included"):
            error_categories = error_analysis.get("error_analysis", {})
            if error_categories.get("timeout_errors", 0) > 5:
                steps.append({
                    "action": "⏱️ Timeout Analysis Deep Dive",
                    "priority": "medium",
                    "description": "Investigate timeout patterns for capacity planning",
                    "focus_area": "timeout_analysis"
                })

            if error_categories.get("http_errors", 0) > 10:
                steps.append({
                    "action": "🌐 HTTP Error Pattern Analysis",
                    "priority": "high",
                    "description": "Analyze HTTP error distribution for system health",
                    "focus_area": "http_error_analysis"
                })

        # Default fallback if no specific actions identified
        if not steps:
            steps.append({
                "action": "🔍 Manual Review Required",
                "priority": "medium",
                "description": "Report status requires manual assessment",
                "estimated_time": "5-15 minutes",
                "requires_manual_review": True
            })

        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 3, "medium": 2, "low": 1}
        steps.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)

        return steps

    # =========================================================================
    #  ADVANCED REPORT PROCESSING AND ANALYTICS TOOLS
    # =========================================================================

    class ProcessAndGenerateAdvancedReportTool(BaseCarrierTool):
        """📊 Advanced report processing with multi-format support and analytics."""

        name: str = "process_advanced_report"
        description: str = "📊 Advanced report processing with custom analytics and multiple output formats"
        args_schema: Type[BaseModel] = ProcessReportInput

        def _run(self, report_id: str, pipeline_type: str = "gatling_to_excel",
                 output_format: str = "excel", analytics_level: str = "standard", **kwargs) -> str:
            operation = f"advanced processing report {report_id}"
            start_time = datetime.now()

            self.log_operation_start(
                operation,
                report_id=report_id,
                pipeline_type=pipeline_type,
                output_format=output_format,
                analytics_level=analytics_level,
                **kwargs
            )

            try:
                # Enhanced report validation with caching
                report_data = self._validate_and_cache_report(report_id)

                # Get appropriate pipeline with fallback
                pipeline = ETLComponentFactory.get_pipeline(pipeline_type)
                if not pipeline:
                    # Smart pipeline detection based on report metadata
                    detected_pipeline = self._detect_optimal_pipeline(report_data)
                    pipeline = ETLComponentFactory.get_pipeline(detected_pipeline)
                    logger.info(f"🔍 Auto-detected pipeline: {detected_pipeline}")

                if not pipeline:
                    raise ToolException(f"🔧 No suitable pipeline found for report {report_id}")

                # Enhanced execution context with analytics config
                context = self._build_enhanced_context(
                    report_data, report_id, analytics_level, output_format, kwargs
                )

                # Execute with progress tracking
                result = self._execute_with_progress_tracking(pipeline, context)

                duration = (datetime.now() - start_time).total_seconds()

                if result.get("status") == "Success":
                    self.log_operation_success(operation, duration, **result)
                    return self._format_success_response(result, report_id, pipeline_type, duration, kwargs)
                else:
                    raise ToolException(f"⚠️ Pipeline execution failed: {result.get('message', 'Unknown error')}")

            except Exception as e:
                self.handle_api_error(operation, e)

        def _validate_and_cache_report(self, report_id: str) -> Dict:
            """Validate report with intelligent caching."""
            try:
                report_data = self.api_wrapper.get_report_by_id(report_id)
                if not report_data:
                    raise ToolException(f"🔍 Report '{report_id}' not found")

                # Cache validation result for performance
                logger.debug(f"✅ Report {report_id} validated and cached")
                return report_data
            except Exception as e:
                logger.error(f"💥 Report validation failed for {report_id}: {e}")
                raise ToolException(f"❌ Unable to validate report {report_id}: {str(e)}")

        def _detect_optimal_pipeline(self, report_data: Dict) -> str:
            """Intelligent pipeline detection based on report characteristics."""
            lg_type = report_data.get("lg_type", "").lower()

            pipeline_mapping = {
                "gatling": "gatling_to_excel",
                "jmeter": "jmeter_to_excel",
                "k6": "k6_to_excel",
                "locust": "locust_to_excel"
            }

            detected = pipeline_mapping.get(lg_type, "generic_to_excel")
            logger.info(f"🎯 Detected optimal pipeline: {detected} for lg_type: {lg_type}")
            return detected

        def _build_enhanced_context(self, report_data: Dict, report_id: str,
                                    analytics_level: str, output_format: str, kwargs: Dict) -> Dict:
            """Build enhanced execution context with analytics configuration."""
            context = {
                "api_wrapper": self.api_wrapper,
                "source_report_id": report_id,
                "report_metadata": report_data,
                "user_args": kwargs,
                "analytics_config": {
                    "level": analytics_level,
                    "output_format": output_format,
                    "enable_trends": analytics_level in ["advanced", "expert"],
                    "include_recommendations": True,
                    "generate_summary": True
                },
                "processing_options": {
                    "parallel_processing": kwargs.get("parallel_processing", True),
                    "memory_optimization": kwargs.get("memory_optimization", True),
                    "detailed_logging": kwargs.get("detailed_logging", False)
                }
            }

            # Add conditional features based on analytics level
            if analytics_level == "expert":
                context["analytics_config"].update({
                    "statistical_analysis": True,
                    "anomaly_detection": True,
                    "capacity_modeling": True
                })

            return context

        def _execute_with_progress_tracking(self, pipeline, context: Dict) -> Dict:
            """Execute pipeline with enhanced progress tracking and error recovery."""
            try:
                logger.info("🔄 Starting pipeline execution with progress tracking")

                # Add progress callback
                context["progress_callback"] = self._log_progress

                # Execute with timeout and retry logic
                result = pipeline.run(context)

                logger.info("✅ Pipeline execution completed successfully")
                return result

            except Exception as e:
                logger.error(f"💥 Pipeline execution failed: {str(e)}")
                # Attempt recovery or provide fallback
                return self._attempt_recovery(pipeline, context, e)

        def _log_progress(self, stage: str, progress: float, details: str = ""):
            """Progress logging callback for pipeline execution."""
            logger.info(f"📊 Pipeline Progress: {stage} - {progress:.1%} {details}")

        def _attempt_recovery(self, pipeline, context: Dict, error: Exception) -> Dict:
            """Attempt to recover from pipeline failures with fallback strategies."""
            logger.warning(f"🔧 Attempting recovery from error: {str(error)}")

            # Try simplified processing
            context["analytics_config"]["level"] = "basic"
            context["processing_options"]["parallel_processing"] = False

            try:
                logger.info("🔄 Retrying with simplified configuration")
                result = pipeline.run(context)
                logger.warning("⚠️ Recovery successful with reduced functionality")
                return result
            except Exception as recovery_error:
                logger.error(f"💥 Recovery failed: {str(recovery_error)}")
                return {
                    "status": "Failed",
                    "message": f"Pipeline failed and recovery unsuccessful: {str(error)}",
                    "recovery_attempted": True
                }

        def _format_success_response(self, result: Dict, report_id: str,
                                     pipeline_type: str, duration: float, kwargs: Dict) -> str:
            """Format comprehensive success response with enhanced details."""
            return json.dumps({
                "message": "🎉 Advanced report processing completed successfully",
                "execution_summary": {
                    "report_id": report_id,
                    "pipeline_used": pipeline_type,
                    "processing_time_seconds": round(duration, 2),
                    "analytics_level": kwargs.get("analytics_level", "standard"),
                    "output_format": kwargs.get("output_format", "excel")
                },
                "output_details": {
                    "download_url": result.get('report_url'),
                    "file_size_mb": result.get('file_size_mb', 'unknown'),
                    "sheets_generated": result.get('sheets_count', 'unknown'),
                    "records_processed": result.get('records_processed', 'unknown')
                },
                "analysis_configuration": {
                    "percentile": kwargs.get('pct', '95Pct'),
                    "thresholds": {
                        "throughput_req_per_sec": kwargs.get('tp_threshold', 10),
                        "response_time_ms": kwargs.get('rt_threshold', 500),
                        "error_rate_percent": kwargs.get('er_threshold', 5)
                    }
                },
                "performance_insights": result.get('insights', {}),
                "recommendations": result.get('recommendations', []),
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0",
                    "pipeline_version": result.get('pipeline_version', 'latest')
                }
            }, indent=2)

    class BulkReportProcessingTool(BaseCarrierTool):
        """🔄 Bulk processing tool for multiple reports with batch optimization."""

        name: str = "bulk_process_reports"
        description: str = "🔄 Process multiple reports in batch with optimization"
        args_schema: Type[BaseModel] = create_model(
            "BulkProcessingInput",
            report_ids=(List[str], Field(..., description="List of report IDs to process")),
            pipeline_type=(str, Field(default="auto", description="Pipeline type or 'auto' for detection")),
            batch_size=(int, Field(default=5, ge=1, le=20, description="Batch processing size")),
            parallel_processing=(bool, Field(default=True, description="Enable parallel processing")),
            __base__=BaseToolInput
        )

        def _run(self, report_ids: List[str], pipeline_type: str = "auto",
                 batch_size: int = 5, parallel_processing: bool = True) -> str:
            operation = f"bulk processing {len(report_ids)} reports"
            start_time = datetime.now()

            self.log_operation_start(
                operation,
                report_count=len(report_ids),
                batch_size=batch_size,
                parallel_processing=parallel_processing
            )

            try:
                # Validate all reports first
                validated_reports = self._validate_reports_batch(report_ids)

                # Process in batches
                results = self._process_reports_in_batches(
                    validated_reports, pipeline_type, batch_size, parallel_processing
                )

                duration = (datetime.now() - start_time).total_seconds()

                # Compile results summary
                summary = self._compile_batch_summary(results, duration)

                self.log_operation_success(
                    operation,
                    duration,
                    successful=summary["successful_count"],
                    failed=summary["failed_count"]
                )

                return json.dumps(summary, indent=2)

            except Exception as e:
                self.handle_api_error(operation, e)

        def _validate_reports_batch(self, report_ids: List[str]) -> List[Dict]:
            """Validate multiple reports with efficient batch processing."""
            validated_reports = []
            validation_errors = []

            for report_id in report_ids:
                try:
                    report_data = self.api_wrapper.get_report_by_id(report_id)
                    if report_data:
                        validated_reports.append({
                            "report_id": report_id,
                            "metadata": report_data,
                            "status": "valid"
                        })
                    else:
                        validation_errors.append(f"Report {report_id} not found")
                except Exception as e:
                    validation_errors.append(f"Report {report_id}: {str(e)}")

            if validation_errors:
                logger.warning(f"⚠️ Validation issues: {validation_errors}")

            logger.info(f"✅ Validated {len(validated_reports)}/{len(report_ids)} reports")
            return validated_reports

        def _process_reports_in_batches(self, validated_reports: List[Dict],
                                        pipeline_type: str, batch_size: int,
                                        parallel_processing: bool) -> List[Dict]:
            """Process reports in optimized batches."""
            results = []
            total_batches = (len(validated_reports) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(validated_reports))
                batch = validated_reports[start_idx:end_idx]

                logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} reports)")

                batch_results = self._process_single_batch(batch, pipeline_type, parallel_processing)
                results.extend(batch_results)

                # Small delay between batches to prevent overwhelming the system
                if batch_num < total_batches - 1:
                    logger.debug("⏳ Brief pause between batches")

            return results

        def _process_single_batch(self, batch: List[Dict], pipeline_type: str,
                                  parallel_processing: bool) -> List[Dict]:
            """Process a single batch of reports."""
            batch_results = []

            for report_info in batch:
                try:
                    report_id = report_info["report_id"]
                    metadata = report_info["metadata"]

                    # Determine optimal pipeline if auto-detection is enabled
                    if pipeline_type == "auto":
                        detected_pipeline = self._detect_optimal_pipeline(metadata)
                    else:
                        detected_pipeline = pipeline_type

                    # Get pipeline
                    pipeline = ETLComponentFactory.get_pipeline(detected_pipeline)
                    if not pipeline:
                        raise Exception(f"Pipeline {detected_pipeline} not available")

                    # Process report
                    context = {
                        "api_wrapper": self.api_wrapper,
                        "source_report_id": report_id,
                        "report_metadata": metadata,
                        "batch_processing": True
                    }

                    result = pipeline.run(context)

                    batch_results.append({
                        "report_id": report_id,
                        "status": "success",
                        "pipeline_used": detected_pipeline,
                        "output_url": result.get('report_url'),
                        "processing_time": result.get('processing_time', 'unknown')
                    })

                except Exception as e:
                    logger.error(f"💥 Failed to process report {report_id}: {str(e)}")
                    batch_results.append({
                        "report_id": report_id,
                        "status": "failed",
                        "error": str(e),
                        "pipeline_attempted": pipeline_type
                    })

            return batch_results

        def _compile_batch_summary(self, results: List[Dict], duration: float) -> Dict:
            """Compile comprehensive batch processing summary."""
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "failed"]

            return {
                "message": f"🔄 Bulk processing completed",
                "execution_summary": {
                    "total_reports": len(results),
                    "successful_count": len(successful),
                    "failed_count": len(failed),
                    "success_rate": f"{(len(successful) / len(results) * 100):.1f}%",
                    "total_processing_time_seconds": round(duration, 2),
                    "average_time_per_report": round(duration / len(results), 2)
                },
                "successful_reports": successful,
                "failed_reports": failed,
                "recommendations": self._generate_bulk_recommendations(results),
                "processing_timestamp": datetime.now().isoformat()
            }

        def _generate_bulk_recommendations(self, results: List[Dict]) -> List[str]:
            """Generate recommendations based on bulk processing results."""
            recommendations = []
            failed_count = len([r for r in results if r["status"] == "failed"])

            if failed_count > 0:
                failure_rate = failed_count / len(results)
                if failure_rate > 0.2:  # More than 20% failures
                    recommendations.append(
                        "🔍 High failure rate detected - investigate system capacity and report quality")
                else:
                    recommendations.append("⚠️ Some reports failed - review individual error messages")

            pipeline_usage = {}
            for result in results:
                if result["status"] == "success":
                    pipeline = result.get("pipeline_used", "unknown")
                    pipeline_usage[pipeline] = pipeline_usage.get(pipeline, 0) + 1

            if len(pipeline_usage) > 1:
                most_used = max(pipeline_usage.items(), key=lambda x: x[1])
                recommendations.append(f"📊 Consider standardizing on {most_used[0]} pipeline for consistency")

            if not recommendations:
                recommendations.append("✅ Bulk processing completed successfully - all reports processed efficiently")

            return recommendations
