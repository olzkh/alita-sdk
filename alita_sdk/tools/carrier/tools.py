import time
import logging
from functools import wraps
from typing import Dict, List, Tuple, Type, Optional

from langchain_core.tools import BaseTool

from .api_wrapper import CarrierAPIWrapper
from .utils.intent_utils import CarrierIntentExtractor, CarrierIntent
from .parameter_extractor import extract_parameters
from .metrics import ToolkitMetrics

# Import all tool classes
from .tickets_tool import FetchTicketsTool, CreateTicketTool
from .backend_reports_tool import GetReportsTool, GetReportByIDTool, ProcessAndGenerateReportTool
from .backend_tests_tool import GetBackendTestsTool, GetTestByIDTool, RunTestByIDTool, CreateBackendTestTool
from .ui_reports_tool import GetUIReportsTool, GetUIReportByIDTool, GetUITestsTool
from .run_ui_test_tool import RunUITestTool
from .update_ui_test_schedule_tool import UpdateUITestScheduleTool
from .create_ui_excel_report_tool import CreateUIExcelReportTool
from .create_ui_test_tool import CreateUITestTool
from .cancel_ui_test_tool import CancelUITestTool

logger = logging.getLogger(__name__)

# List of all atomic tool classes for easy iteration
ATOMIC_TOOL_CLASSES: List[Type[BaseTool]] = [
    FetchTicketsTool, CreateTicketTool,
    GetReportsTool, GetReportByIDTool, ProcessAndGenerateReportTool,
    GetBackendTestsTool, GetTestByIDTool, RunTestByIDTool, CreateBackendTestTool,
    GetUIReportsTool, GetUIReportByIDTool, GetUITestsTool,
    RunUITestTool, UpdateUITestScheduleTool, CreateUIExcelReportTool,
    CreateUITestTool, CancelUITestTool,
]


def circuit_breaker(max_failures: int = 3, reset_timeout: int = 60):
    """
    Circuit breaker pattern for tool execution reliability.
    Prevents cascading failures by temporarily disabling failing operations.
    """

    def decorator(func):
        func._failures = 0
        func._last_failure = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()

            # Reset circuit if timeout has passed
            if func._failures >= max_failures:
                if current_time - func._last_failure > reset_timeout:
                    func._failures = 0
                    logger.info(f"[CircuitBreaker] Reset for {func.__name__}")
                else:
                    logger.warning(f"[CircuitBreaker] Circuit open for {func.__name__}")
                    raise Exception(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                func._failures = 0  # Reset on success
                return result
            except Exception as e:
                func._failures += 1
                func._last_failure = current_time
                logger.error(f"[CircuitBreaker] Failure {func._failures}/{max_failures} for {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


class CarrierIntentMetaTool(BaseTool):
    """
    Production-ready meta-tool that orchestrates intent recognition and tool execution.
    Implements SOLID principles:
    - Single Responsibility: Only handles orchestration
    - Open/Closed: New tools can be added without modifying this class
    - Dependency Inversion: Depends on abstractions (BaseTool) not concrete implementations
    """
    intent_extractor: CarrierIntentExtractor
    api_wrapper: CarrierAPIWrapper
    tool_map: Dict[tuple, Type]
    fallback_enabled: bool = True
    metrics: Optional[ToolkitMetrics] = None

    name: str = "carrier_performance_analytics_meta_tool"
    description: str = """
    Intelligent Performance Analytics toolkit that routes user intents to appropriate Carrier tools.
    Designed to streamline daily activities for Performance Analysis teams.

    Simply describe what you want to do in natural language. Examples:
    - "Show me UI test reports from last week"
    - "Run backend test with ID 5134 and generate Excel report"
    - "Create a ticket for failed performance test"
    - "Schedule UI test to run daily at 2 AM"
    """

    @circuit_breaker(max_failures=3, reset_timeout=30)
    def _run(self, user_message: str) -> str:
        """
        Main execution method with comprehensive error handling.
        Follows a clear pipeline: Extract Intent → Route → Execute → Format Response.
        This method is now the single point of orchestration and error handling.
        """
        if self.metrics:
            self.metrics.intent_extraction_count += 1

        logger.info(f"[CarrierMetaTool] Processing: {user_message[:100]}...")

        try:
            # Step 1: Extract intent directly. The extractor handles its own retries.
            intent = self.intent_extractor.extract_intent(user_message)

            if not intent:
                logger.error("[CarrierMetaTool] Intent extraction failed to return an intent.")
                if self.metrics:
                    self.metrics.failed_tool_calls += 1  # Count as a failed call
                return self._handle_intent_extraction_failure(user_message)

            logger.info(f"[CarrierMetaTool] Intent recognized: {intent.task_type}.{intent.action}")

            # Step 2: Route and execute the identified tool
            result = self._route_and_execute_tool(intent, user_message)

            if self.metrics:
                self.metrics.successful_tool_calls += 1

            return result

        except Exception as e:
            # This is the critical top-level catch-all. It will now catch the
            # TypeError from the extractor and prevent a crash.
            logger.exception("[CarrierMetaTool] A critical, unhandled error occurred during execution.")
            if self.metrics:
                self.metrics.failed_tool_calls += 1
            # Return a formatted, user-friendly error message.
            return self._handle_execution_error(e, user_message)

    def _extract_intent_with_retry(self, user_message: str, retry_count: int = 0) -> CarrierIntent:
        """Extract intent with exponential backoff retry logic"""
        try:
            intent = self.intent_extractor.extract_intent(user_message)
            logger.info(f"[CarrierMetaTool] Intent extracted on attempt {retry_count + 1}")
            return intent


        except Exception as e:

            logger.warning(f"[CarrierMetaTool] Intent extraction failed (attempt {retry_count + 1}): {e}")

            if retry_count < self.max_retries - 1:

                wait_time = (2 ** retry_count) * 0.1  # Exponential backoff

                time.sleep(wait_time)

                return self._extract_intent_with_retry(user_message, retry_count + 1)

            else:

                logger.error(f"[CarrierMetaTool] Intent extraction failed after {self.max_retries} attempts")

                return None

    def _format_error_response(self, error_message: str, tool_name: str = None, suggestions: List[str] = None) -> str:
        """Format error responses with helpful context"""
        response = f"❌ **Performance Analytics Error**\n\n"

        if tool_name:
            response += f"**Tool:** {tool_name}\n"

        response += f"**Error:** {error_message}\n\n"

        if suggestions:
            response += "**💡 Suggestions:**\n"
            for suggestion in suggestions:
                response += f"- {suggestion}\n"
        else:
            # Default suggestions
            response += "**💡 Try:**\n"
            response += "- Check if the ID/reference is correct\n"
            response += "- Ensure you have access to this resource\n"
            response += "- Try rephrasing your request\n"

        return response

    def _route_and_execute_tool(self, intent: CarrierIntent, user_message: str) -> str:
        """
        Routes intent to tool and executes it with comprehensive logging.
        """
        logger.info(f"🎯 [CarrierMetaTool] Routing intent: {intent.task_type}.{intent.action}")
        logger.info(f"   Full intent object: {intent}")

        tool_class = self.tool_map.get((intent.task_type, intent.action))

        if not tool_class:
            logger.warning(f"⚠️ [CarrierMetaTool] No mapping found for: {intent.task_type}.{intent.action}")
            logger.info(f"   Available mappings: {list(self.tool_map.keys())}")
            return self._handle_unmapped_intent(intent, user_message)

        logger.critical(f"🔍 [DEBUG] Tool class name being processed: {tool_class.__name__}")

        tool_kwargs = {}
        # Add this temporary debug in tools.py before parameter extraction:

        logger.critical(f"🔍 [DEBUG] Tool class: {tool_class.__class__.__name__}")
        logger.critical(f"🔍 [DEBUG] Tool.args_schema: {getattr(tool_class, 'args_schema', 'None')}")
        if hasattr(tool_class, 'args_schema') and tool_class.args_schema:
            schema = tool_class.args_schema
            logger.critical(f"🔍 [DEBUG] args_schema type: {type(schema)}")
            if hasattr(schema, 'model_fields'):
                logger.critical(f"🔍 [DEBUG] args_schema.model_fields: {schema.model_fields}")
            if hasattr(schema, '__fields__'):
                logger.critical(f"🔍 [DEBUG] args_schema.__fields__: {schema.__fields__}")

        try:
            # Extract parameters with enhanced logging
            logger.info(f"📋 [CarrierMetaTool] Starting parameter extraction")
            tool_kwargs = extract_parameters(tool_class, intent)

            logger.info(f"📦 [CarrierMetaTool] Extracted parameters: {list(tool_kwargs.keys())}")
            logger.info(f"   Full parameters: {tool_kwargs}")
            schema = getattr(tool_class, 'args_schema', None)
            if schema:
                # Check for any required fields in the tool's input schema
                # This works for both Pydantic v1 (__fields__) and v2 (model_fields)
                schema_fields = getattr(schema, 'model_fields', getattr(schema, '__fields__', {}))
                required_fields = {name for name, field in schema_fields.items() if field.is_required()}

                # If there are required fields that were not extracted, it's an error.
                if required_fields and not tool_kwargs:
                    logger.error(
                        f"❌ [CarrierMetaTool] Missing required parameters for {tool_class.__name__}: {required_fields}")
                    if self.metrics:
                        self.metrics.parameter_extraction_errors += 1

                    suggestions = [f"Please provide a value for '{field}'." for field in required_fields]
                    return self._format_error_response(
                        f"Missing required parameters: {', '.join(required_fields)}",
                        tool_class.__name__,
                        suggestions
                    )
            # Create tool instance
            logger.info(f"🏗️ [CarrierMetaTool] Creating tool instance: {tool_class.__name__}")
            tool_instance = tool_class(api_wrapper=self.api_wrapper)

            # Execute tool
            start_time = time.time()
            logger.info(
                f"🚀 [CarrierMetaTool] Executing {tool_class.__name__} with parameters: {list(tool_kwargs.keys())}")

            result = tool_instance._run(**tool_kwargs)

            execution_time = time.time() - start_time
            logger.info(f"✅ [CarrierMetaTool] Tool executed successfully in {execution_time:.2f}s")

            # Record metrics
            if self.metrics:
                self.metrics.record_action(
                    task_type=intent.task_type,
                    action=intent.action,
                    success=True,
                    execution_time=execution_time
                )
                return self._format_success_response(result, tool_class.__name__, execution_time)

        except TypeError as e:
            # Specific handling for parameter mismatch errors
            logger.error(f"❌ [CarrierMetaTool] Parameter mismatch for {tool_class.__name__}: {e}")
            logger.error(f"   Provided kwargs: {list(tool_kwargs.keys())}")
            logger.error(f"   Full error: {str(e)}")

            if self.metrics:
                self.metrics.parameter_extraction_errors += 1
                self.metrics.record_action(
                    task_type=intent.task_type,
                    action=intent.action,
                    success=False
                )

            # Try to extract missing parameter from error message
            import re
            match = re.search(r"missing \d+ required positional argument(?:s)?: '(.+?)'", str(e))
            if match:
                missing_param = match.group(1)
                logger.error(f"   Missing parameter identified: {missing_param}")

                suggestions = [
                    f"Include {missing_param} in your request",
                    f"Example: 'generate report from [report_id]'",
                    "Check if all required IDs are provided"
                ]

                return self._format_error_response(
                    f"Missing required parameter '{missing_param}'.",
                    tool_class.__name__,
                    suggestions
                )

            return self._format_error_response(str(e), tool_class.__name__)

        except Exception as e:
            logger.exception(f"❌ [CarrierMetaTool] Tool execution failed: {tool_class.__name__}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            logger.error(f"   Tool kwargs: {tool_kwargs}")

            if self.metrics:
                self.metrics.record_action(
                    task_type=intent.task_type,
                    action=intent.action,
                    success=False
                )

            return self._handle_tool_execution_error(e, tool_class, intent)

    def _handle_tool_execution_error(self, error: Exception, tool_class: Type, intent: CarrierIntent) -> str:
        """Handle errors during tool execution with detailed context"""
        error_type = type(error).__name__
        error_msg = str(error)

        # Specific error handling
        if "404" in error_msg or "not found" in error_msg.lower():
            suggestions = [
                "Verify the resource ID is correct",
                "Check if the resource still exists",
                "Try listing available resources first"
            ]
            return self._format_error_response(
                f"Resource not found: {error_msg}",
                tool_class.__name__,
                suggestions
            )
        elif "403" in error_msg or "unauthorized" in error_msg.lower():
            suggestions = [
                "Verify you have access to this resource",
                "Check your permissions",
                "Contact your administrator if access is needed"
            ]
            return self._format_error_response(
                f"Access denied: {error_msg}",
                tool_class.__name__,
                suggestions
            )
        else:
            # Generic error
            return self._format_error_response(
                f"{error_type}: {error_msg}",
                tool_class.__name__
            )

    # === Error Handling Methods (DRY: Centralized error formatting) ===

    def _handle_intent_extraction_failure(self, user_message: str) -> str:

        """Handle complete intent extraction failure with helpful guidance"""

        if self.metrics and self.fallback_enabled:
            self.metrics.fallback_activations += 1

        return self._format_error_response(
            error_message=f"I couldn't understand your request: \"{user_message}\"",
            suggestions=[self._get_usage_examples(),
                         "Be specific about IDs, timeframes, and the type of action you want."]
        )

    def _handle_unmapped_intent(self, intent: CarrierIntent, user_message: str) -> str:

        """Handle understood but unmapped intent"""

        if self.metrics and self.fallback_enabled:
            self.metrics.fallback_activations += 1

        available_actions = self._get_available_actions()

        understood_info = [

            f"**Task Type:** {intent.task_type}",

            f"**Action:** {intent.action}",

            f"**Entities:** {getattr(intent, 'entities', {})}",

            f"**Fields:** {getattr(intent, 'field_requests', [])}"

        ]

        return self._format_error_response(
            error_message="Intent Understood But Action Not Available\n\nWhat I understood:\n" + "\n".join(
                understood_info),
            tool_name=None,
        )

    def _handle_execution_error(self, error: Exception, user_message: str) -> str:

        """Handle general execution errors"""

        return self._format_error_response(
            error_message="Performance Analytics Tool Error\n\n"
                          f"**Request:** {user_message}\n**Error:** {str(error)[:200]}",

        )

    # === Formatting Methods (DRY: Reusable formatting logic) ===

    def _format_success_response(self, result: str, tool_name: str, execution_time: float) -> str:

        """Format successful execution response"""

        return f"""✅ **Performance Analytics Task Completed**


        **Tool:** {tool_name}

        **Execution Time:** {execution_time:.2f}s


        **Results:**

        {result}

        ---
        *Automated by Carrier Performance Analytics Toolkit v2.1.0*"""

        # === Helper Methods (DRY: Reusable data access) ===

    def _get_usage_examples(self) -> str:
        """Get categorized usage examples"""
        categories = {
            "📊 Reports & Analysis": [
                '"Get detailed analysis for report ID 5134"',
                '"Generate Excel for backend report 5134"'
                '"Generate Excel report from report ID 5134"',
                '"Get UI test reports from last 7 days"'
            ],
            "🧪 Test Management": [
                '"Run UI test with ID 5134"',
                '"Create backend test for performance analysis"',
                '"Schedule UI test to run daily"'
            ],
            "🎫 Issue Tracking": [
                '"Create ticket for failed test 5134"',
                '"Show all open tickets for my project"'
            ]
        }

        sections = []
        for category, examples in categories.items():
            sections.append(f"**{category}:**")
            sections.extend([f"- {ex}" for ex in examples])
            sections.append("")

        return "\n".join(sections)

    def _get_available_actions(self) -> str:
        """Get available actions grouped by category"""
        # Group actions by task type
        categories = {}
        for (task_type, action) in self.tool_map.keys():
            if task_type not in categories:
                categories[task_type] = []
            categories[task_type].append(action)

        # Format with icons
        category_icons = {
            'ticket_action': '🎫',
            'backend_analysis': '📊',
            'test_management': '🧪',
            'test_execution': '▶️',
            'ui_analysis': '🖥️',
            'ui_test_management': '🎮',
            'ui_test_execution': '🚀'
        }

        sections = ["**Available Performance Analytics actions:**"]
        for category, actions in sorted(categories.items()):
            icon = category_icons.get(category, '🔧')
            category_name = category.replace('_', ' ').title()
            sections.append(f"\n**{icon} {category_name}:**")

            # Show first 3 actions
            for action in actions[:3]:
                sections.append(f"  - {action.replace('_', ' ')}")

            if len(actions) > 3:
                sections.append(f"  - ... and {len(actions) - 3} more")

        return "\n".join(sections)

    def _get_troubleshooting_steps(self) -> str:
        """Get standard troubleshooting steps"""
        steps = [
            "**🔧 Troubleshooting Steps:**",
            "1. Verify your request format matches the examples",
            "2. Check if required IDs/parameters are valid",
            "3. Ensure you have proper access permissions",
            "4. Try rephrasing your request"
        ]
        return "\n".join(steps)


# === Tool Mapping Configuration (DRY: Single source of truth) ===

ACTION_TOOL_MAP: Dict[Tuple[str, str], Type[BaseTool]] = {
    # Ticket Management
    ("ticket_action", "get_ticket_list"): FetchTicketsTool,
    ("ticket_action", "create_ticket"): CreateTicketTool,

    # Backend Analysis
    ("backend_analysis", "get_reports"): GetReportsTool,
    ("backend_analysis", "get_report_by_id"): GetReportByIDTool,
    ("backend_analysis", "process_report"): ProcessAndGenerateReportTool,

    # Test Management
    ("backend_test_management", "get_backend_tests"): GetBackendTestsTool,
    ("test_management", "get_test_by_id"): GetTestByIDTool,
    ("test_management", "create_backend_test"): CreateBackendTestTool,

    # Test Execution
    ("test_execution", "run_test"): RunTestByIDTool,

    # UI Analysis
    ("ui_analysis", "get_ui_reports"): GetUIReportsTool,
    ("ui_analysis", "get_ui_report_by_id"): GetUIReportByIDTool,
    ("ui_analysis", "create_ui_excel_report"): CreateUIExcelReportTool,

    # UI Test Management

    ("ui_test_management", "create_ui_test"): CreateUITestTool,
    ("ui_test_management", "update_ui_test_schedule"): UpdateUITestScheduleTool,
    ("ui_test_management", "cancel_ui_test"): CancelUITestTool,

    # UI Test Execution
    ("ui_test_execution", "run_ui_test"): RunUITestTool,
}

# === Action Examples (DRY: Centralized examples) ===

ACTION_EXAMPLES = {
    # Backend Analysis & Reports
    "get_reports": "Show me all backend performance reports",
    "get_report_by_id": "Get detailed analysis for report ID 5134",
    "process_report": "Generate comprehensive Excel report from test 5134",

    # UI Testing & Analysis
    "get_ui_reports": "Show UI test reports for performance analysis",
    "get_ui_report_by_id": "Get UI test report 5134 with detailed metrics",
    "run_ui_test": "Execute UI performance test 5134",
    "create_ui_excel_report": "Generate Excel dashboard from UI test 5134",

    # Test Management
    "get_tests": "List all performance tests in project",
    "run_test": "Execute backend performance test 5134",
    "create_backend_test": "Create new performance regression test",

    # Scheduling & Automation
    "update_ui_test_schedule": "Schedule UI test 5134 to run daily at 2 AM",
    "cancel_ui_test": "Cancel running UI performance test 5134",

    # Issue Management
    "get_ticket_list": "Show performance-related tickets",
    "create_ticket": "Create ticket for failed performance test 5134",
}

DEFAULT_EXAMPLE = "Describe your Performance Analytics task - testing, reporting, or issue tracking"


# === Utility Functions (DRY: Reusable helpers) ===

def get_all_tools(api_wrapper: CarrierAPIWrapper) -> List[BaseTool]:
    """
    Instantiate all atomic tools for direct use.
    NOTE: Use the meta-tool for LLM-driven agents, not this function.
    """
    tools = []
    for tool_class in ATOMIC_TOOL_CLASSES:
        try:
            tool = tool_class(api_wrapper=api_wrapper)
            tools.append(tool)
            logger.info(f"[CarrierToolkit] Instantiated: {tool.name}")
        except Exception as e:
            logger.error(f"[CarrierToolkit] Failed to instantiate {tool_class.__name__}: {e}")

    return tools


def get_supported_intents_and_examples(tool_map: Dict[Tuple[str, str], Type[BaseTool]]) -> Tuple[
    List[Tuple[str, str]], List[str]]:
    """
    Extract supported intent pairs and unique example phrases.
    Used for documentation and intent extractor training.
    """
    pairs = list(tool_map.keys())
    seen = set()
    examples = []

    for (task_type, action) in pairs:
        example = ACTION_EXAMPLES.get(action, DEFAULT_EXAMPLE)
        if example not in seen:
            examples.append(example)
            seen.add(example)

    return pairs, examples


__all__ = [
    # Main components
    "CarrierIntentMetaTool",
    "ACTION_TOOL_MAP",
    "ACTION_EXAMPLES",

    # Utility functions
    "get_all_tools",
    "get_supported_intents_and_examples",
    "circuit_breaker",

    # Tool classes (for direct imports if needed)
    "FetchTicketsTool", "CreateTicketTool",
    "GetReportsTool", "GetReportByIDTool", "ProcessAndGenerateReportTool",
    "GetBackendTestsTool", "GetTestByIDTool", "RunTestByIDTool", "CreateBackendTestTool",
    "GetUIReportsTool", "GetUIReportByIDTool", "GetUITestsTool",
    "RunUITestTool", "UpdateUITestScheduleTool", "CreateUIExcelReportTool",
    "CreateUITestTool", "CancelUITestTool",
]
