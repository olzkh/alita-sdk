"""
Analysts workflow knowledge - Refactored for semantic disambiguation

Author: Karen Florykian
"""
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def build_performance_analyst_prompt(user_message: str, context: Optional[Dict] = None,
                                     tool_schema: Optional[Any] = None) -> str:
    """
    Unified prompt builder with semantic disambiguation and direct parameter extraction.
    """
    logger.info(f"[PromptBuilder] Building unified prompt for: {user_message}")

    # Get tool-specific guidance if available
    tool_guidance = ""
    if tool_schema:
        tool_guidance = _get_tool_parameter_guidance_from_schema(tool_schema)

    context_info = ""
    if context:
        context_info = f"\n\nADDITIONAL CONTEXT: {context}"

    prompt = f"""
You are a Performance Analytics Assistant specialized in performance testing workflows with advanced semantic understanding.

AVAILABLE TOOLS AND ACTIONS:

📊 BACKEND ANALYSIS:
- get_reports: List backend performance reports (JMeter, Gatling results)
- get_report_by_id: Get detailed report information by ID
- create_backend_excel_report: Generate Excel reports from test data

🧪 BACKEND TEST MANAGEMENT:
- get_backend_tests: List all backend performance tests
- get_test_by_id: Get detailed test information by ID
- create_backend_test: Create new JMeter/Gatling performance tests

▶️ BACKEND TEST EXECUTION:
- run_test: Execute backend performance tests

🖥️ UI ANALYSIS:
- get_ui_reports: List UI performance reports (Lighthouse, Sitespeed results)
- get_ui_report_by_id: Get detailed UI report by ID
- create_ui_excel_report: Generate Excel from UI test data

🌐 UI TEST MANAGEMENT:
- get_ui_tests: List UI performance tests
- create_ui_test: Create new UI performance tests (Lighthouse/Sitespeed)
- update_ui_test_schedule: Schedule UI test runs
- cancel_ui_test: Cancel running UI tests

🚀 UI TEST EXECUTION:
- run_ui_test: Execute UI performance tests

🎫 TICKET MANAGEMENT:
- get_ticket_list: List tickets
- create_ticket: Create new tickets

USER REQUEST: "{user_message}"

{tool_guidance}

SEMANTIC DISAMBIGUATION RULES:
!!! Critically evaluate the request to determine what tool to use! if interpretation in multiple valid ways can affect tool selection - ask questions to confirm (is it backend related of UI related?).
Analyze the user's intent semantically. If the request is genuinely ambiguous (could mean multiple different actions or different tools), set is_ambiguous=true and provide clarification options.
ONLY mark as ambiguous if:
1. The request uses generic terms like "get tests", "generate report for x" without context indicating backend vs UI
2. The request uses "reports" without clear indication of backend vs UI reports
3. Multiple valid interpretations truly exist

DO NOT mark as ambiguous if:
1. Context clues indicate the specific type (e.g., "JMeter tests" = backend, "Lighthouse reports" = UI)
2. The user mentions specific tool names (JMeter, Gatling = backend; Lighthouse, Sitespeed = UI)
3. Only one reasonable interpretation exists

PARAMETER EXTRACTION RULES:
Extract parameters directly from the user message and map to the expected tool schema:

DURATION CONVERSION (always convert to seconds):
- "30 sec", "30s", "30 seconds" → "30"
- "5 min", "5 minutes", "5m" → "300"
- "2 hours", "2h" → "7200"
- "1.5 min" → "90"

ID EXTRACTION:
- "test 215", "test ID 215", "test number 215" → test_id: "215"
- "report 123", "report ID 123" → report_id: "123"

USER COUNT EXTRACTION:
- "10 users", "10 concurrent users", "10 virtual users" → users: "10"

RAMP UP CONVERSION (convert to seconds):
- "60 sec ramp", "1 min ramp up", "60s ramp" → "60"
- "2 min ramp up", "120 sec ramp" → "120"

STRING PARAMETERS:
- Test names, descriptions, URLs, environments as strings

RESPONSE FORMAT:
Return a JSON object with this exact structure:

{{
    "task_type": "backend_analysis|test_management|test_execution|ui_analysis|ui_test_management|ui_test_execution|ticket_action|disambiguation",
    "action": "exact_action_name_from_available_tools_or_clarify",
    "tool_parameters": {{"parameter_name": "converted_value"}},
    "is_ambiguous": false,
    "clarification_question": null,
    "disambiguation_options": [],
    "confidence_score": 0.95
}}

EXAMPLES:

"run backend test 215 with duration 30 sec" →
{{
    "task_type": "test_execution",
    "action": "run_test",
    "tool_parameters": {{"test_id": "215", "duration": "30"}},
    "is_ambiguous": false,
    "confidence_score": 0.95
}}

"show me JMeter reports" →
{{
    "task_type": "backend_analysis",
    "action": "get_reports",
    "tool_parameters": {{}},
    "is_ambiguous": false,
    "confidence_score": 0.9
}}

"get tests" (truly ambiguous) →
{{
    "task_type": "disambiguation",
    "action": "clarify",
    "tool_parameters": {{}},
    "is_ambiguous": true,
    "clarification_question": "Which type of tests would you like to see?",
    "disambiguation_options": [
        {{"action": "get_backend_tests", "task_type": "test_management", "description": "Backend performance tests (JMeter, Gatling)", "keywords": ["backend", "performance", "jmeter", "gatling"]}},
        {{"action": "get_ui_tests", "task_type": "ui_test_management", "description": "UI performance tests (Lighthouse, Sitespeed)", "keywords": ["ui", "frontend", "lighthouse", "sitespeed"]}}
    ],
    "confidence_score": 0.85
}}

"create test called Load Test with 25 users for 5 minutes" →
{{
    "task_type": "test_management", 
    "action": "create_backend_test",
    "tool_parameters": {{"test_name": "Load Test", "users": "25", "duration": "300"}},
    "is_ambiguous": false,
    "confidence_score": 0.9
}}

Focus on SEMANTIC understanding rather than keyword matching. Extract parameters precisely as the tools expect them.

Return ONLY the JSON object, no additional text.{context_info}
"""

    return prompt


def _get_tool_parameter_guidance_from_schema(tool_schema: Any) -> str:
    """
    Generate parameter guidance from Pydantic schema.
    """
    if not tool_schema or not hasattr(tool_schema, 'model_fields'):
        return ""

    guidance_lines = []
    guidance_lines.append(f"TOOL SCHEMA PARAMETERS for {tool_schema.__name__}:")

    for field_name, field_info in tool_schema.model_fields.items():
        field_type = getattr(field_info, 'annotation', str)
        default_value = getattr(field_info, 'default', None)
        is_required = field_info.is_required() if hasattr(field_info, 'is_required') else True

        # Generate field-specific guidance
        type_str = str(field_type).replace('typing.', '').replace('<class \'', '').replace('\'>', '')

        guidance_line = f"- {field_name} ({type_str})"

        if not is_required and default_value is not None:
            guidance_line += f" [default: {default_value}]"
        elif is_required:
            guidance_line += " [REQUIRED]"

        guidance_lines.append(guidance_line)

    return "\n".join(guidance_lines)


VALID_ACTION_MAPPINGS = {
    'backend_analysis': ['get_reports', 'get_report_by_id', 'create_backend_excel_report'],
    'test_management': ['get_backend_tests', 'get_test_by_id', 'create_backend_test'],
    'test_execution': ['run_test'],
    'ui_analysis': ['get_ui_reports', 'get_ui_report_by_id', 'create_ui_excel_report'],
    'ui_test_management': ['get_ui_tests', 'create_ui_test', 'update_ui_test_schedule', 'cancel_ui_test'],
    'ui_test_execution': ['run_ui_test'],
    'ticket_action': ['get_ticket_list', 'create_ticket'],
    'disambiguation': ['clarify']
}


def validate_action_mapping(task_type: str, action: str) -> bool:
    """
    Validate task_type and action combinations.

    Args:
        task_type: The task category
        action: The specific action to validate

    Returns:
        bool: True if the mapping is valid
    """
    if task_type not in VALID_ACTION_MAPPINGS:
        logger.warning(f"[ActionValidation] Invalid task_type: {task_type}")
        return False

    if action not in VALID_ACTION_MAPPINGS[task_type]:
        logger.warning(f"[ActionValidation] Invalid action '{action}' for task_type '{task_type}'")
        logger.info(f"[ActionValidation] Valid actions for {task_type}: {VALID_ACTION_MAPPINGS[task_type]}")
        return False

    return True


def build_parameter_extraction_examples() -> List[Dict[str, Any]]:
    """
    Build comprehensive examples for parameter extraction validation.
    """
    return [
        {
            "user_message": "run test 215 with duration 30 sec",
            "expected_parameters": {"test_id": "215", "duration": "30"},
            "description": "Basic test execution with duration"
        },
        {
            "user_message": "execute backend test 123 for 5 minutes with 10 users",
            "expected_parameters": {"test_id": "123", "duration": "300", "users": "10"},
            "description": "Complex test execution with multiple parameters"
        },
        {
            "user_message": "start test 456 with 2 min ramp up and 25 concurrent users",
            "expected_parameters": {"test_id": "456", "ramp_up": "120", "users": "25"},
            "description": "Test with ramp up and user count"
        },
        {
            "user_message": "run ui test 789 for 1 hour",
            "expected_parameters": {"test_id": "789", "duration": "3600"},
            "description": "UI test with hour-based duration"
        },
        {
            "user_message": "get report 999",
            "expected_parameters": {"report_id": "999"},
            "description": "Simple report retrieval"
        },
        {
            "user_message": "create test named Load Test with 50 users and 90s duration",
            "expected_parameters": {"test_name": "Load Test", "users": "50", "duration": "90"},
            "description": "Test creation with name and parameters"
        }
    ]


__all__ = [
    'build_performance_analyst_prompt',
    'validate_action_mapping',
    'build_parameter_extraction_examples',
    'VALID_ACTION_MAPPINGS'
]