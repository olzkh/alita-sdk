"""
Analysts workflow knowledge

Author: Karen Florykian
"""
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def build_performance_analyst_prompt(user_message: str, context: Optional[Dict] = None,
                                     tool_schema: Optional[Any] = None) -> str:
    """
    Enhanced prompt builder with smart disambiguation capabilities and parameter extraction
    """
    logger.info(f"[PromptBuilder] Building enhanced prompt for: {user_message}")

    from .intent_utils import detect_ambiguous_intent
    disambiguation_info = detect_ambiguous_intent(user_message)

    if disambiguation_info and not disambiguation_info.get("resolved", False):
        return build_disambiguation_prompt(user_message, disambiguation_info)
    else:
        return build_standard_intent_prompt(user_message, context, tool_schema)


def build_disambiguation_prompt(user_message: str, disambiguation_info: Dict) -> str:
    """
    Build prompt specifically for handling ambiguous requests
    """
    logger.info(f"[PromptBuilder] Building disambiguation prompt for ambiguous request: '{user_message}'")
    prompt = f"""
You are a Performance Analytics Assistant. The user's request is AMBIGUOUS and requires clarification.
What you know about the request: JMeter and Gatling are backend performance testing tools, while Lighthouse and Sitespeed are used for UI performance testing.
USER REQUEST: "{user_message}"

DETECTED AMBIGUITY: This request matches multiple possible actions.

Your task is to return a JSON object that asks for clarification:

{{
    "task_type": "disambiguation",
    "action": "clarify", 
    "entities": [],
    "tool_parameters": {{}},
    "is_ambiguous": true,
    "clarification_question": "{disambiguation_info['clarification_question']}",
    "disambiguation_options": {disambiguation_info['options']},
    "confidence_score": 0.95
}}

IMPORTANT: Always use action: "clarify" for ambiguous requests.

Return ONLY the JSON object, no additional text.
"""

    return prompt


def build_resolved_intent_prompt(user_message: str, disambiguation_info: Dict, context: Optional[Dict] = None,
                                 tool_schema: Optional[Any] = None) -> str:
    """
    Build tool-schema-aware prompt for requests where ambiguity was resolved by context
    """
    logger.info(f"[PromptBuilder] Building resolved intent prompt for action: {disambiguation_info['action']}")

    # Get tool-specific parameter extraction instructions
    tool_guidance = _get_tool_parameter_guidance(disambiguation_info['action'], tool_schema)

    prompt = f"""
You are a Performance Analytics Assistant. The user's request has been resolved from context.

USER REQUEST: "{user_message}"

RESOLVED ACTION: {disambiguation_info['action']}
RESOLVED TASK TYPE: {disambiguation_info['task_type']}

{tool_guidance}

Extract any additional details and parameters, then return JSON:

{{
    "task_type": "{disambiguation_info['task_type']}",
    "action": "{disambiguation_info['action']}",
    "entities": [],  // Extract any entity IDs from the user message
    "tool_parameters": {{}},  // Extract parameter values based on tool schema
    "is_ambiguous": false,
    "clarification_question": null,
    "disambiguation_options": [],
    "confidence_score": 0.9
}}

GENERAL PARAMETER EXTRACTION RULES:
1. DURATION: Convert to seconds ("30 sec" → "30", "5 min" → "300")
2. TEST_ID: Extract numeric IDs ("test 215" → "215")  
3. USER_COUNT: Extract user numbers ("10 users" → "10")
4. RAMP_UP: Convert to seconds ("2 min ramp" → "120")
5. STRINGS: Extract names, descriptions, environments as-is
6. NUMBERS: Extract limits, counts as integers

Extract entities like test IDs, report IDs, etc. from the user message.

Return ONLY the JSON object, no additional text.
"""

    return prompt


def build_standard_intent_prompt(user_message: str, context: Optional[Dict] = None,
                                 tool_schema: Optional[Any] = None) -> str:
    """
    Enhanced standard intent extraction prompt with parameter extraction capabilities
    """
    logger.info(f"[PromptBuilder] Building standard intent prompt")

    # Get tool-specific guidance if available
    tool_guidance = ""
    if tool_schema:
        tool_guidance = _get_tool_parameter_guidance_from_schema(tool_schema)

    prompt = f"""
You are a Performance Analytics Assistant specialized in performance testing workflows and parameter extraction.

AVAILABLE TOOLS:

📊 BACKEND ANALYSIS:
- get_reports: List backend performance reports
- get_report_by_id: Get detailed report information  
- process_report: Generate Excel reports from test data

🧪 BACKEND TEST MANAGEMENT:
- get_backend_tests: List backend tests
- get_test_by_id: Get detailed test information
- create_backend_test: Create new JMeter/Gatling tests

▶️ BACKEND TEST EXECUTION:
- run_test: Execute backend performance tests

🖥️ UI ANALYSIS:
- get_ui_reports: List UI performance reports
- get_ui_report_by_id: Get detailed UI report
- create_ui_excel_report: Generate Excel from UI test data

🌐 UI TEST MANAGEMENT:
- get_ui_tests: List UI tests
- create_ui_test: Create new UI performance tests
- update_ui_test_schedule: Schedule UI test runs
- cancel_ui_test: Cancel running UI tests

🚀 UI TEST EXECUTION:
- run_ui_test: Execute UI performance tests

🎫 TICKET MANAGEMENT:
- get_ticket_list: List tickets
- create_ticket: Create new tickets

USER REQUEST: "{user_message}"

{tool_guidance}

EXTRACTION RULES:
1. Use EXACT action names from the available tools list
2. Extract numeric IDs as entities with proper types
3. For Excel/report requests → use process_report or create_ui_excel_report
4. For get/show requests → use appropriate get_* actions
5. For run/execute → use run_test or run_ui_test
6. For create/new → use appropriate create_* actions

CRITICAL PARAMETER EXTRACTION:
Extract tool parameters from the user message and convert appropriately:

DURATION CONVERSION (always convert to seconds):
- "30 sec", "30s", "30 seconds" → "30"
- "5 min", "5 minutes", "5m" → "300" 
- "2 hours", "2h" → "7200"
- "1.5 min" → "90"

TEST/REPORT ID EXTRACTION:
- "test 215", "test ID 215", "test number 215" → "215"
- "report 123", "report ID 123" → "123"

USER COUNT EXTRACTION:
- "10 users", "10 concurrent users" → "10"
- "50 virtual users" → "50"

RAMP UP CONVERSION (convert to seconds):
- "60 sec ramp", "1 min ramp up", "60s ramp" → "60"
- "2 min ramp up", "120 sec ramp" → "120"

OTHER PARAMETERS:
- Test names, descriptions, URLs, etc. as strings
- Boolean flags (true/false)

Return JSON:
{{
    "task_type": "backend_analysis|test_management|test_execution|ui_analysis|ui_test_management|ui_test_execution|ticket_action",
    "action": "exact_action_name_from_available_tools",
    "entities": [{{"type": "test_id|report_id", "value": "extracted_id"}}],
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
    "entities": [{{"type": "test_id", "value": "215"}}],
    "tool_parameters": {{"test_id": "215", "duration": "30"}},
    "is_ambiguous": false,
    "confidence_score": 0.95
}}

"list all reports for WhateverNameTest'" →
{{
    "task_type": "test_management",
    "action": "get_reports",
    "entities": [{{"type": "test_id", "value": "215"}}],
    "tool_parameters": {{"test_id": "215", "duration": "30"}},
    "is_ambiguous": false,
    "confidence_score": 0.95
}}

"execute test 123 for 5 minutes with 10 users and 2 min ramp up" →
{{
    "task_type": "test_execution", 
    "action": "run_test",
    "entities": [{{"type": "test_id", "value": "123"}}],
    "tool_parameters": {{"test_id": "123", "duration": "300", "users": "10", "ramp_up": "120"}},
    "is_ambiguous": false,
    "confidence_score": 0.9
}}

"get report 456" →
{{
    "task_type": "backend_analysis",
    "action": "get_report_by_id", 
    "entities": [{{"type": "report_id", "value": "456"}}],
    "tool_parameters": {{"report_id": "456"}},
    "is_ambiguous": false,
    "confidence_score": 0.95
}}

"create backend test called API Load Test with 25 users" →
{{
    "task_type": "test_management",
    "action": "create_backend_test",
    "entities": [],
    "tool_parameters": {{"test_name": "API Load Test", "users": "25"}},
    "is_ambiguous": false,
    "confidence_score": 0.85
}}

Focus on EXACT parameter extraction - the user's success depends on accurate parameter parsing.

Return ONLY the JSON object, no additional text.
"""

    if context:
        prompt += f"\n\nADDITIONAL CONTEXT: {context}"

    return prompt


# VALIDATION MAPPINGS - Updated with all available actions
VALID_ACTION_MAPPINGS = {
    'backend_analysis': ['get_reports', 'get_report_by_id', 'process_report'],
    'test_management': ['get_backend_tests', 'get_test_by_id', 'create_backend_test'],
    'test_execution': ['run_test'],
    'ui_analysis': ['get_ui_reports', 'get_ui_report_by_id', 'create_ui_excel_report'],
    'ui_test_management': ['get_ui_tests', 'create_ui_test', 'update_ui_test_schedule', 'cancel_ui_test'],
    'ui_test_execution': ['run_ui_test'],
    'ticket_action': ['get_ticket_list', 'create_ticket'],
    'disambiguation': ['clarify']  # Special task type for clarification requests
}


def validate_action_mapping(task_type: str, action: str) -> bool:
    """
    Enhanced validation that includes disambiguation support
    """
    if task_type not in VALID_ACTION_MAPPINGS:
        logger.warning(f"[ActionValidation] Invalid task_type: {task_type}")
        return False

    if action not in VALID_ACTION_MAPPINGS[task_type]:
        logger.warning(f"[ActionValidation] Invalid action '{action}' for task_type '{task_type}'")
        logger.info(f"[ActionValidation] Valid actions for {task_type}: {VALID_ACTION_MAPPINGS[task_type]}")
        return False

    return True


def get_disambiguation_suggestions(user_message: str) -> List[Dict[str, str]]:
    """
    Get smart suggestions for potentially ambiguous requests
    """
    user_lower = user_message.lower()
    suggestions = []

    # Check for common ambiguous patterns
    if any(word in user_lower for word in ['tests', 'test']):
        suggestions.extend([
            {
                "action": "get_backend_tests",
                "description": "View backend performance tests (JMeter, Gatling)",
                "example": "get backend tests"
            },
            {
                "action": "get_ui_tests",
                "description": "View UI performance tests (Lighthouse, Sitespeed)",
                "example": "get ui tests"
            }
        ])

    if any(word in user_lower for word in ['reports', 'report']):
        suggestions.extend([
            {
                "action": "get_reports",
                "description": "View backend performance reports",
                "example": "get backend reports"
            },
            {
                "action": "get_ui_reports",
                "description": "View UI performance reports",
                "example": "get ui reports"
            }
        ])

    return suggestions


def _get_tool_parameter_guidance(action: str, tool_schema: Optional[Any] = None) -> str:
    """
    Generate tool-specific parameter extraction guidance based on action and schema
    """
    # Tool-specific guidance based on action name
    guidance_map = {
        "get_reports": """
TOOL-SPECIFIC PARAMETERS for get_reports:
- tag_name: Extract tag/category names ("prod reports" → tag_name: "prod")
- limit: Extract number limits ("show 20 reports" → limit: 20, default: 50)
- environment: Extract environment names ("staging reports" → environment: "staging")

Examples:
"get 20 prod reports" → {"tag_name": "prod", "limit": 20}
"show staging reports" → {"environment": "staging"}
"list reports with limit 100" → {"limit": 100}
""",

        "get_report_by_id": """
TOOL-SPECIFIC PARAMETERS for get_report_by_id:
- report_id: Extract numeric report ID ("report 123" → report_id: "123")

Examples:
"get report 456" → {"report_id": "456"}
"show report ID 789" → {"report_id": "789"}
""",

        "run_test": """
TOOL-SPECIFIC PARAMETERS for run_test:
- test_id: Extract numeric test ID ("test 215" → test_id: "215")
- duration: Convert time to seconds ("30 sec" → duration: "30")
- users: Extract user count ("10 users" → users: "10")
- ramp_up: Convert ramp time to seconds ("2 min ramp" → ramp_up: "120")
- environment: Extract environment ("prod" → environment: "prod")

Examples:
"run test 215 with duration 30 sec" → {"test_id": "215", "duration": "30"}
"execute test 123 for 5 min with 10 users" → {"test_id": "123", "duration": "300", "users": "10"}
""",

        "create_backend_test": """
TOOL-SPECIFIC PARAMETERS for create_backend_test:
- test_name: Extract test name ("Load Test" → test_name: "Load Test")
- users: Extract user count ("25 users" → users: "25")
- duration: Convert time to seconds ("5 min" → duration: "300")
- ramp_up: Convert ramp time to seconds ("60 sec ramp" → ramp_up: "60")
- description: Extract test description

Examples:
"create test called API Load Test with 25 users" → {"test_name": "API Load Test", "users": "25"}
""",

        "get_ui_reports": """
TOOL-SPECIFIC PARAMETERS for get_ui_reports:
- limit: Extract number limits ("show 15 ui reports" → limit: 15)
- test_type: Extract test type ("lighthouse reports" → test_type: "lighthouse")

Examples:
"get 15 ui reports" → {"limit": 15}
"show lighthouse reports" → {"test_type": "lighthouse"}
"""
    }

    base_guidance = guidance_map.get(action, "")

    # Add schema-specific guidance if available
    if tool_schema and hasattr(tool_schema, 'model_fields'):
        schema_guidance = _get_tool_parameter_guidance_from_schema(tool_schema)
        if schema_guidance:
            base_guidance += f"\n\nSCHEMA-BASED PARAMETERS:\n{schema_guidance}"

    return base_guidance


def _get_tool_parameter_guidance_from_schema(tool_schema: Any) -> str:
    """
    Generate parameter guidance from Pydantic schema
    """
    if not tool_schema or not hasattr(tool_schema, 'model_fields'):
        return ""
    guidance_lines = []

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

    if guidance_lines:
        return "Expected parameters:\n" + "\n".join(guidance_lines)


def build_parameter_extraction_examples() -> List[Dict[str, Any]]:
    """
    Build comprehensive examples for parameter extraction training/validation
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
    'build_disambiguation_prompt',
    'build_resolved_intent_prompt',
    'build_standard_intent_prompt',
    'validate_action_mapping',
    'get_disambiguation_suggestions',
    'build_parameter_extraction_examples',
    'VALID_ACTION_MAPPINGS'
]
