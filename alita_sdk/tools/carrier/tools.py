import logging
from typing import Dict, Any, Type, Optional, List

from langchain_core.tools import BaseTool
from pydantic import Field

from .api_wrapper import CarrierAPIWrapper
from .utils.intent_utils import CarrierIntentExtractor

# ====================================================================================
# 1. IMPORT ALL ATOMIC TOOL CLASSES
# ====================================================================================
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


# ====================================================================================
# ORCHESTRATION ENGINE WITH PROPER PARAMETER EXTRACTION
# ====================================================================================
# In your tools.py file, replace the CarrierOrchestrationEngine class with this updated version:

class CarrierOrchestrationEngine:
    """
    A robust, stateful orchestrator that manages the entire conversation,
    with enhanced LLM-based parameter extraction.
    """

    def __init__(self, llm: Any, api_wrapper: CarrierAPIWrapper, tool_class_map: Dict[str, Type[BaseTool]]):
        logger.info("[OrchestrationEngine] Initializing with LLM-based parameter extraction.")
        self.intent_extractor = CarrierIntentExtractor(llm)
        self.api_wrapper = api_wrapper
        self.tool_class_map = tool_class_map
        self.session_context: Dict[str, Any] = {}

    def _get_or_create_session(self, session_id: str) -> Dict:
        """Gets or creates a session context."""
        if session_id not in self.session_context:
            logger.critical(f"🧠 [STATE] Creating NEW session: {session_id}")
            self.session_context[session_id] = {
                "state": "START",
                "action": None,
                "task_type": None,
                "entities": {},
                "extracted_parameters": {},
                "clarification_options": [],
                "original_intent": None
            }
        return self.session_context[session_id]

    def _clear_session(self, session_id: str):
        """Resets the session."""
        if session_id in self.session_context:
            logger.critical(f"🧠 [STATE] Clearing session: {session_id}")
            del self.session_context[session_id]

    def process_request(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        """Processes a user request using enhanced LLM-based parameter extraction."""
        session = self._get_or_create_session(session_id)
        logger.critical(f"🧠 [STATE] Processing request. Session '{session_id}' is in state: {session['state']}")
        logger.critical(f"   - Current Action: {session['action']}")
        logger.critical(f"   - Accumulated Entities: {session['entities']}")

        # 1. Get the tool schema if we know the action
        tool_schema = None
        if session['action']:
            tool_class = self.tool_class_map.get(session['action'])
            if tool_class and hasattr(tool_class, 'args_schema'):
                tool_schema = tool_class.args_schema

        # 2. Extract intent AND parameters in a single LLM call
        current_intent = self.intent_extractor.extract_intent_with_parameters(
            user_message,
            tool_schema=tool_schema
        )

        if not current_intent:
            return self._create_error_response("I couldn't understand your request. Could you please rephrase it?")

        # Store the full intent for later use
        session['original_intent'] = current_intent

        # 3. Fuse new entities into the session
        for entity in current_intent.entities:
            session['entities'][entity['type']] = entity['value']

        # 4. Fuse extracted parameters into the session
        session['extracted_parameters'].update(current_intent.tool_parameters)

        logger.critical(f"   - Fused Entities: {session['entities']}")
        logger.critical(f"   - Extracted Parameters: {session['extracted_parameters']}")

        # 5. STATE MACHINE LOGIC (unchanged)
        if session['state'] == 'AWAITING_CLARIFICATION':
            resolved_option = self._find_resolved_option(user_message, session['clarification_options'])
            if resolved_option:
                logger.critical(f"   - CLARIFICATION SUCCEEDED. Resolved to action: {resolved_option['action']}")
                session['action'] = resolved_option['action']
                session['task_type'] = resolved_option['task_type']
                session['state'] = 'AWAITING_PARAMS'
            else:
                logger.warning("   - CLARIFICATION FAILED. Re-prompting.")
                return self._create_clarification_response(clarification_options=session['clarification_options'])

        # 6. If we don't have an action yet, determine one
        if not session['action']:
            if current_intent.needs_clarification():
                logger.critical("   - Intent is AMBIGUOUS. Transitioning to AWAITING_CLARIFICATION.")
                session['state'] = 'AWAITING_CLARIFICATION'
                session['clarification_options'] = current_intent.disambiguation_options
                return self._create_clarification_response(clarification_options=current_intent.disambiguation_options,
                                                           question_text=current_intent.clarification_question)
            else:
                logger.critical(f"   - Intent is UNAMBIGUOUS. Setting action to: {current_intent.action}")
                session['action'] = current_intent.action
                session['task_type'] = current_intent.task_type

        # 7. Check if we have a valid tool for the action
        tool_class = self.tool_class_map.get(session['action'])
        if not tool_class:
            self._clear_session(session_id)
            return self._create_error_response(f"Action '{session['action']}' is not a valid tool.")

        # 8. Use the enhanced parameter extraction from CarrierIntent
        if hasattr(tool_class, 'args_schema'):
            extracted_params = current_intent.get_tool_parameters_for_schema(tool_class.args_schema)
        else:
            extracted_params = session['extracted_parameters']

        # 9. Check for missing required parameters
        logger.critical(f"   - Final extracted parameters: {extracted_params}")
        missing_params = self._get_missing_params(tool_class, extracted_params)

        if not missing_params:
            logger.critical(f"   - All parameters satisfied for action '{session['action']}'. EXECUTING.")
            action_to_execute = session['action']
            self._clear_session(session_id)
            return self._execute_tool_action(action_to_execute, extracted_params)
        else:
            logger.critical(f"   - Missing parameters for '{session['action']}': {missing_params}. AWAITING_PARAMS.")
            session['state'] = 'AWAITING_PARAMS'
            question = f"To run the '{session['action']}' tool, I still need the following information: {', '.join(missing_params)}."
            return self._create_clarification_response(question_text=question)

    def _get_missing_params(self, tool_class: Type[BaseTool], provided_params: Dict) -> List[str]:
        """Checks the tool's args_schema to find what's missing."""
        if not hasattr(tool_class, 'args_schema'):
            return []

        # Get required fields from the Pydantic model
        required_fields = []
        logger.critical(f"🔍 [PARAMS] Checking required fields for {tool_class.__name__}")

        for field_name, field_info in tool_class.args_schema.model_fields.items():
            # Check if field is required (no default value)
            if field_info.is_required():
                required_fields.append(field_name)
                logger.critical(f"🔍 [PARAMS] Required field: {field_name}")

        provided_keys = set(provided_params.keys())
        missing = [field for field in required_fields if field not in provided_keys]

        logger.critical(f"🔍 [PARAMS] Required: {required_fields}")
        logger.critical(f"🔍 [PARAMS] Provided: {list(provided_keys)}")
        logger.critical(f"🔍 [PARAMS] Missing: {missing}")

        return missing

    def _execute_tool_action(self, action: str, params: Dict) -> Dict[str, Any]:
        """Executes the tool with the properly extracted parameters."""
        logger.info(f"[OrchestrationEngine] Executing resolved action: {action}")
        tool_class = self.tool_class_map.get(action)
        try:
            tool_instance = tool_class(api_wrapper=self.api_wrapper)
            logger.info(f"[OrchestrationEngine] Invoking tool '{tool_class.__name__}' with params: {params}")
            result = tool_instance._run(**params)
            return {'type': 'tool_execution_result', 'action': action, 'result': result}
        except Exception as e:
            logger.exception(f"[OrchestrationEngine] An unexpected error occurred during tool execution.")
            return self._create_error_response(f"A system error occurred while running the {action} tool: {str(e)}")

    def _create_clarification_response(self, clarification_options: Optional[List] = None,
                                       question_text: Optional[str] = None) -> Dict[str, Any]:
        """Creates a standardized clarification response."""
        if clarification_options:
            question = question_text or "Could you please clarify?"
            options_text = "\n".join(
                [f"{i + 1}. {opt['description']}" for i, opt in enumerate(clarification_options)])
            message = f"❓ {question}\nHere are the options I found:\n{options_text}"
            return {'type': 'clarification_request', 'message': message}
        return {'type': 'clarification_request', 'message': question_text or "I need more information."}

    def _find_resolved_option(self, user_message: str, options: List[Dict]) -> Optional[Dict]:
        """Finds the matching option from the user's clarification response."""
        user_lower = user_message.lower().strip()
        for option in options:
            if any(keyword in user_lower for keyword in option.get('keywords', [])):
                return option
        return None

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Creates a standardized error dictionary."""
        return {'type': 'error', 'message': message}


# ====================================================================================
# 3. THE ADAPTER META-TOOL
# ====================================================================================
class CarrierIntentMetaTool(BaseTool):
    """
    The single, intelligent entry point for the LangChain agent.
    It uses a powerful orchestration engine to handle complex, multi-turn conversations.
    """

    orchestration_engine: CarrierOrchestrationEngine = Field(...)
    session_id: str = Field(default="default_session", description="Unique ID for the user's conversation.")

    name: str = "carrier_performance_analytics"
    description: str = (
        "Use this for any tasks related to performance testing, analysis, or reporting. "
        "Provide the user's request exactly as it is. For example: 'show me the latest reports', "
        "'run test 123', or 'get tests'."
    )

    def _run(self, user_message: str) -> str:
        """
        Delegates the user's request to the orchestration engine and formats
        the structured response into a single string for the agent.
        """
        logger.info(f"[AdapterMetaTool] Passing request to Orchestration Engine for session '{self.session_id}'.")
        response = self.orchestration_engine.process_request(user_message, self.session_id)

        if response['type'] == 'clarification_request':
            logger.info("[AdapterMetaTool] Engine requires clarification. Returning question to user.")
            return f"❓ {response['message']}"

        elif response['type'] == 'tool_execution_result':
            logger.info("[AdapterMetaTool] Engine executed a tool successfully.")
            result_data = response.get('result', 'Task completed without a specific result.')
            return f"✅ Task '{response.get('action')}' completed successfully.\n\n{result_data}"

        elif response['type'] == 'error':
            logger.error(f"[AdapterMetaTool] Engine reported an error: {response.get('message')}")
            return f"❌ Error: {response.get('message')}"

        else:
            logger.error(f"[AdapterMetaTool] Received unknown response type from engine: {response.get('type')}")
            return "❌ An unexpected issue occurred in the toolkit."


# ====================================================================================
# 4. THE MASTER TOOL MAP
# ====================================================================================
ACTION_TOOL_MAP: Dict[str, Type[BaseTool]] = {
    # Ticket Management
    "get_ticket_list": FetchTicketsTool,
    "create_ticket": CreateTicketTool,

    # Backend Analysis
    "get_reports": GetReportsTool,
    "get_report_by_id": GetReportByIDTool,
    "process_report": ProcessAndGenerateReportTool,

    # Backend Test Management
    "get_backend_tests": GetBackendTestsTool,
    "get_test_by_id": GetTestByIDTool,
    "create_backend_test": CreateBackendTestTool,

    # Backend Test Execution
    "run_test": RunTestByIDTool,

    # UI Analysis
    "get_ui_reports": GetUIReportsTool,
    "get_ui_report_by_id": GetUIReportByIDTool,
    "create_ui_excel_report": CreateUIExcelReportTool,

    # UI Test Management
    "get_ui_tests": GetUITestsTool,
    "create_ui_test": CreateUITestTool,
    "update_ui_test_schedule": UpdateUITestScheduleTool,
    "cancel_ui_test": CancelUITestTool,

    # UI Test Execution
    "run_ui_test": RunUITestTool,
}

# ====================================================================================
# 5. PUBLIC EXPORTS
# ====================================================================================
__all__ = [
    "CarrierIntentMetaTool",
    "ACTION_TOOL_MAP",
    "CarrierOrchestrationEngine"
]
