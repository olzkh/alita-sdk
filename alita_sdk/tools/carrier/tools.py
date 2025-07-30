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
# ORCHESTRATION ENGINE
# ====================================================================================

class CarrierOrchestrationEngine:
    """
    Streamlined orchestrator with explicit error handling and simplified state management.
    """

    def __init__(self, llm: Any, api_wrapper: CarrierAPIWrapper, tool_class_map: Dict[str, Type[BaseTool]]):
        logger.info("[OrchestrationEngine] Initializing streamlined orchestrator")

        try:
            self.intent_extractor = CarrierIntentExtractor(llm)
        except Exception as e:
            logger.error(f"[OrchestrationEngine] Failed to initialize intent extractor: {e}")
            raise RuntimeError(f"Orchestrator initialization failed: {e}")

        self.api_wrapper = api_wrapper
        self.tool_class_map = tool_class_map
        self.session_context: Dict[str, Any] = {}

    def _get_or_create_session(self, session_id: str) -> Dict:
        """Gets or creates a session context."""
        if session_id not in self.session_context:
            logger.info(f"[OrchestrationEngine] Creating new session: {session_id}")
            self.session_context[session_id] = {
                "state": "START",
                "action": None,
                "task_type": None,
                "accumulated_parameters": {},
                "clarification_options": [],
                "original_intent": None
            }
        return self.session_context[session_id]

    def _clear_session(self, session_id: str):
        """Resets the session."""
        if session_id in self.session_context:
            logger.info(f"[OrchestrationEngine] Clearing session: {session_id}")
            del self.session_context[session_id]

    def process_request(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process user request with explicit error handling and simplified flow.
        """
        session = self._get_or_create_session(session_id)
        logger.info(f"[OrchestrationEngine] Processing request in session '{session_id}', state: {session['state']}")

        try:
            # 1. Get the tool schema if we know the action
            tool_schema = None
            if session['action']:
                tool_class = self.tool_class_map.get(session['action'])
                if tool_class and hasattr(tool_class, 'args_schema'):
                    tool_schema = tool_class.args_schema

            # 2. Extract intent with explicit error handling
            try:
                current_intent = self.intent_extractor.extract_intent_with_parameters(
                    user_message,
                    tool_schema=tool_schema
                )
            except RuntimeError as e:
                logger.error(f"[OrchestrationEngine] Intent extraction failed: {e}")
                self._clear_session(session_id)
                return self._create_error_response(
                    "I'm having trouble understanding your request. Please try rephrasing it.")

            # Store the intent for context
            session['original_intent'] = current_intent

            # 3. Update accumulated parameters
            session['accumulated_parameters'].update(current_intent.tool_parameters)

            logger.info(f"[OrchestrationEngine] Accumulated parameters: {session['accumulated_parameters']}")

            # 4. STATE MACHINE LOGIC
            if session['state'] == 'AWAITING_CLARIFICATION':
                resolved_option = self._find_resolved_option(user_message, session['clarification_options'])
                if resolved_option:
                    logger.info(f"[OrchestrationEngine] Clarification resolved to: {resolved_option['action']}")
                    session['action'] = resolved_option['action']
                    session['task_type'] = resolved_option['task_type']
                    session['state'] = 'AWAITING_PARAMS'
                else:
                    logger.warning("[OrchestrationEngine] Clarification failed, re-prompting")
                    return self._create_clarification_response(clarification_options=session['clarification_options'])

            # 5. Determine action if not set
            if not session['action']:
                if current_intent.needs_clarification():
                    logger.info("[OrchestrationEngine] Intent requires clarification")
                    session['state'] = 'AWAITING_CLARIFICATION'
                    session['clarification_options'] = current_intent.disambiguation_options
                    return self._create_clarification_response(
                        clarification_options=current_intent.disambiguation_options,
                        question_text=current_intent.clarification_question
                    )
                else:
                    logger.info(f"[OrchestrationEngine] Intent resolved to action: {current_intent.action}")
                    session['action'] = current_intent.action
                    session['task_type'] = current_intent.task_type

            # 6. Validate tool exists
            tool_class = self.tool_class_map.get(session['action'])
            if not tool_class:
                self._clear_session(session_id)
                return self._create_error_response(f"Action '{session['action']}' is not available.")

            # 7. Check for missing required parameters
            final_params = session['accumulated_parameters']
            missing_params = self._get_missing_params(tool_class, final_params)

            if not missing_params:
                logger.info(f"[OrchestrationEngine] All parameters satisfied, executing: {session['action']}")
                action_to_execute = session['action']
                self._clear_session(session_id)
                return self._execute_tool_action(action_to_execute, final_params)
            else:
                logger.info(f"[OrchestrationEngine] Missing parameters: {missing_params}")
                session['state'] = 'AWAITING_PARAMS'
                question = f"To run the '{session['action']}' tool, I still need: {', '.join(missing_params)}."
                return self._create_clarification_response(question_text=question)

        except Exception as e:
            logger.error(f"[OrchestrationEngine] Unexpected error processing request: {e}")
            self._clear_session(session_id)
            return self._create_error_response("An unexpected error occurred. Please try again.")

    def _get_missing_params(self, tool_class: Type[BaseTool], provided_params: Dict) -> List[str]:
        """Check tool's schema for missing required parameters."""
        if not hasattr(tool_class, 'args_schema'):
            return []

        required_fields = []
        logger.info(f"[OrchestrationEngine] Checking required fields for {tool_class.__name__}")

        try:
            for field_name, field_info in tool_class.args_schema.model_fields.items():
                if field_info.is_required():
                    required_fields.append(field_name)
                    logger.debug(f"[OrchestrationEngine] Required field: {field_name}")

            provided_keys = set(provided_params.keys())
            missing = [field for field in required_fields if field not in provided_keys]

            logger.info(
                f"[OrchestrationEngine] Required: {required_fields}, Provided: {list(provided_keys)}, Missing: {missing}")
            return missing

        except Exception as e:
            logger.error(f"[OrchestrationEngine] Error checking parameters: {e}")
            return []

    def _execute_tool_action(self, action: str, params: Dict) -> Dict[str, Any]:
        """Execute the tool with proper error handling."""
        logger.info(f"[OrchestrationEngine] Executing action: {action}")
        tool_class = self.tool_class_map.get(action)

        if not tool_class:
            return self._create_error_response(f"Tool '{action}' not found.")

        try:
            tool_instance = tool_class(api_wrapper=self.api_wrapper)
            logger.info(f"[OrchestrationEngine] Invoking {tool_class.__name__} with params: {params}")
            result = tool_instance._run(**params)
            return {
                'type': 'tool_execution_result',
                'action': action,
                'result': result
            }
        except Exception as e:
            logger.error(f"[OrchestrationEngine] Tool execution failed: {e}")
            return self._create_error_response(f"Failed to execute {action}: {str(e)}")

    def _create_clarification_response(self, clarification_options: Optional[List] = None,
                                       question_text: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized clarification response."""
        if clarification_options:
            question = question_text or "Could you please clarify?"
            options_text = "\n".join(
                [f"{i + 1}. {opt['description']}" for i, opt in enumerate(clarification_options)]
            )
            message = f"❓ {question}\nHere are the options I found:\n{options_text}"
            return {'type': 'clarification_request', 'message': message}

        return {
            'type': 'clarification_request',
            'message': question_text or "I need more information."
        }

    def _find_resolved_option(self, user_message: str, options: List[Dict]) -> Optional[Dict]:
        """Find matching option from clarification response."""
        user_lower = user_message.lower().strip()
        for option in options:
            if any(keyword in user_lower for keyword in option.get('keywords', [])):
                return option
        return None

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
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
