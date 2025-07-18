import logging
from .api_wrapper import CarrierAPIWrapper
from .utils.intent_utils import CarrierIntentExtractor

# ==== Import All Atomic Tool Classes (1-to-1 with your use-cases) ====
from .tickets_tool import FetchTicketsTool, CreateTicketTool
from .backend_reports_tool import GetReportsTool, GetReportByIDTool, ProcessAndGenerateReportTool
from .backend_tests_tool import GetTestsTool, GetTestByIDTool, RunTestByIDTool, CreateBackendTestTool
from .ui_reports_tool import GetUIReportsTool, GetUIReportByIDTool, GetUITestsTool
from .run_ui_test_tool import RunUITestTool
from .update_ui_test_schedule_tool import UpdateUITestScheduleTool
from .create_ui_excel_report_tool import CreateUIExcelReportTool
from .create_ui_test_tool import CreateUITestTool
from .cancel_ui_test_tool import CancelUITestTool

logger = logging.getLogger(__name__)

# ==== DRY Principle: Maintain One Central List ====
ATOMIC_TOOL_CLASSES = [
    FetchTicketsTool, CreateTicketTool,
    GetReportsTool, GetReportByIDTool, ProcessAndGenerateReportTool,
    GetTestsTool, GetTestByIDTool, RunTestByIDTool, CreateBackendTestTool,
    GetUIReportsTool, GetUIReportByIDTool, GetUITestsTool,
    RunUITestTool, UpdateUITestScheduleTool, CreateUIExcelReportTool,
    CreateUITestTool, CancelUITestTool,
]

# ==== Always provide ONLY tool instances to agent/assistant ====
def get_all_tools(api_wrapper: CarrierAPIWrapper):
    """
    Returns all instantiated tool objects for LLM agent registration.
    Each tool .name must match the LLM/agent invocation.
    """
    tools = [cls(api_wrapper=api_wrapper) for cls in ATOMIC_TOOL_CLASSES]
    # [Optional] Log registered tools for audit
    for t in tools:
        logger.debug(f"[CarrierToolkit] Registered tool: {getattr(t, 'name', type(t).__name__)}")
    return tools

# ==== Map (task_type, action) to the correct tool class (for intent meta-routing, not agent registration) ====
ACTION_TOOL_MAP = {
    ("ticket_action", "get_ticket_list"): FetchTicketsTool,
    ("ticket_action", "create_ticket"): CreateTicketTool,

    ("backend_analysis", "get_reports"): GetReportsTool,
    ("backend_analysis", "get_report_by_id"): GetReportByIDTool,
    ("backend_analysis", "process_report"): ProcessAndGenerateReportTool,

    ("test_management", "get_tests"): GetTestsTool,
    ("test_management", "get_test_by_id"): GetTestByIDTool,
    ("test_execution", "run_test"): RunTestByIDTool,
    ("test_management", "create_backend_test"): CreateBackendTestTool,

    ("ui_analysis", "get_ui_reports"): GetUIReportsTool,
    ("ui_analysis", "get_ui_report_by_id"): GetUIReportByIDTool,
    ("ui_test_management", "get_ui_tests"): GetUITestsTool,
    ("ui_test_execution", "run_ui_test"): RunUITestTool,
    ("ui_test_management", "update_ui_test_schedule"): UpdateUITestScheduleTool,
    ("ui_analysis", "create_ui_excel_report"): CreateUIExcelReportTool,
    ("ui_test_management", "create_ui_test"): CreateUITestTool,
    ("ui_test_management", "cancel_ui_test"): CancelUITestTool,
}

# ==== LLM-powered meta-tool for intent-driven routing ====
from langchain_core.tools import BaseTool

class CarrierIntentMetaTool(BaseTool):
    """
    Meta-tool for LLM-driven Carrier intent handling.
    Given a user message, extracts intent, confirms, and dispatches to correct tool.
    """
    intent_extractor: CarrierIntentExtractor
    api_wrapper: CarrierAPIWrapper
    tool_map: dict  # (task_type, action) => tool class

    name = "carrier_intent_meta_tool"
    description = "Routes user intent to the correct Carrier tool using LLM extraction and confirmation."

    def _run(self, user_message: str):
        intent = self.intent_extractor.extract_intent(user_message)
        print(f"Intent detected: {intent.action} ({intent.task_type})")
        print(f"Entities: {intent.entities}")
        print(f"Fields/Params: {intent.field_requests}")
        print(f"{intent.confirmation_question}")
        user_reply = input("Confirm (yes/no/clarify): ").strip().lower()
        if user_reply != "yes":
            return "❗ User did not confirm; aborting."

        tool_class = self.tool_map.get((intent.task_type, intent.action))
        if not tool_class:
            return f"🚫 No tool found for ({intent.task_type}, {intent.action})"

        tool_kwargs = {}
        if isinstance(intent.entities, dict):
            tool_kwargs.update(intent.entities)
        elif isinstance(intent.entities, list) and intent.entities and isinstance(intent.entities[0], dict):
            tool_kwargs.update(intent.entities[0])

        tool_instance = tool_class(api_wrapper=self.api_wrapper)
        return tool_instance._run(**tool_kwargs)

# ==== __all__ for import * hygiene ====
__all__ = [
    "get_all_tools",
    "ACTION_TOOL_MAP",
    "CarrierIntentMetaTool",
    # ...list tool classes if you want import * to grab them as well
    "FetchTicketsTool", "CreateTicketTool",
    "GetReportsTool", "GetReportByIDTool", "ProcessAndGenerateReportTool",
    "GetTestsTool", "GetTestByIDTool", "RunTestByIDTool", "CreateBackendTestTool",
    "GetUIReportsTool", "GetUIReportByIDTool", "GetUITestsTool",
    "RunUITestTool", "UpdateUITestScheduleTool", "CreateUIExcelReportTool",
    "CreateUITestTool", "CancelUITestTool",
]
