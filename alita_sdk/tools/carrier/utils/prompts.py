import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

PERFORMANCE_ANALYST_INTENT_EXAMPLES = {
    # ===== BACKEND ANALYSIS =====
    "generate_excel_report": {
        "task_type": "backend_analysis",
        "action": "process_report",
        "user_message": "generate excel report from test 5134",
        "variations": [
            "create excel report for test 5134",
            "export report 5134 to excel",
            "generate report in excel for backend test test run with build id 5144"
        ]
    },
    "get_backend_reports": {
        "task_type": "backend_analysis",
        "action": "get_reports",
        "user_message": "show me all backend reports",
        "variations": [
            "list backend reports",
            "get reports list",
            "show performance reports",
            "compare the 5 latest reports"
        ]
    },
    "get_specific_report": {
        "task_type": "backend_analysis",
        "action": "get_report_by_id",
        "user_message": "get report 5134 details",
        "variations": [
            "show report 5134",
            "get detailed info by report id 5134",
            "fetch report with id 5134"
        ]
    },

    # ===== TEST MANAGEMENT =====
    "get_tests_list": {
        "task_type": "test_management",
        "action": "get_tests",
        "user_message": "show all backend tests",
        "variations": [
            "list all tests",
            "get tests list",
            "show available backend tests"
        ]
    },
    "get_test_details": {
        "task_type": "test_management",
        "action": "get_test_by_id",
        "user_message": "get test by id 5144",
        "variations": [
            "show test 5144 details",
            "get detailed test info for 5144",
            "fetch test information 5144"
        ]
    },
    "create_backend_test": {
        "task_type": "test_management",
        "action": "create_backend_test",
        "user_message": "create new backend test",
        "variations": [
            "set up new backend test",
            "create JMeter test",
            "create Gatling test",
            "add new performance test"
        ]
    },

    # ===== TEST EXECUTION =====
    "run_backend_test": {
        "task_type": "test_execution",
        "action": "run_test",
        "user_message": "run test 5144",
        "variations": [
            "execute test 5144",
            "start test run for 5144",
            "run backend test 5144"
        ]
    },

    # ===== UI ANALYSIS =====
    "get_ui_reports": {
        "task_type": "ui_analysis",
        "action": "get_ui_reports",
        "user_message": "show ui reports",
        "variations": [
            "list ui reports",
            "get ui performance reports",
            "show frontend reports"
        ]
    },
    "get_ui_report_details": {
        "task_type": "ui_analysis",
        "action": "get_ui_report_by_id",
        "user_message": "get ui report 5134",
        "variations": [
            "show ui report details 5134",
            "fetch ui report 5134"
        ]
    },
    "create_ui_excel": {
        "task_type": "ui_analysis",
        "action": "create_ui_excel_report",
        "user_message": "generate ui excel report 5134",
        "variations": [
            "create ui excel for 5134",
            "export ui report 5134 to excel"
        ]
    },

    # ===== UI TEST MANAGEMENT =====
    "get_ui_tests": {
        "task_type": "ui_test_management",
        "action": "get_ui_tests",
        "user_message": "show ui tests",
        "variations": [
            "list ui tests",
            "get frontend tests"
        ]
    },
    "create_ui_test": {
        "task_type": "ui_test_management",
        "action": "create_ui_test",
        "user_message": "create new ui test",
        "variations": [
            "set up ui test",
            "create Lighthouse test",
            "create Sitespeed test"
        ]
    },
    "schedule_ui_test": {
        "task_type": "ui_test_management",
        "action": "update_ui_test_schedule",
        "user_message": "schedule ui test 5134",
        "variations": [
            "update test schedule for 5134",
            "set schedule for ui test"
        ]
    },
    "cancel_ui_test": {
        "task_type": "ui_test_management",
        "action": "cancel_ui_test",
        "user_message": "cancel ui test 5134",
        "variations": [
            "stop ui test 5134",
            "abort ui test run"
        ]
    },

    # ===== UI TEST EXECUTION =====
    "run_ui_test": {
        "task_type": "ui_test_execution",
        "action": "run_ui_test",
        "user_message": "run ui test 5134",
        "variations": [
            "execute ui test 5134",
            "start ui test run"
        ]
    },

    # ===== TICKET MANAGEMENT =====
    "get_tickets": {
        "task_type": "ticket_action",
        "action": "get_ticket_list",
        "user_message": "show tickets list",
        "variations": [
            "list all tickets",
            "get tickets by board id",
            "show engagement tickets",
            "filter tickets by timeline"
        ]
    },
    "create_ticket": {
        "task_type": "ticket_action",
        "action": "create_ticket",
        "user_message": "create new ticket",
        "variations": [
            "add new task",
            "create engagement ticket",
            "start project tickets",
            "create new task for board"
        ]
    }
}

# Common performance analyst workflows mapped to available tools
ANALYST_WORKFLOW_PATTERNS = {
    # Daily Analysis Tasks
    "compare_reports": {
        "keywords": ["compare", "latest", "5 reports", "comparison"],
        "suggested_action": "get_reports",
        "task_type": "backend_analysis",
        "workflow": "Get latest reports → Export to Excel → Manual comparison"
    },
    "environment_comparison": {
        "keywords": ["compare", "environment", "env", "vct"],
        "suggested_action": "get_reports",
        "task_type": "backend_analysis",
        "workflow": "Filter reports by environment → Compare performance metrics"
    },
    "error_analysis": {
        "keywords": ["error", "errors", "analysis", "log", "failure"],
        "suggested_action": "get_report_by_id",
        "task_type": "backend_analysis",
        "workflow": "Get report details → Analyze error logs → Create ticket if needed"
    },
    "baseline_creation": {
        "keywords": ["baseline", "new baseline", "threshold"],
        "suggested_action": "get_reports",
        "task_type": "backend_analysis",
        "workflow": "Get historical reports → Analyze trends → Set new baseline"
    },
    "excel_generation": {
        "keywords": ["excel", "export", "report", "generate"],
        "suggested_action": "process_report",
        "task_type": "backend_analysis",
        "workflow": "Process report data → Generate Excel format"
    }
}


def build_performance_analyst_prompt(user_message: str, context: Optional[Dict] = None) -> str:
    """
    Build specialized prompt for Performance Analytics with exact tool mapping
    """
    logger.info(f"[PerformancePrompt] Building prompt for: {user_message}")

    prompt = f"""
    You are a Performance Analytics Assistant specialized in performance testing workflows.

    AVAILABLE TOOLS BY CATEGORY:

    📊 BACKEND ANALYSIS:
    - get_reports: List all backend performance reports
    - get_report_by_id: Get detailed report information  
    - process_report: Generate Excel reports from test data

    🧪 TEST MANAGEMENT:
    - get_tests: List all backend tests
    - get_test_by_id: Get detailed test information
    - create_backend_test: Create new JMeter/Gatling tests

    ▶️ TEST EXECUTION:
    - run_test: Execute backend performance tests

    🖥️ UI ANALYSIS:
    - get_ui_reports: List UI performance reports
    - get_ui_report_by_id: Get detailed UI report
    - create_ui_excel_report: Generate Excel from UI test data

    🌐 UI TEST MANAGEMENT:
    - get_ui_tests: List UI tests (Lighthouse/Sitespeed)
    - create_ui_test: Create new UI performance tests
    - update_ui_test_schedule: Schedule UI test runs
    - cancel_ui_test: Cancel running UI tests

    🚀 UI TEST EXECUTION:
    - run_ui_test: Execute UI performance tests

    🎫 TICKET MANAGEMENT:
    - get_ticket_list: List engagement/project tickets
    - create_ticket: Create new tasks/tickets

    COMMON PERFORMANCE ANALYST WORKFLOWS:

    1️⃣ "Compare 5 latest reports" → get_reports + process_report
    2️⃣ "Generate Excel for test 5134" → process_report 
    3️⃣ "Error analysis for report 5134" → get_report_by_id
    4️⃣ "Run backend test 5144" → run_test
    5️⃣ "Create new Gatling test" → create_backend_test
    6️⃣ "UI performance analysis" → get_ui_reports
    7️⃣ "Schedule UI test" → update_ui_test_schedule
    8️⃣ "Create engagement ticket" → create_ticket

    USER REQUEST: "{user_message}"

    EXTRACTION RULES:
    - Use EXACT action names from the available tools list
    - Extract numeric IDs as entities with proper types
    - For "excel/report" requests → use process_report or create_ui_excel_report
    - For "get/show" requests → use get_* actions  
    - For "run/execute" → use run_test or run_ui_test
    - For "create/new" → use create_* actions

    Return JSON:
    {{
        "task_type": "backend_analysis|test_management|test_execution|ui_analysis|ui_test_management|ui_test_execution|ticket_action",
        "action": "exact_action_name_from_available_tools",
        "entities": [{{"type": "test_id|report_id|board_id", "value": "extracted_id"}}],
        "field_requests": ["excel", "report", "details"],
        "confirmation_question": "Should I [action_description]?",
        "confidence_score": 0.95
    }}

    IMPORTANT: Use exact action names. No custom actions allowed.
    """

    return prompt


def get_action_suggestions_for_analyst(user_message: str) -> List[Dict[str, str]]:
    """
    Provide smart suggestions based on common analyst workflows
    """
    user_lower = user_message.lower()
    suggestions = []

    # Excel/Report Generation
    if any(word in user_lower for word in ['excel', 'report', 'generate', 'export']):
        suggestions.extend([
            {
                "action": "process_report",
                "task_type": "backend_analysis",
                "description": "Generate Excel report from backend test data",
                "example": "process_report for test ID 5134"
            },
            {
                "action": "create_ui_excel_report",
                "task_type": "ui_analysis",
                "description": "Generate Excel report from UI test data",
                "example": "create_ui_excel_report for UI test 5134"
            }
        ])

    # Test Execution
    if any(word in user_lower for word in ['run', 'execute', 'start']):
        suggestions.extend([
            {
                "action": "run_test",
                "task_type": "test_execution",
                "description": "Execute backend performance test",
                "example": "run_test with ID 5144"
            },
            {
                "action": "run_ui_test",
                "task_type": "ui_test_execution",
                "description": "Execute UI performance test",
                "example": "run_ui_test with ID 5134"
            }
        ])

    # Analysis and Reporting
    if any(word in user_lower for word in ['show', 'get', 'list', 'analysis']):
        suggestions.extend([
            {
                "action": "get_reports",
                "task_type": "backend_analysis",
                "description": "List all backend performance reports",
                "example": "get_reports for comparison analysis"
            },
            {
                "action": "get_ui_reports",
                "task_type": "ui_analysis",
                "description": "List all UI performance reports",
                "example": "get_ui_reports for frontend analysis"
            }
        ])

    # Test Management
    if any(word in user_lower for word in ['create', 'new', 'setup']):
        suggestions.extend([
            {
                "action": "create_backend_test",
                "task_type": "test_management",
                "description": "Create new backend performance test",
                "example": "create_backend_test with JMeter/Gatling"
            },
            {
                "action": "create_ui_test",
                "task_type": "ui_test_management",
                "description": "Create new UI performance test",
                "example": "create_ui_test with Lighthouse/Sitespeed"
            }
        ])

    return suggestions


def validate_action_mapping(task_type: str, action: str) -> bool:
    """
    Validate that action exists in available tool mappings
    """
    # Available mappings from your log
    VALID_MAPPINGS = {
        'ticket_action': ['get_ticket_list', 'create_ticket'],
        'backend_analysis': ['get_reports', 'get_report_by_id', 'process_report'],
        'test_management': ['get_tests', 'get_test_by_id', 'create_backend_test'],
        'test_execution': ['run_test'],
        'ui_analysis': ['get_ui_reports', 'get_ui_report_by_id', 'create_ui_excel_report'],
        'ui_test_management': ['get_ui_tests', 'create_ui_test', 'update_ui_test_schedule', 'cancel_ui_test'],
        'ui_test_execution': ['run_ui_test']
    }

    if task_type not in VALID_MAPPINGS:
        logger.warning(f"[ActionValidation] Invalid task_type: {task_type}")
        return False

    if action not in VALID_MAPPINGS[task_type]:
        logger.warning(f"[ActionValidation] Invalid action '{action}' for task_type '{task_type}'")
        logger.info(f"[ActionValidation] Valid actions for {task_type}: {VALID_MAPPINGS[task_type]}")
        return False

    return True


def build_carrier_prompt_(user_message: str, context: Optional[Dict] = None) -> str:
    """
    Enhanced prompt builder with performance analyst focus
    """
    return build_performance_analyst_prompt(user_message, context)


# Performance Analyst Task Automation Recommendations
AUTOMATION_RECOMMENDATIONS = {
    "daily_report_comparison": {
        "manual_process": "Manually download and compare 5 latest reports",
        "automated_solution": "get_reports → process_report → automated Excel comparison",
        "tools_needed": ["get_reports", "process_report"],
        "complexity": "Medium",
        "time_saved": "2-3 hours daily"
    },

    "error_analysis_workflow": {
        "manual_process": "Check each report for errors, analyze logs manually",
        "automated_solution": "get_report_by_id → automated error extraction → create_ticket for issues",
        "tools_needed": ["get_report_by_id", "create_ticket"],
        "complexity": "High",
        "time_saved": "1-2 hours per analysis"
    },

    "baseline_management": {
        "manual_process": "Manually track performance trends and set baselines",
        "automated_solution": "get_reports → trend analysis → automated baseline suggestions",
        "tools_needed": ["get_reports", "process_report"],
        "complexity": "High",
        "time_saved": "4-5 hours weekly"
    },

    "test_execution_monitoring": {
        "manual_process": "Manually trigger tests and monitor results",
        "automated_solution": "run_test → automated monitoring → get_report_by_id when complete",
        "tools_needed": ["run_test", "get_report_by_id"],
        "complexity": "Low",
        "time_saved": "1 hour per test cycle"
    },

    "environment_comparison": {
        "manual_process": "Manually filter and compare different environment results",
        "automated_solution": "get_reports with filters → automated environment comparison",
        "tools_needed": ["get_reports", "process_report"],
        "complexity": "Medium",
        "time_saved": "2-3 hours per comparison"
    },

    "ui_performance_tracking": {
        "manual_process": "Manually run UI tests and compile performance metrics",
        "automated_solution": "run_ui_test → get_ui_report_by_id → create_ui_excel_report",
        "tools_needed": ["run_ui_test", "get_ui_report_by_id", "create_ui_excel_report"],
        "complexity": "Low",
        "time_saved": "1.5 hours per UI analysis"
    },

    "engagement_ticket_management": {
        "manual_process": "Manually create and track engagement tickets",
        "automated_solution": "get_ticket_list → create_ticket with templates → automated tracking",
        "tools_needed": ["get_ticket_list", "create_ticket"],
        "complexity": "Low",
        "time_saved": "30 minutes per engagement setup"
    },

    "test_schedule_optimization": {
        "manual_process": "Manually schedule and manage test runs",
        "automated_solution": "update_ui_test_schedule → automated execution → results compilation",
        "tools_needed": ["update_ui_test_schedule", "run_ui_test", "get_ui_reports"],
        "complexity": "Medium",
        "time_saved": "2 hours weekly"
    }
}


def get_automation_recommendations(user_request: str) -> List[Dict[str, Any]]:
    """
    Provide automation recommendations based on user request
    """
    user_lower = user_request.lower()
    recommendations = []

    # Match user request to automation opportunities
    if any(word in user_lower for word in ['compare', 'comparison', 'latest']):
        recommendations.append({
            "task": "Daily Report Comparison Automation",
            "current_manual_effort": "2-3 hours daily",
            "automation_approach": "Chain get_reports → process_report → automated comparison",
            "implementation": "Create scheduled workflow that pulls latest 5 reports and generates comparison Excel",
            "roi": "High - saves 10-15 hours weekly"
        })

    if any(word in user_lower for word in ['error', 'analysis', 'log']):
        recommendations.append({
            "task": "Automated Error Analysis",
            "current_manual_effort": "1-2 hours per report",
            "automation_approach": "get_report_by_id → parse error patterns → auto-create tickets",
            "implementation": "Set up error pattern recognition and automatic ticket creation for known issues",
            "roi": "Medium - saves 5-8 hours weekly"
        })

    if any(word in user_lower for word in ['baseline', 'threshold', 'trend']):
        recommendations.append({
            "task": "Intelligent Baseline Management",
            "current_manual_effort": "4-5 hours weekly",
            "automation_approach": "get_reports → trend analysis → adaptive baseline suggestions",
            "implementation": "ML-based trend analysis to automatically suggest baseline updates",
            "roi": "High - saves 15-20 hours monthly"
        })

    if any(word in user_lower for word in ['schedule', 'run', 'execute']):
        recommendations.append({
            "task": "Smart Test Execution",
            "current_manual_effort": "1 hour per test cycle",
            "automation_approach": "Intelligent scheduling based on environment availability and historical performance",
            "implementation": "Auto-schedule tests during optimal times, chain execution with result analysis",
            "roi": "Medium - saves 3-5 hours weekly"
        })

    return recommendations


# Export functions and constants
__all__ = [
    'PERFORMANCE_ANALYST_INTENT_EXAMPLES',
    'ANALYST_WORKFLOW_PATTERNS',
    'AUTOMATION_RECOMMENDATIONS',
    'build_performance_analyst_prompt',
    'build_carrier_prompt_',
    'get_action_suggestions_for_analyst',
    'validate_action_mapping',
    'get_automation_recommendations'
]

# Logging for debugging
logger.info("[PerformancePrompts] Performance Analytics prompts module loaded")
logger.info(f"[PerformancePrompts] Configured {len(PERFORMANCE_ANALYST_INTENT_EXAMPLES)} intent examples")
logger.info(f"[PerformancePrompts] Defined {len(AUTOMATION_RECOMMENDATIONS)} automation recommendations")
