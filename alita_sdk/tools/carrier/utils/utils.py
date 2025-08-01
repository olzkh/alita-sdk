import os
import json
import logging
from dataclasses import dataclass, field

from typing import List, Dict, Any, Optional
import zipfile
import io
from enum import Enum

from langchain_core.tools import ToolException
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

import functools
import traceback

# =================================================================================
# MODULE-LEVEL LOGGER
# =================================================================================
logger = logging.getLogger(__name__)


# =================================================================================
# 1. APPLICATION-WIDE CONFIGURATION MODELS (SRP: Storing Configuration)
# =================================================================================
@dataclass(frozen=True)
class GatlingConfig:
    """Configuration constants specific to Gatling report processing."""
    DEFAULT_THINK_TIME: str = "5,0-10,0"
    LOG_LINE_PREFIX_USER: str = "USER"
    LOG_LINE_PREFIX_REQUEST: str = "REQUEST"
    LOG_LINE_PREFIX_GROUP: str = "GROUP"
    LOG_LINE_STATUS_OK: str = "OK"
    LOG_LINE_STATUS_START: str = "START"
    REPORTING_TIMEZONE: str = "US/Eastern"
    REPORT_TYPE: str = "GATLING"
    TIMESTAMP_DIVISOR: int = 1000
    MIN_LOG_PARTS: int = 4
    REQUEST_MIN_PARTS: int = 6
    GROUP_MIN_PARTS: int = 6

# =================================================================================
# EXCEL STYLING THEME
# =================================================================================
@dataclass(frozen=True)
class ExcelStyleTheme:
    """A centralized, immutable theme for Excel report styling."""
    _HEX_GREEN_FILL: str = 'AFF2C9'
    _HEX_YELLOW_FILL: str = 'F7F7A9'
    _HEX_RED_FILL: str = 'F7A9A9'
    _HEX_TEAL_HEADER: str = '7FD5D8'
    _HEX_TEAL_SUMMARY: str = 'CDEBEA'
    _HEX_BLUE_FONT: str = '291A75'
    _HEX_GREEN_FONT: str = '2BBD4D'
    _HEX_RED_FONT: str = 'F90808'
    _HEX_GRAY_BORDER: str = '040404'
    _HEX_WHITE_FONT: str = 'FFFFFF'

    FILL_STATUS_PASSED: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_GREEN_FILL, fill_type='solid'))
    FILL_STATUS_WARNING: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_YELLOW_FILL, fill_type='solid'))
    FILL_STATUS_FAILED: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_RED_FILL, fill_type='solid'))
    FILL_TABLE_HEADER: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_TEAL_HEADER, fill_type='solid'))
    FILL_SUMMARY_HEADER: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_TEAL_SUMMARY, fill_type='solid'))

    # UI-specific fills using the same color scheme
    FILL_UI_HEADER: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_TEAL_HEADER, fill_type='solid'))
    FILL_UI_GOOD: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_GREEN_FILL, fill_type='solid'))
    FILL_UI_WARNING: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_YELLOW_FILL, fill_type='solid'))
    FILL_UI_POOR: PatternFill = field(
        default_factory=lambda: PatternFill(start_color=ExcelStyleTheme._HEX_RED_FILL, fill_type='solid'))

    FONT_HEADER: Font = field(default_factory=lambda: Font(bold=True, color=ExcelStyleTheme._HEX_WHITE_FONT))
    FONT_SUMMARY_LABEL: Font = field(default_factory=lambda: Font(bold=True, color=ExcelStyleTheme._HEX_BLUE_FONT))
    FONT_STATUS_PASSED: Font = field(default_factory=lambda: Font(bold=True, color=ExcelStyleTheme._HEX_GREEN_FONT))
    FONT_STATUS_FAILED: Font = field(default_factory=lambda: Font(bold=True, color=ExcelStyleTheme._HEX_RED_FONT))
    FONT_HYPERLINK: Font = field(
        default_factory=lambda: Font(bold=True, underline="single", color=ExcelStyleTheme._HEX_BLUE_FONT))

    # UI-specific fonts
    FONT_UI_BOLD: Font = field(default_factory=lambda: Font(bold=True, size=16))
    FONT_UI_HEADER: Font = field(default_factory=lambda: Font(bold=True))

    BORDER_DEFAULT_THIN: Border = field(
        default_factory=lambda: Border(left=Side(style="thin", color=ExcelStyleTheme._HEX_GRAY_BORDER),
                                       right=Side(style="thin", color=ExcelStyleTheme._HEX_GRAY_BORDER),
                                       top=Side(style="thin", color=ExcelStyleTheme._HEX_GRAY_BORDER),
                                       bottom=Side(style="thin", color=ExcelStyleTheme._HEX_GRAY_BORDER)))

    ALIGN_CENTER_ALL: Alignment = field(
        default_factory=lambda: Alignment(horizontal="center", vertical="center", wrap_text=True))
    ALIGN_LEFT_CENTER: Alignment = field(
        default_factory=lambda: Alignment(horizontal="left", vertical="center", wrap_text=True))
    ALIGN_JUSTIFY_TOP: Alignment = field(
        default_factory=lambda: Alignment(horizontal="justify", vertical="top", wrap_text=True))

    FORMAT_PERCENTAGE: str = '0.00%'
    FORMAT_SECONDS: str = '0.000'


# Global instances for easy, consistent import.
GATLING_CONFIG = GatlingConfig()
REPORT_THEME = ExcelStyleTheme()


# =================================================================================
# 4. PURE HELPER FUNCTIONS (SRP: Reusable, Stateless Logic)
# =================================================================================
# These functions have no side effects and only depend on their inputs.

def aggregate_errors(test_errors: list) -> dict:
    """Aggregates error counts from a list of error dictionaries."""
    aggregated_errors = {}
    for error_dict in test_errors:
        for err, details in error_dict.items():
            if err not in aggregated_errors:
                aggregated_errors[err] = details.copy()
            else:
                count = aggregated_errors[err].get('Error count', 0)
                aggregated_errors[err]['Error count'] = int(count) + int(details.get('Error count', 0))
    return aggregated_errors


def exclude_from_dict(data: dict, keys_to_exclude_str: str) -> dict:
    """Removes keys from a dictionary if they contain any of the substrings."""
    if not keys_to_exclude_str:
        return data
    exclude_substrings = list(filter(None, keys_to_exclude_str.split(',')))

    filtered_data = data.copy()
    for sub in exclude_substrings:
        filtered_data = {k: v for k, v in filtered_data.items() if sub not in k}
    return filtered_data


def parse_config_from_string(config_str: str) -> dict:
    """Parses configuration from a JSON string or a key:value newline-separated string."""
    try:
        # First, attempt to parse as a valid JSON object.
        return json.loads(config_str)
    except (json.JSONDecodeError, TypeError):
        # If it fails, fall back to parsing as key: value pairs.
        config = {}
        if isinstance(config_str, str):
            for line in config_str.splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
        return config


# =================================================================================
# 5. FUNCTIONS WITH EXTERNAL DEPENDENCIES / SIDE EFFECTS
# =================================================================================
# This section is for functions that interact with the file system, network, etc.

def get_latest_log_file(root_dir: str, log_file_name: str) -> str:
    """Finds the most recently modified sub-folder in a directory and returns the path to a log file within it."""
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory for logs not found: {root_dir}")
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    try:
        # Get all subdirectories and sort by modification time.
        subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                      os.path.isdir(os.path.join(root_dir, d))]
        if not subfolders:
            raise FileNotFoundError(f"No subdirectories found in {root_dir}")

        latest_folder = max(subfolders, key=os.path.getmtime)
        simulation_log_file = os.path.join(latest_folder, log_file_name)

        if not os.path.isfile(simulation_log_file):
            logger.error(f"Log file not found in latest directory: {simulation_log_file}")
            raise FileNotFoundError(f"File not found: {simulation_log_file}")

        logger.info(f"Found latest log file: {simulation_log_file}")
        return simulation_log_file

    except Exception as e:
        logger.error(f"Error while searching for log file in '{root_dir}': {e}", exc_info=True)
        raise


@dataclass(frozen=True)
class AnalysisRulesConfig:
    """
    Configuration for the business rules used in performance analysis.
    This centralizes all "magic numbers" for easy tuning.
    """
    # Response Time Severity Multipliers (based on the primary threshold)
    RT_CRITICAL_MULTIPLIER: float = 3.0
    RT_HIGH_MULTIPLIER: float = 2.0

    # Error Rate Severity Multiplier
    ER_HIGH_MULTIPLIER: float = 2.0

    # Throughput Severity Threshold (as a percentage below target)
    TP_HIGH_SEVERITY_DEFICIT_PCT: float = 50.0

    # System Reliability Thresholds (as a percentage success rate)
    RELIABILITY_STANDARD_SUCCESS_RATE: float = 99.0
    RELIABILITY_HIGH_PRIORITY_THRESHOLD: float = 95.0

    # Imbalance Detection
    IMBALANCE_MIN_TRANSACTIONS: int = 5
    IMBALANCE_FACTOR: float = 5.0  # How many times slower the slowest are compared to the fastest

    # Maximum number of recommendations to return
    MAX_RECOMMENDATIONS: int = 5


# Add a global instance for easy import
ANALYSIS_CONFIG = AnalysisRulesConfig()


class CarrierArtifactUploader:
    """
    A dedicated, reusable utility for handling file I/O and uploading artifacts
    to Carrier. It encapsulates the legacy requirement of zipping files before upload.
    """

    def __init__(self, api_wrapper: Any):
        if not api_wrapper:
            raise ToolException("CarrierArtifactUploader requires a valid api_wrapper")
        self.api = api_wrapper
        self.logger = logging.getLogger(__name__)

    def upload_leg(self, file_bytes: bytes, bucket_name: str, remote_filename: str) -> bool:
        """
        Handles the complete upload process, including zipping, saving to a
        temporary file, and cleanup, based on the legacy requirement.

        Args:
            file_bytes: The raw bytes of the file to upload (e.g., the Excel report).
            bucket_name: The target bucket in Carrier.
            remote_filename: The desired filename on the remote server (e.g., "report.xlsx").

        Returns:
            True if the upload was successful, False otherwise.

        Raises:
            ToolException: If upload fails for any reason.
        """
        if not file_bytes:
            raise ToolException("File bytes cannot be empty")
        if not bucket_name:
            raise ToolException("Bucket name is required")
        if not remote_filename:
            raise ToolException("Remote filename is required")

        temp_zip_path = f"/tmp/{os.path.splitext(remote_filename)[0]}.zip"
        self.logger.info(
            f"Preparing to upload '{remote_filename}' by creating a temporary zip archive at '{temp_zip_path}'.")

        # Step 1: Create a zip archive in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the Excel file bytes to the zip archive with its desired name
            zipf.writestr(remote_filename, file_bytes)
        zip_bytes = zip_buffer.getvalue()

        # Step 2: Write the zip archive bytes to a temporary file
        with open(temp_zip_path, 'wb') as temp_file:
            temp_file.write(zip_bytes)

        self.logger.debug(f"Temporary zip file '{temp_zip_path}' created successfully.")

        # Step 3: Upload the temporary zip file using the provided API wrapper method
        success = self.api.upload_file(bucket_name, temp_zip_path)

        if not success:
            raise ToolException(f"Failed to upload '{temp_zip_path}' to bucket '{bucket_name}'")

        self.logger.info(f"Successfully uploaded '{temp_zip_path}' to bucket '{bucket_name}'.")
        return True


from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel
from abc import ABC, abstractmethod

# ==============================================================================
# PROMPT TEMPLATES - Pattern
# ==============================================================================

REPORT_ANALYSIS_PROMPT = """
You are a Performance Analysis expert tasked with analyzing test reports and providing insights.
Your task is to generate a structured analysis of performance test results including:

Guidelines:
- report_type: Classify the report type (baseline, comparison, ui_performance)
- key_metrics: Extract and summarize key performance indicators
- issues_found: List any performance issues or anomalies detected
- recommendations: Provide actionable recommendations for improvement
- comparison_insights: If comparing reports, highlight differences and trends

Content to analyze: {report_content}
Baseline data (if available): {baseline_data}
"""


class ReportInsights(LangchainBaseModel):
    """Performance report analysis insights"""
    report_type: str
    key_metrics: List[str]
    issues_found: List[str]
    recommendations: List[str]
    comparison_insights: Optional[List[str]] = None


class ReportAnalysis(LangchainBaseModel):
    """Complete report analysis structure"""
    insights: ReportInsights
    confidence_score: float
    analysis_timestamp: str


# ==============================================================================
# INTERFACE DEFINITIONS - Interface Segregation Principle
# ==============================================================================

class IReportProcessor(ABC):
    """Interface for report processing operations"""

    @abstractmethod
    def process_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_insights(self, report_data: Dict[str, Any]) -> ReportAnalysis:
        pass


# ==============================================================================
# CONSTANTS & ENUMS - DRY Principle
# ==============================================================================

class FileType(Enum):
    """File type enumeration"""
    HTML = ".html"
    JSON = ".json"
    EXCEL = ".xlsx"
    ZIP = ".zip"



def tool_logger(func):
    """
    A decorator that provides extended, structured logging for tool execution.
    It logs entry, exit, arguments, return values, and exceptions.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        tool_name = self.name
        # Combine args and kwargs for a complete picture of the input
        all_args = {**kwargs}
        # Handle positional arguments if any (rare for these tools)
        arg_names = func.__code__.co_varnames[1:func.__code__.co_argcount] # [1:] to skip 'self'
        for i, arg in enumerate(args):
            all_args[arg_names[i]] = arg

        logger.info(f"[{tool_name}] ---> Entering tool execution.")
        logger.debug(f"[{tool_name}] Arguments: {all_args}")

        try:
            result = func(self, *args, **kwargs)
            logger.debug(f"[{tool_name}] Raw result: {str(result)[:250]}...") # Log a preview of the result
            logger.info(f"[{tool_name}] <--- Tool execution successful.")
            return result
        except Exception as e:
            # Use ToolException for controlled errors, log others more severely
            if isinstance(e, ToolException):
                logger.warning(f"[{tool_name}] <--- Tool execution failed with ToolException: {e}")
            else:
                # Log the full stack trace for unexpected errors
                stack_trace = traceback.format_exc()
                logger.error(f"[{tool_name}] <--- Tool execution failed with unhandled exception: {e}\n{stack_trace}")
            # Re-raise the exception so the framework can handle it
            raise
    return wrapper