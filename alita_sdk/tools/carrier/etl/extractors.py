import io
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .etl_pipeline import BaseExtractor
from langchain_core.tools import ToolException

# Configure logger
logger = logging.getLogger(__name__)


class GatlingLogExtractor(BaseExtractor):
    """
    🎯 Production-ready Gatling log extractor with comprehensive validation,
    error handling, and user guidance following legacy patterns.
    """

    def __init__(self):
        super().__init__()
        logger.info("🎯 GatlingLogExtractor initialized")

    def extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Extracts Gatling log content from LOCAL files created by get_report_file_name.
        No longer tries to download files that only exist locally.
        """
        logger.info("🚀 Starting robust Gatling log extraction process...")

        try:
            # Step 1: Validate context
            validated_context = self._validate_extraction_context(context)
            api_wrapper = validated_context["api_wrapper"]
            report_id = validated_context["report_id"]

            # Step 2: Get report metadata and LOCAL file paths
            report_info = self._get_and_validate_report(api_wrapper, report_id)
            bucket_name = report_info.get("name", "").replace("_", "").replace(" ", "").lower()
            if not bucket_name:
                raise ToolException(f"Could not determine bucket name from report metadata for report ID {report_id}.")

            # ✅ FIX: get_report_file_name returns LOCAL file paths, not remote filenames
            logger.info("📁 Getting local file paths from get_report_file_name...")
            _, test_log_filepath, errors_log_filepath = api_wrapper.get_report_file_name(report_id)

            if not test_log_filepath:
                raise ToolException(f"API did not return a test log filepath for report ID {report_id}.")

            logger.info(f"📄 Local test log file path: {test_log_filepath}")
            logger.info(f"📄 Local error log file path: {errors_log_filepath}")

            # ✅ FIX: Read directly from local file (no download needed)
            if not os.path.exists(test_log_filepath):
                raise ToolException(f"Local test log file not found: {test_log_filepath}")

            logger.info(f"📖 Reading local log file: {test_log_filepath}")
            with open(test_log_filepath, 'r', encoding='utf-8') as log_file:
                log_content = log_file.read()

            logger.info(f"✅ Successfully read {len(log_content)} characters from local log file.")

            # Step 3: Prepare the result with the actual content
            extraction_result = self._prepare_extraction_result(
                log_content,
                report_info,
                {
                    "test_log_path": test_log_filepath,
                    "error_log_path": errors_log_filepath
                }
            )

            logger.info("✅ Gatling log extraction completed successfully")
            return extraction_result

        except ToolException:
            raise
        except Exception as e:
            logger.error(f"💥 Unexpected extraction failure: {e}", exc_info=True)
            raise ToolException(f"🚨 Gatling log extraction failed unexpectedly: {str(e)}")

    def _validate_extraction_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        🛡️ Validates extraction context with detailed error guidance.
        """
        logger.debug("🛡️ Validating extraction context...")

        # Validate API wrapper
        api_wrapper = context.get("api_wrapper")
        if not api_wrapper:
            logger.error("❌ Missing API wrapper in context")
            raise ToolException(
                "🔧 API wrapper not available\n"
                "💡 This is a system configuration issue\n"
                "🔧 Solution: Ensure Carrier API is properly initialized\n"
                "📞 Contact administrator if this persists"
            )

        # Validate report ID with flexible key names
        report_id = context.get("report_id") or context.get("source_report_id")
        if not report_id:
            logger.error("❌ Missing report ID in context")
            available_keys = list(context.keys())
            raise ToolException(
                "📋 Report ID is required for extraction\n"
                f"🔍 Available context keys: {available_keys}\n"
                "💡 Solution: Provide either 'report_id' or 'source_report_id'\n"
                "🔧 Example: Use get_reports tool to find valid report IDs"
            )

        # Validate report ID format
        try:
            report_id_int = int(report_id)
            if report_id_int <= 0:
                raise ValueError("Report ID must be positive")
        except (ValueError, TypeError):
            logger.error(f"❌ Invalid report ID format: {report_id}")
            raise ToolException(
                f"🔢 Invalid report ID format: '{report_id}'\n"
                "✅ Report ID must be a positive integer\n"
                "💡 Example: 27, 156, 2847\n"
                "🔧 Use get_reports tool to find valid report IDs"
            )

        logger.debug(f"✅ Context validation passed for report ID: {report_id}")
        return {
            "api_wrapper": api_wrapper,
            "report_id": str(report_id_int),
            "original_context": context
        }

    def _get_and_validate_report(self, api_wrapper, report_id: str) -> Dict[str, Any]:
        """
        📋 Retrieves and validates report metadata with user guidance.
        """
        logger.info(f"📋 Fetching report metadata for ID: {report_id}")

        try:
            # Get reports list using the correct API method
            reports_data = api_wrapper.get_reports_list()

            # Handle both string and list responses
            if isinstance(reports_data, list):
                reports = reports_data
                logger.debug("📄 Reports data already parsed as list")
            else:
                try:
                    reports = json.loads(reports_data)
                    logger.debug("📄 Reports data parsed from JSON string")
                except json.JSONDecodeError as e:
                    logger.error(f"💥 Failed to parse reports JSON: {e}")
                    raise ToolException(
                        "🚨 Invalid response format from Carrier API\n"
                        "💥 Unable to parse reports list\n"
                        "🔧 Next Steps:\n"
                        "   1. 🌐 Check Carrier platform status\n"
                        "   2. 🔄 Try again in a few moments\n"
                        "   3. 📞 Contact support if issue persists"
                    )

            logger.debug(f"📊 Retrieved {len(reports)} total reports")

            # Find the specific report
            report_info = None
            try:
                report_id_int = int(report_id)
                report_info = next((r for r in reports if r.get('id') == report_id_int), None)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid report ID for comparison: {report_id}")

            if not report_info:
                logger.warning(f"🔍 Report ID {report_id} not found in {len(reports)} available reports")

                # Provide helpful suggestions
                available_ids = [str(r.get('id', 'unknown')) for r in reports[:5]]
                raise ToolException(
                    f"📋 Report not found: ID '{report_id}'\n"
                    f"🔍 Found {len(reports)} total reports\n"
                    f"💡 Recent report IDs: {', '.join(available_ids)}\n"
                    "🔧 Next Steps:\n"
                    "   1. 📋 Use get_reports tool to see all available reports\n"
                    "   2. 🔍 Verify the report ID is correct\n"
                    "   3. ✅ Ensure the report belongs to your project"
                )

            # Validate report has required fields
            self._validate_report_structure(report_info, report_id)

            logger.info(f"✅ Report metadata validated for '{report_info.get('name', 'unknown')}'")
            return report_info

        except ToolException:
            raise
        except Exception as e:
            logger.error(f"💥 Failed to retrieve report metadata: {e}", exc_info=True)
            raise ToolException(
                f"🚨 Failed to retrieve report information\n"
                f"💥 Error: {str(e)}\n"
                f"🔧 Next Steps:\n"
                f"   1. 🌐 Check Carrier platform connectivity\n"
                f"   2. 🔍 Verify report ID '{report_id}' exists\n"
                f"   3. 🔄 Try again in a few moments"
            )

    def _validate_report_structure(self, report_info: Dict[str, Any], report_id: str):
        """
        🔍 Validates report structure and provides guidance for missing fields.
        """
        required_fields = ['id', 'name']
        missing_fields = [field for field in required_fields if not report_info.get(field)]

        if missing_fields:
            logger.error(f"❌ Report structure validation failed: missing {missing_fields}")
            raise ToolException(
                f"🚨 Invalid report structure for ID '{report_id}'\n"
                f"❌ Missing required fields: {', '.join(missing_fields)}\n"
                "💡 This may indicate:\n"
                "   • 🔧 Report is still being processed\n"
                "   • 📋 Report metadata is incomplete\n"
                "   • 🚨 API response format changed\n"
                "🔧 Solution: Use get_report_by_id tool for detailed report status"
            )

        # Validate load generator type
        lg_type = report_info.get('lg_type', '').lower()
        if lg_type and lg_type not in ['gatling', 'jmeter']:
            logger.warning(f"⚠️ Unexpected load generator type: {lg_type}")

        logger.debug(f"✅ Report structure validation passed for {required_fields}")

    def _prepare_extraction_result(self, log_content: str, report_info: Dict[str, Any],
                                   file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        ✨ Prepares the final extraction result with comprehensive metadata.
        """
        logger.info("✨ Preparing extraction result with metadata...")

        # Validate log content is not empty
        if not log_content.strip():
            logger.warning("⚠️ Log file appears to be empty")
            raise ToolException(
                "📄 Log file is empty\n"
                "🔍 Possible reasons:\n"
                "   • 🔄 Test is still generating logs\n"
                "   • ❌ Test failed immediately\n"
                "   • 📋 No requests were executed\n"
                "🔧 Solution: Check test configuration and execution status"
            )
        log_stream = io.StringIO(log_content)
        error_log_path = file_paths.get("error_log_path", "")
        has_error_log = bool(error_log_path) and os.path.exists(error_log_path)

        # Prepare comprehensive result
        extraction_result = {
            "log_content_stream": log_stream,
            "log_content": log_content,
            "report_metadata": report_info,
            "file_paths": file_paths,
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "extractor_type": "GatlingLogExtractor",
                "content_size": len(log_content),
                "lg_type": report_info.get('lg_type', 'gatling'),
                "report_name": report_info.get('name', 'unknown'),
                "success": True
            }
        }

        # Add validation summary
        validation_summary = {
            "has_content": len(log_content.strip()) > 0,
            "content_size_bytes": len(log_content.encode('utf-8')),
            "estimated_lines": log_content.count('\n') + 1,
            "has_error_log": has_error_log,
            "report_id": report_info.get('id'),
            "report_status": report_info.get('test_status', 'unknown')
        }

        extraction_result["validation_summary"] = validation_summary

        logger.info("📊 Extraction result summary:")
        logger.info(f"   📄 Content size: {validation_summary['content_size_bytes']} bytes")
        logger.info(f"   📝 Estimated lines: {validation_summary['estimated_lines']}")
        logger.info(f"   🚨 Has error log: {validation_summary['has_error_log']}")
        logger.info(f"   📋 Report status: {validation_summary['report_status']}")

        return extraction_result

    def get_missing_input_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        💡 Provides guided response when required inputs are missing.
        Following legacy pattern for parameter confirmation.
        """
        logger.info("💡 Generating missing input guidance...")

        # Check what's missing and provide specific guidance
        missing_items = []
        available_items = []

        if not context.get("report_id") and not context.get("source_report_id"):
            missing_items.append("report_id")
        else:
            available_items.append("report_id")

        if not context.get("api_wrapper"):
            missing_items.append("api_wrapper (system configuration)")
        else:
            available_items.append("api_wrapper")

        # Get default parameters following legacy pattern
        default_parameters = self._get_default_extraction_parameters()

        return {
            "message": "🔧 Missing required inputs for Gatling log extraction",
            "status": "awaiting_input",
            "missing_inputs": missing_items,
            "available_inputs": available_items,
            "required_parameters": {
                "report_id": {
                    "description": "📋 The ID of the report to extract logs from",
                    "type": "integer",
                    "example": "27",
                    "required": True,
                    "how_to_get": "Use get_reports tool to find available report IDs"
                }
            },
            "optional_parameters": default_parameters,
            "next_steps": [
                "📋 Use get_reports tool to find available reports",
                "🔍 Copy the report ID you want to process",
                "🚀 Call process_and_generate_report with the report ID",
                "💡 Default extraction parameters will be used if not specified"
            ],
            "examples": [
                {
                    "description": "🎯 Extract logs from report ID 27",
                    "command": "process_and_generate_report",
                    "parameters": {
                        "report_id": "27",
                        "pipeline_type": "gatling_to_excel"
                    }
                }
            ]
        }

    def _get_default_extraction_parameters(self) -> Dict[str, Any]:
        """
        📋 Returns default extraction parameters following legacy pattern.
        """
        return {
            "think_time": {
                "value": "2,0-5,0",
                "description": "🕐 Think time configuration for analysis",
                "type": "string"
            },
            "pct": {
                "value": "95Pct",
                "description": "📊 Percentile for response time analysis",
                "type": "string",
                "options": ["95Pct", "99Pct", "50Pct"]
            },
            "tp_threshold": {
                "value": 10,
                "description": "🎯 Throughput threshold (requests/second)",
                "type": "integer"
            },
            "rt_threshold": {
                "value": 500,
                "description": "⏱️ Response time threshold (milliseconds)",
                "type": "integer"
            },
            "er_threshold": {
                "value": 5,
                "description": "🚨 Error rate threshold (percentage)",
                "type": "integer"
            }
        }
