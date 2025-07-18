import io
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
        📥 Extracts Gatling log file content with comprehensive validation and error handling.

        Args:
            context: Dictionary containing extraction parameters

        Returns:
            Dictionary with log content stream and metadata

        Raises:
            ToolException: With detailed user guidance for resolution
        """
        logger.info("🚀 Starting Gatling log extraction process...")

        try:
            # 🛡️ Step 1: Validate and extract context parameters
            validated_context = self._validate_extraction_context(context)
            api_wrapper = validated_context["api_wrapper"]
            report_id = validated_context["report_id"]

            # 📋 Step 2: Get and validate report metadata
            report_info = self._get_and_validate_report(api_wrapper, report_id)

            # 🔍 Step 3: Determine file paths using legacy approach
            file_paths = self._determine_file_paths(api_wrapper, report_id, report_info)

            # 📥 Step 4: Extract log content using local file path (legacy method)
            log_content = self._extract_log_content(file_paths["test_log_path"])

            # ✅ Step 5: Prepare extraction result
            extraction_result = self._prepare_extraction_result(
                log_content,
                report_info,
                file_paths
            )

            logger.info("✅ Gatling log extraction completed successfully")
            return extraction_result

        except ToolException:
            raise
        except Exception as e:
            logger.error(f"💥 Unexpected extraction failure: {e}", exc_info=True)
            raise ToolException(
                f"🚨 Gatling log extraction failed unexpectedly\n"
                f"💥 Error: {str(e)}\n"
                f"🔧 Next Steps:\n"
                f"   1. 🔍 Use get_report_by_id tool to verify report status\n"
                f"   2. 📋 Check if report ID '{context.get('report_id', 'unknown')}' exists\n"
                f"   3. 🌐 Verify Carrier platform connectivity\n"
                f"   4. 📞 Contact support if issue persists"
            )

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

    def _determine_file_paths(self, api_wrapper, report_id: str, report_info: Dict[str, Any]) -> Dict[str, str]:
        """
        🔍 Determines file paths using legacy Carrier API approach (no download_report_as_bytes).
        """
        logger.info("🔍 Determining file paths using legacy API approach...")

        try:
            # Use legacy API method to get file paths
            report, test_log_file_path, errors_log_file_path = api_wrapper.get_report_file_name(report_id)

            logger.info(f"📁 File paths determined:")
            logger.info(f"   📊 Test log: {test_log_file_path}")
            logger.info(f"   🚨 Error log: {errors_log_file_path}")

            # Validate paths exist
            if not test_log_file_path:
                logger.error("❌ Test log file path is empty")
                raise ToolException(
                    "📁 Test log file not available\n"
                    "🔍 Possible reasons:\n"
                    "   • 🔄 Test is still running\n"
                    "   • ❌ Test failed before generating logs\n"
                    "   • 📋 Log files were not uploaded\n"
                    "🔧 Solution: Use get_report_by_id to check test status"
                )

            return {
                "test_log_path": test_log_file_path,
                "error_log_path": errors_log_file_path,
                "report_metadata": report
            }

        except Exception as e:
            logger.error(f"💥 Failed to determine file paths: {e}", exc_info=True)
            raise ToolException(
                f"🚨 Unable to locate log files for report '{report_id}'\n"
                f"💥 Error: {str(e)}\n"
                "🔍 Possible causes:\n"
                "   • 📋 Report is still being processed\n"
                "   • 🔄 Test execution is incomplete\n"
                "   • 📁 Log files are not yet available\n"
                "🔧 Next Steps:\n"
                "   1. ⏳ Wait for test completion if still running\n"
                "   2. 🔍 Use get_report_by_id to check status\n"
                "   3. 🔄 Try again after test finishes"
            )

    def _extract_log_content(self, test_log_path: str) -> str:
        """
        📥 Extracts log content from local file path (legacy approach - no bytes download).
        """
        logger.info(f"📥 Extracting log content from: {test_log_path}")

        try:
            # Read log file content directly (legacy approach)
            # Note: In legacy code, parsers read directly from file paths
            # We simulate this by reading the file if it's accessible

            with open(test_log_path, 'r', encoding='utf-8') as log_file:
                log_content = log_file.read()

            logger.info(f"✅ Successfully extracted {len(log_content)} characters from log file")
            logger.debug(f"📊 Log preview (first 200 chars): {log_content[:200]}...")

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

            return log_content

        except FileNotFoundError:
            logger.error(f"❌ Log file not found: {test_log_path}")
            raise ToolException(
                f"📁 Log file not accessible\n"
                f"📍 Path: {test_log_path}\n"
                "🔍 This indicates:\n"
                "   • 📋 Test is still running\n"
                "   • 🔄 Log files not yet uploaded\n"
                "   • 🚨 File system access issue\n"
                "🔧 Solution: Use legacy download method or wait for completion"
            )
        except PermissionError:
            logger.error(f"❌ Permission denied accessing: {test_log_path}")
            raise ToolException(
                f"🔒 Access denied to log file\n"
                f"📍 Path: {test_log_path}\n"
                "🔧 This is a system permission issue\n"
                "📞 Contact administrator for file access resolution"
            )
        except UnicodeDecodeError as e:
            logger.error(f"❌ Encoding error reading log file: {e}")
            raise ToolException(
                "📄 Log file encoding issue\n"
                "💥 Unable to read file content as UTF-8\n"
                "🔧 Possible solutions:\n"
                "   • 📋 File may be corrupted\n"
                "   • 🔄 Try regenerating the report\n"
                "   • 📞 Contact support for file analysis"
            )
        except Exception as e:
            logger.error(f"💥 Unexpected error reading log file: {e}", exc_info=True)
            raise ToolException(
                f"🚨 Unexpected error accessing log file\n"
                f"💥 Error: {str(e)}\n"
                f"📍 Path: {test_log_path}\n"
                "🔧 Next Steps:\n"
                "   1. 🔄 Try again in a few moments\n"
                "   2. 🔍 Check file system status\n"
                "   3. 📞 Contact support if issue persists"
            )

    def _prepare_extraction_result(self, log_content: str, report_info: Dict[str, Any],
                                   file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        ✨ Prepares the final extraction result with comprehensive metadata.
        """
        logger.info("✨ Preparing extraction result with metadata...")

        # Create string IO stream for compatibility with legacy parsers
        log_stream = io.StringIO(log_content)

        # Prepare comprehensive result
        extraction_result = {
            "log_content_stream": log_stream,
            "log_content": log_content,  # Raw content for debugging
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
            "has_error_log": bool(file_paths.get("error_log_path")),
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

# class JMeterLogExtractor(BaseExtractor):
#     """
#     🎯 Production-ready JMeter log extractor with comprehensive validation.
#     """
#
#     def __init__(self):
#         super().__init__()
#         logger.info("⚡ JMeterLogExtractor initialized")
#
#     def extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         📥 Extracts JMeter log file content with validation and error handling.
#         """
#         logger.info("⚡ Starting JMeter log extraction process...")
#
#         try:
#             # Reuse validation logic from Gatling extractor
#             gatling_extractor = GatlingLogExtractor()
#             validated_context = gatling_extractor._validate_extraction_context(context)
#
#             api_wrapper = validated_context["api_wrapper"]
#             report_id = validated_context["report_id"]
#
#             # Get report metadata
#             report_info = gatling_extractor._get_and_validate_report(api_wrapper, report_id)
#
#             # Validate this is a JMeter report
#             lg_type = report_info.get('lg_type', '').lower()
#             if lg_type and lg_type != 'jmeter':
#                 logger.warning(f"⚠️ Expected JMeter report, found: {lg_type}")
#                 raise ToolException(
#                     f"🔧 Load generator type mismatch\n"
#                     f"📋 Expected: JMeter\n"
#                     f"📊 Found: {lg_type}\n"
#                     "💡 Solution: Use the correct pipeline type:\n"
#                     "   • ⚡ For JMeter reports: 'jmeter_to_excel
