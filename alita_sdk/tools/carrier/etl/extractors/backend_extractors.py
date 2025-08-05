"""
Extractors from Carrier logic

Author: Karen Florykian
"""
import logging
from typing import Dict, Any
from datetime import datetime

from ..etl_pipeline import BaseExtractor
from alita_sdk.tools.carrier.api_wrapper import CarrierAPIWrapper
from langchain_core.tools import ToolException

logger = logging.getLogger(__name__)


class CarrierArtifactExtractor(BaseExtractor):
    """
    Extracts report metadata and the list of artifact files required for processing.
    """

    def __init__(self):
        super().__init__()
        logger.info("CarrierArtifactExtractor initialized")

    def extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts report metadata and a list of artifact filenames from Carrier
        using the provided API wrapper.
        """
        logger.info("Starting artifact metadata extraction...")
        try:
            # Step 1: Get required components from context.
            api_wrapper: CarrierAPIWrapper = context["api_wrapper"]
            report_id = str(context["report_id"])
            logger.info(f"Extracting metadata for report_id: {report_id}")

            # Step 2: Get report metadata using the clean wrapper method.
            report_metadata = api_wrapper.get_report_metadata(report_id)
            build_id = report_metadata.get("build_id")
            bucket_name = report_metadata.get("name", "").replace("_", "").replace(" ", "").lower()

            if not all([build_id, bucket_name]):
                raise ToolException("Report metadata is missing 'build_id' or 'name', cannot proceed.")

            # Step 3: Get the list of all files in the report's bucket via the wrapper.
            # This call is now valid because we implemented the method in the wrapper and client.
            all_artifacts_response = api_wrapper.list_artifacts(bucket_name)
            all_filenames = [artifact.get("name") for artifact in all_artifacts_response.get("rows", [])]

            # Step 4: Filter for the relevant report archives.
            report_archive_prefix = f"reports_test_results_{build_id}"
            artifact_filenames = [
                name for name in all_filenames
                if name and name.startswith(report_archive_prefix) and "excel_report" not in name
            ]

            if not artifact_filenames:
                raise ToolException(
                    f"No report artifact files found for report {report_id} with prefix '{report_archive_prefix}'.")

            logger.info(f"Found {len(artifact_filenames)} artifact files to be processed.")

            # Step 5: Prepare the structured output using our helper method.
            return self._prepare_extraction_result(report_metadata, artifact_filenames, bucket_name)

        except KeyError as e:
            raise ToolException(f"Extraction context is missing a required key: {e}.")
        except Exception as e:
            logger.error(f"Artifact metadata extraction failed for report {report_id}: {e}", exc_info=True)
            raise ToolException(f"Failed to extract artifact metadata: {str(e)}")

    def _prepare_extraction_result(self, report_metadata: Dict, artifact_filenames: list, bucket_name: str) -> Dict[
        str, Any]:
        """
        Prepares the final, structured dictionary to be passed to the transformer.
        This dictionary contains all information needed for the next ETL stage.
        """
        return {
            "report_metadata": report_metadata,
            "artifact_filenames": artifact_filenames,
            "bucket_name": bucket_name,
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "extractor_type": "CarrierArtifactExtractor",
                "artifacts_found": len(artifact_filenames),
                "report_id": report_metadata.get("id"),
                "success": True
            }
        }


class ComparisonExtractor(BaseExtractor):
    """
    A simple extractor for comparison pipelines. It doesn't fetch new data,
    but instead validates and passes through the list of report metadata
    provided in the initial context.
    """

    def __init__(self):
        super().__init__()
        logger.info("ComparisonExtractor initialized")

    def extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts the list of report metadata from the context.
        """
        logger.info("Starting comparison extraction...")
        try:
            reports_meta = context["reports_meta"]
            if not isinstance(reports_meta, list) or len(reports_meta) < 2:
                raise ToolException("Comparison requires at least two reports in 'reports_meta'.")

            logger.info(f"Found {len(reports_meta)} reports to compare.")

            # The "extracted" data is simply the list of reports to be processed.
            return {"reports_meta": reports_meta}

        except KeyError:
            raise ToolException("Comparison context is missing the required 'reports_meta' key.")
        except Exception as e:
            logger.error(f"Comparison extraction failed: {e}", exc_info=True)
            raise ToolException(f"Failed to extract comparison data: {str(e)}")
