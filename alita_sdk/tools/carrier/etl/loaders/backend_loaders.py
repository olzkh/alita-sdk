import io
import os
import logging
from datetime import datetime
from typing import Dict, Any
from langchain_core.tools import ToolException
import zipfile
from ..etl_pipeline import BaseLoader
from ..reporting.core.chart_builder import ChartBuilder
from ..reporting.core.data_models import PerformanceReport
from ..reporting.backend_excel_reporter import ExcelReporter
from ..reporting.core.markdown_builder import MarkdownReportBuilder
from ..reporting.backend_excel_reporter import ComparisonReporter

from ...utils.utils import CarrierArtifactUploader, DateTimeUtils


class CarrierExcelLoader(BaseLoader):
    """
    🎯 Excel loader that generates reports and provides download links.
    Uses existing API wrapper methods following DRY principles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: PerformanceReport, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        💾 Generates Excel report and uploads as ZIP using legacy pattern.
        """
        self.logger.info("Initiating Loader phase...")

        try:
            # Validate context
            api_wrapper = context.get("api_wrapper")
            report_metadata = context.get("report_metadata", {})

            if not api_wrapper:
                raise ToolException("API wrapper missing from context.")

            # Step 1: Generate Excel report in memory
            self.logger.info("📊 Generating Excel report...")
            excel_reporter = ExcelReporter()
            report_workbook = excel_reporter.generate_workbook(transformed_data)

            # Convert to bytes
            excel_buffer = io.BytesIO()
            report_workbook.save(excel_buffer)
            excel_bytes = excel_buffer.getvalue()
            excel_size = len(excel_bytes)

            self.logger.info(f"✅ Successfully rendered Excel report ({excel_size} bytes).")

            # Step 2: Prepare upload details
            bucket_name, file_name = self._get_upload_details(report_metadata)

            # Step 3: Use legacy ZIP upload method
            uploader = CarrierArtifactUploader(api_wrapper)

            self.logger.info(f"📤 Uploading '{file_name}' as ZIP to bucket '{bucket_name}'...")

            upload_success = uploader.upload_leg(excel_bytes, bucket_name, file_name)

            if not upload_success:
                raise ToolException("Report was generated but upload failed.")

            # Step 4: ✅: Generate ZIP download link
            zip_file_name = f"{os.path.splitext(file_name)[0]}.zip"  # Change .xlsx to .zip
            download_url = self._generate_download_link(api_wrapper, bucket_name, zip_file_name)

            # Step 5: Return comprehensive result
            self.logger.info("✅ Loading phase completed successfully.")

            return {
                "status": "success",
                "report_url": getattr(transformed_data, 'carrier_report_url', 'Unknown'),
                "message": "Excel report generated and uploaded as ZIP successfully",
                "file_name": zip_file_name,
                "excel_file_name": file_name,
                "bucket_name": bucket_name,
                "file_size_bytes": excel_size,
                "download_url": download_url,
                "upload_success": upload_success,
                "metadata": {
                    "report_id": report_metadata.get("id"),
                    "report_name": report_metadata.get("name"),
                    "build_status": getattr(transformed_data, 'build_status', 'Unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "loader_type": "CarrierExcelLoader",
                    "archive_type": "ZIP"
                }
            }

        except Exception as e:
            self.logger.error(f"💥 Loading phase failed: {e}", exc_info=True)
            raise ToolException(f"Excel loading failed: {str(e)}")

    def _get_upload_details(self, metadata: dict) -> tuple[str, str]:
        """
        📋 Extract upload details from metadata using existing patterns.
        """
        # Use same bucket naming as existing code
        bucket_name = metadata.get("name", "default_bucket").replace("_", "").replace(" ", "").lower()

        # Generate filename using existing pattern
        build_id = metadata.get("build_id")
        if not build_id:
            # Fallback to report ID if no build_id
            report_id = metadata.get("id", "unknown")
            build_id = f"{report_id}"

        file_name = f"reports_test_results_{build_id}_excel_report.xlsx"

        self.logger.debug(f"📁 Upload details: bucket='{bucket_name}', file='{file_name}'")
        return bucket_name, file_name

    def _generate_download_link(self, api_wrapper, bucket_name: str, file_name: str) -> str:
        """
        Generates the artifact download URL using the same logic as CarrierExcelLoader.
        """
        try:
            if hasattr(api_wrapper, 'carrier_client'):
                project_id = api_wrapper.carrier_client.credentials.project_id
                base_url = api_wrapper.carrier_client.credentials.url.rstrip('/')
            else:
                project_id = api_wrapper._client.credentials.project_id
                base_url = api_wrapper._client.credentials.url.rstrip('/')

            download_url = f"{base_url}/api/v1/artifacts/artifact/default/{project_id}/{bucket_name}/{file_name}?integration_id=1&is_local=False"
            self.logger.info(f"Successfully generated download link: {download_url}")
            return download_url
        except Exception as e:
            self.logger.warning(f"Could not generate download link: {e}. Providing manual access info.")
            return f"Upload successful. Access file '{file_name}' in bucket '{bucket_name}'."


class CarrierPPTXLoader(BaseLoader):
    """
    📊 PowerPoint loader - placeholder for future implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("📊 CarrierPPTXLoader is not yet implemented.")
        raise NotImplementedError("PPTX loading not implemented yet")


class CarrierDocxLoader(BaseLoader):
    """
    📝 Word document loader - placeholder for future implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning("📝 CarrierDocxLoader is not yet implemented.")
        raise NotImplementedError("DOCX loading not implemented yet")


class ComparisonExcelLoader:
    """
    Orchestrates the generation of a complete comparison package (ZIP),
    including an interactive Excel report, markdown analysis, and charts.
    """

    def __init__(self, history_limit: int = 20):
        self.logger = logging.getLogger(__name__)
        self.history_limit = history_limit
        self.reporter = ComparisonReporter()
        self.markdown_builder = MarkdownReportBuilder()
        self.chart_builder = ChartBuilder()
        self.uploader = None

    def _generate_specific_charts(self, chart_data: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Generates all required charts by calling specific methods on the ChartBuilder.
        This avoids assuming a generic 'generate_all_charts' method.
        """
        self.logger.info("Generating specific performance charts...")
        chart_files = {}

        # 1. Generate the Performance Trend Chart
        try:
            trend_buffer = io.BytesIO()
            if self.chart_builder.create_performance_trend_chart(chart_data, trend_buffer):
                trend_buffer.seek(0)
                chart_files['performance_trends.png'] = trend_buffer.getvalue()
                self.logger.info("Successfully generated 'performance_trends.png'.")
            else:
                self.logger.warning("Performance trend chart could not be generated (check data).")
        except Exception as e:
            self.logger.error(f"Failed to create performance trend chart: {e}", exc_info=True)

        # 2. Generate the Side-by-Side Comparison Chart
        try:
            comparison_buffer = io.BytesIO()
            if self.chart_builder.create_comparison_chart(chart_data, comparison_buffer):
                comparison_buffer.seek(0)
                chart_files['test_comparison.png'] = comparison_buffer.getvalue()
                self.logger.info("Successfully generated 'test_comparison.png'.")
            else:
                self.logger.warning("Test comparison chart could not be generated (check data).")
        except Exception as e:
            self.logger.error(f"Failed to create test comparison chart: {e}", exc_info=True)

        self.logger.info(f"Finished chart generation. Total charts created: {len(chart_files)}.")
        return chart_files

    def load(self, transformed_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point to generate, package, and upload the comparison report.
        """
        self.logger.info("Orchestrating comparison report package generation...")
        api_wrapper = context["api_wrapper"]
        self.uploader = CarrierArtifactUploader(api_wrapper)
        test_name = context.get("test_name", "default_test")
        bucket_name = context.get("comparison_bucket", "performance-comparisons")

        # 1. Generate Excel Workbook Object, including AI analysis
        reports = transformed_data.get("all_reports", [])
        ai_analysis = transformed_data.get("comparison_analysis")
        workbook_obj = self.reporter.generate_comparison_workbook(reports, ai_analysis)

        excel_buffer = io.BytesIO()
        workbook_obj.save(excel_buffer)
        excel_bytes = excel_buffer.getvalue()

        # 2. Generate Markdown Content
        markdown_content = self.markdown_builder.generate_markdown_content(transformed_data)

        # 3. Generate Charts
        chart_data = self._extract_chart_data(transformed_data)
        charts = self._generate_specific_charts(chart_data)

        # 4. Create ZIP Package
        zip_filename = f"comparison_{test_name}_{DateTimeUtils.get_current_timestamp()}.zip"
        zip_bytes = self._create_zip_package(excel_bytes, markdown_content, charts, test_name)

        # 5. Upload ZIP
        upload_success = self.uploader.upload(bucket_name, zip_filename, zip_bytes)
        if not upload_success:
            raise ToolException(f"Failed to upload ZIP package to {bucket_name}")

        # 6. Generate and return final result
        download_url = self.uploader.generate_download_url(bucket_name, zip_filename)
        print(download_url)
        return {
            "status": "success",
            "message": "Comparison package created and uploaded successfully.",
            "download_url": download_url,
        }

    def _create_zip_package(self, excel_bytes: bytes, md_content: str, charts: Dict[str, bytes],
                            test_name: str) -> bytes:
        """Packages all artifacts into a single ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{test_name}_report.xlsx", excel_bytes)
            zf.writestr("analysis.md", md_content.encode('utf-8'))
            for filename, chart_bytes in charts.items():
                zf.writestr(f"charts/{filename}", chart_bytes)
        return zip_buffer.getvalue()

    def _extract_chart_data(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and transforms performance data into the precise structure
        required by the ChartBuilder. This method acts as an adapter between
        the PerformanceReport data model and the visualization module.
        """
        self.logger.info("Extracting and transforming data for chart generation...")
        all_reports = transformed_data.get("all_reports", [])
        if not all_reports:
            self.logger.warning("No reports found in transformed_data to extract chart data from.")
            return {}

        chart_data = {}
        for i, report in enumerate(all_reports):
            summary = getattr(report, 'summary', None)
            if not summary:
                self.logger.warning(f"Skipping report {i} for chart data due to missing summary.")
                continue

            # Generate a unique and descriptive key for this report run
            report_key = self._generate_report_key(report)

            # Map the PerformanceReport data to the flat structure ChartBuilder expects
            chart_data[report_key] = {
                # Timestamp for sorting chronologically
                'timestamp': summary.date_start,

                # Top-level summary metrics required by the trend chart
                'throughput': getattr(summary, 'throughput', 0.0),
                'error_rate': getattr(summary, 'error_rate', 0.0),
                # The chart expects response times in seconds, so we convert from ms
                'avg_response_time': getattr(report.transactions.get("Total", object()), 'avg', 0.0) / 1000.0,
                'p90_response_time': getattr(report.transactions.get("Total", object()), 'pct90', 0.0) / 1000.0,

                # Detailed transaction data required by the comparison chart
                'transactions': self._extract_transaction_chart_data(report.transactions)
            }

        self.logger.info(f"Successfully extracted chart data for {len(chart_data)} reports.")
        return chart_data

    def _generate_report_key(self, report: PerformanceReport) -> str:
        """
        Generates a unique, human-readable key for a report to be used in charts.
        This is independent of the Excel sheet naming logic.
        """
        summary = report.summary
        date_obj = DateTimeUtils.parse_datetime(summary.date_start)
        # Use a concise format for chart labels
        return f"Run_{date_obj.strftime('%m-%d_%H:%M')}"

    def _extract_transaction_chart_data(self, transactions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extracts and maps transaction data to the structure expected by the
        comparison chart, converting metrics to the correct units (seconds).
        """
        tx_chart_data = {}
        for tx_name, metrics in transactions.items():
            if tx_name == "Total": continue
            tx_chart_data[tx_name] = {
                'avg_response_time': getattr(metrics, 'avg', 0.0) / 1000.0,
                'p90_response_time': getattr(metrics, 'pct90', 0.0) / 1000.0,
                'error_rate': getattr(metrics, 'error_rate', 0.0)
            }
        return tx_chart_data
