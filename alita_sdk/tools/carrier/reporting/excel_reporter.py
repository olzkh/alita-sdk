"""
Excel Workbook Generator - Production Ready with Legacy Formatting Support

This module maintains full compatibility with legacy Excel formatting while
implementing DRY and SOLID principles. All conditional formatting is applied
at spreadsheet level using Excel's native conditional formatting rules.

Author: Karen Florykian
"""

import logging
from typing import List, Dict, Any
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Font, Border, Side, Alignment
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule

from .core.data_models import PerformanceReport, TransactionMetrics, PerformanceStatus, ThresholdConfig, \
    TransactionAnalysis

logger = logging.getLogger(__name__)

RED_COLOR = 'F7A9A9'
GREEN_COLOR = 'AFF2C9'
YELLOW_COLOR = 'F7F7A9'
RED_COLOR_FONT = '00F90808'
GREEN_COLOR_FONT = '002BBD4D'
RED_FILL = PatternFill(start_color=RED_COLOR, end_color=RED_COLOR, fill_type='solid')
GREEN_FILL = PatternFill(start_color=GREEN_COLOR, end_color=GREEN_COLOR, fill_type='solid')
YELLOW_FILL = PatternFill(start_color=YELLOW_COLOR, end_color=YELLOW_COLOR, fill_type='solid')


# =================================================================================
# 1. BUSINESS INSIGHTS GENERATOR (SRP: Convert technical data to business insights)
# =================================================================================

class BusinessInsightsGenerator:
    """
    Converts technical performance data into actionable business insights.
    Maintains legacy justification format while adding modern insight generation.
    """

    @staticmethod
    def generate_key_insights(report: PerformanceReport) -> List[str]:
        """Generate modern business insights from performance data."""
        insights = []

        # Overall performance assessment
        if report.build_status == PerformanceStatus.PASSED:
            insights.append("✅ Overall system performance meets defined thresholds")
        else:
            insights.append("🔴 System performance degradation detected - immediate attention required")

        # Throughput analysis
        if report.summary.throughput < 10.0:
            slow_transactions = [name for name, metrics in report.transactions.items()
                                 if name != "Total" and metrics.pct_95 > 2000]  # >2s
            if slow_transactions:
                transaction_list = ", ".join(slow_transactions[:3])
                more_count = len(slow_transactions) - 3
                if more_count > 0:
                    transaction_list += f" and {more_count} more"
                insights.append(f"🔻 Throughput below 10.0 req/s affected by slow transactions: {transaction_list}")

        # Error rate analysis
        if report.summary.error_rate > 0:
            insights.append(
                f"⚠️ System error rate at {report.summary.error_rate:.2f}% - investigate failed transactions")
        else:
            insights.append("✅ Zero error rate - all transactions completing successfully")

        # Response time analysis
        critical_transactions = [name for name, metrics in report.transactions.items()
                                 if name != "Total" and metrics.pct_95 > 3000]  # >3s
        if critical_transactions:
            insights.append(
                f"🔴 Response times > 3 seconds by 95 pct detected in {len(critical_transactions)} transaction(s): {critical_transactions}")

        return insights

    @staticmethod
    def parse_legacy_justification(justification: str) -> str:
        """Parse legacy justification format and clean up incomplete sentences."""
        if not justification:
            return "All performance metrics within acceptable thresholds"

        # Clean up incomplete sentences and formatting
        cleaned = justification.replace(" and Response Time", ". Response Time")
        cleaned = cleaned.replace(" by 95", " (95th percentile)")

        # Fix incomplete transaction lists
        if "transaction(s):" in cleaned and cleaned.count(",") > 2:
            parts = cleaned.split("transaction(s):")
            if len(parts) > 1:
                transaction_part = parts[1].strip()
                transactions = [t.strip() for t in transaction_part.split(",")]
                if len(transactions) > 3:
                    cleaned = parts[
                                  0] + f"transaction(s): {', '.join(transactions[:3])} and {len(transactions) - 3} others"

        return cleaned


# =================================================================================
# 2. LEGACY EXCEL FORMATTER (SRP: Excel-specific formatting with legacy compatibility)
# =================================================================================

class LegacyExcelFormatter:
    """
    Handles all Excel formatting to maintain exact legacy appearance.
    Applies conditional formatting at spreadsheet level, not row level.
    """

    def __init__(self):
        self.header_mapping = {
            'Transaction': 'request_name',
            'Req, count': 'Total',
            'KO, count': 'KO',
            'KO, %': 'Error_pct',
            'Min, sec': 'min',
            'Avg, sec': 'average',
            '90p, sec': 'pct_90',
            '95p, sec': 'pct_95',
            'Max, sec': 'max',
            '📝 Notes': 'notes'  # New column
        }

        self.title_mapping = {
            'Users': 'max_user_count',
            'Ramp Up, min': 'ramp_up_period',
            'Duration, min': 'duration',
            'Think time, sec': 'think_time',
            'Start Date, EST': 'date_start',
            'End Date, EST': 'date_end',
            'Throughput, req/sec': 'throughput',
            'Error rate, %': 'error_rate',
            'Carrier report': 'carrier_report',
            'Build status': 'build_status',
            'Justification': 'justification',
            'Key Insights': 'key_insights',
            'Overall system performance': 'overall_performance'
        }

    def apply_title_section_formatting(self, ws: Worksheet, report: PerformanceReport, insights: List[str]):
        """Apply legacy title section formatting with new insights."""
        # Prepare data with new fields
        report_data = {
            'max_user_count': getattr(report.summary, 'max_user_count', 'N/A'),
            'ramp_up_period': getattr(report.summary, 'ramp_up_period', 'N/A'),
            'duration': getattr(report.summary, 'duration', 'N/A'),
            'think_time': getattr(report.summary, 'think_time', 'N/A'),
            'date_start': report.summary.date_start.strftime(
                '%Y-%m-%d %H:%M:%S') if report.summary.date_start else 'N/A',
            'date_end': report.summary.date_end.strftime('%Y-%m-%d %H:%M:%S') if report.summary.date_end else 'N/A',
            'throughput': report.summary.throughput,
            'error_rate': report.summary.error_rate / 100.0,  # Convert to decimal for percentage
            'carrier_report': getattr(report.summary, 'carrier_report', 'N/A'),
            'build_status': report.build_status.value,
            'justification': BusinessInsightsGenerator.parse_legacy_justification(report.analysis_summary),
            'key_insights': '\n'.join(insights),
            'overall_performance': 'HEALTHY' if report.build_status == PerformanceStatus.PASSED else 'DEGRADED'
        }

        # Apply legacy title formatting
        for i, (title_name, data_key) in enumerate(self.title_mapping.items()):
            row_num = i + 1

            # Title name cell
            title_cell = ws.cell(row=row_num, column=1, value=title_name)
            title_cell.font = Font(bold=True, color='00291A75')
            title_cell.fill = PatternFill("solid", fgColor='00CDEBEA')
            title_cell.alignment = Alignment(horizontal="left", vertical="center")

            # Title value cell
            value_cell = ws.cell(row=row_num, column=2, value=report_data[data_key])
            value_cell.fill = PatternFill("solid", fgColor='00CDEBEA')
            value_cell.alignment = Alignment(horizontal="center", vertical="center")

            # Apply borders
            border_style = Side(border_style="thin", color="040404")
            border = Border(top=border_style, left=border_style, right=border_style, bottom=border_style)
            title_cell.border = border
            value_cell.border = border

            # Special formatting for specific fields
            if data_key == 'error_rate':
                value_cell.number_format = '0.00%'
            elif data_key == 'carrier_report' and report_data[data_key] != 'N/A':
                value_cell.hyperlink = report_data[data_key]
                value_cell.value = "Carrier report"
                value_cell.font = Font(bold=True, underline="single", color='00291A75')
            elif data_key == 'build_status':
                color = GREEN_COLOR_FONT if report_data[data_key] == 'PASSED' else RED_COLOR_FONT
                value_cell.font = Font(bold=True, color=color)
            elif data_key in ['justification', 'key_insights'] and len(str(report_data[data_key])) > 125:
                value_cell.alignment = Alignment(horizontal="center", vertical="justify", wrap_text=True)
                ws.row_dimensions[row_num].height = 50

            # Merge cells across header columns
            ws.merge_cells(start_row=row_num, start_column=2, end_row=row_num, end_column=len(self.header_mapping))

        return len(self.title_mapping) + 1  # Return next available row

    def apply_transaction_table_formatting(self, ws: Worksheet, report: PerformanceReport, start_row: int,
                                           thresholds: Dict[str, float]):
        """Apply legacy transaction table formatting with conditional formatting at spreadsheet level."""
        header_row = start_row

        # Create headers
        for col_idx, header_text in enumerate(self.header_mapping.keys(), 1):
            cell = ws.cell(row=header_row, column=col_idx, value=header_text)
            cell.alignment = Alignment(horizontal="center")
            cell.fill = PatternFill("solid", fgColor='007FD5D8')

            border_style = Side(border_style="thin", color="040404")
            cell.border = Border(top=border_style, left=border_style, right=border_style, bottom=border_style)

        # Write transaction data
        current_row = header_row + 1
        content_start = f"A{header_row}"
        content_end = None

        # Sort transactions (Total last)
        sorted_transactions = sorted([name for name in report.transactions.keys() if name != "Total"])
        if "Total" in report.transactions:
            sorted_transactions.append("Total")

        for tx_name in sorted_transactions:
            metrics = report.transactions[tx_name]

            for col_idx, (header_name, data_key) in enumerate(self.header_mapping.items(), 1):
                if data_key == 'notes':
                    # New notes column - empty for now, can be filled by analysts
                    cell_value = ""
                else:
                    cell_value = getattr(metrics, data_key, 'N/A')

                cell = ws.cell(row=current_row, column=col_idx, value=cell_value)

                # Apply borders
                border_style = Side(border_style="thin", color="040404")
                cell.border = Border(top=border_style, left=border_style, right=border_style, bottom=border_style)

                # Format time values (convert ms to seconds)
                if data_key in ['min', 'average', 'pct_90', 'pct_95', 'max'] and isinstance(cell_value, (int, float)):
                    cell.value = round(cell_value / 1000, 3)

                # Format percentage values
                if data_key == 'Error_pct' and isinstance(cell_value, (int, float)):
                    cell.number_format = '0.00%'
                    cell.value = cell_value / 100.0

                # Special formatting for Total row
                if tx_name == "Total":
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill("solid", fgColor='007FD5D8')

                content_end = f"{get_column_letter(col_idx)}{current_row}"

            current_row += 1

        # Apply conditional formatting at spreadsheet level (legacy approach)
        self._apply_conditional_formatting(ws, header_row + 1, current_row - 1, thresholds)

        # Apply legacy table features
        self._apply_legacy_table_features(ws, content_start, content_end, header_row)

        return current_row

    def _apply_conditional_formatting(self, ws: Worksheet, start_row: int, end_row: int, thresholds: Dict[str, float]):
        """Apply conditional formatting rules at spreadsheet level (legacy approach)."""
        rt_threshold = thresholds.get('response_time', 500.0)  # Default 500ms

        # Convert to seconds for display
        if rt_threshold > 1:
            threshold_seconds = rt_threshold / 1000
        else:
            threshold_seconds = rt_threshold

        # Apply conditional formatting to time columns (Min, Avg, 90p, 95p, Max)
        time_columns = [5, 6, 7, 8, 9]  # Columns E through I

        for col_idx in time_columns:
            col_letter = get_column_letter(col_idx)
            range_coords = f"{col_letter}{start_row}:{col_letter}{end_row}"

            # Green: Below threshold
            ws.conditional_formatting.add(range_coords,
                                          CellIsRule(operator='lessThan',
                                                     formula=[threshold_seconds],
                                                     fill=GREEN_FILL))

            # Red: Above 1.5x threshold
            ws.conditional_formatting.add(range_coords,
                                          CellIsRule(operator='greaterThan',
                                                     formula=[threshold_seconds * 1.5],
                                                     fill=RED_FILL))

            # Yellow: Between threshold and 1.5x threshold
            ws.conditional_formatting.add(range_coords,
                                          CellIsRule(operator='between',
                                                     formula=[threshold_seconds, threshold_seconds * 1.5],
                                                     fill=YELLOW_FILL))

    def _apply_legacy_table_features(self, ws: Worksheet, content_start: str, content_end: str, header_row: int):
        """Apply legacy table features: auto-filter, freeze panes, column widths."""
        # Auto-filter
        ws.auto_filter.ref = f"{content_start}:{content_end}"

        # Freeze panes
        freeze_cell = ws[f'B{header_row + 1}']
        ws.freeze_panes = freeze_cell

        # Column width adjustments (legacy approach)
        for i, header in enumerate(self.header_mapping.keys(), 1):
            if header == "Transaction":
                # Calculate width based on longest transaction name
                max_length = max(len(str(cell.value)) for cell in ws[get_column_letter(i)])
                ws.column_dimensions[get_column_letter(i)].width = max_length + 5
            else:
                ws.column_dimensions[get_column_letter(i)].width = len(header) + 5


# =================================================================================
# 3. PRODUCTION EXCEL REPORTER (Orchestrator with legacy compatibility)
# =================================================================================

class ExcelReporter:
    """
    Production-ready Excel reporter maintaining full legacy compatibility.
    Orchestrates report generation using DRY principles while preserving
    exact legacy formatting for business continuity.
    """

    def __init__(self):
        self.formatter = LegacyExcelFormatter()
        self.insights_generator = BusinessInsightsGenerator()
        logger.info("Production ExcelReporter initialized with legacy formatting support.")

    def generate_workbook(self, report: PerformanceReport) -> Workbook:
        """Generate production-ready Excel workbook with legacy formatting."""
        logger.info(f"Generating production Excel report for build status: {report.build_status.value}")

        wb = Workbook()
        ws = wb.active
        ws.title = "Test results"  # Legacy sheet name

        # Generate insights
        insights = self.insights_generator.generate_key_insights(report)

        # Extract threshold values for conditional formatting
        thresholds = self._extract_thresholds(report.thresholds)

        # Apply formatting sections
        next_row = self.formatter.apply_title_section_formatting(ws, report, insights)
        self.formatter.apply_transaction_table_formatting(ws, report, next_row, thresholds)

        logger.info("Production Excel report generated successfully.")
        return wb

    def update_report_with_new_sheet(self, workbook_path: str, new_report: PerformanceReport) -> bool:
        """Update existing workbook with new sheet (legacy compatibility)."""
        try:
            wb = load_workbook(workbook_path)
            logger.info(f"Loaded existing workbook from '{workbook_path}'.")
        except FileNotFoundError:
            logger.warning(f"Workbook not found at '{workbook_path}'. Creating new workbook.")
            wb = Workbook()
            if wb.active.title == "Sheet":
                wb.remove(wb.active)

        # Create new sheet with timestamp
        sheet_name = new_report.summary.date_start.strftime('%Y-%m-%d_%H%M')
        if sheet_name in wb.sheetnames:
            logger.warning(f"Sheet '{sheet_name}' already exists. Replacing it.")
            wb.remove(wb[sheet_name])

        ws_new = wb.create_sheet(sheet_name)

        # Generate insights and format new sheet
        insights = self.insights_generator.generate_key_insights(new_report)
        thresholds = self._extract_thresholds(new_report.thresholds)

        next_row = self.formatter.apply_title_section_formatting(ws_new, new_report, insights)
        self.formatter.apply_transaction_table_formatting(ws_new, new_report, next_row, thresholds)

        # Update or create tests overview sheet
        self._update_tests_overview_sheet(wb, sheet_name, new_report)

        wb.save(workbook_path)
        logger.info(f"Successfully updated workbook saved to '{workbook_path}'.")
        return True

    def _extract_thresholds(self, threshold_configs: List[ThresholdConfig]) -> Dict[str, float]:
        """Extract threshold values from ThresholdConfig objects."""
        thresholds = {}
        if threshold_configs:
            for config in threshold_configs:
                thresholds[config.target] = config.threshold_value
        return thresholds

    def _update_tests_overview_sheet(self, wb: Workbook, new_sheet_name: str, report: PerformanceReport):
        """Update tests overview sheet with hyperlinked entry."""
        sheet_title = "Tests Overview"  # Legacy sheet name

        if sheet_title not in wb.sheetnames:
            ws = wb.create_sheet(sheet_title, 0)
            headers = ["Test Date", "Report Sheet", "Build Status", "Throughput (req/s)",
                       "P95 Response Time (ms)", "Error Rate", "Key Issues"]
            ws.append(headers)
        else:
            ws = wb[sheet_title]

        # Create hyperlink formula
        hyperlink_formula = f'=HYPERLINK("#{new_sheet_name}!A1", "{new_sheet_name}")'

        # Extract key issues for overview
        key_issues = "None"
        if report.build_status == PerformanceStatus.FAILED:
            if report.summary.error_rate > 0:
                key_issues = f"Error Rate: {report.summary.error_rate:.2f}%"
            if report.summary.throughput < 10.0:
                key_issues += f", Low Throughput: {report.summary.throughput:.1f} req/s"

        new_row = [
            report.summary.date_start.strftime('%Y-%m-%d %H:%M'),
            hyperlink_formula,
            report.build_status.value,
            report.summary.throughput,
            report.transactions.get("Total", TransactionMetrics(request_name="Total")).pct_95,
            report.summary.error_rate / 100.0,
            key_issues
        ]

        ws.append(new_row)

        # Format the new row
        row_num = ws.max_row
        ws.cell(row=row_num, column=6).number_format = '0.00%'  # Error rate
        ws.cell(row=row_num, column=2).font = Font(bold=True, underline="single", color='00291A75')  # Hyperlink

        # Status color coding
        status_cell = ws.cell(row=row_num, column=3)
        if report.build_status == PerformanceStatus.PASSED:
            status_cell.font = Font(bold=True, color=GREEN_COLOR_FONT)
        else:
            status_cell.font = Font(bold=True, color=RED_COLOR_FONT)


class TransactionAnalyzer:
    """
    SOLID: Single Responsibility - Analyze transaction performance.
    DRY: Eliminates duplicate analysis methods.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._severity_weights = {
            'Critical': 3,
            'Warning': 2,
            'Good': 1
        }

    def analyze_performance(self, metrics: TransactionMetrics,
                            thresholds: ThresholdConfig) -> ThresholdConfig:
        """
        Comprehensive transaction performance analysis.

        Args:
            metrics: Transaction metrics to analyze
            thresholds: Performance thresholds for comparison

        Returns:
            TransactionAnalysis with status, severity, and detailed notes
        """
        self.logger.debug(f"Analyzing transaction performance: error_pct={metrics.error_pct}, p95_ms={metrics.p95_ms}")

        issues = []
        severity = 'Good'

        try:
            # Error Rate Analysis
            error_analysis = self._analyze_error_rate(metrics, thresholds)
            if error_analysis['severity'] != 'Good':
                issues.extend(error_analysis['issues'])
                severity = self._get_higher_severity(severity, error_analysis['severity'])

            # Response Time Analysis
            rt_analysis = self._analyze_response_time(metrics, thresholds)
            if rt_analysis['severity'] != 'Good':
                issues.extend(rt_analysis['issues'])
                severity = self._get_higher_severity(severity, rt_analysis['severity'])

            # Generate final status
            status = self._determine_status(severity, issues)
            notes = self._format_notes(issues)

            self.logger.debug(f"Analysis complete: status={status}, severity={severity}, issues_count={len(issues)}")

            return TransactionAnalysis(
                status=status,
                severity=severity,
                notes=notes,
                issues=issues
            )

        except Exception as e:
            self.logger.error(f"Error in transaction analysis: {e}")
            return TransactionAnalysis(
                status="❓ Error",
                severity="Unknown",
                notes=f"Analysis failed: {str(e)}",
                issues=[]
            )

    def _analyze_error_rate(self, metrics: TransactionMetrics,
                            thresholds: ThresholdConfig) -> Dict[str, Any]:
        """Analyze error rate performance."""
        if metrics.error_pct > thresholds.error_rate:
            return {
                'severity': 'Critical',
                'issues': [f"High error rate ({metrics.error_pct:.1f}%)"]
            }
        return {'severity': 'Good', 'issues': []}

    def _analyze_response_time(self, metrics: TransactionMetrics,
                               thresholds: ThresholdConfig) -> Dict[str, Any]:
        """Analyze response time performance."""
        rt_issues = []
        severity = 'Good'

        # P95 Analysis
        if metrics.p95_ms > thresholds.response_time * 3:
            rt_issues.append("P95 critical")
            severity = 'Critical'
        elif metrics.p95_ms > thresholds.response_time * 2:
            rt_issues.append("P95 high")
            severity = 'Warning'
        elif metrics.p95_ms > thresholds.response_time:
            rt_issues.append("P95 elevated")
            severity = 'Warning'

        # Average Analysis
        if metrics.avg_ms > thresholds.response_time:
            rt_issues.append("Avg elevated")
            severity = 'Warning' if severity == 'Good' else severity

        formatted_issues = [f"RT: {', '.join(rt_issues)}"] if rt_issues else []

        return {
            'severity': severity,
            'issues': formatted_issues
        }

    def _get_higher_severity(self, current: str, new: str) -> str:
        """Determine higher severity between two levels."""
        current_weight = self._severity_weights.get(current, 0)
        new_weight = self._severity_weights.get(new, 0)
        return new if new_weight > current_weight else current

    def _determine_status(self, severity: str, issues: List[str]) -> str:
        """Determine status based on severity and issues."""
        if not issues:
            return "✅ Pass"
        return "❌ Fail" if severity == 'Critical' else "⚠️ Warn"

    def _format_notes(self, issues: List[str]) -> str:
        """Format issues into readable notes."""
        return " | ".join(issues) if issues else "Within thresholds"

    # ADD TO EXISTING TransactionAnalyzer class in excel_reporter.py

    def generate_evidence_based_justification(self, transactions: Dict[str, TransactionMetrics],
                                              thresholds: ThresholdConfig) -> str:
        """Generate data-driven justification with specific evidence."""
        total_transactions = len([t for t in transactions.keys() if t != "Total"])

        # Response Time Analysis
        rt_passed = sum(1 for name, metrics in transactions.items()
                        if name != "Total" and metrics.p95_ms <= thresholds.response_time)
        rt_evidence = f"{rt_passed}/{total_transactions} passed Response Time SLA of {thresholds.response_time}ms"

        # Error Rate Analysis
        er_passed = sum(1 for name, metrics in transactions.items()
                        if name != "Total" and metrics.error_pct <= thresholds.error_rate)
        er_evidence = f"{er_passed}/{total_transactions} passed Error Rate SLA of {thresholds.error_rate}%"

        # Overall Assessment
        overall_passed = sum(1 for name, metrics in transactions.items()
                             if name != "Total" and
                             metrics.p95_ms <= thresholds.response_time and
                             metrics.error_pct <= thresholds.error_rate)

        if overall_passed == total_transactions:
            return f"✅ All systems operational: {rt_evidence}, {er_evidence}"
        else:
            failing_areas = []
            if rt_passed < total_transactions:
                failing_areas.append(f"Response Time: {rt_evidence}")
            if er_passed < total_transactions:
                failing_areas.append(f"Error Rate: {er_evidence}")

            return f"🔴 Performance degradation detected: {', '.join(failing_areas)}. Overall: {overall_passed}/{total_transactions} transactions passed all SLAs"

    def generate_transaction_validation_notes(self, metrics: TransactionMetrics,
                                              thresholds: ThresholdConfig) -> str:
        """Generate specific validation notes for each transaction."""
        notes = []

        # Response Time Validation
        if metrics.p95_ms > thresholds.response_time:
            overage = ((metrics.p95_ms - thresholds.response_time) / thresholds.response_time) * 100
            notes.append(f"P95 {overage:.1f}% over SLA")

        # Error Rate Validation
        if metrics.error_pct > thresholds.error_rate:
            notes.append(f"Error rate {metrics.error_pct:.1f}% exceeds {thresholds.error_rate}% SLA")

        # Performance Classification
        if metrics.p95_ms <= thresholds.response_time * 0.8:
            notes.append("Excellent performance")
        elif metrics.p95_ms <= thresholds.response_time:
            notes.append("Within SLA")

        return " | ".join(notes) if notes else "All validations passed"


class TextFormatter:
    """
    SOLID: Single Responsibility - Format text consistently.
    DRY: Eliminates duplicate formatting methods.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def format_transaction_name(self, name: str) -> str:
        """
        Format transaction names for better readability.

        Args:
            name: Raw transaction name

        Returns:
            Formatted, readable transaction name
        """
        try:
            # Remove common prefixes
            cleaned = name.replace("POST_", "").replace("GET_", "").replace("PUT_", "").replace("DELETE_", "")

            # Add spaces before capitals for camelCase
            import re
            formatted = re.sub(r'(?<!^)(?=[A-Z])', ' ', cleaned)

            self.logger.debug(f"Formatted transaction name: '{name}' -> '{formatted}'")
            return formatted

        except Exception as e:
            self.logger.warning(f"Failed to format transaction name '{name}': {e}")
            return name

    def parse_justification_to_insights(self, justification: str) -> List[str]:
        """
        Convert technical justification to business insights.

        Args:
            justification: Technical justification string

        Returns:
            List of formatted business insights
        """
        if not justification:
            return ["No specific insights available."]

        try:
            insights = []
            parts = justification.split(" | ")

            for part in parts:
                part = part.strip()

                # Transform technical messages to business insights
                if "success rate" in part.lower():
                    insights.append(part)
                elif "critical rt issues" in part.lower():
                    insights.append(part.replace("RT", "Response Time"))
                elif "throughput" in part.lower() and "below" in part.lower():
                    insights.append(f"🔻 {part}")
                elif "error" in part.lower() and "above" in part.lower():
                    insights.append(f"🚨 {part}")
                elif "focus:" in part.lower():
                    insights.append(f"🎯 Recommended Actions: {part.split(':', 1)[1].strip()}")
                elif "excellent" in part.lower() or "all" in part.lower() and "passed" in part.lower():
                    insights.append(f"🎉 {part}")
                elif len(part) > 10 and not any(skip in part.lower() for skip in ["focus", "recommendations"]):
                    insights.append(part)

            result = insights[:5]  # Limit to top 5 insights for readability
            self.logger.debug(f"Parsed {len(parts)} justification parts into {len(result)} insights")
            return result

        except Exception as e:
            self.logger.error(f"Failed to parse justification: {e}")
            return [f"Error parsing insights: {str(e)}"]

    def generate_business_insights(self, report: Any, thresholds: ThresholdConfig) -> List[str]:
        """Generate actionable business insights from performance data."""
        insights = []

        # Overall Performance Assessment
        total_transactions = len([t for t in report.transactions.keys() if t != "Total"])
        passed_transactions = sum(1 for name, metrics in report.transactions.items()
                                  if name != "Total" and
                                  metrics.p95_ms <= thresholds.response_time and
                                  metrics.error_pct <= thresholds.error_rate)

        pass_rate = (passed_transactions / total_transactions) * 100 if total_transactions > 0 else 0

        if pass_rate >= 90:
            insights.append(
                f"✅ Excellent system health: {passed_transactions}/{total_transactions} ({pass_rate:.1f}%) transactions meeting all SLAs")
        elif pass_rate >= 70:
            insights.append(
                f"⚠️ System performance acceptable: {passed_transactions}/{total_transactions} ({pass_rate:.1f}%) transactions meeting SLAs")
        else:
            insights.append(
                f"🔴 Critical performance issues: Only {passed_transactions}/{total_transactions} ({pass_rate:.1f}%) transactions meeting SLAs")

        # Throughput Analysis
        if hasattr(report.summary, 'throughput'):
            if report.summary.throughput < thresholds.throughput:
                insights.append(
                    f"🔻 Throughput below target: {report.summary.throughput:.1f} req/s vs {thresholds.throughput:.1f} req/s target")
            else:
                insights.append(
                    f"🚀 Throughput healthy: {report.summary.throughput:.1f} req/s exceeds {thresholds.throughput:.1f} req/s target")

        # Error Rate Analysis
        if hasattr(report.summary, 'error_rate'):
            if report.summary.error_rate > thresholds.error_rate:
                insights.append(
                    f"🚨 Error rate elevated: {report.summary.error_rate:.1f}% exceeds {thresholds.error_rate}% threshold")
            else:
                insights.append(
                    f"✅ Error rate healthy: {report.summary.error_rate:.1f}% within {thresholds.error_rate}% threshold")

        # Critical Transactions Analysis
        critical_transactions = [name for name, metrics in report.transactions.items()
                                 if name != "Total" and metrics.p95_ms > thresholds.response_time * 2]

        if critical_transactions:
            insights.append(
                f"🔥 Critical attention required: {len(critical_transactions)} transaction(s) with P95 > {thresholds.response_time * 2}ms")

        return insights[:5]
