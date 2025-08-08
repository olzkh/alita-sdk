import logging
from typing import Dict, Any, List, Optional

from alita_sdk.tools.carrier.etl.reporting.backend_excel_reporter import BusinessInsightsGenerator
from alita_sdk.tools.carrier.etl.reporting.core.data_models import PerformanceReport, PerformanceAnalysisResult
from alita_sdk.tools.carrier.utils.utils import ExcelFormattingConfig


class MarkdownReportBuilder:
    """
    Generates a comprehensive and readable markdown report by formatting
    pre-existing AI analysis and business insights.

    This class follows the Single Responsibility Principle: it only formats data,
    it does not perform any analysis itself.
    """

    def __init__(self, config: ExcelFormattingConfig = None):
        self.config = config or ExcelFormattingConfig()
        self.insights_generator = BusinessInsightsGenerator(self.config)
        self.logger = logging.getLogger(__name__)

    def generate_markdown_content(self, transformed_data: Dict[str, Any]) -> str:
        """
        Orchestrates the creation of the markdown report from transformed data.
        """
        self.logger.info("Generating comprehensive markdown report...")

        all_reports = transformed_data.get("all_reports", [])
        # Use the AI analysis object directly from the transformer
        ai_analysis: Optional[PerformanceAnalysisResult] = transformed_data.get("comparison_analysis")

        content = []

        # Build each section of the report
        content.extend(self._generate_header())

        if ai_analysis:
            content.extend(self._generate_ai_summary_section(ai_analysis))
            content.extend(self._generate_ai_trends_section(ai_analysis))
            content.extend(self._generate_ai_recommendations_section(ai_analysis))
        else:
            content.append("## AI Analysis\n\n- AI-powered analysis was not available for this comparison.\n")

        # Add rule-based insights for the most recent test for detailed context
        if all_reports:
            content.extend(self._generate_business_insights_section(all_reports[-1]))

        content.extend(self._generate_chart_interpretation_section())

        self.logger.info("Markdown report content generated successfully.")
        return "\n".join(content)

    def _generate_header(self) -> List[str]:
        """Generates the main title for the markdown report."""
        return ["# Performance Comparison Analysis", "---"]

    def _generate_ai_summary_section(self, ai_analysis: PerformanceAnalysisResult) -> List[str]:
        """Formats the Executive Summary and Risk Assessment from the AI response."""
        content = ["## 📊 Executive Summary (AI-Powered)", ""]
        content.append(ai_analysis.summary)
        content.append("")

        risk = ai_analysis.risk_assessment
        if risk and risk.get("overall_risk"):
            risk_level = risk["overall_risk"].upper()
            icon = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            content.append(f"**AI Assessed Risk Level:** {icon} {risk_level}")
            if risk.get("risk_factors"):
                factors = ", ".join(risk["risk_factors"])
                content.append(f"**Primary Risk Factors:** {factors}")
        content.append("")
        return content

    def _generate_ai_trends_section(self, ai_analysis: PerformanceAnalysisResult) -> List[str]:
        """Formats the Performance Trends section from the AI response."""
        content = ["## 📈 AI-Identified Performance Trends", ""]
        trends = ai_analysis.performance_trends
        if not trends:
            return content

        trend_map = {
            "response_time_trend": "Response Time",
            "throughput_trend": "Throughput",
            "error_rate_trend": "Error Rate"
        }

        for key, label in trend_map.items():
            trend_value = trends.get(key, "stable")
            icon = "📉" if trend_value == "degrading" else "📈" if trend_value == "improving" else "➡️"
            content.append(f"- **{label}:** {icon} {trend_value.capitalize()}")

        content.append("")
        content.append(f"**Overall Trend Narrative:** {trends.get('overall_trend', 'Not specified.')}")
        content.append("")
        return content

    def _generate_ai_recommendations_section(self, ai_analysis: PerformanceAnalysisResult) -> List[str]:
        """Formats the Recommendations and Key Findings from the AI response."""
        content = ["## 🎯 AI Recommendations & Key Findings", ""]

        content.append("### Key Findings")
        if ai_analysis.key_findings:
            for finding in ai_analysis.key_findings:
                content.append(f"1. {finding}")
        else:
            content.append("- No specific findings were highlighted by the AI.")

        content.append("\n### Actionable Recommendations")
        if ai_analysis.recommendations:
            for rec in ai_analysis.recommendations:
                content.append(f"1. {rec}")
        else:
            content.append("- No specific recommendations were generated by the AI.")

        content.append("")
        return content

    def _generate_business_insights_section(self, latest_report: PerformanceReport) -> List[str]:
        """
        Generates detailed insights for the latest report by reusing BusinessInsightsGenerator.
        """
        content = ["## 📝 Detailed Insights for Latest Test", ""]

        # REUSE the existing generator. No duplicated logic.
        insights = self.insights_generator.generate_key_insights(latest_report)

        if insights:
            for insight in insights:
                content.append(f"- {insight}")
        else:
            content.append("- No specific rule-based insights were triggered for the latest test.")

        content.append("")
        return content

    def _generate_chart_interpretation_section(self) -> List[str]:
        """Provides a static guide on how to interpret the visual artifacts."""
        return [
            "---",
            "## ❓ How to Interpret Visuals",
            "- **Trend Charts:** Show performance metrics over multiple runs to identify patterns.",
            "- **Comparison Charts:** Provide a side-by-side view of the two most recent tests.",
            "- **Excel Report:** Contains raw data, dynamic formulas, and detailed transaction breakdowns for deep-dive analysis."
        ]