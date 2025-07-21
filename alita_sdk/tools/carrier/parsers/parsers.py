import io
import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict
import numpy as np

from ..utils.utils import GATLING_CONFIG
from ..reporting.core.data_models import (
    TransactionMetrics,
    ReportSummary,
    PerformanceReport,
    create_transaction_metrics_from_stats,
    create_empty_transaction_metrics,
    validate_performance_report
)

logger = logging.getLogger(__name__)


class BaseReportParser(ABC):
    """
    An abstract base class that defines the UNIFIED contract for all report parsers.
    It ensures all parsers have a single entry point `.parse()` that returns a
    validated PerformanceReport object, making the ETL pipeline type-safe.
    """

    def __init__(self, content_stream: io.StringIO, **kwargs):
        """Shared constructor verifies the input is a valid in-memory text stream."""
        if not isinstance(content_stream, io.StringIO):
            raise TypeError("Parser input must be an in-memory io.StringIO object.")

        self.stream = content_stream
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Parser initialized for in-memory stream processing.")

    @abstractmethod
    def parse(self) -> PerformanceReport:
        """
        --- NEW CONTRACT ---
        Parses the stream and returns a single, complete, and validated
        PerformanceReport object. This is the sole entry point for the transformer.
        """
        raise NotImplementedError

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """DRY: Centralized safe division to prevent ZeroDivisionError."""
        if denominator == 0:
            self.logger.warning(f"Division by zero prevented: {numerator}/{denominator}")
            return default
        return numerator / denominator

    def _calculate_error_percentage(self, errors: int, total: int) -> float:
        """DRY: Standardized error percentage calculation."""
        return round(self._safe_divide(errors, total) * 100, 4)


class GatlingReportParser(BaseReportParser):
    """
    Merges the robust parsing logic from the legacy tool with the modern,
    Pandas-based architecture.
    """

    def __init__(self, content_stream: io.StringIO, **kwargs):
        super().__init__(content_stream)
        self.calculated_think_time = kwargs.get("think_times", GATLING_CONFIG.DEFAULT_THINK_TIME)
        self.logger.info(f"Enhanced Gatling parser initialized with think_time: {self.calculated_think_time}")

    def _get_metric_category(self, metric_name: str) -> str:
        """
        Categorizes metrics for conditional formatting in Excel reports.
        """
        if metric_name.startswith("Group_"):
            return "GROUP"
        elif metric_name == "Total":
            return "TOTAL"
        else:
            return "REQUEST"

    def parse(self) -> PerformanceReport:
        """
        Enhanced to support both requests and groups using DRY principles.
        """
        self.logger.info("Starting enhanced Gatling parsing to create PerformanceReport object.")

        try:
            # Phase 1: Extract raw data including groups
            raw_requests, raw_groups, users, date_start, date_end, ramp_up_period = self._parse_log_file()

            # Phase 2: Create TransactionMetrics using DRY principle
            transaction_metrics = {}

            # Process requests
            for name, entries in raw_requests.items():
                transaction_metrics[name] = self._calculate_single_metric(name, entries)

            # Process groups using same calculation method (DRY)
            for name, entries in raw_groups.items():
                transaction_metrics[name] = self._calculate_single_metric(name, entries)

            # Add total metric
            all_entries = [entry for entries in raw_requests.values() for entry in entries]
            transaction_metrics["Total"] = self._calculate_single_metric("Total", all_entries)

            # Phase 3: Create summary and report
            duration = date_end - date_start if date_start and date_end else timedelta(seconds=0)
            summary = self._create_summary(users, date_start, date_end, ramp_up_period, duration, transaction_metrics)

            report = PerformanceReport(
                summary=summary,
                transactions=transaction_metrics,
                report_type=GATLING_CONFIG.REPORT_TYPE
            )
            validation_errors = validate_performance_report(report)
            if validation_errors:
                self.logger.warning(f"Post-parsing validation found issues: {validation_errors}")

            self.logger.info("Successfully created and validated PerformanceReport with groups support.")
            return report

        except Exception as e:
            self.logger.error(f"Critical error during Gatling parsing: {e}", exc_info=True)
            raise

    def _parse_log_file(self):
        """
        Uses the proven line-by-line parsing logic with DRY configuration constants.
        """
        requests = defaultdict(list)
        groups = defaultdict(list)
        users, ramp_start, ramp_end = 0, None, None
        timestamps = []

        self.stream.seek(0)
        for line in self.stream:
            parts = line.strip().split('\t')
            if not parts or len(parts) < GATLING_CONFIG.MIN_LOG_PARTS:
                continue

            try:
                current_ts = int(parts[3])
                timestamps.append(current_ts)

                if line.startswith(GATLING_CONFIG.LOG_LINE_PREFIX_REQUEST) and len(
                        parts) >= GATLING_CONFIG.REQUEST_MIN_PARTS:
                    requests[parts[2]].append({
                        "response_time": int(parts[4]) - int(parts[3]),
                        "status": parts[5]
                    })
                elif line.startswith(GATLING_CONFIG.LOG_LINE_PREFIX_GROUP) and len(
                        parts) >= GATLING_CONFIG.GROUP_MIN_PARTS:
                    groups[parts[1]].append({
                        "response_time": int(parts[4]),
                        "status": parts[5]
                    })
                elif line.startswith(
                        GATLING_CONFIG.LOG_LINE_PREFIX_USER) and GATLING_CONFIG.LOG_LINE_STATUS_START in parts:
                    users += 1
                    if ramp_start is None:
                        ramp_start = current_ts
                    ramp_end = current_ts

            except (ValueError, IndexError):
                continue

        date_start = datetime.fromtimestamp(min(timestamps) / GATLING_CONFIG.TIMESTAMP_DIVISOR) if timestamps else None
        date_end = datetime.fromtimestamp(max(timestamps) / GATLING_CONFIG.TIMESTAMP_DIVISOR) if timestamps else None
        ramp_up_period = timedelta(milliseconds=(ramp_end - ramp_start)) if ramp_start and ramp_end else timedelta(
            seconds=0)

        return requests, groups, users, date_start, date_end, ramp_up_period

    def _calculate_single_metric(self, name: str, entries: list) -> TransactionMetrics:
        """
        Uses numpy and configuration constants for consistent calculations.
        """
        if not entries:
            return create_empty_transaction_metrics(name)

        response_times = [d["response_time"] for d in entries]
        ok_count = len([d for d in entries if d["status"] == GATLING_CONFIG.LOG_LINE_STATUS_OK])
        total_count = len(entries)
        ko_count = total_count - ok_count

        stats = {
            'Total': total_count,
            'OK': ok_count,
            'KO': ko_count,
            'min': round(min(response_times), 3) if response_times else 0,
            'max': round(max(response_times), 3) if response_times else 0,
            'average': round(np.mean(response_times), 3) if response_times else 0,
            'median': round(np.median(response_times), 3) if response_times else 0,
            'Error%': self._calculate_error_percentage(ko_count, total_count),
            '90Pct': round(np.percentile(response_times, 90), 3) if response_times else 0,
            '95Pct': round(np.percentile(response_times, 95), 3) if response_times else 0,
        }
        return create_transaction_metrics_from_stats(name, stats)

    def _create_summary(self, users, date_start, date_end, ramp_up, duration, metrics) -> ReportSummary:
        """Assembles the final, validated ReportSummary object."""
        total_metrics = metrics.get("Total")
        if not total_metrics:
            return ReportSummary()

        throughput = self._safe_divide(total_metrics.OK, duration.total_seconds())

        return ReportSummary(
            max_user_count=users,
            ramp_up_period=ramp_up,
            error_rate=total_metrics.Error_pct,
            date_start=date_start,
            date_end=date_end,
            throughput=round(throughput, 2),
            duration=duration,
            think_time=self.calculated_think_time
        )


class JMeterReportParser(BaseReportParser):
    """
    Production-ready JMeter parser conforming to the unified BaseReportParser contract.
    Its single public method `parse()` returns a validated PerformanceReport object.
    """

    def __init__(self, content_stream: io.StringIO, **kwargs):
        super().__init__(content_stream, **kwargs)
        self.df = None
        self.calculated_think_time = kwargs.get("think_times", 0)
        self.logger.info("Unified JMeter parser initialized.")

    def parse(self) -> PerformanceReport:
        """
        Orchestrates the entire parsing process from raw JMeter CSV to a validated
        PerformanceReport object.
        """
        self.logger.info("Starting unified JMeter parsing to create PerformanceReport object.")
        try:
            # Phase 1: Load JMeter CSV into a validated DataFrame.
            self._load_and_validate_dataframe()

            # Phase 2: Calculate metrics for each transaction, creating TransactionMetrics objects.
            transaction_metrics = self._calculate_transaction_metrics()

            # Phase 3: Create the validated ReportSummary object.
            summary = self._create_summary()

            # Phase 4: Assemble the final, validated PerformanceReport object.
            report = PerformanceReport(
                summary=summary,
                transactions=transaction_metrics,
                report_type="JMETER"
            )

            # Phase 5: Run internal validation and log any issues.
            validation_errors = validate_performance_report(report)
            if validation_errors:
                self.logger.warning(f"Post-parsing report validation found issues: {validation_errors}")

            self.logger.info("Successfully created and validated PerformanceReport from JMeter CSV.")
            return report

        except Exception as e:
            self.logger.error(f"Critical error during JMeter parsing: {e}", exc_info=True)
            raise

    def _load_and_validate_dataframe(self):
        """(Private helper) Loads and validates the JMeter CSV data."""
        self.logger.debug("Loading JMeter data into DataFrame...")
        try:
            self.stream.seek(0)
            self.df = pd.read_csv(self.stream, sep=',')
            required_cols = ['label', 'elapsed', 'success', 'timeStamp', 'allThreads']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols: raise ValueError(f"Missing required JMeter columns: {missing_cols}")
            self.df['timeStamp'] = pd.to_numeric(self.df['timeStamp'], errors='coerce')
            self.logger.info(f"Loaded {len(self.df)} records from JMeter CSV.")
        except Exception as e:
            self.logger.error(f"Failed to load or validate JMeter DataFrame: {e}", exc_info=True)
            raise

    def _calculate_transaction_metrics(self) -> Dict[str, TransactionMetrics]:
        """(Private helper) Calculates metrics and returns a dictionary of TransactionMetrics objects."""
        if self.df is None or self.df.empty: return {}

        transaction_metrics = {}
        for label in self.df['label'].unique():
            transaction_df = self.df[self.df['label'] == label]
            transaction_metrics[label] = self._create_metrics_from_df(transaction_df, label)

        transaction_metrics['Total'] = self._create_metrics_from_df(self.df, 'Total')
        self.logger.info(f"Calculated metrics for {len(transaction_metrics) - 1} transactions plus Total.")
        return transaction_metrics

    def _create_metrics_from_df(self, df: pd.DataFrame, name: str) -> TransactionMetrics:
        """(Private helper) Creates a single TransactionMetrics object from a DataFrame slice."""
        if df.empty: return create_empty_transaction_metrics(name)

        total = len(df)
        ok = len(df[df['success'] == True])
        ko = total - ok

        stats = {
            'Total': total, 'OK': ok, 'KO': ko,
            'min': round(float(df['elapsed'].min()), 3),
            'max': round(float(df['elapsed'].max()), 3),
            'average': round(float(df['elapsed'].mean()), 3),
            'median': round(float(df['elapsed'].median()), 3),
            'Error%': self._calculate_error_percentage(ko, total),
            '90Pct': round(float(df['elapsed'].quantile(0.9)), 3),
            '95Pct': round(float(df['elapsed'].quantile(0.95)), 3),
        }
        return create_transaction_metrics_from_stats(name, stats)

    def _create_summary(self) -> ReportSummary:
        """(Private helper) Creates a validated ReportSummary object from the DataFrame."""
        if self.df is None or self.df.empty:
            self.logger.warning("Cannot create summary from empty DataFrame.")
            # Return a default empty summary to avoid crashes downstream
            return ReportSummary(max_user_count=0, ramp_up_period=0, error_rate=0, date_start="", date_end="",
                                 throughput=0, duration=0, think_time=0)

        df_sorted = self.df.sort_values(by=['timeStamp'])
        start_ts = df_sorted['timeStamp'].iloc[0]
        end_ts = df_sorted['timeStamp'].iloc[-1]
        duration_seconds = (end_ts - start_ts) / 1000.0

        total_requests = len(df_sorted)
        ok_requests = len(df_sorted[df_sorted['success'] == True])
        error_count = total_requests - ok_requests

        throughput = self._safe_divide(ok_requests, duration_seconds)
        error_rate = self._calculate_error_percentage(error_count, total_requests)
        max_user_count = int(df_sorted['allThreads'].max())

        try:
            summary = ReportSummary(
                max_user_count=max_user_count,
                ramp_up_period=0,  # JMeter logs don't have a clear ramp-up period like Gatling
                error_rate=error_rate,
                date_start=datetime.fromtimestamp(start_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                date_end=datetime.fromtimestamp(end_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                throughput=round(throughput, 2),
                duration=round(duration_seconds, 2),
                think_time=str(self.calculated_think_time)
            )
            self.logger.info(f"JMeter summary created - Throughput: {summary.throughput}")
            return summary
        except Exception as e:
            self.logger.error(f"Failed to create JMeter ReportSummary object: {e}", exc_info=True)
            raise


# =================================================================================
# 3. REFACTORED PLACEHOLDER PARSERS
# =================================================================================

class LighthouseJsonParser(BaseReportParser):
    """Placeholder parser conforming to the unified contract."""

    def parse(self) -> PerformanceReport:
        self.logger.warning("LighthouseJsonParser is not yet implemented.")
        raise NotImplementedError("Lighthouse JSON parsing not yet supported")


class PptxTextExtractor(BaseReportParser):
    """Placeholder parser conforming to the unified contract."""

    def parse(self) -> PerformanceReport:
        self.logger.warning("PptxTextExtractor is not yet implemented.")
        raise NotImplementedError("PPTX text extraction not yet supported")


class DocxTextExtractor(BaseReportParser):
    """Placeholder parser conforming to the unified contract."""

    def parse(self) -> PerformanceReport:
        self.logger.warning("DocxTextExtractor is not yet implemented.")
        raise NotImplementedError("DOCX text extraction not yet supported")
