"""
Chart Builder
Author: Karen Florykian
"""

import logging
from typing import Dict, Any
import io


def _safe_float(value: Any) -> float:
    """Safely convert value to float."""
    try:
        if isinstance(value, str):
            # Remove percentage signs and other non-numeric characters
            value = value.strip().replace('%', '').replace(',', '')
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0


def _add_bar_labels(ax, bars, format_str='{:.2f}'):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add label if height is positive
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    format_str.format(height),
                    ha='center', va='bottom', fontsize=8)


class ChartBuilder:
    """Handles chart generation with type safety."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_performance_trend_chart(self, chart_data: Dict[str, Any], output_buffer: io.BytesIO) -> bool:
        """Create performance trend chart with type safety."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Validate we have data
            if not chart_data or len(chart_data) < 2:
                self.logger.info("Not enough data points for trend chart (need at least 2)")
                return False

            # Extract and validate data
            test_names = []
            throughputs = []
            error_rates = []
            avg_response_times = []
            p90_response_times = []

            for test_name, data in sorted(chart_data.items()):
                if isinstance(data, dict):
                    # Validate all required fields are numeric
                    try:
                        throughput = float(data.get('throughput', 0))
                        error_rate = float(data.get('error_rate', 0))
                        avg_rt = float(data.get('avg_response_time', 0))
                        p90_rt = float(data.get('p90_response_time', 0))

                        test_names.append(test_name)
                        throughputs.append(throughput)
                        error_rates.append(error_rate)
                        avg_response_times.append(avg_rt)
                        p90_response_times.append(p90_rt)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Skipping {test_name} due to invalid data: {e}")
                        continue

            if len(test_names) < 2:
                self.logger.info("Not enough valid data points after filtering")
                return False

            # Convert to numpy arrays for safe arithmetic
            throughputs = np.array(throughputs, dtype=np.float64)
            error_rates = np.array(error_rates, dtype=np.float64)
            avg_response_times = np.array(avg_response_times, dtype=np.float64)
            p90_response_times = np.array(p90_response_times, dtype=np.float64)

            # Create the chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Response time trends
            x = np.arange(len(test_names))
            ax1.plot(x, avg_response_times, 'b-o', label='Average', linewidth=2, markersize=8)
            ax1.plot(x, p90_response_times, 'r-s', label='90th Percentile', linewidth=2, markersize=8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(test_names, rotation=45, ha='right')
            ax1.set_ylabel('Response Time (seconds)')
            ax1.set_title('Response Time Trends')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Throughput vs Error Rate
            ax2_twin = ax2.twinx()

            # Bar width
            bar_width = 0.35

            # Throughput bars
            bars1 = ax2.bar(x - bar_width / 2, throughputs, bar_width,
                            label='Throughput', color='green', alpha=0.7)

            # Error rate bars
            bars2 = ax2_twin.bar(x + bar_width / 2, error_rates, bar_width,
                                 label='Error Rate', color='red', alpha=0.7)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}', ha='center', va='bottom')

            for bar in bars2:
                height = bar.get_height()
                ax2_twin.text(bar.get_x() + bar.get_width() / 2., height,
                              f'{height:.1f}%', ha='center', va='bottom')

            ax2.set_xticks(x)
            ax2.set_xticklabels(test_names, rotation=45, ha='right')
            ax2.set_ylabel('Throughput (req/sec)', color='green')
            ax2_twin.set_ylabel('Error Rate (%)', color='red')
            ax2.set_title('Throughput vs Error Rate')

            # Add legends
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(output_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info("Successfully created performance trend chart")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create trend chart: {e}", exc_info=True)
            return False

    def create_comparison_chart(self, chart_data: Dict[str, Any], output_buffer: io.BytesIO) -> bool:
        """Create comparison chart between two test runs."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Validate we have data for comparison
            if not chart_data or len(chart_data) < 2:
                self.logger.info("Not enough data points for comparison chart (need at least 2)")
                return False

            # Extract the two most recent test runs for comparison
            sorted_tests = sorted(chart_data.items(),
                                  key=lambda x: x[1].get('timestamp', ''),
                                  reverse=True)

            if len(sorted_tests) < 2:
                self.logger.info("Need at least 2 test runs for comparison")
                return False

            test1_name, test1_data = sorted_tests[1]  # Previous test
            test2_name, test2_data = sorted_tests[0]  # Latest test

            # Validate data structure
            if not isinstance(test1_data, dict) or not isinstance(test2_data, dict):
                self.logger.error("Invalid data structure for comparison chart")
                return False

            # Extract transaction data for comparison
            test1_transactions = test1_data.get('transactions', {})
            test2_transactions = test2_data.get('transactions', {})

            if not test1_transactions or not test2_transactions:
                self.logger.warning("No transaction data found for comparison")
                return False

            # Find common transactions between both tests
            common_transactions = set(test1_transactions.keys()) & set(test2_transactions.keys())

            if not common_transactions:
                self.logger.warning("No common transactions found between test runs")
                return False

            # Limit to top 10 transactions by average response time
            sorted_transactions = sorted(
                common_transactions,
                key=lambda tx: max(
                    float(test1_transactions[tx].get('avg_response_time', 0)),
                    float(test2_transactions[tx].get('avg_response_time', 0))
                ),
                reverse=True
            )[:10]

            # Prepare data for plotting
            transaction_names = []
            test1_avg_times = []
            test1_p90_times = []
            test1_error_rates = []
            test2_avg_times = []
            test2_p90_times = []
            test2_error_rates = []

            for tx_name in sorted_transactions:
                tx1 = test1_transactions[tx_name]
                tx2 = test2_transactions[tx_name]

                # Truncate long transaction names
                display_name = tx_name[:20] + "..." if len(tx_name) > 20 else tx_name
                transaction_names.append(display_name)

                # Extract metrics with safe conversion
                test1_avg_times.append(_safe_float(tx1.get('avg_response_time', 0)))
                test1_p90_times.append(_safe_float(tx1.get('p90_response_time', 0)))
                test1_error_rates.append(_safe_float(tx1.get('error_rate', 0)))

                test2_avg_times.append(_safe_float(tx2.get('avg_response_time', 0)))
                test2_p90_times.append(_safe_float(tx2.get('p90_response_time', 0)))
                test2_error_rates.append(_safe_float(tx2.get('error_rate', 0)))

            # Convert to numpy arrays for plotting
            test1_avg_times = np.array(test1_avg_times, dtype=np.float64)
            test1_p90_times = np.array(test1_p90_times, dtype=np.float64)
            test1_error_rates = np.array(test1_error_rates, dtype=np.float64)
            test2_avg_times = np.array(test2_avg_times, dtype=np.float64)
            test2_p90_times = np.array(test2_p90_times, dtype=np.float64)
            test2_error_rates = np.array(test2_error_rates, dtype=np.float64)

            # Create the comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Performance Comparison: {test1_name} vs {test2_name}',
                         fontsize=16, fontweight='bold')

            x = np.arange(len(transaction_names))
            bar_width = 0.35

            # 1. Average Response Time Comparison
            bars1 = ax1.bar(x - bar_width / 2, test1_avg_times, bar_width,
                            label=f'Previous ({test1_name[:15]})',
                            color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x + bar_width / 2, test2_avg_times, bar_width,
                            label=f'Latest ({test2_name[:15]})',
                            color='lightcoral', alpha=0.8)

            ax1.set_xlabel('Transactions')
            ax1.set_ylabel('Average Response Time (seconds)')
            ax1.set_title('Average Response Time Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(transaction_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            _add_bar_labels(ax1, bars1)
            _add_bar_labels(ax1, bars2)

            # 2. 90th Percentile Response Time Comparison
            bars3 = ax2.bar(x - bar_width / 2, test1_p90_times, bar_width,
                            label=f'Previous ({test1_name[:15]})',
                            color='lightgreen', alpha=0.8)
            bars4 = ax2.bar(x + bar_width / 2, test2_p90_times, bar_width,
                            label=f'Latest ({test2_name[:15]})',
                            color='orange', alpha=0.8)

            ax2.set_xlabel('Transactions')
            ax2.set_ylabel('90th Percentile Response Time (seconds)')
            ax2.set_title('90th Percentile Response Time Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(transaction_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add value labels on bars
            _add_bar_labels(ax2, bars3)
            _add_bar_labels(ax2, bars4)

            # 3. Error Rate Comparison
            bars5 = ax3.bar(x - bar_width / 2, test1_error_rates, bar_width,
                            label=f'Previous ({test1_name[:15]})',
                            color='gold', alpha=0.8)
            bars6 = ax3.bar(x + bar_width / 2, test2_error_rates, bar_width,
                            label=f'Latest ({test2_name[:15]})',
                            color='crimson', alpha=0.8)

            ax3.set_xlabel('Transactions')
            ax3.set_ylabel('Error Rate (%)')
            ax3.set_title('Error Rate Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(transaction_names, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels on bars
            _add_bar_labels(ax3, bars5, format_str='{:.1f}%')
            _add_bar_labels(ax3, bars6, format_str='{:.1f}%')

            # 4. Performance Change Indicators (Percentage Change)
            avg_time_changes = []
            for i in range(len(test1_avg_times)):
                if test1_avg_times[i] > 0:
                    change = ((test2_avg_times[i] - test1_avg_times[i]) / test1_avg_times[i]) * 100
                else:
                    change = 0
                avg_time_changes.append(change)

            avg_time_changes = np.array(avg_time_changes)

            # Color bars based on performance change (green for improvement, red for degradation)
            colors = ['green' if change <= 0 else 'red' for change in avg_time_changes]

            bars7 = ax4.bar(x, avg_time_changes, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax4.set_xlabel('Transactions')
            ax4.set_ylabel('Performance Change (%)')
            ax4.set_title('Average Response Time Change (Latest vs Previous)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(transaction_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            _add_bar_labels(ax4, bars7, format_str='{:.1f}%')

            # Add legend for performance change
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Improvement'),
                Patch(facecolor='red', alpha=0.7, label='Degradation')
            ]
            ax4.legend(handles=legend_elements)

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Save to buffer
            plt.savefig(output_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Successfully created comparison chart with {len(transaction_names)} transactions")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create comparison chart: {e}", exc_info=True)
            return False
