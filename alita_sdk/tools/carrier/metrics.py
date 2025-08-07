import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolkitMetrics:
    """
    Metrics collector for monitoring toolkit performance and usage patterns.
    Helps identify bottlenecks and optimize the most-used features.
    """

    def __init__(self):
        self.intent_extraction_count = 0
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0
        self.fallback_activations = 0
        self.parameter_extraction_errors = 0
        self.total_execution_time = 0.0
        self.start_time = datetime.now()

        # Track per-action metrics
        self.action_counts: Dict[str, int] = {}
        self.action_errors: Dict[str, int] = {}

    def record_action(self, task_type: str, action: str, success: bool, execution_time: float = 0.0):
        """Record metrics for a specific action"""
        action_key = f"{task_type}.{action}"

        if action_key not in self.action_counts:
            self.action_counts[action_key] = 0
            self.action_errors[action_key] = 0

        self.action_counts[action_key] += 1

        if success:
            self.successful_tool_calls += 1
            self.total_execution_time += execution_time
        else:
            self.failed_tool_calls += 1
            self.action_errors[action_key] += 1

    def log_metrics(self):
        """Log current metrics for operational insight"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        avg_execution_time = (self.total_execution_time / self.successful_tool_calls
                              if self.successful_tool_calls > 0 else 0.0)

        logger.info(
            f"[CarrierMetrics] Runtime: {runtime:.0f}s | "
                        f"Intents: {self.intent_extraction_count} | "
            f"Success: {self.successful_tool_calls} | "
            f"Failed: {self.failed_tool_calls} | "
            f"Fallbacks: {self.fallback_activations} | "
            f"Param Errors: {self.parameter_extraction_errors} | "
            f"Avg Time: {avg_execution_time:.2f}s"
        )

        # Log top actions if available
        if self.action_counts:
            top_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"[CarrierMetrics] Top Actions: {', '.join(f'{k}({v})' for k, v in top_actions)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dictionary"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        return {
            'runtime_seconds': runtime,
            'intent_extraction_count': self.intent_extraction_count,
            'successful_tool_calls': self.successful_tool_calls,
            'failed_tool_calls': self.failed_tool_calls,
            'fallback_activations': self.fallback_activations,
            'parameter_extraction_errors': self.parameter_extraction_errors,
            'avg_execution_time': (self.total_execution_time / self.successful_tool_calls
                                     if self.successful_tool_calls > 0 else 0.0),
            'top_actions': dict(sorted(self.action_counts.items(),
                                     key=lambda x: x[1], reverse=True)[:10])
        }

    def reset(self):
        """Reset metrics for new reporting period"""
        self.__init__()


# Singleton instance for consistent metrics across the toolkit
toolkit_metrics = ToolkitMetrics()