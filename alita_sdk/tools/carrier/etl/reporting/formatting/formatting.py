import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class TextFormatter:
    """
    SRP: A dedicated class for all text formatting and parsing logic.
    This makes text manipulation reusable across the application.
    """

    @staticmethod
    def format_transaction_name(name: str) -> str:
        """Formats a raw transaction name into a human-readable version."""
        try:
            # Remove common HTTP method prefixes
            name = re.sub(r'^(GET|POST|PUT|DELETE)_', '', name, flags=re.IGNORECASE)
            # Add spaces before capitals for camelCase to PascalCase
            name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
            return name.strip().title()  # Capitalize each word
        except Exception as e:
            logger.warning(f"Failed to format transaction name '{name}': {e}")
            return name

    @staticmethod
    def parse_justification_to_insights(justification: str) -> List[str]:
        """Converts a technical justification string into a user-friendly list of insights."""
        if not justification:
            return ["No specific insights available."]

        insights = []
        parts = justification.split(" | ")
        for part in parts:
            part = part.strip()
            if not part: continue

            if "exceeded the threshold" in part.lower():
                insights.append(f"🚨 {part}")
            elif "below the threshold" in part.lower():
                insights.append(f"📉 {part}")
            elif "all performance metrics" in part.lower():
                insights.append(f"✅ {part}")
            else:
                insights.append(f"• {part}")

        return insights[:5]