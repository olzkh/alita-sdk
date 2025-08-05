"""
Performance Analyst helper for maintainability and reliability

Author: Karen Florykian
"""
import logging
import time
import json
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Type

logger = logging.getLogger(__name__)


class CarrierIntent(BaseModel):
    """
    Simplified intent model with direct parameter extraction capabilities.
    """
    task_type: str = Field(..., description="Primary task category")
    action: str = Field(..., description="Specific action to perform OR 'clarify' for ambiguous requests")
    tool_parameters: Dict[str, Any] = Field(default_factory=dict, description="Direct tool parameters")

    # DISAMBIGUATION FIELDS
    is_ambiguous: bool = Field(default=False, description="True if request needs clarification")
    clarification_question: Optional[str] = Field(None, description="Question to ask user for clarification")
    disambiguation_options: List[Dict[str, Any]] = Field(default_factory=list, description="Available options")

    # METADATA
    confidence_score: Optional[float] = Field(None, description="Confidence in interpretation (0-1)")

    @validator('disambiguation_options', pre=True)
    def ensure_disambiguation_options_is_list(cls, v):
        return v or []

    @validator('tool_parameters', pre=True)
    def ensure_tool_parameters_is_dict(cls, v):
        return v or {}

    def needs_clarification(self) -> bool:
        """Check if this intent requires user clarification."""
        return self.is_ambiguous and self.action == 'clarify'

    def get_clarification_prompt(self) -> str:
        """Generate a user-friendly clarification prompt."""
        if not self.needs_clarification():
            return ""

        prompt = self.clarification_question or "Could you please clarify your request?"
        if self.disambiguation_options:
            prompt += "\nHere are the options I found:"
            for i, option in enumerate(self.disambiguation_options, 1):
                action_desc = option.get('description', option.get('action', 'Unknown'))
                prompt += f"\n{i}. {action_desc}"
        return prompt


class CarrierIntentExtractor:
    """
    Streamlined intent extractor with direct parameter extraction.
    """

    def __init__(self, llm, max_retries: int = 2, timeout: int = 30):
        """Initialize with simplified error handling"""
        logger.info("[IntentExtractor] Initializing streamlined extractor")

        self.llm = llm
        self.max_retries = max_retries
        self.timeout = timeout

        # Create structured LLM with explicit error handling
        self.structured_llm = self._create_structured_llm()

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_extractions': 0,
            'disambiguation_requests': 0,
            'failed_extractions': 0,
            'average_response_time': 0.0
        }

    def _create_structured_llm(self):
        """Create structured LLM with proper error handling"""
        try:
            structured_llm = self.llm.with_structured_output(
                schema=CarrierIntent,
                method="json_mode",
                include_raw=False
            )
            logger.info("[IntentExtractor] Structured LLM created successfully")
            return structured_llm
        except Exception as e:
            logger.error(f"[IntentExtractor] Failed to create structured LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")

    def extract_intent_with_parameters(self, user_message: str, tool_schema: Optional[Type[BaseModel]] = None,
                                       context: Optional[Dict] = None,
                                       skip_action_validation: bool = False) -> CarrierIntent:
        """
        Extract intent and parameters in a single, robust LLM call.

        Args:
            user_message: The input message from the user or system.
            tool_schema: The Pydantic schema for structured data extraction.
            context: Additional context for the prompt.
            skip_action_validation: If True, bypasses the action-to-task mapping validation.

        Raises:  RuntimeError: If extraction fails after all retries.
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1

        logger.info(f"[IntentExtractor] Processing: '{user_message}'")
        if tool_schema:
            logger.info(f"[IntentExtractor] Using schema: {tool_schema.__name__}")

        # Build schema-aware prompt
        from .prompts import build_performance_analyst_prompt
        prompt = build_performance_analyst_prompt(user_message, context, tool_schema)

        # Execute extraction with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"[IntentExtractor] Attempt {attempt + 1}/{self.max_retries}")

                # Invoke structured LLM
                result = self.structured_llm.invoke(prompt)

                # Process and validate result
                intent = self._process_llm_result(result, user_message)

                if intent and self._validate_intent(intent, skip_action_validation=skip_action_validation):
                    response_time = time.time() - start_time
                    self._update_success_metrics(response_time, intent)
                    logger.info(f"[IntentExtractor] Success on attempt {attempt + 1}")
                    return intent
                else:
                    logger.warning(f"[IntentExtractor] Invalid intent on attempt {attempt + 1}")
                    last_error = "Invalid intent returned"

            except Exception as e:
                logger.warning(f"[IntentExtractor] Attempt {attempt + 1} failed: {e}")
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))

        # All attempts failed - explicit error
        self.metrics['failed_extractions'] += 1
        error_msg = f"Intent extraction failed after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(f"[IntentExtractor] {error_msg}")
        raise RuntimeError(error_msg)

    def extract_intent(self, user_message: str, context: Optional[Dict] = None) -> CarrierIntent:
        """
        Backward compatibility method.

        Raises:
            RuntimeError: If extraction fails
        """
        return self.extract_intent_with_parameters(user_message, None, context)

    def _process_llm_result(self, result: Any, user_message: str) -> Optional[CarrierIntent]:
        """
        Process LLM result with explicit error handling.
        """
        logger.debug(f"[IntentExtractor] Processing result type: {type(result)}")

        if isinstance(result, CarrierIntent):
            logger.debug("[IntentExtractor] Result is already CarrierIntent")
            return result

        elif isinstance(result, dict):
            logger.debug("[IntentExtractor] Result is dict, creating CarrierIntent")
            return self._create_intent_from_dict(result)

        elif hasattr(result, 'content'):
            logger.debug("[IntentExtractor] Result has content, parsing JSON")
            return self._parse_content_to_intent(result.content)

        else:
            logger.error(f"[IntentExtractor] Unexpected result type: {type(result)}")
            return None

    def extract_structured_data(self, user_message: str, tool_schema: Type[BaseModel],
                                context: Optional[Dict] = None) -> Optional[BaseModel]:
        """
        Directly extracts structured data into a given Pydantic schema.
        This is the preferred method for internal, non-routing analysis tasks.
        """
        logger.info(
            f"[IntentExtractor] Performing direct structured data extraction for schema: {tool_schema.__name__}")
        start_time = time.time()

        try:
            # Configure the LLM to directly output the target schema
            structured_llm = self.llm.with_structured_output(schema=tool_schema)

            from .prompts import build_performance_analyst_prompt
            prompt = build_performance_analyst_prompt(user_message, context, tool_schema)

            # Invoke the LLM and get the structured object directly
            result = structured_llm.invoke(prompt)

            response_time = time.time() - start_time
            logger.info(f"Direct data extraction successful in {response_time:.2f}s.")

            # The result should already be an instance of our target schema
            if isinstance(result, tool_schema):
                return result
            else:
                logger.error(f"LLM did not return the expected schema type. Got: {type(result)}")
                return None

        except Exception as e:
            logger.error(f"Direct structured data extraction failed: {e}", exc_info=True)
            return None

    def _create_intent_from_dict(self, data: Dict) -> Optional[CarrierIntent]:
        """
        Safely create CarrierIntent from dictionary.
        """
        try:
            # Ensure required fields exist
            if 'task_type' not in data or 'action' not in data:
                logger.error("[IntentExtractor] Missing required fields in result")
                return None

            safe_data = {
                'task_type': data['task_type'],
                'action': data['action'],
                'tool_parameters': data.get('tool_parameters', {}),
                'is_ambiguous': data.get('is_ambiguous', False),
                'clarification_question': data.get('clarification_question'),
                'disambiguation_options': data.get('disambiguation_options', []),
                'confidence_score': data.get('confidence_score')
            }

            logger.debug(f"[IntentExtractor] Creating CarrierIntent: {safe_data}")
            return CarrierIntent(**safe_data)

        except Exception as e:
            logger.error(f"[IntentExtractor] Failed to create CarrierIntent: {e}")
            return None

    def _parse_content_to_intent(self, content: str) -> Optional[CarrierIntent]:
        """
        Parse string content to CarrierIntent.
        """
        try:
            if isinstance(content, str):
                parsed_data = json.loads(content)
                return self._create_intent_from_dict(parsed_data)
            else:
                logger.error(f"[IntentExtractor] Content is not string: {type(content)}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"[IntentExtractor] JSON parsing failed: {e}")
            return None

    def _validate_intent(self, intent: CarrierIntent, skip_action_validation: bool = False) -> bool:
        """
        Validate intent with explicit checks.
        """
        if not intent:
            return False

        # Basic field validation
        if not intent.task_type or not intent.action:
            logger.warning("[IntentExtractor] Missing required fields")
            return False

        # Special validation for disambiguation
        if intent.needs_clarification():
            if not intent.clarification_question:
                logger.warning("[IntentExtractor] Disambiguation intent missing clarification question")
                return False
            logger.info("[IntentExtractor] Valid disambiguation intent")
            return True

        # Standard action validation
        from .prompts import validate_action_mapping
        if not validate_action_mapping(intent.task_type, intent.action):
            logger.warning(f"[IntentExtractor] Invalid action mapping: {intent.task_type} -> {intent.action}")
            return False

        if skip_action_validation:
            logger.debug("[IntentExtractor] Bypassing action validation for internal parsing task.")
        else:
            from .prompts import validate_action_mapping
            if not validate_action_mapping(intent.task_type, intent.action):
                logger.warning(f"[IntentExtractor] Invalid action mapping: {intent.task_type} -> {intent.action}")
                return False

        # Confidence check
        if intent.confidence_score is not None and intent.confidence_score < 0.3:
            logger.warning(f"[IntentExtractor] Low confidence: {intent.confidence_score}")
            return False

        logger.debug("[IntentExtractor] Intent validation passed")
        return True

    def _update_success_metrics(self, response_time: float, intent: CarrierIntent):
        """
        Update performance metrics.
        """
        self.metrics['successful_extractions'] += 1

        if intent.needs_clarification():
            self.metrics['disambiguation_requests'] += 1

        # Update average response time
        total_success = self.metrics['successful_extractions']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = ((current_avg * (total_success - 1)) + response_time) / total_success

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics.
        """
        total = self.metrics['total_requests']
        return {
            **self.metrics,
            'success_rate': self.metrics['successful_extractions'] / max(total, 1),
            'disambiguation_rate': self.metrics['disambiguation_requests'] / max(total, 1),
            'failure_rate': self.metrics['failed_extractions'] / max(total, 1)
        }

    def reset_metrics(self):
        """Reset all performance metrics"""
        logger.info("[IntentExtractor] Resetting performance metrics")
        self.metrics = {
            'total_requests': 0,
            'successful_extractions': 0,
            'disambiguation_requests': 0,
            'failed_extractions': 0,
            'average_response_time': 0.0
        }


def create_performance_analytics_extractor(llm) -> CarrierIntentExtractor:
    """
    Factory function to create the streamlined intent extractor.

    Raises:
        RuntimeError: If extractor creation fails
    """
    logger.info("[Factory] Creating streamlined intent extractor")

    try:
        extractor = CarrierIntentExtractor(
            llm=llm,
            max_retries=2,
            timeout=30
        )
        logger.info("[Factory] Extractor created successfully")
        return extractor

    except Exception as e:
        logger.error(f"[Factory] Failed to create extractor: {e}")
        raise RuntimeError(f"Extractor creation failed: {e}")


__all__ = [
    'CarrierIntent',
    'CarrierIntentExtractor',
    'create_performance_analytics_extractor'
]
