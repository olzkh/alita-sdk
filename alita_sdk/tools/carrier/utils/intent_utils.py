import logging
import time
from typing import Optional, Dict, Any, List
from functools import wraps

from langchain_core.pydantic_v1 import BaseModel, Field, validator
from .prompts import build_carrier_prompt_, PERFORMANCE_ANALYST_INTENT_EXAMPLES

logger = logging.getLogger(__name__)


class CarrierIntent(BaseModel):
    """
    Enhanced intent model with validation for Performance Analytics workflows
    """
    task_type: str = Field(..., description="Primary task category (e.g., backend_analysis, ui_test_management)")
    action: str = Field(..., description="Specific action to perform (e.g., get_reports, run_test)")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities with parameters")
    field_requests: List[str] = Field(default_factory=list, description="Specific fields requested by user")
    confirmation_question: str = Field(..., description="Question to confirm intent understanding")
    rephrased_suggestion: Optional[str] = Field(None, description="Alternative phrasing suggestion")
    confidence_score: Optional[float] = Field(None, description="Intent extraction confidence (0-1)")

    @validator('task_type')
    def validate_task_type(cls, v):
        """Validate task_type against known Performance Analytics categories"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating task_type: {repr(v)} (type: {type(v)})")
        valid_task_types = {
            'ticket_action', 'backend_analysis', 'ticket_management',
            'ui_analysis', 'ui_test_management', 'ui_test_execution', 'backend_test_management',
            'backend_test_execution'
        }
        if v not in valid_task_types:
            logger.warning(f"[CarrierIntent] Unknown task_type: {v}")
        logger.critical(f"🔍 [DIAGNOSTIC] task_type validation passed: {v}")
        return v

    @validator('action')
    def validate_action(cls, v):
        """Add validation logging for action field"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating action: {repr(v)} (type: {type(v)})")
        logger.critical(f"🔍 [DIAGNOSTIC] action validation passed: {v}")
        return v

    @validator('entities', pre=True)
    def validate_entities(cls, v):
        """Add extensive validation logging for entities field"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating entities (PRE): {repr(v)} (type: {type(v)})")

        if v is None:
            logger.critical(f"🔍 [DIAGNOSTIC] entities is None, converting to empty list")
            return []

        if not isinstance(v, list):
            logger.critical(f"🔍 [DIAGNOSTIC] entities is not a list, type: {type(v)}, value: {v}")
            if isinstance(v, dict):
                logger.critical(f"🔍 [DIAGNOSTIC] Converting dict to list of dicts: {v}")
                return [v] if v else []
            else:
                logger.critical(f"🔍 [DIAGNOSTIC] Unknown entities type, using empty list")
                return []

        logger.critical(f"🔍 [DIAGNOSTIC] entities is list with {len(v)} items")
        for i, item in enumerate(v):
            logger.critical(f"🔍 [DIAGNOSTIC] entities[{i}]: {type(item)} = {repr(item)}")
            if not isinstance(item, dict):
                logger.critical(f"🔍 [DIAGNOSTIC] WARNING: entities[{i}] is not a dict!")

        logger.critical(f"🔍 [DIAGNOSTIC] entities validation passed: {v}")
        return v

    @validator('field_requests', pre=True)
    def validate_field_requests(cls, v):
        """Add validation logging for field_requests"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating field_requests (PRE): {repr(v)} (type: {type(v)})")

        if v is None:
            logger.critical(f"🔍 [DIAGNOSTIC] field_requests is None, converting to empty list")
            return []

        if not isinstance(v, list):
            logger.critical(f"🔍 [DIAGNOSTIC] field_requests is not a list: {type(v)}, converting")
            if isinstance(v, str):
                return [v]
            else:
                return []

        logger.critical(f"🔍 [DIAGNOSTIC] field_requests validation passed: {v}")
        return v

    @validator('confirmation_question')
    def validate_confirmation_question(cls, v):
        """Add validation logging for confirmation_question"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating confirmation_question: {repr(v)} (type: {type(v)})")
        if not v:
            logger.critical(f"🔍 [DIAGNOSTIC] Empty confirmation_question!")
        logger.critical(f"🔍 [DIAGNOSTIC] confirmation_question validation passed")
        return v

    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Ensure confidence score is between 0 and 1"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating confidence_score: {repr(v)} (type: {type(v)})")
        if v is not None and not (0 <= v <= 1):
            logger.warning(f"[CarrierIntent] Invalid confidence score: {v}")
            return None
        logger.critical(f"🔍 [DIAGNOSTIC] confidence_score validation passed: {v}")
        return v

    def __str__(self):
        return f"CarrierIntent(task={self.task_type}, action={self.action}, confidence={self.confidence_score})"


def with_timeout(timeout_seconds: int = 30):
    """Decorator to add timeout handling to LLM operations"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                logger.critical(f"🔍 [DIAGNOSTIC] Starting {func.__name__} with timeout {timeout_seconds}s")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"[IntentExtractor] {func.__name__} completed in {execution_time:.2f}s")
                logger.critical(f"🔍 [DIAGNOSTIC] {func.__name__} completed successfully")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.critical(
                    f"🔍 [DIAGNOSTIC] {func.__name__} failed after {execution_time:.2f}s: {type(e).__name__}: {e}")
                if execution_time >= timeout_seconds:
                    logger.error(f"[IntentExtractor] {func.__name__} timed out after {execution_time:.2f}s")
                    raise TimeoutError(f"LLM operation timed out after {timeout_seconds}s")
                raise

        return wrapper

    return decorator


class CarrierIntentExtractor:
    """
    Production-ready intent extractor optimized for Performance Analytics workflows
    with comprehensive error handling, retry logic, and performance monitoring
    """

    def __init__(self, llm, intent_examples: Dict = None, max_retries: int = 3, timeout: int = 30):
        """
        Initialize enhanced intent extractor with production configurations
        """
        logger.critical(f"🔍 [DIAGNOSTIC] Initializing CarrierIntentExtractor")
        logger.critical(f"  LLM type: {type(llm)}")
        logger.critical(f"  Max retries: {max_retries}")
        logger.critical(f"  Timeout: {timeout}s")

        self.llm = llm
        self.intent_examples = intent_examples or PERFORMANCE_ANALYST_INTENT_EXAMPLES
        self.max_retries = max_retries
        self.timeout = timeout

        try:
            logger.critical(f"🔍 [DIAGNOSTIC] Creating structured LLM...")
            self.structured_llm = self.llm.with_structured_output(schema=CarrierIntent)
            logger.critical(f"🔍 [DIAGNOSTIC] Structured LLM created: {type(self.structured_llm)}")
            logger.info(f"[IntentExtractor] Initialized with structured output for Performance Analytics")
        except Exception as e:
            logger.critical(f"🔍 [DIAGNOSTIC] Failed to create structured LLM: {type(e).__name__}: {e}")
            logger.error(f"[IntentExtractor] Failed to initialize structured output: {e}")
            # Fallback to regular LLM if structured output fails
            self.structured_llm = self.llm

        # PRODUCTION: Performance metrics
        self.extraction_metrics = {
            'total_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'retry_activations': 0,
            'average_execution_time': 0.0
        }

        logger.critical(f"🔍 [DIAGNOSTIC] CarrierIntentExtractor initialization complete")
        logger.info(f"[IntentExtractor] Production extractor initialized - retries: {max_retries}, timeout: {timeout}s")

    @with_timeout(30)
    def extract_intent(self, user_message: str, context: Optional[Dict] = None) -> Optional[CarrierIntent]:
        """
        Extract intent with comprehensive error handling and retry logic
        """
        logger.critical(f"🔍 [DIAGNOSTIC] Starting extract_intent")
        logger.critical(f"  User message: '{user_message}'")
        logger.critical(f"  Context: {context}")

        self.extraction_metrics['total_attempts'] += 1
        start_time = time.time()

        logger.info(f"[IntentExtractor] Processing Performance Analytics request: {user_message[:100]}...")

        for attempt in range(self.max_retries):
            logger.critical(f"🔍 [DIAGNOSTIC] Attempt {attempt + 1}/{self.max_retries}")

            try:
                logger.critical(f"🔍 [DIAGNOSTIC] Calling _attempt_extraction...")
                intent = self._attempt_extraction(user_message, context, attempt)
                logger.critical(f"🔍 [DIAGNOSTIC] _attempt_extraction returned: {type(intent)} = {intent}")

                if intent and self._validate_extracted_intent(intent):
                    execution_time = time.time() - start_time
                    self._update_success_metrics(execution_time)

                    logger.critical(f"🔍 [DIAGNOSTIC] Intent extraction successful on attempt {attempt + 1}")
                    logger.info(f"[IntentExtractor] Intent extracted successfully on attempt {attempt + 1}")
                    logger.debug(f"[IntentExtractor] Result: {intent}")

                    return intent
                else:
                    logger.critical(f"🔍 [DIAGNOSTIC] Intent validation failed on attempt {attempt + 1}")
                    logger.warning(f"[IntentExtractor] Invalid intent extracted on attempt {attempt + 1}")

            except Exception as e:
                logger.critical(f"🔍 [DIAGNOSTIC] Exception in attempt {attempt + 1}: {type(e).__name__}: {e}")

                # Check for the specific error we're hunting
                if "'NoneType' object is not iterable" in str(e):
                    logger.critical(f"🎯 [DIAGNOSTIC] FOUND THE ITERATION ERROR in attempt {attempt + 1}!")
                    import traceback
                    tb = traceback.format_exc()
                    logger.critical(f"🎯 [DIAGNOSTIC] Full traceback:")
                    for line_num, line in enumerate(tb.split('\n')):
                        if line.strip():
                            logger.critical(f"      {line_num:2d}: {line}")

                logger.warning(f"[IntentExtractor] Attempt {attempt + 1} failed: {str(e)[:100]}...")

                if attempt < self.max_retries - 1:
                    self.extraction_metrics['retry_activations'] += 1
                    # PRODUCTION: Exponential backoff with jitter
                    wait_time = (2 ** attempt) * 0.1 + (time.time() % 0.1)
                    logger.critical(f"🔍 [DIAGNOSTIC] Retrying in {wait_time:.2f}s...")
                    logger.debug(f"[IntentExtractor] Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    logger.critical(f"🔍 [DIAGNOSTIC] All {self.max_retries} attempts exhausted")
                    logger.error(f"[IntentExtractor] All {self.max_retries} attempts failed")

            self.extraction_metrics['failed_extractions'] += 1
            execution_time = time.time() - start_time

            logger.critical(f"🔍 [DIAGNOSTIC] Intent extraction completely failed after {execution_time:.2f}s")
            logger.error(f"[IntentExtractor] Intent extraction completely failed after {execution_time:.2f}s")
            return None

    def _attempt_extraction(self, user_message: str, context: Optional[Dict], attempt: int) -> Optional[
        CarrierIntent]:
        """Single attempt at intent extraction with enhanced prompt engineering"""
        logger.critical(f"🔍 [DIAGNOSTIC] Starting _attempt_extraction (attempt {attempt})")

        try:
            # PRODUCTION: Force structured output with explicit schema
            if hasattr(self.llm, 'with_structured_output'):
                logger.critical(f"🔍 [DIAGNOSTIC] LLM has with_structured_output, creating structured LLM...")

                structured_llm = self.llm.with_structured_output(
                    schema=CarrierIntent,
                    method="json_mode",  # Force JSON mode
                    include_raw=False
                )
                logger.critical(f"🔍 [DIAGNOSTIC] Structured LLM created: {type(structured_llm)}")

                # Create focused prompt for structured output
                logger.critical(f"🔍 [DIAGNOSTIC] Building prompt...")
                prompt = build_carrier_prompt_(user_message, context)
                logger.critical(f"🔍 [DIAGNOSTIC] Prompt built, length: {len(prompt)}")
                logger.critical(f"🔍 [DIAGNOSTIC] Prompt preview: {prompt[:200]}...")

                logger.critical(f"🔍 [DIAGNOSTIC] Invoking structured LLM...")
                result = structured_llm.invoke(prompt)
                logger.critical(f"🔍 [DIAGNOSTIC] LLM returned: {type(result)} = {repr(result)}")

                if isinstance(result, CarrierIntent):
                    logger.critical(f"🔍 [DIAGNOSTIC] Result is already CarrierIntent - SUCCESS!")
                    logger.critical(f"  task_type: {result.task_type}")
                    logger.critical(f"  action: {result.action}")
                    logger.critical(f"  entities: {result.entities}")
                    logger.critical(f"  field_requests: {result.field_requests}")
                    return result
                else:
                    logger.critical(f"🔍 [DIAGNOSTIC] Result is not CarrierIntent, type: {type(result)}")

                    # Try to handle different result types
                    if hasattr(result, 'content'):
                        logger.critical(f"🔍 [DIAGNOSTIC] Result has content: {result.content}")
                        # Try to parse the content as JSON and create CarrierIntent
                        import json
                        try:
                            if isinstance(result.content, str):
                                parsed = json.loads(result.content)
                                logger.critical(f"🔍 [DIAGNOSTIC] Parsed JSON from content: {parsed}")
                                logger.critical(f"🔍 [DIAGNOSTIC] Creating CarrierIntent from parsed JSON...")
                                intent = CarrierIntent(**parsed)
                                logger.critical(f"🔍 [DIAGNOSTIC] CarrierIntent created from JSON content!")
                                return intent
                        except Exception as parse_error:
                            logger.critical(f"🔍 [DIAGNOSTIC] Failed to parse content as JSON: {parse_error}")

                    # Try direct conversion if it's a dict
                    if isinstance(result, dict):
                        logger.critical(f"🔍 [DIAGNOSTIC] Result is dict, trying direct CarrierIntent creation...")
                        logger.critical(f"🔍 [DIAGNOSTIC] Dict keys: {list(result.keys())}")
                        for key, value in result.items():
                            logger.critical(f"🔍 [DIAGNOSTIC] {key}: {type(value)} = {repr(value)}")

                        try:
                            intent = CarrierIntent(**result)
                            logger.critical(f"🔍 [DIAGNOSTIC] CarrierIntent created from dict!")
                            return intent
                        except Exception as dict_error:
                            logger.critical(
                                f"🔍 [DIAGNOSTIC] Failed to create CarrierIntent from dict: {type(dict_error).__name__}: {dict_error}")

                            # Check if this is our target error
                            if "'NoneType' object is not iterable" in str(dict_error):
                                logger.critical(
                                    f"🎯 [DIAGNOSTIC] FOUND THE ERROR during CarrierIntent creation from dict!")
                                logger.critical(f"🎯 [DIAGNOSTIC] This happens in Pydantic validation")
                                logger.critical(f"🎯 [DIAGNOSTIC] Dict being processed: {result}")
                                # Let's examine each field that could cause iteration issues
                                logger.critical(f"🎯 [DIAGNOSTIC] Examining each field for iteration issues:")
                                for field_name in ['entities', 'field_requests']:
                                    if field_name in result:
                                        field_value = result[field_name]
                                        logger.critical(f"🎯 [DIAGNOSTIC] Field '{field_name}':")
                                        logger.critical(f"    Type: {type(field_value)}")
                                        logger.critical(f"    Value: {repr(field_value)}")
                                        logger.critical(f"    Is None: {field_value is None}")

                                        if field_value is not None:
                                            logger.critical(f"    Has __iter__: {hasattr(field_value, '__iter__')}")
                                            if hasattr(field_value, '__iter__') and not isinstance(field_value,
                                                                                                   (str, bytes)):
                                                try:
                                                    logger.critical(f"    Testing iteration...")
                                                    list_result = list(field_value)
                                                    logger.critical(f"    Iteration successful: {list_result}")
                                                except Exception as iter_err:
                                                    logger.critical(
                                                        f"    🚨 ITERATION FAILED: {type(iter_err).__name__}: {iter_err}")
                                                    logger.critical(f"    🚨 This is likely the source of our error!")

                            raise  # Re-raise the original error

            else:
                logger.critical(f"🔍 [DIAGNOSTIC] LLM does not have with_structured_output, using fallback")

            # PRODUCTION: Enhanced fallback parsing with pattern matching
            logger.critical(f"🔍 [DIAGNOSTIC] Falling back to enhanced parsing...")
            return self._enhanced_fallback_parsing(user_message, context)

        except Exception as e:
            logger.critical(f"🔍 [DIAGNOSTIC] Exception in _attempt_extraction: {type(e).__name__}: {e}")

            # Special handling for our target error
            if "'NoneType' object is not iterable" in str(e):
                logger.critical(f"🎯 [DIAGNOSTIC] CAUGHT THE ITERATION ERROR in _attempt_extraction!")
                import traceback
                tb = traceback.format_exc()
                logger.critical(f"🎯 [DIAGNOSTIC] Detailed traceback:")
                for i, line in enumerate(tb.split('\n')):
                    if line.strip():
                        logger.critical(f"    {i:2d}: {line}")

            logger.warning(f"[IntentExtractor] Extraction attempt failed: {e}")
            raise

    def _build_structured_prompt(self, user_message: str, context: Optional[Dict], attempt: int) -> str:
        """Build prompt specifically designed for structured JSON output"""
        logger.critical(f"🔍 [DIAGNOSTIC] Building structured prompt for attempt {attempt}")

        prompt = f"""
        You are a Performance Analytics intent extractor. Extract intent from user requests and return ONLY valid JSON.

        REQUIRED JSON FORMAT:
        {{
            "task_type": "backend_analysis|ui_analysis|test_management|test_execution|ui_test_management|ui_test_execution|ticket_action",
            "action": "get_reports|get_report_by_id|process_report|run_test|create_ticket|etc",
            "entities": [{{"id": "5134", "type": "report_id"}}],
            "field_requests": [],
            "confirmation_question": "Should I [action description]?",
            "confidence_score": 0.9
        }}

        MAPPING RULES:
        - "generate excel report from 5134" → task_type: "backend_analysis", action: "process_report"
        - "run test 5134" → task_type: "test_execution", action: "run_test"
        - "show ui reports" → task_type: "ui_analysis", action: "get_ui_reports"
        - "create ticket" → task_type: "ticket_action", action: "create_ticket"

        USER REQUEST: {user_message}

        Return ONLY the JSON object, no additional text.
        """

        logger.critical(f"🔍 [DIAGNOSTIC] Structured prompt created, length: {len(prompt)}")
        return prompt

    def _enhanced_fallback_parsing(self, user_message: str, context: Optional[Dict]) -> Optional[CarrierIntent]:
        """Enhanced fallback parsing with better pattern recognition"""
        logger.critical(f"🔍 [DIAGNOSTIC] Starting enhanced fallback parsing")

        patterns = {
            r'generate.*excel.*report.*(\d+)': ('backend_analysis', 'process_report'),
            r'excel.*report.*(\d+)': ('backend_analysis', 'process_report'),
            r'run.*test.*(\d+)': ('test_execution', 'run_test'),
            r'show.*ui.*report': ('ui_analysis', 'get_ui_reports'),
            r'get.*report.*(\d+)': ('backend_analysis', 'get_report_by_id'),
            r'create.*ticket': ('ticket_action', 'create_ticket'),
        }

        user_lower = user_message.lower()
        logger.critical(f"🔍 [DIAGNOSTIC] Testing patterns against: '{user_lower}'")

        for pattern, (task_type, action) in patterns.items():
            import re
            match = re.search(pattern, user_lower)
            if match:
                logger.critical(f"🔍 [DIAGNOSTIC] Pattern matched: {pattern} -> {task_type}, {action}")

                entities = []
                if match.groups():
                    # Extract ID from pattern
                    entity_id = match.group(1)
                    entities = [{"id": entity_id, "type": "test_id"}]
                    logger.critical(f"🔍 [DIAGNOSTIC] Extracted entity: {entities}")

                logger.critical(f"🔍 [DIAGNOSTIC] Creating CarrierIntent from pattern match...")
                try:
                    intent = CarrierIntent(
                        task_type=task_type,
                        action=action,
                        entities=entities,
                        field_requests=[],
                        confirmation_question=f"Should I {action.replace('_', ' ')} as requested?",
                        confidence_score=0.8
                    )
                    logger.critical(f"🔍 [DIAGNOSTIC] CarrierIntent created successfully from pattern!")
                    return intent
                except Exception as pattern_error:
                    logger.critical(
                        f"🔍 [DIAGNOSTIC] Failed to create CarrierIntent from pattern: {type(pattern_error).__name__}: {pattern_error}")

                    if "'NoneType' object is not iterable" in str(pattern_error):
                        logger.critical(f"🎯 [DIAGNOSTIC] ITERATION ERROR in pattern fallback!")
                        logger.critical(f"🎯 [DIAGNOSTIC] Entities being passed: {entities}")
                        logger.critical(f"🎯 [DIAGNOSTIC] Field_requests being passed: []")

                    raise

            logger.critical(f"🔍 [DIAGNOSTIC] No pattern match found")
            logger.warning(f"[IntentExtractor] No pattern match found for: {user_message}")
            return None

    def _validate_extracted_intent(self, intent: CarrierIntent) -> bool:
        """Validate extracted intent for completeness and correctness"""
        logger.critical(f"🔍 [DIAGNOSTIC] Validating extracted intent: {intent}")

        if not intent:
            logger.critical(f"🔍 [DIAGNOSTIC] Validation failed: No intent object")
            logger.debug("[IntentExtractor] Validation failed: No intent object")
            return False

        # PRODUCTION: Check required fields
        if not intent.task_type or not intent.action:
            logger.critical(f"🔍 [DIAGNOSTIC] Validation failed: Missing required fields")
            logger.critical(f"  task_type: {intent.task_type}")
            logger.critical(f"  action: {intent.action}")
            logger.debug(
                f"[IntentExtractor] Validation failed: Missing required fields - task_type: {intent.task_type}, action: {intent.action}")
            return False

        # PRODUCTION: Check confirmation question exists
        if not intent.confirmation_question:
            logger.critical(f"🔍 [DIAGNOSTIC] Validation failed: Missing confirmation question")
            logger.debug("[IntentExtractor] Validation failed: Missing confirmation question")
            return False

        # PRODUCTION: Log confidence score if available
        if intent.confidence_score is not None:
            logger.critical(f"🔍 [DIAGNOSTIC] Intent confidence: {intent.confidence_score}")
            logger.debug(f"[IntentExtractor] Intent confidence: {intent.confidence_score}")

            # Consider low confidence as validation failure
            if intent.confidence_score < 0.3:
                logger.critical(f"🔍 [DIAGNOSTIC] Validation failed: Low confidence {intent.confidence_score}")
                logger.warning(f"[IntentExtractor] Low confidence intent: {intent.confidence_score}")
                return False

        logger.critical(f"🔍 [DIAGNOSTIC] Intent validation passed successfully")
        logger.debug("[IntentExtractor] Intent validation passed")
        return True

    def _parse_unstructured_response(self, llm_response) -> Optional[CarrierIntent]:
        """Parse unstructured LLM response as fallback when structured output fails"""
        logger.critical(f"🔍 [DIAGNOSTIC] Parsing unstructured response: {type(llm_response)}")
        logger.debug("[IntentExtractor] Parsing unstructured LLM response")

        try:
            response_text = str(llm_response.content if hasattr(llm_response, 'content') else llm_response)
            logger.critical(f"🔍 [DIAGNOSTIC] Response text: {response_text[:200]}...")
            print("Response text:", response_text[:100])  # Debugging output
            import re

            # Extract task_type and action using patterns
            task_match = re.search(r'task_type["\']?\s*:\s*["\']?([^"\',\s]+)', response_text, re.IGNORECASE)
            action_match = re.search(r'action["\']?\s*:\s*["\']?([^"\',\s]+)', response_text, re.IGNORECASE)

            if task_match and action_match:
                logger.critical(
                    f"🔍 [DIAGNOSTIC] Extracted task_type: {task_match.group(1)}, action: {action_match.group(1)}")

                try:
                    intent = CarrierIntent(
                        task_type=task_match.group(1),
                        action=action_match.group(1),
                        entities=[],
                        field_requests=[],
                        confirmation_question="Please confirm this interpretation is correct.",
                        confidence_score=0.5  # Lower confidence for parsed responses
                    )
                    logger.critical(f"🔍 [DIAGNOSTIC] Successfully created CarrierIntent from unstructured response")
                    return intent
                except Exception as unstructured_error:
                    logger.critical(
                        f"🔍 [DIAGNOSTIC] Failed to create CarrierIntent from unstructured response: {unstructured_error}")

                    if "'NoneType' object is not iterable" in str(unstructured_error):
                        logger.critical(f"🎯 [DIAGNOSTIC] ITERATION ERROR in unstructured parsing!")

                    raise

            logger.critical(f"🔍 [DIAGNOSTIC] Could not extract task_type/action from unstructured response")
            logger.warning("[IntentExtractor] Could not parse unstructured response")
            return None

        except Exception as e:
            logger.critical(f"🔍 [DIAGNOSTIC] Error in unstructured parsing: {type(e).__name__}: {e}")
            logger.error(f"[IntentExtractor] Error parsing unstructured response: {e}")
            return None

    def _update_success_metrics(self, execution_time: float):
        """Update performance metrics on successful extraction"""
        logger.critical(f"🔍 [DIAGNOSTIC] Updating success metrics, execution_time: {execution_time}")

        self.extraction_metrics['successful_extractions'] += 1

        # Update rolling average execution time
        total_successful = self.extraction_metrics['successful_extractions']
        current_avg = self.extraction_metrics['average_execution_time']

        new_avg = ((current_avg * (total_successful - 1)) + execution_time) / total_successful
        self.extraction_metrics['average_execution_time'] = new_avg

        logger.critical(f"🔍 [DIAGNOSTIC] Metrics updated - success rate: {self.get_success_rate():.2%}")
        logger.debug(f"[IntentExtractor] Updated metrics - success rate: {self.get_success_rate():.2%}")

    def get_success_rate(self) -> float:
        """Calculate current success rate for monitoring"""
        total = self.extraction_metrics['total_attempts']
        if total == 0:
            return 0.0
        return self.extraction_metrics['successful_extractions'] / total

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for monitoring and debugging"""
        return {
            **self.extraction_metrics,
            'success_rate': self.get_success_rate(),
            'retry_rate': (self.extraction_metrics['retry_activations'] /
                           max(self.extraction_metrics['total_attempts'], 1))
        }

    def reset_metrics(self):
        """Reset metrics counters (useful for testing or periodic resets)"""
        logger.critical(f"🔍 [DIAGNOSTIC] Resetting performance metrics")
        logger.info("[IntentExtractor] Resetting performance metrics")
        self.extraction_metrics = {
            'total_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'retry_activations': 0,
            'average_execution_time': 0.0
        }


def create_performance_analytics_extractor(llm, custom_examples: Optional[Dict] = None) -> CarrierIntentExtractor:
    """
    Factory function to create optimized intent extractor for Performance Analytics
    """
    logger.critical(f"🔍 [DIAGNOSTIC] Creating performance analytics extractor")
    logger.critical(f"  LLM type: {type(llm)}")
    logger.critical(f"  Custom examples provided: {custom_examples is not None}")

    # Merge custom examples with defaults
    examples = {**PERFORMANCE_ANALYST_INTENT_EXAMPLES}
    if custom_examples:
        examples.update(custom_examples)
        logger.critical(f"🔍 [DIAGNOSTIC] Merged {len(custom_examples)} custom examples")

    try:
        extractor = CarrierIntentExtractor(
            llm=llm,
            intent_examples=examples,
            max_retries=1,
            timeout=30
        )
        logger.critical(f"🔍 [DIAGNOSTIC] Performance analytics extractor created successfully")
        return extractor
    except Exception as e:
        logger.critical(f"🔍 [DIAGNOSTIC] Failed to create extractor: {type(e).__name__}: {e}")

        if "'NoneType' object is not iterable" in str(e):
            logger.critical(f"🎯 [DIAGNOSTIC] ITERATION ERROR during extractor creation!")

        raise


def validate_intent_examples(examples: Dict) -> bool:
    """Validate intent examples structure for consistency"""
    logger.critical(f"🔍 [DIAGNOSTIC] Validating {len(examples)} intent examples")

    required_keys = ['task_type', 'action', 'user_message']

    for example_name, example_data in examples.items():
        logger.critical(f"🔍 [DIAGNOSTIC] Validating example '{example_name}': {example_data}")

        if not all(key in example_data for key in required_keys):
            logger.critical(f"🔍 [DIAGNOSTIC] Example '{example_name}' missing required keys")
            logger.error(f"[IntentExtractor] Invalid example '{example_name}': missing required keys")
            return False

    logger.critical(f"🔍 [DIAGNOSTIC] All {len(examples)} examples validated successfully")
    logger.info(f"[IntentExtractor] Validated {len(examples)} intent examples")
    return True


__all__ = [
    'CarrierIntent',
    'CarrierIntentExtractor',
    'create_performance_analytics_extractor',
    'validate_intent_examples',
    'with_timeout'
]
