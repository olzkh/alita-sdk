"""
Performance Analyst helper

Author: Karen Florykian
"""
import logging
import time
import json
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Type

from .prompts import validate_action_mapping

logger = logging.getLogger(__name__)

DISAMBIGUATION_PATTERNS = {
    "get_tests": {
        "trigger_keywords": ["get tests", "show tests", "list tests", "show me the tests"],
        "clarification_question": "Which type of tests would you like to see?",
        "options": [
            {
                "action": "get_backend_tests",
                "task_type": "backend_test_management",
                "description": "Backend performance tests (JMeter, Gatling)",
                "keywords": ["backend", "performance", "jmeter", "gatling", "api", "load"]
            },
            {
                "action": "get_ui_tests",
                "task_type": "ui_test_management",
                "description": "UI/Frontend tests (Lighthouse, Sitespeed)",
                "keywords": ["ui", "frontend", "lighthouse", "sitespeed", "web"]
            }
        ]
    },
    "get_reports": {
        "trigger_keywords": ["get reports", "show reports", "list reports"],
        "clarification_question": "Which type of reports would you like to see?",
        "options": [
            {
                "action": "get_reports",
                "task_type": "backend_analysis",
                "description": "Backend performance reports",
                "keywords": ["backend", "performance", "load"]
            },
            {
                "action": "get_ui_reports",
                "task_type": "ui_analysis",
                "description": "UI/Frontend performance reports",
                "keywords": ["ui", "frontend", "lighthouse"]
            }
        ]
    },
    "create_test": {
        "trigger_keywords": ["create test", "new test", "add test"],
        "clarification_question": "What type of test would you like to create?",
        "options": [
            {
                "action": "create_backend_test",
                "task_type": "test_management",
                "description": "Backend performance test (JMeter/Gatling)",
                "keywords": ["backend", "api", "load", "stress"]
            },
            {
                "action": "create_ui_test",
                "task_type": "ui_test_management",
                "description": "UI performance test (Lighthouse/Sitespeed)",
                "keywords": ["ui", "frontend", "page", "web"]
            }
        ]
    },
    "run_test": {
        "trigger_keywords": ["run test", "execute test", "start test"],
        "clarification_question": "Which test would you like to run? Please provide the test ID or specify the type (backend or UI).",
        "options": [
            {
                "action": "run_test",
                "task_type": "test_execution",
                "description": "Run a backend performance test",
                "keywords": ["backend", "performance"]
            },
            {
                "action": "run_ui_test",
                "task_type": "ui_test_execution",
                "description": "Run a UI performance test",
                "keywords": ["ui", "frontend"]
            }
        ]
    }
}


def detect_ambiguous_intent(user_message: str) -> Optional[Dict]:
    """
    Detects if a user request is ambiguous by checking for trigger phrases.
    """
    user_lower = user_message.lower()

    for intent_key, config in DISAMBIGUATION_PATTERNS.items():
        if any(phrase in user_lower for phrase in config.get("trigger_keywords", [])):
            logger.info(f"[DisambiguationDetector] Matched ambiguous intent '{intent_key}'")
            return {"resolved": False, **config}

    return None


class CarrierIntent(BaseModel):
    """
    Intent model with smart disambiguation capabilities.
    """
    task_type: str = Field(..., description="Primary task category")
    action: str = Field(..., description="Specific action to perform OR 'clarify' for ambiguous requests")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    tool_parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted tool parameters")

    # DISAMBIGUATION FIELDS
    is_ambiguous: bool = Field(default=False, description="True if request needs clarification")
    clarification_question: Optional[str] = Field(None, description="Question to ask user for clarification")
    disambiguation_options: List[Dict[str, Any]] = Field(default_factory=list,
                                                         description="Available options for ambiguous requests")

    # METADATA
    confidence_score: Optional[float] = Field(None, description="Confidence in interpretation (0-1)")

    @validator('disambiguation_options', pre=True)
    def ensure_disambiguation_options_is_list(cls, v):
        if v is None: return []
        return v

    @validator('entities', pre=True)
    def ensure_entities_is_list(cls, v):
        if v is None: return []
        return v

    @validator('tool_parameters', pre=True)
    def ensure_tool_parameters_is_dict(cls, v):
        if v is None: return {}
        return v

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

    def get_tool_parameters_for_schema(self, tool_schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        Map extracted tool parameters to the specific tool's schema requirements.
        This replaces the fragile ParameterExtractor.
        """
        if not hasattr(tool_schema, 'model_fields'):
            return {}

        mapped_params = {}

        # Direct parameter mapping first
        for field_name, field_info in tool_schema.model_fields.items():
            if field_name in self.tool_parameters:
                mapped_params[field_name] = self.tool_parameters[field_name]
                continue

        # Entity to parameter mapping for common cases
        entity_to_param_mapping = {
            'test_id': ['test_id', 'id'],
            'report_id': ['report_id', 'id'],
            'duration': ['duration', 'test_duration'],
            'users': ['users', 'user_count', 'concurrent_users'],
            'ramp_up': ['ramp_up', 'rampUp', 'ramp_up_time'],
            'test_name': ['test_name', 'name'],
            'description': ['description', 'desc']
        }

        for entity in self.entities:
            entity_type = entity.get('type')
            entity_value = entity.get('value')

            if entity_type in entity_to_param_mapping:
                for possible_param in entity_to_param_mapping[entity_type]:
                    if possible_param in tool_schema.model_fields and possible_param not in mapped_params:
                        mapped_params[possible_param] = entity_value
                        break

        logger.info(f"[CarrierIntent] Mapped parameters for {tool_schema.__name__}: {mapped_params}")
        return mapped_params


class CarrierIntentExtractor:
    """
    Intent extractor with built-in parameter extraction capabilities.
    """

    def __init__(self, llm, max_retries: int = 2, timeout: int = 30):
        """Initialize with parameter extraction support"""
        logger.info(f"[EnhancedExtractor] Initializing with parameter extraction support")

        self.llm = llm
        self.max_retries = max_retries
        self.timeout = timeout

        # Create structured LLM with enhanced error handling
        try:
            self.structured_llm = self.llm.with_structured_output(
                schema=CarrierIntent,
                method="json_mode",
                include_raw=False
            )
            logger.info("[EnhancedExtractor] Structured LLM created successfully")
        except Exception as e:
            logger.warning(f"[EnhancedExtractor] Structured output failed, using fallback: {e}")
            self.structured_llm = self.llm

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_extractions': 0,
            'disambiguation_requests': 0,
            'failed_extractions': 0,
            'average_response_time': 0.0,
            'parameter_extractions': 0
        }

    def extract_intent_with_parameters(self, user_message: str, tool_schema: Optional[Type[BaseModel]] = None,
                                       context: Optional[Dict] = None) -> Optional[CarrierIntent]:
        """
        Extract intent and parameters in a single LLM call with tool schema awareness.
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1

        logger.info(f"[EnhancedExtractor] Processing with schema-aware parameter extraction: '{user_message}'")
        if tool_schema:
            logger.info(f"[EnhancedExtractor] Using tool schema: {tool_schema.__name__}")

        try:
            # Build schema-aware prompt
            prompt = self._build_parameter_aware_prompt(user_message, tool_schema, context)

            # Execute extraction with retries
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"[EnhancedExtractor] Attempt {attempt + 1}/{self.max_retries}")

                    # Invoke structured LLM
                    result = self.structured_llm.invoke(prompt)

                    # Process result
                    intent = self._process_llm_result(result, user_message)

                    if intent and self._validate_intent(intent):
                        # Update metrics
                        response_time = time.time() - start_time
                        self._update_success_metrics(response_time, intent)

                        if intent.tool_parameters:
                            self.metrics['parameter_extractions'] += 1
                            logger.info(f"[EnhancedExtractor] Extracted parameters: {intent.tool_parameters}")

                        logger.info(f"[EnhancedExtractor] Success on attempt {attempt + 1}: {intent}")
                        return intent
                    else:
                        logger.warning(f"[EnhancedExtractor] Invalid intent on attempt {attempt + 1}")

                except Exception as e:
                    logger.warning(f"[EnhancedExtractor] Attempt {attempt + 1} failed: {e}")

                    if attempt < self.max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))

            # All attempts failed
            self.metrics['failed_extractions'] += 1
            logger.error(f"[EnhancedExtractor] All attempts failed for: '{user_message}'")
            return None

        except Exception as e:
            self.metrics['failed_extractions'] += 1
            logger.error(f"[EnhancedExtractor] Critical error: {e}")
            return None

    def extract_intent(self, user_message: str, context: Optional[Dict] = None) -> Optional[CarrierIntent]:
        """
        Backward compatibility method - calls the enhanced version without tool schema.
        """
        return self.extract_intent_with_parameters(user_message, None, context)

    def _build_parameter_aware_prompt(self, user_message: str, tool_schema: Optional[Type[BaseModel]],
                                      context: Optional[Dict]) -> str:
        """
        Use the enhanced prompt builder that's now tool-schema aware
        """
        # Import here to avoid circular imports
        from .prompts import build_performance_analyst_prompt

        # Use the enhanced prompt builder with tool schema
        return build_performance_analyst_prompt(user_message, context, tool_schema)

    def _process_llm_result(self, result: Any, user_message: str) -> Optional[CarrierIntent]:
        """
        Process LLM result with parameter extraction support.
        """
        logger.debug(f"[EnhancedExtractor] Processing result type: {type(result)}")

        try:
            if isinstance(result, CarrierIntent):
                logger.debug("[EnhancedExtractor] Result is already CarrierIntent")
                return result

            elif isinstance(result, dict):
                logger.debug("[EnhancedExtractor] Result is dict, creating CarrierIntent")
                return self._create_intent_from_dict(result)

            elif hasattr(result, 'content'):
                logger.debug("[EnhancedExtractor] Result has content, parsing JSON")
                return self._parse_content_to_intent(result.content)

            else:
                logger.warning(f"[EnhancedExtractor] Unexpected result type: {type(result)}")
                return self._fallback_parsing(user_message)

        except Exception as e:
            logger.error(f"[EnhancedExtractor] Error processing result: {e}")
            return self._fallback_parsing(user_message)

    def _create_intent_from_dict(self, data: Dict) -> Optional[CarrierIntent]:
        """
        Safely create CarrierIntent from dictionary with parameter support.
        """
        try:
            safe_data = {
                'task_type': data.get('task_type', 'unknown'),
                'action': data.get('action', 'unknown'),
                'entities': data.get('entities') or [],
                'tool_parameters': data.get('tool_parameters') or {},  # NEW
                'is_ambiguous': data.get('is_ambiguous', False),
                'clarification_question': data.get('clarification_question'),
                'disambiguation_options': data.get('disambiguation_options') or [],
                'confidence_score': data.get('confidence_score')
            }

            logger.debug(f"[EnhancedExtractor] Creating CarrierIntent with parameters: {safe_data}")
            return CarrierIntent(**safe_data)

        except Exception as e:
            logger.error(f"[EnhancedExtractor] Failed to create CarrierIntent from dict: {e}")
            logger.error(f"[EnhancedExtractor] Problem data: {data}")
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
                logger.warning(f"[EnhancedExtractor] Content is not string: {type(content)}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"[EnhancedExtractor] JSON parsing failed: {e}")
            return None

    def _fallback_parsing(self, user_message: str) -> Optional[CarrierIntent]:
        """
        Enhanced fallback parsing with parameter extraction.
        """
        logger.info(f"[EnhancedExtractor] Using fallback parsing for: '{user_message}'")

        # Check for disambiguation first
        disambiguation_info = detect_ambiguous_intent(user_message)

        if disambiguation_info and not disambiguation_info.get("resolved", False):
            logger.info("[EnhancedExtractor] Fallback detected ambiguous request")
            return CarrierIntent(
                task_type="disambiguation",
                action="clarify",
                entities=[],
                tool_parameters={},
                is_ambiguous=True,
                clarification_question=disambiguation_info["clarification_question"],
                disambiguation_options=disambiguation_info["options"],
                confidence_score=0.8
            )

        # Enhanced pattern matching with parameter extraction
        import re
        user_lower = user_message.lower()

        # Pattern: run test X with duration Y
        run_test_pattern = r'run.*test.*?(\d+)(?:.*?(?:duration|for).*?(\d+)\s*(sec|min|minute|s|m))?'
        match = re.search(run_test_pattern, user_lower)
        if match:
            test_id = match.group(1)
            tool_parameters = {"test_id": test_id}

            # Extract duration if present
            if match.group(2):
                duration_value = int(match.group(2))
                duration_unit = match.group(3) or 'sec'

                # Convert to seconds
                if duration_unit in ['min', 'minute', 'm']:
                    duration_seconds = duration_value * 60
                else:
                    duration_seconds = duration_value

                tool_parameters["duration"] = str(duration_seconds)

            entities = [{"type": "test_id", "value": test_id}]

            logger.info(f"[EnhancedExtractor] Fallback extracted parameters: {tool_parameters}")
            return CarrierIntent(
                task_type="test_execution",
                action="run_test",
                entities=entities,
                tool_parameters=tool_parameters,
                is_ambiguous=False,
                confidence_score=0.7
            )

        # Add more enhanced patterns as needed...

        logger.warning(f"[EnhancedExtractor] No fallback pattern matched for: '{user_message}'")
        return None

    def _validate_intent(self, intent: CarrierIntent) -> bool:
        """
        Enhanced validation with parameter validation.
        """
        if not intent:
            return False

        # Basic field validation
        if not intent.task_type or not intent.action:
            logger.warning("[EnhancedExtractor] Missing required fields")
            return False

        # Special validation for disambiguation
        if intent.needs_clarification():
            if not intent.clarification_question:
                logger.warning("[EnhancedExtractor] Disambiguation intent missing clarification question")
                return False
            logger.info("[EnhancedExtractor] Valid disambiguation intent")
            return True

        # Standard action validation
        if not validate_action_mapping(intent.task_type, intent.action):
            logger.warning(f"[EnhancedExtractor] Invalid action mapping: {intent.task_type} -> {intent.action}")
            return False

        # Confidence check
        if intent.confidence_score is not None and intent.confidence_score < 0.3:
            logger.warning(f"[EnhancedExtractor] Low confidence: {intent.confidence_score}")
            return False

        logger.debug("[EnhancedExtractor] Intent validation passed")
        return True

    def _update_success_metrics(self, response_time: float, intent: CarrierIntent):
        """
        Update performance metrics with parameter extraction tracking.
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
        Get comprehensive metrics including parameter extraction stats.
        """
        total = self.metrics['total_requests']
        return {
            **self.metrics,
            'success_rate': self.metrics['successful_extractions'] / max(total, 1),
            'disambiguation_rate': self.metrics['disambiguation_requests'] / max(total, 1),
            'failure_rate': self.metrics['failed_extractions'] / max(total, 1),
            'parameter_extraction_rate': self.metrics['parameter_extractions'] / max(
                self.metrics['successful_extractions'], 1)
        }

    def reset_metrics(self):
        """Reset all performance metrics"""
        logger.info("[EnhancedExtractor] Resetting performance metrics")
        self.metrics = {
            'total_requests': 0,
            'successful_extractions': 0,
            'disambiguation_requests': 0,
            'failed_extractions': 0,
            'average_response_time': 0.0,
            'parameter_extractions': 0
        }


def create_performance_analytics_extractor(llm) -> CarrierIntentExtractor:
    """
    Factory function to create the enhanced intent extractor with parameter extraction.
    """
    logger.info("[Factory] Creating enhanced intent extractor with parameter extraction support")

    try:
        extractor = CarrierIntentExtractor(
            llm=llm,
            max_retries=2,
            timeout=30
        )
        logger.info("[Factory] Enhanced extractor created successfully")
        return extractor

    except Exception as e:
        logger.error(f"[Factory] Failed to create enhanced extractor: {e}")
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
]
