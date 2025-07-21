import logging
from typing import Dict, Any, Type, Optional, List, get_type_hints
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.tools import BaseTool

from alita_sdk.tools.carrier.utils.intent_utils import CarrierIntent

logger = logging.getLogger(__name__)


class ParameterMappingStrategy(ABC):
    """Abstract base for parameter mapping strategies (SOLID: Open/Closed)"""

    @abstractmethod
    def can_handle(self, field_name: str, field_type: Any) -> bool:
        """Check if this strategy can handle the field"""
        pass

    @abstractmethod
    def map_parameters(self, field_name: str, intent: Any) -> Optional[Any]:
        """Map intent data to parameter value"""
        pass


class IDMappingStrategy(ParameterMappingStrategy):
    """Maps various ID fields from intent entities."""

    def can_handle(self, field_name: str, field_type: Any) -> bool:
        return 'id' in field_name.lower()

    def map_parameters(self, field_name: str, intent: Any) -> Optional[Any]:
        # SIMPLIFIED: No longer need to check for hasattr, as the CarrierIntent
        # validator guarantees 'entities' is a dictionary.
        entities = intent.entities

        # Direct match from entities dictionary
        if field_name in entities:
            return entities[field_name]

        # Common ID mappings (DRY: centralized mapping logic)
        id_mappings = {
            'report_id': ['id', 'report', 'test_id', 'build_id'],
            'test_id': ['id', 'test', 'build_id'],
            'ticket_id': ['id', 'ticket'],
            'build_id': ['id', 'build'],
        }

        # SIMPLIFIED: Logic now iterates through mapping keys, which is more direct.
        if field_name in id_mappings:
            for key in id_mappings[field_name]:
                if key in entities:
                    logger.info(f"Mapped entity '{key}' to parameter '{field_name}': {entities[key]}")
                    return entities[key]
        return None


class TimeRangeMappingStrategy(ParameterMappingStrategy):
    """Maps time-related fields"""

    def can_handle(self, field_name: str, field_type: Any) -> bool:
        time_fields = ['days', 'hours', 'start_date', 'end_date', 'time_range', 'period']
        return any(tf in field_name.lower() for tf in time_fields)

    def map_parameters(self, field_name: str, intent: Any) -> Optional[Any]:
        if not hasattr(intent, 'entities') or not intent.entities:
            return None

        entities = intent.entities

        # Direct match
        if field_name in entities:
            return entities[field_name]

        # Time mappings
        if 'days' in field_name and 'time_range' in entities:
            # Extract days from phrases like "last 7 days"
            time_range = str(entities['time_range']).lower()
            import re
            match = re.search(r'(\d+)\s*day', time_range)
            if match:
                return int(match.group(1))

        return None


class DefaultValueStrategy(ParameterMappingStrategy):
    """Uses field defaults from schema"""

    def can_handle(self, field_name: str, field_type: Any) -> bool:
        return True  # Can handle any field as last resort

    def map_parameters(self, field_name: str, intent: Any) -> Optional[Any]:
        # This strategy doesn't map, just returns None to use defaults
        return None


class ParameterExtractor:
    """
    Centralized parameter extraction using Strategy pattern.
    Single Responsibility: Only handles parameter extraction from intents.
    """

    def __init__(self):
        # Initialize strategies in order of precedence
        self.strategies: List[ParameterMappingStrategy] = [
            IDMappingStrategy(),
            TimeRangeMappingStrategy(),
            DefaultValueStrategy(),
        ]
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

    def get_expected_args(self, tool_class: Type[BaseTool]) -> Dict[str, Any]:
        """
        Extract expected arguments from tool's args_schema.
        Handles inheritance and various schema access patterns.
        """
        class_name = tool_class.__name__

        if class_name in self._schema_cache:
            logger.info(f"✅ [ParameterExtractor] Using cached schema for {class_name}")
            return self._schema_cache[class_name]

        # Initialize args_schema to None
        args_schema = None

        # Method 1: Direct class attribute (check __class__ attributes)
        try:
            # Check class dict directly
            if 'args_schema' in tool_class.__dict__:
                args_schema = tool_class.__dict__['args_schema']
                logger.info(f"✅ [ParameterExtractor] Found args_schema in __dict__ for {class_name}")
            elif hasattr(tool_class, 'args_schema'):
                args_schema = getattr(tool_class, 'args_schema')
                logger.info(f"✅ [ParameterExtractor] Found args_schema via getattr for {class_name}")
        except Exception as e:
            logger.warning(f"⚠️ [ParameterExtractor] Failed to access args_schema directly: {e}")

        # Method 2: Check for model_fields (Pydantic v2 BaseModel)
        if not args_schema and hasattr(tool_class, 'model_fields'):
            # For tools using Pydantic BaseModel directly
            logger.info(f"🔍 [ParameterExtractor] Checking model_fields for {class_name}")
            # Look for ProcessReportInput or similar input models
            for attr_name in dir(tool_class):
                if 'Input' in attr_name and not attr_name.startswith('_'):
                    try:
                        potential_schema = getattr(tool_class, attr_name)
                        if isinstance(potential_schema, type) and issubclass(potential_schema, BaseModel):
                            args_schema = potential_schema
                            logger.info(f"✅ [ParameterExtractor] Found schema class {attr_name}")
                            break
                    except:
                        pass

        # Method 3: Check parent classes
        if not args_schema:
            logger.info(f"🔄 [ParameterExtractor] Checking MRO for {class_name}")
            for base_class in tool_class.__mro__[1:]:  # Skip the class itself
                if hasattr(base_class, 'args_schema'):
                    try:
                        args_schema = getattr(base_class, 'args_schema')
                        if args_schema:  # Make sure it's not None
                            logger.info(f"✅ [ParameterExtractor] Found args_schema in parent {base_class.__name__}")
                            break
                    except Exception as e:
                        logger.info(f"⚠️ [ParameterExtractor] Failed to access from {base_class.__name__}: {e}")

        # Extract fields from schema
        expected_args = {}

        # Handle Pydantic BaseModel schemas
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            logger.info(f"📋 [ParameterExtractor] Processing Pydantic schema: {args_schema}")

            logger.info(f"📋 [ParameterExtractor] Processing Pydantic schema: {args_schema.__name__}")

            # For Pydantic v2
            if hasattr(args_schema, 'model_fields'):
                fields = args_schema.model_fields
                for field_name, field_info in fields.items():
                    expected_args[field_name] = {
                        'type': field_info.annotation,
                        'required': field_info.is_required(),
                        'default': field_info.default if hasattr(field_info, 'default') else None,
                        'description': field_info.description if hasattr(field_info, 'description') else None
                    }
                    logger.info(
                        f"  📌 Field: {field_name} (required={field_info.is_required()}, type={field_info.annotation})")
                    # For Pydantic v1
            elif hasattr(args_schema, '__fields__'):
                fields = args_schema.__fields__
                for field_name, field_info in fields.items():
                    expected_args[field_name] = {
                        'type': field_info.type_,
                        'required': field_info.required,
                        'default': field_info.default if field_info.default is not None else None,
                        'description': field_info.field_info.description if hasattr(field_info,
                                                                                    'field_info') else None
                    }
                    logger.info(
                        f"  📌 Field: {field_name} (required={field_info.required}, type={field_info.type_})")
        # Cache the results
        self._schema_cache[class_name] = expected_args
        logger.info(f"✅ [ParameterExtractor] Successfully extracted {len(expected_args)} args for {class_name}")

        return expected_args

    def extract_parameters(self, intent: Any, tool_class: Type[BaseTool]) -> Dict[str, Any]:
        """
        Main extraction method that coordinates strategies.
        Returns parameters ready for tool execution.
        """
        logger.info(f"🎯 [ParameterExtractor] Extracting parameters for {tool_class.__name__}")
        logger.info(f"🎯 [ParameterExtractor] Extracting parameters for {intent}")
        logger.info(f"  Intent type: {intent.task_type}, action: {intent.action}")
        logger.info(f"  Intent entities: {getattr(intent, 'entities', {})}")

        expected_args = self.get_expected_args(tool_class)
        if not expected_args:
            logger.warning(f"⚠️ [ParameterExtractor] No expected args found for {tool_class.__name__}")
            return {}

        extracted_params = {}
        missing_required = []

        # Process each expected argument
        for arg_name, arg_info in expected_args.items():
            logger.info(f"🔍 [ParameterExtractor] Processing arg: {arg_name}")

            value = None
            strategy_used = None

            # Try each strategy in order
            for strategy in self.strategies:
                if strategy.can_handle(arg_name, arg_info.get('type')):
                    try:
                        value = strategy.map_parameters(arg_name, intent)
                        if value is not None:
                            strategy_used = strategy.__class__.__name__
                            logger.info(f"  ✅ {strategy_used} provided value: {value}")
                            break
                    except Exception as e:
                        logger.warning(f"  ⚠️ {strategy.__class__.__name__} failed: {e}")

            # Handle the extracted value
            if value is not None:
                extracted_params[arg_name] = value
                logger.info(f"  ✅ [ParameterExtractor] Mapped {arg_name} = {value} using {strategy_used}")
            elif arg_info.get('default') is not None:
                # Use default value
                extracted_params[arg_name] = arg_info['default']
                logger.info(f"  📌 [ParameterExtractor] Using default for {arg_name} = {arg_info['default']}")
            elif arg_info.get('required', True):
                # Required field is missing
                missing_required.append(arg_name)
                logger.warning(f"  ❌ [ParameterExtractor] Missing required parameter: {arg_name}")

        # Log extraction summary
        if missing_required:
            logger.error(
                f"❌ [ParameterExtractor] Missing required fields for {tool_class.__name__}: {missing_required}")
            # Try to provide helpful context
            if hasattr(intent, 'entities'):
                logger.error(f"   Available entities: {list(intent.entities.keys())}")
            logger.error(f"   Extracted params: {list(extracted_params.keys())}")
        else:
            logger.info(f"✅ [ParameterExtractor] Successfully extracted all parameters for {tool_class.__name__}")

        return extracted_params

    # alita_sdk/tools/carrier/parameter_extractor.py

    def _get_expected_args(self, tool) -> Dict[str, Any]:
        """Get expected arguments - check args_schema first"""

        # PRIORITY 1: Check args_schema (this is where tool input parameters are defined)
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema_obj = tool.args_schema

            # Check for Pydantic v2 model_fields
            if hasattr(schema_obj, 'model_fields'):
                logger.critical(f"✅ Found args_schema.model_fields: {list(schema_obj.model_fields.keys())}")
                return schema_obj.model_fields

            # Check for Pydantic v1 __fields__
            if hasattr(schema_obj, '__fields__'):
                logger.critical(f"✅ Found args_schema.__fields__: {list(schema_obj.__fields__.keys())}")
                return schema_obj.__fields__

            # If args_schema is a class, instantiate it
            if isinstance(schema_obj, type):
                try:
                    instance = schema_obj()
                    if hasattr(instance, 'model_fields'):
                        logger.critical(
                            f"✅ Found instantiated args_schema.model_fields: {list(instance.model_fields.keys())}")
                        return instance.model_fields
                except Exception as e:
                    logger.warning(f"Could not instantiate args_schema: {e}")

        # PRIORITY 2: Check tool's _run method signature
        if hasattr(tool, '_run'):
            import inspect
            sig = inspect.signature(tool._run)
            params = {name: param for name, param in sig.parameters.items()
                      if name not in ['self', 'kwargs']}
            if params:
                logger.critical(f"✅ Found _run parameters: {list(params.keys())}")
                return params

        logger.critical(f"❌ No input schema found for {tool.__class__.__name__}")
        return {}

    def extract_parameters_from_intent(self, tool, intent: CarrierIntent) -> Dict[str, Any]:
        """Extract parameters focusing on tool input schema"""

        logger.critical(f"🔍 [ParameterExtractor] Tool: {tool.__class__.__name__}")
        logger.critical(f"🔍 [ParameterExtractor] Tool.args_schema: {getattr(tool, 'args_schema', 'None')}")

        expected_args = self._get_expected_args(tool)
        logger.critical(f"🔍 Expected args: {list(expected_args.keys())}")
        logger.critical(f"🔍 Available entities: {intent.entities}")

        if not expected_args:
            return {}

        extracted_params = {}

        # Enhanced entity mapping
        for entity in (intent.entities or []):
            entity_type = entity.get('type', '')
            entity_value = entity.get('value', '')

            # Common parameter name patterns
            param_patterns = {
                'test_id': ['test_id', 'id', 'test_name', 'name'],
                'report_id': ['report_id', 'id', 'name'],
                'build_id': ['build_id', 'id', 'name'],
                'board_id': ['board_id', 'id']
            }

            # Try exact matches first
            if entity_type in expected_args:
                extracted_params[entity_type] = entity_value
                logger.critical(f"✅ Direct match: {entity_type}={entity_value}")
                continue

            # Try pattern matching
            patterns = param_patterns.get(entity_type, [entity_type])
            for pattern in patterns:
                for param_name in expected_args:
                    if pattern.lower() in param_name.lower():
                        extracted_params[param_name] = entity_value
                        logger.critical(f"✅ Pattern match: {entity_type}={entity_value} → {param_name}")
                        break
                if param_name in extracted_params:
                    break

        logger.critical(f"🎯 Final extracted params: {extracted_params}")
        return extracted_params




_parameter_extractor = ParameterExtractor()


def extract_parameters_leg(intent: Any, tool_class: Type[BaseTool]) -> Dict[str, Any]:
    """
    Public API for parameter extraction.
    Uses singleton instance for efficiency.
    """
    return _parameter_extractor.extract_parameters(intent, tool_class)

def extract_parameters(tool, intent: CarrierIntent) -> Dict[str, Any]:
    """
    Public API for parameter extraction.
    Matches the existing method signature.
    """
    return _parameter_extractor.extract_parameters_from_intent(tool, intent)