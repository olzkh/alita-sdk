import logging
from typing import Dict, List, Optional
from langchain_core.tools import BaseToolkit, BaseTool
from pydantic import create_model, BaseModel, ConfigDict, Field, SecretStr
from functools import lru_cache

from .api_wrapper import CarrierAPIWrapper
from .tools import ACTION_TOOL_MAP, CarrierIntentMetaTool
from .utils.intent_utils import CarrierIntentExtractor
from .utils.prompts import PERFORMANCE_ANALYST_INTENT_EXAMPLES as INTENT_EXAMPLES
from .metrics import ToolkitMetrics, toolkit_metrics

logger = logging.getLogger(__name__)
name = 'carrier'


class AlitaCarrierToolkit(BaseToolkit):
    """
    Intent-driven Performance Analytics toolkit using a single intelligent meta-tool.

    This toolkit implements enterprise patterns:
    - Single entry point for all operations (meta-tool pattern)
    - LLM-powered intent recognition for natural language commands
    - Robust parameter extraction with fallback mechanisms
    - Comprehensive metrics and monitoring
    - Circuit breaker pattern for reliability

    The toolkit routes user requests through natural language understanding,
    eliminating the need for users to know specific tool names or parameters.
    """
    tools: List[BaseTool] = []

    @classmethod
    @lru_cache(maxsize=1)  # Cache schema as it's static
    def toolkit_config_schema(cls) -> BaseModel:
        """
        Define configuration schema for the toolkit.

        Key differences from legacy approach:
        - Requires LLM for intent recognition (not optional)
        - Simpler configuration focused on connection details
        - No need to list individual tools (handled internally)
        """
        return create_model(
            name,
            url=(str, Field(description="Carrier Platform Base URL")),
            organization=(str, Field(description="Carrier Organization Name")),
            private_token=(SecretStr, Field(description="Carrier Platform Authentication Token")),
            project_id=(Optional[str], Field(None, description="Optional project ID for scoped operations")),
            llm=(object, Field(description="LLM instance required for intent recognition")),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Carrier Performance Analytics Toolkit",
                    "version": "2.1.0",
                    "icon_url": "carrier.svg",
                    "categories": ["testing", "analytics", "performance", "intent-driven"],
                    "production_ready": True,
                    "requires_llm": True,
                    "architecture": "meta-tool-pattern"
                }
            })
        )

    @classmethod
    def get_toolkit(
            cls,
            url: str,
            organization: str,
            private_token: SecretStr,
            llm: object,
            project_id: Optional[str] = None,
            **kwargs
    ) -> 'AlitaCarrierToolkit':
        """
        Factory method to create toolkit instance.

        Creates a single meta-tool that handles all operations through
        intent recognition and intelligent routing.

        Args:
            url: Carrier platform API endpoint
            organization: Organization identifier
            private_token: Authentication token
            llm: Language model for intent recognition (required)
            project_id: Optional project scope
            **kwargs: Additional configuration options

        Returns:
            Configured toolkit with meta-tool

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if llm is None:
            logger.critical("[AlitaCarrierToolkit] LLM is mandatory for intent recognition")
            raise ValueError("An LLM instance is required for the intent-driven Carrier toolkit")

        logger.info("[AlitaCarrierToolkit] Initializing v2.1.0 with intent-driven architecture")

        try:
            # Initialize shared API wrapper (DRY: single instance for all tools)
            carrier_api_wrapper = CarrierAPIWrapper(
                url=url,
                organization=organization,
                private_token=private_token,
                project_id=project_id,
                **kwargs
            )
            intent_extractor = CarrierIntentExtractor(
                llm=llm,
                intent_examples=INTENT_EXAMPLES,
                max_retries=1,
                timeout=15,
            )

            # Create the meta-tool (single entry point for all operations)
            meta_tool = CarrierIntentMetaTool(
                intent_extractor=intent_extractor,
                api_wrapper=carrier_api_wrapper,
                tool_map=ACTION_TOOL_MAP,
                metrics=toolkit_metrics,
                fallback_enabled=True,
                max_retries=1
            )

            # The toolkit contains only the meta-tool
            tools = [meta_tool]

            logger.info(
                f"[AlitaCarrierToolkit] Toolkit ready with {len(ACTION_TOOL_MAP)} actions "
                f"accessible through natural language interface"
            )

            return cls(tools=tools)

        except Exception as e:
            logger.exception("[AlitaCarrierToolkit] Critical initialization error")
            raise ValueError(f"Toolkit initialization failed: {str(e)}")

    def get_tools(self) -> List[BaseTool]:
        """
        Returns the list of tools (single meta-tool in this architecture).

        This method is called by the agent framework to get available tools.
        The meta-tool pattern means agents only see one tool but can access
        all functionality through natural language.
        """
        logger.info(f"[AlitaCarrierToolkit] Providing {len(self.tools)} meta-tool(s) to agent")

        # Log metrics periodically when tools are accessed
        if hasattr(self, '_access_count'):
            self._access_count += 1
            if self._access_count % 10 == 0:  # Log every 10 accesses
                toolkit_metrics.log_metrics()
        else:
            self._access_count = 1

        return self.tools


def get_tools(tool_config: Dict) -> List[BaseTool]:
    """
    Main entry point for the Alita framework to configure the toolkit.

    This function validates configuration and returns configured tools.
    It implements defensive programming with comprehensive validation
    and meaningful error messages.

    Args:
            Args:
        tool_config: Configuration dictionary with 'settings' key containing:
            - url: Carrier API endpoint
            - organization: Organization name
            - private_token: Authentication token
            - llm: Language model instance
            - project_id: Optional project ID

    Returns:
        List containing the configured meta-tool

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    logger.info("[CarrierToolkit] Configuring Performance Analytics toolkit")

    try:
        # Extract settings with defensive defaults
        settings = tool_config.get('settings', {})

        if not isinstance(settings, dict):
            raise ValueError("Configuration 'settings' must be a dictionary")

        # Validate required fields (DRY: centralized validation)
        required_fields = {
            'url': 'Carrier Platform Base URL',
            'organization': 'Organization Name',
            'private_token': 'Authentication Token',
            'llm': 'Language Model Instance'
        }

        missing_fields = []
        for field, description in required_fields.items():
            if not settings.get(field):
                missing_fields.append(f"{field} ({description})")

        if missing_fields:
            raise ValueError(
                f"Missing required configuration for Carrier toolkit:\n"
                f"{chr(10).join(f'  - {field}' for field in missing_fields)}"
            )

        # Validate URL format
        url = settings['url'].rstrip('/')  # Normalize URL
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {url}. Must start with http:// or https://")

        # Build configuration for toolkit factory
        toolkit_config = {
            'url': url,
            'organization': settings['organization'],
            'private_token': settings['private_token'],
            'llm': settings['llm'],
            'project_id': settings.get('project_id'),
        }

        # Add any additional settings that aren't in required fields
        for key, value in settings.items():
            if key not in toolkit_config and key not in ['settings']:
                toolkit_config[key] = value

        # Create toolkit using factory method
        toolkit = AlitaCarrierToolkit.get_toolkit(**toolkit_config)
        tools = toolkit.get_tools()

        logger.info(
            f"[CarrierToolkit] Successfully configured. "
            f"Agent can now process Performance Analytics requests in natural language."
        )

        # Log initial metrics
        toolkit_metrics.log_metrics()

        return tools

    except ValueError as ve:
        # Re-raise ValueError with original message
        logger.error(f"[CarrierToolkit] Configuration error: {ve}")
        raise ve

    except Exception as e:
        # Wrap unexpected errors with context
        logger.exception("[CarrierToolkit] Unexpected error during configuration")
        raise ValueError(f"Failed to configure Performance Analytics toolkit: {str(e)}")


# Public API exports
__all__ = [
    # Main toolkit class
    'AlitaCarrierToolkit',

    # Configuration function for Alita framework
    'get_tools',

    # Metrics for monitoring (from metrics module)
    'toolkit_metrics',
    'ToolkitMetrics',

    # Version info
    'name',
]

# Version marker for compatibility checking
__version__ = "2.1.0"