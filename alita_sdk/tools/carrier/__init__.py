"""
Carrier

Author: Karen Florykian
"""
import logging
from typing import Dict, List, Optional
from functools import lru_cache

from langchain_core.tools import BaseToolkit, BaseTool
from pydantic import create_model, BaseModel, ConfigDict, Field, SecretStr

from .api_wrapper import CarrierAPIWrapper
from .tools import ACTION_TOOL_MAP, CarrierIntentMetaTool, CarrierOrchestrationEngine
from .metrics import ToolkitMetrics, toolkit_metrics

logger = logging.getLogger(__name__)
name = 'carrier'


class AlitaCarrierToolkit(BaseToolkit):
    """
    An intelligent, conversational Performance Analytics toolkit.

    This toolkit's architecture is built on modern AI patterns:
    - **Orchestration Engine:** A central "brain" handles complex logic.
    - **Adapter Pattern:** A `BaseTool` acts as a bridge to the LangChain framework.
    - **Stateful Disambiguation:** The toolkit asks for clarification when requests are
      ambiguous, enabling natural, multi-turn conversations.
    - **Single Entry Point:** The agent interacts with a single, powerful meta-tool.
    """
    tools: List[BaseTool] = []

    @classmethod
    @lru_cache(maxsize=1)
    def toolkit_config_schema(cls) -> BaseModel:
        """
        Defines the configuration schema for the toolkit.
        Requires an LLM for its conversational and intent-recognition capabilities.
        """
        return create_model(
            name,
            url=(str, Field(description="Carrier Platform Base URL")),
            organization=(str, Field(description="Carrier Organization Name")),
            private_token=(SecretStr, Field(description="Carrier Platform Authentication Token")),
            project_id=(Optional[str], Field(None, description="Optional project ID for scoped operations")),
            llm=(object, Field(description="LLM instance required for intent recognition and conversation.")),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Carrier Performance Analytics Toolkit",
                    "version": "3.0.0",  # Version bump to reflect new architecture
                    "icon_url": "carrier.svg",
                    "categories": ["testing", "analytics", "performance", "conversational"],
                    "production_ready": True,
                    "requires_llm": True,
                    "architecture": "orchestration-engine-adapter-pattern"
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
        Factory method to create the fully configured toolkit instance.

        This method constructs the core components:
        1. A shared API wrapper for all tool operations.
        2. The powerful `CarrierOrchestrationEngine` that contains the core logic.
        3. The `CarrierIntentMetaTool` which acts as an adapter to the LangChain framework.
        """
        if llm is None:
            logger.critical("[AlitaCarrierToolkit] LLM is mandatory for the conversational engine.")
            raise ValueError("An LLM instance is required for the Carrier toolkit.")

        logger.info("[AlitaCarrierToolkit] Initializing v3.0.0 with conversational orchestration engine.")

        try:
            # 1. Initialize the shared API wrapper.
            carrier_api_wrapper = CarrierAPIWrapper(
                url=url,
                organization=organization,
                private_token=private_token,
                project_id=project_id,
                **kwargs
            )

            # 2. Instantiate the powerful OrchestrationEngine.
            # This is the "brain" that handles the actual work.
            orchestration_engine = CarrierOrchestrationEngine(
                llm=llm,
                api_wrapper=carrier_api_wrapper,
                tool_class_map=ACTION_TOOL_MAP
            )

            # 3. Create the meta-tool adapter and pass the engine to it.
            # This is the single tool the LangChain agent will see.
            meta_tool = CarrierIntentMetaTool(
                orchestration_engine=orchestration_engine
            )

            # The toolkit contains only the single meta-tool adapter.
            tools = [meta_tool]

            logger.info(
                f"[AlitaCarrierToolkit] Toolkit ready. {len(ACTION_TOOL_MAP)} actions are available "
                f"through the intelligent orchestration engine."
            )

            return cls(tools=tools)

        except Exception as e:
            logger.exception("[AlitaCarrierToolkit] Critical initialization error.")
            raise ValueError(f"Toolkit initialization failed: {str(e)}")

    def get_tools(self) -> List[BaseTool]:
        """
        Returns the list of tools for the agent to use.
        In this architecture, it's always a single, powerful meta-tool.
        """
        logger.info(f"[AlitaCarrierToolkit] Providing {len(self.tools)} meta-tool(s) to agent.")
        return self.tools


def get_tools(tool_config: Dict) -> List[BaseTool]:
    """
    Main entry point for the Alita framework to configure and get the toolkit.

    This function validates the provided configuration and uses the factory
    method to construct and return the toolkit's tools.
    """
    logger.info("[CarrierToolkit] Configuring Performance Analytics toolkit.")

    try:
        settings = tool_config.get('settings', {})
        if not isinstance(settings, dict):
            raise ValueError("Configuration 'settings' must be a dictionary.")

        # Centralized validation for required fields.
        required_fields = {
            'url': 'Carrier Platform Base URL',
            'organization': 'Organization Name',
            'private_token': 'Authentication Token',
            'llm': 'Language Model Instance'
        }
        missing_fields = [desc for field, desc in required_fields.items() if not settings.get(field)]
        if missing_fields:
            raise ValueError(
                f"Missing required configuration for Carrier toolkit:\n"
                f"{chr(10).join(f'  - {field}' for field in missing_fields)}"
            )

        # Normalize and validate URL.
        url = settings['url'].rstrip('/')
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {url}. Must start with http:// or https://.")

        # Build the configuration for the toolkit factory.
        toolkit_config = {
            'url': url,
            'organization': settings['organization'],
            'private_token': settings['private_token'],
            'llm': settings['llm'],
            'project_id': settings.get('project_id'),
            **{k: v for k, v in settings.items() if k not in required_fields and k != 'settings'}
        }

        # Create toolkit using the class factory method.
        toolkit = AlitaCarrierToolkit.get_toolkit(**toolkit_config)
        tools = toolkit.get_tools()

        logger.info(
            "[CarrierToolkit] Successfully configured. "
            "Agent can now process Performance Analytics requests conversationally."
        )

        # Log initial metrics upon successful setup.
        toolkit_metrics.log_metrics()

        return tools

    except ValueError as ve:
        logger.error(f"[CarrierToolkit] Configuration error: {ve}")
        raise ve
    except Exception as e:
        logger.exception("[CarrierToolkit] Unexpected error during configuration.")
        raise ValueError(f"Failed to configure Performance Analytics toolkit: {str(e)}")


# Public API exports
__all__ = [
    'AlitaCarrierToolkit',
    'get_tools',
    'toolkit_metrics',
    'ToolkitMetrics',
    'name',
]

# Version marker for compatibility checking
__version__ = "3.0.0"
