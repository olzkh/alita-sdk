import logging
import importlib
from copy import deepcopy as copy
from typing import Any, Optional
from langchain.agents import (
    AgentExecutor, create_openai_tools_agent,
    create_json_chat_agent)
from langgraph.store.base import BaseStore

from .agents.xml_chat import create_xml_chat_agent
from .langraph_agent import create_graph
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage
)
from langchain_core.prompts import MessagesPlaceholder
from .constants import REACT_ADDON, REACT_VARS, XML_ADDON
from .chat_message_template import Jinja2TemplatedChatMessagesTemplate
from ..tools.echo import EchoTool
from langchain_core.tools import BaseTool, ToolException

logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self,
                 alita: 'AlitaClient',
                 data: dict,
                 client: 'LLMLikeObject',
                 chat_history: list[BaseMessage] = [],
                 app_type: str = "openai",
                 tools: Optional[list] = [],
                 memory: Optional[Any] = None,
                 store: Optional[BaseStore] = None):

        self.client = copy(client)
        self.client.max_tokens = data['llm_settings']['max_tokens']
        self.client.temperature = data['llm_settings']['temperature']
        self.client.top_p = data['llm_settings']['top_p']
        self.client.top_k = data['llm_settings']['top_k']
        self.client.model_name = data['llm_settings']['model_name']
        self.client.integration_uid = data['llm_settings']['integration_uid']

        self.app_type = app_type
        self.memory = memory
        self.store = store

        logger.debug("Data for agent creation: %s", data)
        logger.info("App type: %s", app_type)

        model_type = data["llm_settings"]["indexer_config"]["ai_model"]
        model_params = data["llm_settings"]["indexer_config"]["ai_model_params"]
        #
        target_pkg, target_name = model_type.rsplit(".", 1)
        target_cls = getattr(
            importlib.import_module(target_pkg),
            target_name
        )
        self.client = target_cls(**model_params)
        # validate agents compatibility: non-pipeline agents cannot have pipelines as toolkits
        if app_type != "pipeline" and any(tool['agent_type'] == 'pipeline' for tool in data['tools']):
            raise ToolException("Non-pipeline agents cannot have pipelines as a toolkits. "
                                "Review toolkits configuration or use pipeline as master agent.")

        # configure memory store if memory tool is defined
        memory_tool = next((tool for tool in data['tools'] if tool['type'] == 'memory'), None)
        self._configure_store(memory_tool)
        
        # Lazy import to avoid circular dependency
        from ..toolkits.tools import get_tools
        self.tools = get_tools(data['tools'], alita_client=alita, llm=self.client, memory_store=self.store)
        if app_type == "pipeline":
            self.prompt = data['instructions']
        else:
            self.tools += tools
            messages = [SystemMessage(content=data['instructions'])]
            messages.append(MessagesPlaceholder("chat_history"))
            if app_type == "react":
                messages.append(HumanMessage(REACT_ADDON))
            elif app_type == "xml":
                messages.append(HumanMessage(XML_ADDON))
            elif app_type in ['openai', 'dial']:
                messages.append(HumanMessage("{{input}}"))
            messages.append(MessagesPlaceholder("agent_scratchpad"))
            variables = {}
            input_variables = []
            for variable in data['variables']:
                if variable['value'] != "":
                    variables[variable['name']] = variable['value']
                else:
                    input_variables.append(variable['name'])
            if app_type in ["react", "xml"]:
                input_variables = list(set(input_variables + REACT_VARS))

            if chat_history and isinstance(chat_history, list):
                messages.extend(chat_history)
            self.prompt = Jinja2TemplatedChatMessagesTemplate(messages=messages)
            if input_variables:
                self.prompt.input_variables = input_variables
            if variables:
                self.prompt.partial_variables = variables
            try:
                logger.info(f"Client was created with client setting: temperature - {self.client._get_model_default_parameters}")
            except Exception as e:
                logger.info(f"Client was created with client setting: temperature - {self.client.temperature} : {self.client.max_tokens}")

    def _configure_store(self, memory_tool: dict | None) -> None:
        """
        Configure the memory store based on a memory_tool definition.
        Only creates a new store if one does not already exist.
        """
        if not memory_tool or self.store is not None:
            return
        from .store_manager import get_manager
        conn_str = memory_tool.get('settings', {}).get('connection_string', '')
        store = get_manager().get_store(conn_str)
        self.store = store

    def runnable(self):
        if self.app_type == 'pipeline':
            return self.pipeline()
        elif self.app_type == 'openai':
            return self.getOpenAIToolsAgentExecutor()
        elif self.app_type == 'xml':
            return self.getXMLAgentExecutor()
        else:
            self.tools = [EchoTool()] + self.tools
            return self.getAgentExecutor()

    def _agent_executor(self, agent: Any):
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools,
                                                  verbose=True, handle_parsing_errors=True,
                                                  max_execution_time=None, return_intermediate_steps=True)

    def getAgentExecutor(self):
        # Exclude compiled graph runnables from simple tool agents
        simple_tools = [t for t in self.tools if isinstance(t, BaseTool)]
        agent = create_json_chat_agent(llm=self.client, tools=simple_tools, prompt=self.prompt)
        return self._agent_executor(agent)


    def getXMLAgentExecutor(self):
        # Exclude compiled graph runnables from simple tool agents
        simple_tools = [t for t in self.tools if isinstance(t, BaseTool)]
        agent = create_xml_chat_agent(llm=self.client, tools=simple_tools, prompt=self.prompt)
        return self._agent_executor(agent)

    def getOpenAIToolsAgentExecutor(self):
        # Exclude compiled graph runnables from simple tool agents
        simple_tools = [t for t in self.tools if isinstance(t, BaseTool)]
        agent = create_openai_tools_agent(llm=self.client, tools=simple_tools, prompt=self.prompt)
        return self._agent_executor(agent)

    def pipeline(self):
        memory = self.memory
        #
        if memory is None:
            from langgraph.checkpoint.memory import MemorySaver  # pylint: disable=E0401,C0415
            memory = MemorySaver()
        #
        agent = create_graph(
            client=self.client, tools=self.tools,
            yaml_schema=self.prompt, memory=memory
        )
        #
        return agent

    # This one is used only in Alita and OpenAI
    def apredict(self, messages: list[BaseMessage]):
        yield from self.client.ainvoke(messages)

    def predict(self, messages: list[BaseMessage]):
        return self.client.invoke(messages)
