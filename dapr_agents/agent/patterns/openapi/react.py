from dapr_agents.tool.utils.openapi import OpenAPISpecParser, openapi_spec_to_openai_fn
from dapr_agents.agent.patterns.react import ReActAgent
from dapr_agents.storage import VectorStoreBase
from dapr_agents.tool.storage import VectorToolStore
from typing import Dict, Optional, List, Any
from pydantic import Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

class OpenAPIReActAgent(ReActAgent):
    """
    Extends ReActAgent with OpenAPI handling capabilities, including tools for managing API calls.
    """

    role: str = Field(default="OpenAPI Expert", description="The agent's role in the interaction.")
    goal: str = Field(
        default="Help users work with OpenAPI specifications and API integrations.",
        description="The main objective of the agent."
    )
    instructions: List[str] = Field(
        default=[
            "You are an expert assistant specialized in working with OpenAPI specifications and API integrations.",
            "Your goal is to help users identify the correct API endpoints and execute API calls efficiently and accurately.",
            "You must first help users explore potential APIs by analyzing OpenAPI definitions, then assist in making authenticated API requests.",
            "Ensure that all API calls are executed with the correct parameters, authentication, and methods.",
            "Your responses should be concise, clear, and focus on guiding the user through the steps of working with APIs, including retrieving API definitions, understanding endpoint parameters, and handling errors.",
            "You only respond to questions directly related to your role."
        ],
        description="Instructions to guide the agent's behavior."
    )
    spec_parser: OpenAPISpecParser = Field(..., description="Parser for handling OpenAPI specifications.")
    api_vector_store: VectorStoreBase = Field(..., description="Vector store for storing API definitions.")
    auth_header: Optional[Dict] = Field(None, description="Authentication headers for executing API calls.")
    
    tool_vector_store: Optional[VectorToolStore] = Field(default=None, init=False, description="Internal vector store for OpenAPI tools.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization setup for OpenAPIReActAgent, including vector stores and OpenAPI tools.
        """
        logger.info("Setting up VectorToolStore for OpenAPIReActAgent...")

        # Initialize tool vector store using the api_vector_store
        self.tool_vector_store = VectorToolStore(vector_store=self.api_vector_store)

        # Load OpenAPI specifications into the tool vector store
        function_list = openapi_spec_to_openai_fn(self.spec_parser)
        self.tool_vector_store.add_tools(function_list)

        # Generate OpenAPI-specific tools
        from .tools import generate_api_call_executor, generate_get_openapi_definition
        openapi_tools = [
            generate_get_openapi_definition(self.tool_vector_store),
            generate_api_call_executor(self.spec_parser, self.auth_header)
        ]

        # Extend tools with OpenAPI tools
        self.tools.extend(openapi_tools)

        # Call parent model_post_init for additional setup
        super().model_post_init(__context)