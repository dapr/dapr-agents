from openai.types.create_embedding_response import CreateEmbeddingResponse
from dapr_agents.llm.openai.client.base import OpenAIClientBase
from typing import Union, Dict, Any, Literal, List, Optional
from pydantic import Field, model_validator
import logging

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(OpenAIClientBase):
    """
    Client for handling OpenAI's embedding functionalities, supporting both OpenAI and Azure OpenAI configurations.

    Attributes:
        model (str): The ID of the model to use for embedding. Defaults to `text-embedding-ada-002` if not specified.
        encoding_format (Optional[Literal["float", "base64"]]): The format of the embeddings. Defaults to 'float'.
        dimensions (Optional[int]): Number of dimensions for the output embeddings. Only supported in specific models like `text-embedding-3`.
        user (Optional[str]): A unique identifier representing the end-user.
    """

    model: str = Field(
        default=None, description="ID of the model to use for embedding."
    )
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        "float", description="Format for the embeddings. Defaults to 'float'."
    )
    dimensions: Optional[int] = Field(
        None,
        description="Number of dimensions for the output embeddings. Supported in text-embedding-3 and later models.",
    )
    user: Optional[str] = Field(
        None, description="Unique identifier representing the end-user."
    )

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures that the 'model' attribute is set during validation.
        If 'model' is not provided, defaults to 'text-embedding-ada-002' or uses 'azure_deployment' if available.

        Args:
            values (Dict[str, Any]): Dictionary of model attributes to validate and initialize.

        Returns:
            Dict[str, Any]: Updated dictionary of validated attributes.
        """
        if "model" not in values or values["model"] is None:
            values["model"] = values.get("azure_deployment", "text-embedding-ada-002")
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization setup for private attributes.

        This method configures the API endpoint for embedding operations.

        Args:
            __context (Any): Context provided during model initialization.
        """
        self._api = "embeddings"
        return super().model_post_init(__context)

    def create_embedding(
        self,
        input: Union[str, List[Union[str, List[int]]]],
        model: Optional[str] = None,
    ) -> CreateEmbeddingResponse:
        """
        Generate embeddings for the given input text(s).

        Args:
            input (Union[str, List[Union[str, List[int]]]]): Input text(s) or tokenized input(s) to generate embeddings for.
                - A single string for one input.
                - A list of strings or tokenized lists for multiple inputs.
            model (Optional[str]): Model to use for embedding. Overrides the default model if provided.

        Returns:
            CreateEmbeddingResponse: A response object containing the generated embeddings and associated metadata.

        Raises:
            ValueError: If the client fails to generate embeddings.
        """
        logger.info(f"Using model '{self.model}' for embedding generation.")

        # If a model is provided, override the default model
        model = model or self.model

        response = self.client.embeddings.create(
            model=model,
            input=input,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            user=self.user,
        )
        return response
