from openai.types.create_embedding_response import CreateEmbeddingResponse
from dapr_agents.llm.nvidia.client import NVIDIAClientBase
from typing import Union, Dict, Any, Literal, List, Optional
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class NVIDIAEmbeddingClient(NVIDIAClientBase):
    """
    Client for handling NVIDIA's embedding functionalities.

    Attributes:
        model (str): The ID of the model to use for embedding. Required for NVIDIA embeddings.
        encoding_format (Optional[Literal["float", "base64"]]): The format of the embeddings. Defaults to 'float'.
        dimensions (Optional[int]): Number of dimensions for the output embeddings. Not supported by all models.
        input_type (Optional[Literal["query", "passage"]]): Specifies the mode of operation for embeddings.
            'query' for generating embeddings during querying.
            'passage' for generating embeddings during indexing.
        truncate (Optional[Literal["NONE", "START", "END"]]): Specifies handling for inputs exceeding the model's max token length. Defaults to 'NONE'.
    """

    model: str = Field(
        "nvidia/nv-embedqa-e5-v5", description="ID of the model to use for embedding."
    )
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        "float", description="Format for the embeddings. Defaults to 'float'."
    )
    dimensions: Optional[int] = Field(
        None,
        description="Number of dimensions for the output embeddings. Not supported by all models.",
    )
    input_type: Optional[Literal["query", "passage"]] = Field(
        "passage", description="Mode of operation: 'query' or 'passage'."
    )
    truncate: Optional[Literal["NONE", "START", "END"]] = Field(
        "NONE",
        description="Handling for inputs exceeding max token length. Defaults to 'NONE'.",
    )

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
        input: Union[str, List[str]],
        model: Optional[str] = None,
        input_type: Optional[Literal["query", "passage"]] = None,
        truncate: Optional[Literal["NONE", "START", "END"]] = None,
        encoding_format: Optional[Literal["float", "base64"]] = None,
        dimensions: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> CreateEmbeddingResponse:
        """
        Generate embeddings for the given input text(s).

        Args:
            input (Union[str, List[str]]): Input text(s) to generate embeddings for.
                - A single string for one input.
                - A list of strings for multiple inputs.
            model (Optional[str]): Model to use for embedding. Overrides the default model if provided.
            input_type (Optional[Literal["query", "passage"]]): Specifies the mode of operation. Overrides the default if provided.
            truncate (Optional[Literal["NONE", "START", "END"]]): Handling for inputs exceeding max token length.
            encoding_format (Optional[Literal["float", "base64"]]): Format for the embeddings. Defaults to the instance setting.
            dimensions (Optional[int]): Number of dimensions for the embeddings. Only supported by certain models.
            extra_body (Optional[Dict[str, Any]]): Additional parameters to pass in the request body.

        Returns:
            Dict[str, Any]: A response object containing the generated embeddings and associated metadata.

        Raises:
            ValueError: If the client fails to generate embeddings.
        """
        logger.info(f"Using model '{self.model}' for embedding generation.")

        # If a model is provided, override the default model
        model = model or self.model

        # Prepare request parameters
        body = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format or self.encoding_format,
            "extra_body": extra_body or {},
        }

        # Add optional parameters if provided
        if input_type:
            body["extra_body"]["input_type"] = input_type
        if truncate:
            body["extra_body"]["truncate"] = truncate
        if dimensions:
            body["dimensions"] = dimensions

        logger.debug(f"Embedding request payload: {body}")

        # Send the request to the NVIDIA embeddings endpoint
        try:
            response = self.client.embeddings.create(**body)
            logger.info("Embedding generation successful.")
            return response
        except Exception as e:
            logger.error(f"An error occurred while generating embeddings: {e}")
            raise ValueError(f"Failed to generate embeddings: {e}")
