from typing import Union, Dict, Any, Type, Optional, Iterator, Literal, get_args
from dapr_agents.llm.utils import StreamHandler, StructureHandler
from dataclasses import is_dataclass, asdict
from dapr_agents.types import ChatCompletion
from collections.abc import Iterable
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class ResponseHandler:
    """
    Handles the processing of responses from language models.
    """

    @staticmethod
    def process_response(
        response: Any,
        llm_provider: str,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["json", "function_call"] = "json",
        stream: bool = False,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the response from the language model.

        Args:
            response: The response object from the language model.
            llm_provider: The LLM provider (e.g., 'openai').
            response_format: A pydantic model to parse and validate the structured response.
            structured_mode: The mode of the structured response: 'json' or 'function_call'.
            stream: Whether the response is a stream.

        Returns:
            Union[Iterator[Dict[str, Any]], Dict[str, Any]]: The processed response.
        """
        if stream:
            return StreamHandler.process_stream(
                stream=response,
                llm_provider=llm_provider,
                response_format=response_format,
            )
        else:
            if response_format:
                structured_response_json = StructureHandler.extract_structured_response(
                    response=response, llm_provider=llm_provider, structured_mode=structured_mode
                )

                # Processing Iterable Models
                is_iterable = isinstance(response_format, Iterable)
                if is_iterable:
                    item_model = get_args(response_format)[0]
                    response_format = StructureHandler.create_iterable_model(item_model)
                
                # Validating Response
                structured_response_instance = StructureHandler.validate_response(
                    structured_response_json, response_format
                )

                if isinstance(structured_response_instance, response_format):
                    logger.info("Structured output was successfully validated.")
                    if is_iterable:
                        logger.debug(f"Returning objects from an instance of {type(structured_response_instance)}.")
                        return structured_response_instance.objects
                    else:
                        logger.debug(f"Returning an instance of {type(structured_response_instance)}.")
                        return structured_response_instance
                else:
                    logger.error("Validation failed for structured response.")

            # Convert response to dictionary
            if isinstance(response, dict):
                # Already a dictionary
                response_dict = response
            elif is_dataclass(response):
                # Dataclass instance
                response_dict = asdict(response)
            elif isinstance(response, BaseModel):
                # Pydantic object
                response_dict = response.model_dump()
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")

            completion = ChatCompletion(**response_dict)
            logger.debug(f"Chat completion response: {completion}")
            return completion