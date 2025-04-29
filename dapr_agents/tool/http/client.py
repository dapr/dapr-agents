from typing import Dict, Optional, Set, Any, Union
import logging
import requests

from pydantic import BaseModel, Field, PrivateAttr
from dapr_agents.types import ToolError
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class DaprHTTPClient(BaseModel):
    """
    Client for sending HTTP requests to Dapr endpoints.
    """

    dapr_app_id: Optional[str] = Field(
        default="", description="Optional name of the Dapr App ID to invoke."
    )

    dapr_http_endpoint: Optional[str] = Field(
        default="",
        description="Optional name of the HTTPEndpoint to call for invocation",
    )

    http_endpoint: Optional[str] = Field(
        default="", description="Optional FQDN URL to request to."
    )

    method: Optional[str] = Field(
        default="", description="Optional name of the method to invoke."
    )

    headers: Optional[Dict[str, str]] = Field(
        default={},
        description="Default headers to include in all requests.",
    )

    # Private attributes not exposed in model schema
    _base_url: str = PrivateAttr(default="http://localhost:3500/v1.0/invoke")

    def model_post_init(self, __context: Any) -> None:
        """Initialize the client after the model is created."""
        logger.debug("Initializing DaprHTTPClient client")
        super().model_post_init(__context)

    def post(
        self,
        payload: dict[str, str],
        endpoint: str = "",
        method: str = "",
    ) -> Union[tuple[int, str] | ToolError]:
        """
        Send a POST request to the specified endpoint with the given input.

        Args:
            endpoint_url (str): The URL of the endpoint to send the request to.
            payload (dict[str, str]): The payload to include in the request.
            method (str): The method to invoke.
        Returns:
            A tuple with the http status code and respose or a ToolError.
        """

        try:
            url = self._validate_endpoint_type(endpoint=endpoint, method=method)
        except ToolError as e:
            logger.error(f"Error validating endpoint: {e}")
            raise e

        logger.debug(
            f"[HTTP] Sending POST request to '{url}' with input '{payload}' and headers '{self.headers}"
        )

        # We can safely typecast the url to str, since we caught the possible ToolError
        response = requests.post(url=str(url), headers=self.headers, data=payload)

        logger.debug(
            f"Request returned status code '{response.status_code}' and '{response.text}'"
        )

        if not response.ok:
            raise ToolError(
                f"Error occured sending the request. Received '{response.status_code}' - '{response.text}'"
            )

        return (response.status_code, response.text)

    def get(
        self,
        endpoint: str = "",
        method: str = "",
    ) -> Union[tuple[int, str] | ToolError]:
        """
        Send a GET request to the specified endpoint.

        Args:
            endpoint_url (str): The URL of the endpoint to send the request to.
            method (str): The method to invoke.
        Returns:
            A tuple with the http status code and respose or a ToolError.
        """

        try:
            url = self._validate_endpoint_type(endpoint=endpoint, method=method)
        except ToolError as e:
            logger.error(f"Error validating endpoint: {e}")
            raise e

        logger.debug(
            f"[HTTP] Sending GET request to '{url}' with headers '{self.headers}"
        )

        # We can safely typecast the url to str, since we caught the possible ToolError
        response = requests.get(url=str(url), headers=self.headers)

        if not response.ok:
            raise ToolError(
                f"Error occured sending the request. Received '{response.status_code}' - '{response.text}'"
            )

        return (response.status_code, response.text)

    def _validate_endpoint_type(
        self, endpoint: str, method: str
    ) -> Union[str | ToolError]:
        if method == "" and self.method == "":
            raise ToolError("No method provided. Please provide a valid method.")

        if self.dapr_app_id != "":
            # Prefered option
            url = f"{self._base_url}/{self.dapr_app_id}/method{self.method if method == '' else method}"
        elif self.dapr_http_endpoint != "":
            # Dapr HTTPEndpoint
            url = f"{self._base_url}/{self.dapr_http_endpoint}/method{self.method if method == '' else method}"
        elif self.http_endpoint != "":
            # FQDN URL
            url = f"{self._base_url}/{self.http_endpoint}/method{self.method if method == '' else method}"
        elif endpoint != "":
            # Fallback to default
            url = f"{self._base_url}/{endpoint}/method{self.method if method == '' else method}"
        else:
            raise ToolError(
                "No endpoint provided. Please provide a valid dapr-app-id, HTTPEndpoint or endpoint."
            )

        if not self._validate_url(url):
            raise ToolError(f"'{url}' is not a valid URL.")

        return url

    def _validate_url(self, url) -> bool:
        """
        Valides URL for HTTP requests
        """
        logger.debug(f"[HTTP] Url to be validated: {url}")
        try:
            parsed_url = urlparse(url=url)
            return all([parsed_url.scheme, parsed_url.netloc])
        except AttributeError:
            return False
