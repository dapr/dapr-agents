import os
from typing import Dict, Optional, Any, Union
from distutils.util import strtobool
import logging
import requests

from pydantic import BaseModel, Field, PrivateAttr
from dapr_agents.types import ToolError
from urllib.parse import urlparse
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider


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

    path: Optional[str] = Field(
        default="", description="Optional name of the path to invoke."
    )

    headers: Optional[Dict[str, str]] = Field(
        default={},
        description="Default headers to include in all requests.",
    )

    _base_url: str = PrivateAttr(default="http://localhost:3500/v1.0/invoke")

    def model_post_init(self, __context: Any) -> None:
        """Initialize the client after the model is created."""

        try:
            otel_enabled: bool = bool(
                strtobool(os.getenv("DAPR_AGENTS_OTEL_ENABLED", "True"))
            )
        except ValueError:
            otel_enabled = False

        if otel_enabled:
            from dapr_agents.agents.telemetry.otel import DaprAgentsOTel  # type: ignore[import-not-found]

            otel_client = DaprAgentsOTel(
                service_name=os.getenv("OTEL_SERVICE_NAME", "dapr-http-client"),
                otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
            )
            tracer = otel_client.create_and_instrument_tracer_provider()
            trace.set_tracer_provider(tracer)

            otel_logger = otel_client.create_and_instrument_logging_provider(
                logger=logger,
            )
            set_logger_provider(otel_logger)

            RequestsInstrumentor().instrument()

        logger.debug("Initializing DaprHTTPClient client")

        super().model_post_init(__context)

    def do_http_request(
        self,
        payload: dict[str, str],
        endpoint: str = "",
        path: str = "",
        verb: str = "GET",
    ) -> Union[tuple[int, str] | ToolError]:
        """
        Send a POST request to the specified endpoint with the given input.

        Args:
            endpoint_url (str): The host of the URI to send the request to.
            payload (dict[str, str]): The payload to include in the request.
            path (str): The path of the URI to invoke including any query strings appended.
            verb (str): The HTTP verb. Either GET or POST.
        Returns:
            A tuple with the http status code and respose or a ToolError.
        """

        try:
            url = self._validate_endpoint_type(
                endpoint=endpoint, path=self.path if path == "" else path
            )
        except ToolError as e:
            logger.error(f"Error validating endpoint: {e}")
            raise e

        logger.debug(
            f"[HTTP] Sending POST request to '{url}' with input '{payload}' and headers '{self.headers}"
        )

        match verb.upper():
            case "GET":
                response = requests.get(url=str(url), headers=self.headers)
            case "POST":
                response = requests.post(
                    url=str(url), headers=self.headers, json=payload
                )
            case _:
                raise ValueError(
                    f"Value for 'verb' not in expected format ['GET'|'POST']: {verb}"
                )

        logger.debug(
            f"Request returned status code '{response.status_code}' and '{response.text}'"
        )

        if not response.ok:
            raise ToolError(
                f"Error occured sending the request. Received '{response.status_code}' - '{response.text}'"
            )

        return (response.status_code, response.text)

    def _validate_endpoint_type(
        self, endpoint: str, path: Optional[str | None]
    ) -> Union[str | ToolError]:
        if path == "":
            raise ToolError("No path provided. Please provide a valid path.")

        if isinstance(path, str) and path.startswith("/"):
            # Remove leading slash
            path = path[1:]

        try:
            if self.dapr_app_id != "":
                # Prefered option
                if isinstance(self.dapr_app_id, str) and self.dapr_app_id.endswith("/"):
                    # Remove trailing slash
                    self.dapr_app_id = self.dapr_app_id[:-1]
                url = f"{self._base_url}/{self.dapr_app_id}/method/{self.path if path == '' else path}"
            elif self.dapr_http_endpoint != "":
                # Dapr HTTPEndpoint
                if isinstance(
                    self.dapr_http_endpoint, str
                ) and self.dapr_http_endpoint.endswith("/"):
                    # Remove trailing slash
                    self.dapr_http_endpoint = self.dapr_http_endpoint[:-1]
                url = f"{self._base_url}/{self.dapr_http_endpoint}/method/{self.path if path == '' else path}"
            elif self.http_endpoint != "":
                # FQDN URL
                if isinstance(self.http_endpoint, str) and self.http_endpoint.endswith(
                    "/"
                ):
                    # Remove trailing slash
                    self.http_endpoint = self.http_endpoint[:-1]
                url = f"{self._base_url}/{self.http_endpoint}/method/{self.path if path == '' else path}"
            elif endpoint != "":
                # Fallback to default
                if isinstance(endpoint, str) and endpoint.endswith("/"):
                    # Remove trailing slash
                    endpoint = endpoint[:-1]
                url = f"{self._base_url}/{endpoint}/method/{self.path if path == '' else path}"
            else:
                raise ToolError(
                    "No endpoint provided. Please provide a valid dapr-app-id, HTTPEndpoint or endpoint."
                )
        except Exception as e:
            logger.error(f"Error validating endpoint: {e}")
            raise ToolError(
                "Error occured while validating the endpoint. Please check the provided values."
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
