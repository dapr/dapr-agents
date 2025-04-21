
from dapr_agents.tool.utils.openapi import OpenAPISpecParser
from dapr_agents.tool.storage import VectorToolStore
from dapr_agents.tool.base import tool
from pydantic import BaseModel ,Field, ConfigDict
from typing import Optional, Any, Dict
from urllib.parse import urlparse
import json
import requests

def extract_version(path: str) -> str:
    """Extracts the version prefix from a path if it exists, assuming it starts with 'v' followed by digits."""
    parts = path.strip('/').split('/')
    if parts and parts[0].startswith('v') and parts[0][1:].isdigit():
        return parts[0]
    return ''

def generate_get_openapi_definition(tool_vector_store: VectorToolStore):
    @tool
    def get_openapi_definition(user_input: str):
        """
        Get potential APIs for the user to use to accompish task.
        You have to choose the right one after getting a response.
        This tool MUST be used before calling any APIs.
        """
        similatiry_result =  tool_vector_store.get_similar_tools(query_texts=[user_input], k=5)
        documents = similatiry_result['documents'][0]
        return documents
    return get_openapi_definition

def generate_api_call_executor(spec_parser: OpenAPISpecParser, auth_header: Dict = None):
    base_url = spec_parser.spec.servers[0].url
    
    class OpenAPIExecutorInput(BaseModel):
        path_template: str = Field(description="Template of the API path that may include placeholders.")
        method: str = Field(description="The HTTP method to be used for the API call (e.g., 'GET', 'POST').")
        path_params: Dict[str, Any] = Field(default={}, description="Path parameters to be replaced in the path template.")
        data: Dict[str, Any] = Field(default={}, description="Data to be sent in the body of the request, applicable for POST, PUT methods.")
        headers: Optional[Dict[str, Any]] = Field(default=None, description="HTTP headers to send with the request.")
        params: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters to append to the URL.")
        
        model_config = ConfigDict(extra="allow")

    @tool(args_model=OpenAPIExecutorInput)
    def open_api_call_executor(
        path_template: str,
        method: str,
        path_params: Dict[str, Any],
        data: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Execute an API call based on provided parameters and configuration.
        It MUST be used after the get_openapi_definition to call APIs.
        Make sure to include the right header values to authenticate to the API if needed.
        """
        
        # Format the path with path_params
        formatted_path = path_template.format(**path_params)
        
        # Parse the base_url and extract the version
        parsed_url = urlparse(base_url)
        origin = f"{parsed_url.scheme}://{parsed_url.netloc}"
        base_path = parsed_url.path.strip('/')
        
        base_version = extract_version(base_path)
        path_version = extract_version(formatted_path)

        # Avoid duplication of the version in the final URL
        if base_version and path_version == base_version:
            formatted_path = formatted_path[len(f"/{path_version}"):]
        
        # Ensure there is a single slash between origin, base_path, and formatted_path
        final_url = f"{origin}/{base_path}/{formatted_path}".replace('//', '/')
        # Fix the issue by ensuring the correct scheme with double slashes
        if not final_url.startswith('https://') and parsed_url.scheme == 'https':
            final_url = final_url.replace('https:/', 'https://')
        
        # Initialize the headers with auth_header if provided
        final_headers = auth_header if auth_header else {}
        # Update the final_headers with additional headers passed to the function
        if headers:
            final_headers.update(headers)

        if data:
            data = json.dumps(data)  # Convert data to JSON string if not empty

        request_kwargs = {
            "headers": final_headers,
            "params": params,
            "data": data,
            **kwargs
        }

        print(f"Base Url: {base_url}")
        print(f"Requested Url: {final_url}")
        print(f"Requested Parameters: {params}")

        # Filter out None values to avoid sending them to requests
        request_kwargs = {k: v for k, v in request_kwargs.items() if v is not None}

        response = requests.request(method, final_url, **request_kwargs)
        return response.json()

    return open_api_call_executor