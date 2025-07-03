from __future__ import annotations
import json
import yaml
from pathlib import Path
from typing import Union, Type, List, Dict, Callable, Any, Tuple, Optional
import requests
from pydantic import ValidationError
from openapi_pydantic import (
    OpenAPI as OpenAPI_3_1_0,
    Operation as Operation_3_1_0,
    Reference as Reference_3_1_0,
    Parameter as Parameter_3_1_0,
    Schema as Schema_3_1_0,
    RequestBody as RequestBody_3_1_0,
)

from openapi_pydantic.v3.v3_0_3 import (
    OpenAPI as OpenAPI_3_0,
    Operation as Operation_3_0,
    Reference as Reference_3_0,
    Parameter as Parameter_3_0,
    Schema as Schema_3_0,
    RequestBody as RequestBody_3_0,
)
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OpenAPISpecParser:
    """
    A class to parse and handle OpenAPI specifications with support for multiple versions.
    """

    openapi_versions = {
        "3.1.0": OpenAPI_3_1_0,
        "3.0.3": OpenAPI_3_0,
    }

    def __init__(
        self, openapi_spec: Union[OpenAPI_3_1_0, OpenAPI_3_0], openapi_version: str
    ):
        self.spec = openapi_spec
        self.openapi_version = openapi_version

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> OpenAPISpecParser:
        """Load an OpenAPI spec from a local file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")
        with path.open("r") as file:
            return cls.from_string(file.read())

    @classmethod
    def from_url(cls, url: str) -> OpenAPISpecParser:
        """Load an OpenAPI spec from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        return cls.from_string(response.text)

    @classmethod
    def from_string(cls, data: str) -> OpenAPISpecParser:
        """Parse a string containing an OpenAPI spec in JSON or YAML format."""
        try:
            spec_dict = json.loads(data)
        except json.JSONDecodeError:
            spec_dict = yaml.safe_load(data)

        openapi_version = spec_dict.get("openapi", "")
        OpenAPI_class: Type[
            Union[OpenAPI_3_1_0, OpenAPI_3_0]
        ] = cls.openapi_versions.get(openapi_version, OpenAPI_3_0)
        if OpenAPI_class is None:
            raise ValueError(f"Unsupported OpenAPI version: {openapi_version}")

        return cls(OpenAPI_class.model_validate(spec_dict), openapi_version)

    @property
    def endpoints(self) -> List[Tuple[str, str, dict]]:
        """Generate a list of endpoints with method, path, description, and detailed spec.

        Returns:
            A list of tuples, each containing the HTTP method, path, description, and the full spec for that operation.
        """
        endpoints = []
        for path, path_item in self.spec.paths.items():
            for method in [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "options",
                "head",
                "trace",
            ]:
                operation = getattr(path_item, method, None)
                if operation:
                    _ = getattr(operation, "operationId", f"{method.upper()} {path}")
                    description = getattr(
                        operation, "description", "No description provided."
                    )
                    # Collect full operation details or specific parts of the operation as needed
                    endpoints.append(
                        (f"{method.upper()} {path}", description, operation)
                    )
        return endpoints

    def validate_spec(self):
        """
        Validate the entire OpenAPI specification. Raises ValidationError on issues.
        """
        try:
            self.spec.model_validate(self.spec.model_dump())
            print("Specification is valid.")
        except ValidationError as e:
            print("Specification validation failed:", e)

    def get_operation(
        self, path: str, method: str
    ) -> Union[Operation_3_1_0, Operation_3_0]:
        """
        Get a specific operation for a given path and HTTP method.
        """
        path_item = self.spec.paths.get(path)
        if not path_item:
            raise ValueError(f"Path '{path}' not found in the specification.")
        operation = getattr(path_item, method, None)
        if isinstance(operation, Operation_3_1_0) or isinstance(
            operation, Operation_3_0
        ):
            return operation

    def get_parameters_for_operation(
        self, operation: Union[Operation_3_1_0, Operation_3_0]
    ) -> List[Union[Parameter_3_1_0, Parameter_3_0, dict]]:
        """
        Retrieve all parameters for a given operation, resolving references if necessary.
        """
        parameters = []
        if hasattr(operation, "parameters") and operation.parameters:
            for param in operation.parameters:
                if isinstance(param, Reference_3_1_0) or isinstance(
                    param, Reference_3_0
                ):
                    resolved_param = self.resolve_reference(param)
                    if resolved_param:
                        parameters.append(resolved_param)
                else:
                    parameters.append(param)
        return parameters

    def get_request_body_for_operation(
        self, operation: Union[Operation_3_1_0, Operation_3_0]
    ) -> Optional[Union[RequestBody_3_1_0, RequestBody_3_0, Dict]]:
        """
        Retrieve a requestBody for a given operation, resolving references if necessary.
        """
        if hasattr(operation, "requestBody") and operation.requestBody:
            request_body = operation.requestBody
            if isinstance(request_body, Reference_3_1_0) or isinstance(
                request_body, Reference_3_0
            ):
                resolved_request_body = self.resolve_reference(request_body)
                if resolved_request_body:
                    return resolved_request_body
            else:
                return request_body

    def get_methods_for_path(self, path: str) -> List[str]:
        """
        Retrieve all HTTP methods available for a specified path.
        Ensures that the methods are present in the path item.
        """
        path_item = self.spec.paths.get(path)
        if not path_item:
            raise ValueError(f"Path '{path}' not found in the specification.")

        results = []
        # List of possible HTTP methods that could be present in an OpenAPI path item
        possible_methods = [
            "get",
            "put",
            "post",
            "delete",
            "options",
            "head",
            "patch",
            "trace",
        ]
        for method in possible_methods:
            operation = getattr(path_item, method, None)
            if operation is not None:
                results.append(method)
        return results

    def get_parameters_for_path(
        self, path: str
    ) -> List[Union[Parameter_3_1_0, Parameter_3_0, dict]]:
        """
        Retrieve all parameters associated with a given path.
        """
        path_item = self.spec.paths.get(path)
        if not path_item:
            raise ValueError(f"Path '{path}' not found in the specification.")

        # Aggregate parameters from all operations in the path item
        parameters = []
        for operation in [
            getattr(path_item, method) for method in self.get_methods_for_path(path)
        ]:
            if hasattr(operation, "parameters") and operation.parameters:
                for param in operation.parameters:
                    if isinstance(param, Reference_3_1_0) or isinstance(
                        param, Reference_3_0
                    ):
                        resolved_param = self.resolve_reference(param)
                        if resolved_param:
                            parameters.append(resolved_param)
                    else:
                        parameters.append(param)
        return parameters

    def resolve_reference(
        self, ref: Union[Reference_3_1_0, Reference_3_0]
    ) -> Union[Parameter_3_1_0, Parameter_3_0, Schema_3_1_0, Schema_3_0]:
        """
        Resolve a reference to a Parameter or Schema within the spec components.
        """
        parts = ref.ref.split("/")
        if parts[0] != "#":
            raise ValueError("Currently only local references are supported.")
        try:
            return getattr(self.spec.components, parts[-2])[parts[-1]]
        except KeyError:
            raise ValueError(f"Reference not found: {ref.ref}")

    def get_schema(
        self,
        ref_or_schema: Union[
            Reference_3_1_0, Reference_3_0, Schema_3_1_0, Schema_3_0, dict
        ],
    ) -> str:
        """
        Retrieve and return a schema from a Reference or a direct schema dictionary.
        """
        if isinstance(ref_or_schema, Reference_3_1_0) or isinstance(
            ref_or_schema, Reference_3_0
        ):
            return json.loads(
                self.resolve_reference(ref_or_schema).model_dump_json(exclude_none=True)
            )
        elif isinstance(ref_or_schema, dict):
            if self.openapi_version == "3.1.0":
                return json.loads(
                    Schema_3_1_0(**ref_or_schema).model_dump_json(exclude_none=True)
                )
            else:
                return json.loads(
                    Schema_3_0(**ref_or_schema).model_dump_json(exclude_none=True)
                )
        elif isinstance(ref_or_schema, Schema_3_1_0) or isinstance(
            ref_or_schema, Schema_3_0
        ):
            return json.loads(ref_or_schema.model_dump_json(exclude_none=True))
        else:
            raise TypeError("Invalid type for schema retrieval.")

    def get_endpoint_definitions(self) -> List[Dict[str, Any]]:
        """
        Generate a list of endpoint definitions with documents and metadata.

        Each document includes the API name, HTTP method, path, description, and summary.
        Metadata includes the API name and tags.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing documents and metadata for each endpoint.
        """
        endpoint_definitions = []
        for path, description, details in self.endpoints:
            method, _ = path.split()
            api_name = details.get("operationId", "")
            summary = details.get("summary", "")

            # If description is not provided, use summary, if summary is not provided, use empty string
            if not description:
                description = summary if summary else ""

            # Construct the document
            document = (
                (
                    f"API Name: {api_name}. "
                    f"Method: {method}. "
                    f"Path: {path}. "
                    f"Summary: {summary}. "
                    f"Description: {description}"
                )
                .replace("\n", " ")
                .strip()
            )

            # Construct metadata
            tags = ", ".join(details.get("tags", []))
            metadata = {
                "endpoint_path": path,
                "description": description,
                "api_name": api_name,
                "tags": tags,
            }
            endpoint_definitions.append({"document": document, "metadata": metadata})

        return endpoint_definitions


def openapi_params_to_json_schema(
    params: List[Union[Parameter_3_1_0, Parameter_3_0, dict]], spec: OpenAPISpecParser
) -> dict:
    properties = {}
    required = []
    for p in params:
        schema_dict = None
        if hasattr(p, "param_schema") and p.param_schema:
            schema_dict = spec.get_schema(p.param_schema)
        elif hasattr(p, "content") and p.content:
            media_type_schema = list(p.content.values())[
                0
            ]  # Accessing first media type schema.
            schema_dict = spec.get_schema(media_type_schema)

        if schema_dict:
            if (
                p.description
            ):  # Optionally add description if it's not already in the schema
                schema_dict["description"] = (
                    p.description
                    if "description" not in schema_dict
                    else schema_dict["description"]
                )
            properties[p.name] = schema_dict

        if p.required:
            required.append(p.name)

    return {"type": "object", "properties": properties, "required": required}


def openapi_spec_to_openai_fn(
    spec_parser: OpenAPISpecParser,
) -> Tuple[List[Dict[str, Any]], Callable]:
    """Convert a valid OpenAPI spec to the JSON Schema format expected for OpenAI functions.
        Reference: https://github.com/langchain-ai/langchain/blob/fd546196ef0fafa4a4cd7bb7ebb1771ef599f372/libs/langchain/langchain/chains/openai_functions/openapi.py#L90

    Args:
        spec: OpenAPI spec to convert.

    Returns:
        Tuple of the OpenAI functions JSON schema and a default function for executing
            a request based on the OpenAI function schema.
    """
    if not spec_parser.spec.paths:
        return [], lambda: None

    functions = []

    for path in spec_parser.spec.paths:
        path_params = {
            (p.name, p.param_in): p for p in spec_parser.get_parameters_for_path(path)
        }
        for method in spec_parser.get_methods_for_path(path):
            request_args = {}
            op = spec_parser.get_operation(path, method)
            op_params = path_params.copy()
            for param in spec_parser.get_parameters_for_operation(op):
                op_params[(param.name, param.param_in)] = param
            params_by_type = defaultdict(list)
            for name_loc, p in op_params.items():
                params_by_type[name_loc[1]].append(p)
            param_loc_to_arg_name = {
                "query": "params",
                "header": "headers",
                "cookie": "cookies",
                "path": "path_params",
            }
            for param_loc, arg_name in param_loc_to_arg_name.items():
                if params_by_type[param_loc]:
                    request_args[arg_name] = openapi_params_to_json_schema(
                        params_by_type[param_loc], spec_parser
                    )
            request_body = spec_parser.get_request_body_for_operation(op)
            # TODO: Support more MIME types.
            if request_body and request_body.content:
                media_types = {}
                for media_type, media_type_object in request_body.content.items():
                    if media_type_object.media_type_schema:
                        schema = spec_parser.get_schema(
                            media_type_object.media_type_schema
                        )
                        media_types[media_type] = schema
                if len(media_types) == 1:
                    media_type, schema_dict = list(media_types.items())[0]
                    key = "json" if media_type == "application/json" else "data"
                    request_args[key] = schema_dict
                elif len(media_types) > 1:
                    request_args["data"] = {"anyOf": list(media_types.values())}

            # Add method and url as part of the parameters in the function metadata
            func_name = getattr(op, "operationId", f"{path}_{method}")
            functions.append(
                {
                    "definition": {
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "description": op.description,
                            "parameters": {
                                "type": "object",
                                "properties": request_args,
                            },
                        },
                    },
                    "metadata": {
                        "method": method,
                        "url": spec_parser.spec.servers[0].url + path,
                    },
                }
            )
    return functions
