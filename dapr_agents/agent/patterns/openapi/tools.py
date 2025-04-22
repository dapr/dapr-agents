import json, logging, requests
from urllib.parse import urlparse
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field, ConfigDict

from dapr_agents.tool.base import tool
from dapr_agents.tool.storage import VectorToolStore
from dapr_agents.tool.utils.openapi import OpenAPISpecParser

logger = logging.getLogger(__name__)


def _extract_version(path: str) -> str:
    """Extracts the version prefix from a path if it exists, assuming it starts with 'v' followed by digits."""
    seg = path.lstrip("/").split("/", 1)[0]
    return seg if seg.startswith("v") and seg[1:].isdigit() else ""


def _join_url(base: str, path: str) -> str:
    """
    Join *base* and *path* while avoiding duplicated version segments
    and double slashes. Assumes base already ends at the **/servers[0].url**.
    """
    parsed = urlparse(base)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    base_path = parsed.path.strip("/")

    b_ver, p_ver = _extract_version(base_path), _extract_version(path)
    if b_ver and b_ver == p_ver:
        path = path[len(f"/{p_ver}") :]

    pieces = [p for p in (base_path, path.lstrip("/")) if p]
    return f"{origin}/" + "/".join(pieces).replace("//", "/")


def _fmt_candidate(doc: str, meta: Dict[str, Any]) -> str:
    """Return a single nice, log-friendly candidate string."""
    meta_line = f"url={meta.get('url')} | method={meta.get('method', '').upper()} | name={meta.get('name')}"
    return f"{doc.strip()}\n{meta_line}"


class GetDefinitionInput(BaseModel):
    """Free-form query describing *one* desired operation (e.g. "multiply two numbers")."""
    user_input: str = Field(..., description="Natural-language description of ONE desired API operation.")


def generate_get_openapi_definition(store: VectorToolStore):
    @tool(args_model=GetDefinitionInput)
    def get_openapi_definition(user_input: str) -> List[str]:
        """
        Search the vector store for OpenAPI *operation IDs / paths* most relevant
        to **one** user task.

        Always call this **once per new task** *before* attempting an
        `open_api_call_executor`. Returns up to 5 candidate operations.
        """
        result = store.get_similar_tools(query_texts=[user_input], k=5)
        docs: List[str] = result["documents"][0]
        metas: List[Dict[str, Any]] = result["metadatas"][0]

        return [_fmt_candidate(d, m) for d, m in zip(docs, metas)]

    return get_openapi_definition


class OpenAPIExecutorInput(BaseModel):
    path_template: str = Field(..., description="Path template, may contain `{placeholder}` segments.")
    method: str = Field(..., description="HTTP verb, upper‑case.")
    path_params: Dict[str, Any] = Field(default_factory=dict, description="Replacements for path placeholders.")
    data: Dict[str, Any] = Field(default_factory=dict, description="JSON body for POST/PUT/PATCH.")
    headers: Optional[Dict[str, Any]] = Field(default=None, description="Extra request headers.")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Query params (?key=value).")

    model_config = ConfigDict(extra="allow")


def generate_api_call_executor(spec: OpenAPISpecParser, auth_header: Optional[Dict[str, str]] = None):
    base_url = spec.spec.servers[0].url  # assumes at least one server entry

    @tool(args_model=OpenAPIExecutorInput)
    def open_api_call_executor(
        *,
        path_template: str,
        method: str,
        path_params: Dict[str, Any],
        data: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **req_kwargs: Any,
    ) -> Any:
        """
        Execute **one** REST call described by an OpenAPI operation.

        Use this only *after* `get_openapi_definition` has returned a matching
        `path_template`/`method`.

        Authentication: merge `auth_header` given at agent-init time with
        any per-call `headers` argument (per-call overrides duplicates).
        """

        url = _join_url(base_url, path_template.format(**path_params))

        final_headers = (auth_header or {}).copy()
        if headers:
            final_headers.update(headers)

        # redact auth key in debug logs
        safe_hdrs = {k: ("***" if "auth" in k.lower() or "key" in k.lower() else v)
                     for k, v in final_headers.items()}
        
        # Only convert data to JSON if we're doing a request that requires a body
        # and there's actually data to send
        body = None
        if method.upper() in ["POST", "PUT", "PATCH"] and data:
            body = json.dumps(data)
        
        # Add more detailed logging similar to old implementation
        logger.debug("→ %s %s | headers=%s params=%s data=%s", 
                    method, url, safe_hdrs, params, 
                    "***" if body else None)
        
        # For debugging purposes, similar to the old implementation
        print(f"Base Url: {base_url}")
        print(f"Requested Url: {url}")
        print(f"Requested Method: {method}")
        print(f"Requested Parameters: {params}")
        
        resp = requests.request(method, url, headers=final_headers,
                                params=params, data=body, **req_kwargs)
        resp.raise_for_status()
        return resp.json()

    return open_api_call_executor