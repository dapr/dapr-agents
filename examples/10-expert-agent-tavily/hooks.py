#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
before_llm_call hook that web-searches the user's question with Tavily and
injects the results into the prompt as a system message before the LLM runs.

This is "RAG via hook" — the model gets up-to-the-minute context for every
turn without the agent needing a `web_search` tool the model has to choose
to call. The hook fires inside the call_llm activity, so the Tavily network
call is safe under workflow replay: the activity's recorded output is what
gets replayed, not the hook itself.
"""

import logging
import os
from functools import lru_cache

from dapr_agents import HookContext, HookDecision, Modify, Proceed
from tavily import TavilyClient

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> TavilyClient:
    # Lazy so the module imports cleanly even before load_dotenv() runs.
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def enrich_with_tavily(ctx: HookContext) -> HookDecision:
    """Prepend Tavily search results as a system message before the LLM call."""
    messages = ctx.payload.get("messages", [])
    if not messages or messages[-1].get("role") != "user":
        # No user question on the last turn (e.g. an internal continuation turn
        # following a tool call); nothing to enrich.
        return Proceed()

    question = messages[-1]["content"]
    logger.info("[hook] Tavily search: %r", question)
    results = _client().search(query=question, max_results=3)

    snippets = "\n".join(
        f"- {r['title']}: {r['content']}" for r in results.get("results", [])
    )
    if not snippets:
        return Proceed()

    enriched_messages = [
        *messages[:-1],
        {
            "role": "system",
            "content": f"Fresh web context (Tavily):\n{snippets}",
        },
        messages[-1],
    ]
    return Modify(payload={**ctx.payload, "messages": enriched_messages})
