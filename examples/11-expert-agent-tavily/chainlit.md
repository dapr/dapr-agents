# Expert Agent

Welcome to the Expert Agent, a conversational assistant powered by Dapr Agents.

Every question you ask is silently enriched with fresh web context via a `before_llm_call` hook that queries [Tavily](https://tavily.com) — so the model can answer questions about recent events, current versions, and anything that postdates its training cutoff.

## How to use

Just ask a question. The agent will:

1. Run a Tavily web search on your question (via a `before_llm_call` hook).
2. Inject the top results into the prompt as fresh context.
3. Ask the LLM, which now has up-to-the-minute information to ground its answer.

Try things like *"What's the latest Dapr release?"* or *"Who won the most recent F1 race?"* — questions the model would otherwise hedge on.
