# Orchestrators

Available Workflow options to orchestrate communication between agents:

- LLM-based: Uses a large language model (e.g., GPT-4o) to determine the most suitable agent based on the message and context.
- Random: Selects an agent randomly for each task.
- RoundRobin: Cycles through agents in a fixed order, ensuring each agent gets an equal opportunity to process tasks.

## Visual representation of each orchestration option:
![Orchestrator workflows visualized](./orchestratorWorkflows.png)