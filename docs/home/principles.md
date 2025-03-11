# Core Principles

![](../img/concepts-agents-overview.png)

## 1. Agent-Centric Design

Dapr Agents is designed to place agents, powered by LLMs, at the core of task execution and workflow orchestration. This principle emphasizes:

* LLM-Powered Agents: Dapr Agents enables the creation of agents that leverage LLMs for reasoning, dynamic decision-making, and natural language interactions.
* Adaptive Task Handling: Agents in Dapr Agents are equipped with flexible patterns like tool calling and reasoning loops (e.g., ReAct), allowing them to autonomously tackle complex and evolving tasks.
* Seamless Integration: Dapr Agents’ framework allows agents to act as modular, reusable building blocks that integrate seamlessly into workflows, whether they operate independently or collaboratively.

While Dapr Agents centers around agents, it also recognizes the versatility of using LLMs directly in deterministic workflows or simpler task sequences. In scenarios where the agent's built-in task-handling patterns, like `tool calling` or `ReAct` loops, are unnecessary, LLMs can act as core components for reasoning and decision-making. This flexibility ensures users can adapt Dapr Agents to suit diverse needs without being confined to a single approach.

!!! info
    Agents are not standalone; they are building blocks in larger, orchestrated workflows.

## 2. Decoupled Infrastructure Design

Dapr Agents ensures a clean separation between agents and the underlying infrastructure, emphasizing simplicity, scalability, and adaptability:

* Agent Simplicity: Agents focus purely on reasoning and task execution, while Pub/Sub messaging, routing, and validation are managed externally by modular infrastructure components.
* Scalable and Adaptable Systems: By offloading non-agent-specific responsibilities, Dapr Agents allows agents to scale independently and adapt seamlessly to new use cases or integrations.

!!! info
    Decoupling infrastructure keeps agents focused on tasks while enabling seamless scalability and integration across systems.

![](../img/home_concepts_principles_decoupled.png)

## 3. Modular Component Model

Dapr Agents utilizes [Dapr's pluggable component framework](https://docs.dapr.io/concepts/components-concept/) and building blocks to simplify development and enhance flexibility:

* Building Blocks for Core Functionality: Dapr provides API building blocks, such as Pub/Sub messaging, state management, service invocation, and more, to address common microservice challenges and promote best practices.
* Interchangeable Components: Each building block operates on swappable components (e.g., Redis, Kafka, Azure CosmosDB), allowing you to replace implementations without changing application code.
* Seamless Transitions: Develop locally with default configurations and deploy effortlessly to cloud environments by simply updating component definitions.
* Scalable Foundations: Build resilient and adaptable architectures using Dapr’s modular, production-ready building blocks.

!!! info
    Developers can easily switch between different components (e.g., Redis to DynamoDB) based on their deployment environment, ensuring portability and adaptability.

![](../img/home_concepts_principles_modular.png)

## 4. Actor-Based Model for Agents

Dapr Agents leverages [Dapr’s Virtual Actor model](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-overview/) to enable agents to function efficiently and flexibly within distributed environments. Each agent in Dapr Agents is instantiated as an instance of a class, wrapped and managed by a virtual actor. This design offers:

* Stateful Agents: Virtual actors allow agents to store and recall information across tasks, maintaining context and continuity for workflows.
* Dynamic Lifecycle Management: Virtual actors are automatically instantiated when invoked and deactivated when idle. This eliminates the need for explicit creation or cleanup, ensuring resource efficiency and simplicity.
* Location Transparency: Agents can be accessed and operate seamlessly, regardless of where they are located in the system. The underlying runtime handles their mobility, enabling fault-tolerance and dynamic load balancing.
* Scalable Execution: Agents process one task at a time, avoiding concurrency issues, and scale dynamically across nodes to meet workload demands.

This model ensures agents remain focused on their core logic, while the infrastructure abstracts complexities like state management, fault recovery, and resource optimization.

!!! info
    Dapr Agents’ use of virtual actors makes agents always addressable and highly scalable, enabling them to operate reliably and efficiently in distributed, high-demand environments.

## 5. Message-Driven Communication

Dapr Agents emphasizes the use of Pub/Sub messaging for event-driven communication between agents. This principle ensures:

* Decoupled Architecture: Asynchronous communication for scalability and modularity.
* Real-Time Adaptability: Agents react dynamically to events for faster, more flexible task execution.
* Seamless Collaboration: Agents share updates, distribute tasks, and respond to events in a highly coordinated way.

!!! info
    Pub/Sub messaging serves as the backbone for Dapr Agents’ event-driven workflows, enabling agents to communicate and collaborate in real time.

![](../img/home_concepts_principles_message.png)

## 6. Workflow-Oriented Design

Dapr Agents embraces workflows as a foundational concept, integrating [Dapr Workflows](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-overview/) to support both deterministic and event-driven task orchestration. This dual approach enables robust and adaptive systems:

* Deterministic Workflows: Dapr Agents uses Dapr Workflows for stateful, predictable task sequences. These workflows ensure reliable execution, fault tolerance, and state persistence, making them ideal for structured, multi-step processes that require clear, repeatable logic.
* Event-Driven Workflows: By combining Dapr Workflows with Pub/Sub messaging, Dapr Agents supports workflows that adapt to real-time events. This facilitates decentralized, asynchronous collaboration between agents, allowing workflows to dynamically adjust to changing scenarios.

By integrating these paradigms, Dapr Agents enables workflows that combine the reliability of deterministic execution with the adaptability of event-driven processes, ensuring flexibility and resilience in a wide range of applications.

!!! info
    Dapr Agents workflows blend structured, predictable logic with the dynamic responsiveness of event-driven systems, empowering both centralized and decentralized workflows.

![](../img/home_concepts_principles_workflows.png)
