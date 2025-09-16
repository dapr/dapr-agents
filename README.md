# Dapr Agents: A Framework for Agentic AI Systems

[![PyPI - Version](https://img.shields.io/pypi/v/dapr-agents?style=flat&logo=pypi&logoColor=white&label=Latest%20version)](https://pypi.org/project/dapr-agents/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/dapr-agents?style=flat&logo=pypi&logoColor=white&label=Downloads)](https://pypi.org/project/dapr-agents/) 
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/dapr/dapr-agents/.github%2Fworkflows%2Fbuild.yaml?branch=main&label=Build&logo=github)](https://github.com/dapr/dapr-agents/actions/workflows/build.yaml) 
[![GitHub License](https://img.shields.io/github/license/dapr/dapr-agents?style=flat&label=License&logo=github)](https://github.com/dapr/dapr-agents/blob/main/LICENSE) 
[![Discord](https://img.shields.io/discord/778680217417809931?label=Discord&style=flat&logo=discord)](http://bit.ly/dapr-discord) 
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UCtpSQ9BLB_3EXdWAUQYwnRA?style=flat&label=YouTube%20views&logo=youtube)](https://youtube.com/@daprdev)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/daprdev?logo=x&style=flat)](https://twitter.com/daprdev)

Dapr Agents is a developer framework designed to build production-grade resilient AI agent systems that operate at scale. Built on top of the battle-tested Dapr project, it enables software developers to create AI agents that reason, act, and collaborate using Large Language Models (LLMs), while leveraging built-in observability and stateful workflow execution to guarantee agentic workflows complete successfully, no matter how complex.

![](./docs/img/logo-workflows.png)

## Key Features

- **Scale and Efficiency**: Run thousands of agents efficiently on a single core. Dapr distributes single and multi-agent apps transparently across fleets of machines and handles their lifecycle.
- **Workflow Resilience**: Automatically retries agentic workflows and ensures task completion.
- **Kubernetes-Native**: Easily deploy and manage agents in Kubernetes environments.
- **Data-Driven Agents**: Directly integrate with databases, documents, and unstructured data by connecting to dozens of different data sources.
- **Multi-Agent Systems**: Secure and observable by default, enabling collaboration between agents.
- **Vendor-Neutral & Open Source**: Avoid vendor lock-in and gain flexibility across cloud and on-premises deployments.
- **Platform-Ready**: Built-in RBAC, access scopes and declarative resources enable platform teams to integrate Dapr agents into their systems. 

## Why Choose Dapr Agents?

### Scalable Workflows as a First Class Citizen

Dapr Agents uses a [durable-execution workflow engine](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-overview/) that guarantees each agent task executes to completion in the face of network interruptions, node crashes and other types of disruptive failures. Developers do not need to know about the underlying concepts of the workflow engine - simply write an agent that performs any number of tasks and these will get automatically distributed across the cluster. If any task fails, it will be retried and recover its state from where it left off.

### Cost-Effective AI Adoption

Dapr Agents builds on top of Dapr's Workflow API, which under the hood represents each agent as an actor, a single unit of compute and state that is thread-safe and natively distributed, lending itself well to an agentic Scale-To-Zero architecture. This minimizes infrastructure costs, making AI adoption accessible to everyone. The underlying virtual actor model allows thousands of agents to run on demand on a single core machine with double-digit millisecond latency when scaling from zero. When unused, the agents are reclaimed by the system but retain their state until the next time they are needed. With this design, there's no trade-off between performance and resource efficiency.

### Data-Centric AI Agents

With built-in connectivity to over 50 enterprise data sources, Dapr Agents efficiently handles structured and unstructured data. From basic [PDF extraction](https://v1-16.docs.dapr.io/developing-applications/dapr-agents/dapr-agents-integrations/#arxiv-fetcher) to large-scale database interactions, it enables seamless data-driven AI workflows with minimal code changes. Dapr's [bindings](https://docs.dapr.io/reference/components-reference/supported-bindings/) and [state stores](https://docs.dapr.io/reference/components-reference/supported-state-stores/) provide access to a large number of data sources that can be used to ingest data to an agent.

### Accelerated Development

Dapr Agents provides a set of AI features that give developers a complete API surface to tackle common problems. Some of these include:

- Multi-agent communications
- Structured outputs
- Multiple LLM providers
- Contextual memory
- Flexible prompting
- Intelligent tool selection
- [MCP integration](https://docs.anthropic.com/en/docs/agents-and-tools/mcp).

### Integrated Security and Reliability

By building on top of Dapr, platform and infrastructure teams can apply Dapr's [resiliency policies](https://docs.dapr.io/operations/resiliency/resiliency-overview/) to the database and/or message broker of their choice that are used by Dapr Agents. These policies include timeouts, retry/backoffs and circuit breakers. When it comes to security, Dapr provides the option to scope access to a given database or message broker to one or more agentic app deployments. In addition, Dapr Agents uses mTLS to encrypt the communication layer of its underlying components. 

### Built-in Messaging and State Infrastructure

* 🎯 **Service-to-Service Invocation**: Facilitates direct communication between agents with built-in service discovery, error handling, and distributed tracing. Agents can leverage this for synchronous messaging in multi-agent workflows.
* ⚡️ **Publish and Subscribe**: Supports loosely coupled collaboration between agents through a shared message bus. This enables real-time, event-driven interactions critical for task distribution and coordination.
* 🔄 **Durable Workflow**: Defines long-running, persistent workflows that combine deterministic processes with LLM-based decision-making. Dapr Agents uses this to orchestrate complex multi-step agentic workflows seamlessly.
* 🧠 **State Management**: Provides a flexible key-value store for agents to retain context across interactions, ensuring continuity and adaptability during workflows.
* 🤖 **Actors**: Implements the Virtual Actor pattern, allowing agents to operate as self-contained, stateful units that handle messages sequentially. This eliminates concurrency concerns and enhances scalability in agentic systems.

### Vendor-Neutral and Open Source

As a part of **CNCF**, Dapr Agents is vendor-neutral, eliminating concerns about lock-in, intellectual property risks, or proprietary restrictions. Organizations gain full flexibility and control over their AI applications using open-source software they can audit and contribute to.

## Roadmap

Here are some of the major features we're working on:

### Q2 2025
- **MCP Support** - Integration with Anthropic's MCP platform ([#50](https://github.com/dapr/dapr-agents/issues/50) ✅ )
- **Agent Interaction Tracing** - Enhanced observability of agent interactions with LLMs and tools ([#79](https://github.com/dapr/dapr-agents/issues/79))
- **Streaming LLM Output** - Real-time streaming capabilities for LLM responses ([#80](https://github.com/dapr/dapr-agents/issues/80))
- **HTTP Endpoint Tools** - Support for using Dapr's HTTP endpoint capabilities for tool calling ([#81](https://github.com/dapr/dapr-agents/issues/81))
- **DSL Cleanup** - Streamlining the domain-specific language and removing actor dependencies ([#65](https://github.com/dapr/dapr-agents/issues/65))
- **Samples Registry** - A dedicated repository for Dapr Agents examples and use cases

### Q3/Q4 2025
- **Human-in-the-Loop Support**
- **Conversation API Progressed to Beta** 
- **Vector API** - Vector operations support in Dapr and Dapr Agents

For more details about these features and other planned work, please check out our [GitHub issues](https://github.com/dapr/dapr-agents/issues).

### Language Support

| Language | Current Status | Development Status | Stable Status |
|----------|---------------|-------------|--------|
| Python   | In Development | Q2 2025 | Q3 2025 |
| .NET     | Planning | Q3 2025 | Q4 2025 |
| Other Languages | Coming Soon | TBD | TBD |

## Documentation

- [Development Guide](docs/development/README.md) - For developers and contributors

## Community

### Contributing to Dapr Agents

Please refer to our [Dapr Community Code of Conduct](https://github.com/dapr/community/blob/master/CODE-OF-CONDUCT.md)

For development setup and guidelines, see our [Development Guide](docs/development/README.md).

## Getting Started

Prerequisites:

- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/)
- [Python 3.10](https://www.python.org/downloads/release/python-3100/)

### Install Dapr Agents

```bash
dapr init
pip install dapr-agents
```

### Run The Quickstarts

To start running Dapr Agents locally, see our [quickstarts](./quickstarts/README.md).

## Get Involved

Dapr Agents is an open-source project under the CNCF umbrella, and we welcome contributions from developers and organizations worldwide!

- GitHub Repository: [https://github.com/dapr/dapr-agents](https://github.com/dapr/dapr-agents)
- Documentation: [https://dapr.github.io/dapr-agents/](https://dapr.github.io/dapr-agents/)
- Community Discord: [Join the discussion](https://bit.ly/dapr-discord). 
- Contribute: Open an issue or submit a PR to help improve Dapr Agents!
