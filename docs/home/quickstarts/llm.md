# LLM Inference Client

In `Dapr Agents`, the LLM Inference Client is responsible for interacting with language models. It serves as the interface through which the agent communicates with the LLM, generating responses based on the input provided.

!!! info
    By default, `Dapr Agents` uses the `OpenAIChatClient` to interact with the OpenAI Chat endpoint. By default, the `OpenAIChatClient` uses the `gpt-4o` model

## Set Environment Variables

Create an `.env` file for your API keys and other environment variables with sensitive information that you do not want to hardcode.

```
OPENAI_API_KEY="XXXXXX"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

Use [Python-dotenv](https://pypi.org/project/python-dotenv/) to load environment variables from `.env`.

```
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
```

## Basic Example

By default, you can easily initialize the `OpenAIChatClient` without additional configuration. It uses the `OpenAI API` key from your environment variables.

```python
from dapr_agents import OpenAIChatClient

llm = OpenAIChatClient()

llm.generate('Name a famous dog!')
```

This will generate a response using the `OpenAI` model, querying for the name of a famous dog.

```
ChatCompletion(choices=[Choice(finish_reason='stop', index=0, message=MessageContent(content='One famous dog is Lassie, the Rough Collie who became popular through films, television series, and books starting in the 1940s.', role='assistant'), logprobs=None)], created=1732713689, id='chatcmpl-AYCGvJxgP61OPy96Z3YW2dLAz7IJW', model='gpt-4o-2024-08-06', object='chat.completion', usage={'completion_tokens': 30, 'prompt_tokens': 12, 'total_tokens': 42, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}})
```

## Structured Output

Once again, initialize `OpenAIChatClient`.

```python
from dapr_agents import OpenAIChatClient

llmClient = OpenAIChatClient()
```

Define the response structure. You can use [pydantic](https://docs.pydantic.dev/latest/) models.

```python
from pydantic import BaseModel

class dog(BaseModel):
    name: str
    breed: str
    reason: str
```

Finally, you can pass the response model to the LLM Client call.

```python
from dapr_agents.types import UserMessage

response = llmClient.generate(
    messages=[UserMessage("One famous dog in history.")],
    response_model=dog
)
response
```

```
dog(name='Hachiko', breed='Akita', reason="known for his remarkable loyalty to his owner, even many years after his owner's death")
```