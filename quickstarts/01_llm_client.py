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

from dapr_agents.llm import DaprChatClient
from dapr_agents.types import LLMChatResponse

# Basic chat completion
llm = DaprChatClient(component_name="llm-provider")
response: LLMChatResponse = llm.generate(
    "Guess what is the weather in London right now!"
)

print("Response: ", response.get_message().content)
