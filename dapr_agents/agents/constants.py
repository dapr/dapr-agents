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

from dapr_agents.types.agent import ToolChoice, ToolExecutionMode

# ============================================================================
# Agent Execution Defaults
# ============================================================================

AGENT_DEFAULT_MAX_ITERATIONS = 10
AGENT_DEFAULT_TOOL_CHOICE = ToolChoice.AUTO
AGENT_DEFAULT_TOOL_EXECUTION_MODE = ToolExecutionMode.PARALLEL


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    "AGENT_DEFAULT_MAX_ITERATIONS",
    "AGENT_DEFAULT_TOOL_CHOICE",
    "AGENT_DEFAULT_TOOL_EXECUTION_MODE",
]
