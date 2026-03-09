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


class AgentToolExecutorError(Exception):
    """Custom exception for AgentToolExecutor specific errors."""


class AgentError(Exception):
    """Custom exception for Agent specific errors, used to handle errors specific to agent operations."""


class ToolError(Exception):
    """Custom exception for tool-related errors."""


class StructureError(Exception):
    """Custom exception for errors related to structured handling."""


class FunCallBuilderError(Exception):
    """Custom exception for errors related to structured handling."""


class NotSupportedError(Exception):
    """Custom exception for errors related to not supported features or versions."""


class DaprRuntimeVersionNotSupportedError(NotSupportedError):
    """Custom exception for errors related to not supported Dapr runtime versions."""
