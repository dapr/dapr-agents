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

"""Public-facing types for the Drasi extension.

Public types are sourced from `dapr_agents.ext.drasi.schemas` (owned by Drasi)
and re-exported under Drasi-prefixed names; this avoids naming conflicts
and provides a stable API for external consumers.
"""

from __future__ import annotations

from typing import Literal

from dapr_agents.ext.drasi.schemas import ChangeNotification

DrasiOperation = Literal["i", "u", "d"]


class DrasiChangeEvent(ChangeNotification):
    """A Drasi event that represents a change emitted by a Drasi query."""

    pass
