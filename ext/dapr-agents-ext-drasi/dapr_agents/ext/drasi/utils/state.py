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

from __future__ import annotations

import json
from typing import Any

from dapr.clients import DaprClient


# TODO: can we replace these with existing storage APIs from the core?
def _decode_state_value(response: Any) -> Any:
    if response is None:
        return None
    if hasattr(response, "json"):
        try:
            return response.json()
        except Exception:
            pass
    data = getattr(response, "data", None)
    if data is None:
        return None
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = str(data)
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def read_state_value(store_name: str, key: str) -> Any:
    """Read a JSON-encoded Dapr state value."""
    with DaprClient() as client:
        response = client.get_state(
            store_name=store_name,
            key=key,
        )
        value = _decode_state_value(response)
        if value is not None:
            return value
    return None


def write_state_value(store_name: str, key: str, value: Any) -> None:
    """Persist a JSON-serializable value through Dapr state."""
    with DaprClient() as client:
        client.save_state(
            store_name=store_name,
            key=key,
            value=json.dumps(value),
            state_metadata={"contentType": "application/json"},
        )
