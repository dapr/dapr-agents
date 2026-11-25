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

"""Scripted local model and temporary action receiver, never a real LLM."""

from __future__ import annotations

import json
from threading import Event, Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
_lock = Lock()
_gate = Event()
_gate.set()
_calls: list[dict[str, Any]] = []
_records: list[dict[str, str]] = []


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    stream: bool = False


class Record(BaseModel):
    task: str


class Gate(BaseModel):
    blocked: bool


@app.get("/evidence")
def evidence() -> dict[str, Any]:
    with _lock:
        return {"calls": list(_calls), "records": list(_records)}


@app.post("/gate")
def gate(request: Gate) -> dict[str, bool]:
    if request.blocked:
        _gate.clear()
    else:
        _gate.set()
    return {"blocked": request.blocked}


@app.post("/records")
def record(request: Record) -> dict[str, bool]:
    with _lock:
        _records.append(request.model_dump())
    return {"recorded": True}


@app.post("/v1/chat/completions")
def completion(request: ChatRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(400, "This fixture only supports non-streaming calls.")
    names = [tool["function"]["name"] for tool in request.tools]
    required = {"record_event", "list_drasi_subscriptions"}
    if not required.issubset(names):
        raise HTTPException(422, "Ordinary or Drasi tools are missing.")
    tasks = [
        (index, message["content"])
        for index, message in enumerate(request.messages)
        if message["role"] == "user"
    ]
    if not tasks or not isinstance(tasks[-1][1], str):
        raise HTTPException(422, "Expected a self-contained text task.")
    turn_start, task = tasks[-1]
    with _lock:
        _calls.append({"task": task, "tools": names})
    if not _gate.wait(timeout=60):
        raise HTTPException(503, "The integration model gate was not released.")

    called = {
        call["function"]["name"]
        for message in request.messages[turn_start + 1 :]
        for call in message.get("tool_calls") or []
    }
    message: dict[str, Any] = {"role": "assistant", "content": "Recorded."}
    finish_reason = "stop"
    if task == "LIFECYCLE_SUBSCRIBE":
        name = next(
            item for item in names if item.startswith("subscribe_service-errors")
        )
        if name not in called:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                {
                                    "operations": ["i"],
                                    "instructions": "Lifecycle drainage verification.",
                                }
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
    elif not required.issubset(called):
        name = (
            "list_drasi_subscriptions"
            if "list_drasi_subscriptions" not in called
            else "record_event"
        )
        arguments = {"task": task} if name == "record_event" else {}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": 0,
        "model": "scripted",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
