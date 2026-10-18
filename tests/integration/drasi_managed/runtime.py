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

"""An exclusively owned Compose project; no existing Dapr/cluster resources."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx

HERE = Path(__file__).resolve().parent
ROUTER_REVISION = "7e3f4f2aec6574ba1f487d910315e4265670ddc8"
QUERY = "service-errors"
OTHER_QUERY = "rollout-status"
CONSUMER = "managed-agent"
T = TypeVar("T")


def wait_for(
    predicate: Callable[[], T],
    description: str,
    *,
    timeout: float = 45,
) -> T:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            last_error = error
        time.sleep(0.2)
    raise AssertionError(f"Timed out waiting for {description}.") from last_error


def command(
    arguments: list[str], *, environment: dict[str, str], timeout: int = 180
) -> str:
    result = subprocess.run(
        arguments,
        cwd=HERE,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {arguments!r}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def pairs(value: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if len(value) % 2:
        raise ValueError(f"Expected Redis key/value pairs, got {value!r}.")
    return dict(zip(value[::2], value[1::2]))


@dataclass(frozen=True)
class Images:
    agent: str
    router: str
    core_version: str


def command_environment() -> dict[str, str]:
    """Pass only local CLI/daemon coordinates, never model credentials."""
    return {
        name: os.environ[name]
        for name in (
            "PATH",
            "HOME",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "DOCKER_CERT_PATH",
            "DOCKER_TLS_VERIFY",
            "XDG_RUNTIME_DIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "UV_SYSTEM_CERTS",
            "UV_NATIVE_TLS",
        )
        if name in os.environ
    }


class Runtime:
    def __init__(self, directory: Path, images: Images) -> None:
        self.directory = directory
        self.project = f"drasi-i2-{uuid4().hex}"
        self.queries = directory / "queries"
        self.queries.mkdir()
        self.queries.chmod(0o755)
        for query_id, title, description in (
            (QUERY, "Service errors", "Individual service error rows."),
            (OTHER_QUERY, "Rollout status", "Individual deployment status rows."),
        ):
            path = self.queries / query_id
            path.write_text(
                f"title: {title}\ndescription: {description}\n", encoding="utf-8"
            )
            path.chmod(0o644)
        coordinates = {
            "COMPOSE_PROJECT_NAME": self.project,
            "DRASI_AGENT_IMAGE": images.agent,
            "DRASI_ROUTER_IMAGE": images.router,
            "DRASI_CORE_VERSION": images.core_version,
            "DRASI_QUERY_DIRECTORY": str(self.queries),
        }
        (directory / "compose.env").write_text(
            "".join(
                f"{name}={json.dumps(value)}\n" for name, value in coordinates.items()
            ),
            encoding="utf-8",
        )
        self.environment = {
            **command_environment(),
            **coordinates,
            "COMPOSE_DISABLE_ENV_FILE": "1",
        }
        self.client = httpx.Client(timeout=40, trust_env=False)
        self.agent_url = ""
        self.agent_dapr = ""
        self.router_url = ""
        self.router_dapr = ""
        self.model_url = ""
        self.info: dict[str, Any] = {}
        self._started = False

    def compose(self, *arguments: str, timeout: int = 180) -> str:
        return command(
            [
                "docker",
                "compose",
                "--progress",
                "quiet",
                "--env-file",
                os.devnull,
                "--project-name",
                self.project,
                "--file",
                str(HERE / "compose.yaml"),
                *arguments,
            ],
            environment=self.environment,
            timeout=timeout,
        )

    def url(self, service: str, port: int) -> str:
        address = self.compose("port", service, str(port))
        if not address.startswith("127.0.0.1:") or "\n" in address:
            raise ValueError(f"Expected one loopback-only port, got {address!r}.")
        return f"http://{address}"

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    def start(self) -> None:
        self.compose("up", "--detach", "--no-build", "--wait", "--wait-timeout", "150")
        self.agent_url = self.url("agent-dapr", 8000)
        self.agent_dapr = self.url("agent-dapr", 3500)
        self.router_url = self.url("router-dapr", 8000)
        self.router_dapr = self.url("router-dapr", 3500)
        self.model_url = self.url("model", 8001)
        self.info = self.request("GET", f"{self.agent_url}/ready")
        self.record_versions()
        self._started = True

    def record_versions(self) -> None:
        containers = self.compose("ps", "--all", "--quiet").splitlines()
        inspected = json.loads(
            command(["docker", "inspect", *containers], environment=self.environment)
        )
        redis_info = self.compose(
            "exec", "-T", "broker", "redis-cli", "--raw", "INFO", "server"
        )
        redis_version = dict(
            line.split(":", 1) for line in redis_info.splitlines() if ":" in line
        )["redis_version"]
        report = {
            "project": self.project,
            "router_revision": ROUTER_REVISION,
            "contract_version": self.info["versions"]["drasi-agent-router-contracts"],
            "source_revision": command(
                ["git", "rev-parse", "HEAD"], environment=self.environment
            ),
            "source_dirty": bool(
                command(["git", "status", "--porcelain"], environment=self.environment)
            ),
            "packages": self.info["versions"],
            "dapr": self.request("GET", f"{self.agent_dapr}/v1.0/metadata")[
                "runtimeVersion"
            ],
            "redis": redis_version,
            "images": [
                {
                    "container": item["Name"],
                    "reference": item["Config"]["Image"],
                    "id": item["Image"],
                }
                for item in inspected
            ],
        }
        (self.directory / "versions.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    def capture(self) -> None:
        logs = self.compose("logs", "--no-color", "--timestamps", timeout=30)
        (self.directory / "containers.log").write_text(logs + "\n", encoding="utf-8")
        if self._started:
            (self.directory / "artifacts.json").write_text(
                json.dumps(
                    {
                        "scheduling": self.scheduling(),
                        "receiver": self.evidence(),
                        "inbox": self.stream(self.info["inbox"]),
                        "dlt": self.stream(self.info["dlt"]),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def close(self) -> None:
        try:
            self.compose("down", "--volumes", "--remove-orphans", "--timeout", "20")
            for kind in ("container", "network", "volume"):
                remaining = command(
                    [
                        "docker",
                        kind,
                        "ls",
                        "--quiet",
                        *(["--all"] if kind == "container" else []),
                        "--filter",
                        f"label=com.docker.compose.project={self.project}",
                    ],
                    environment=self.environment,
                )
                if remaining:
                    raise RuntimeError(
                        f"Cleanup left {kind} resources for {self.project}: {remaining}"
                    )
        finally:
            self.client.close()

    def redis(self, *arguments: str) -> Any:
        return json.loads(
            self.compose("exec", "-T", "broker", "redis-cli", "--json", *arguments)
        )

    def stream(self, topic: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (entry_id, json.loads(pairs(fields)["data"]))
            for entry_id, fields in self.redis("XRANGE", topic, "-", "+")
        ]

    def consumed(self, topic: str, entry_id: str) -> bool:
        groups = self.redis("XINFO", "GROUPS", topic)
        for group in groups:
            current = pairs(group)
            if current["name"] != CONSUMER:
                continue
            cursor = tuple(
                int(part) for part in current["last-delivered-id"].split("-")
            )
            expected = tuple(int(part) for part in entry_id.split("-"))
            return (
                cursor >= expected and self.redis("XPENDING", topic, CONSUMER)[0] == 0
            )
        return False

    def wait_consumed(self, entry_id: str) -> None:
        wait_for(
            lambda: self.consumed(self.info["inbox"], entry_id),
            f"inbox acknowledgement of {entry_id}",
        )

    def evidence(self) -> dict[str, Any]:
        return self.request("GET", f"{self.model_url}/evidence")

    def scheduling(self) -> list[dict[str, Any]]:
        return self.request("GET", f"{self.agent_url}/scheduling")

    def intent(self) -> dict[str, Any]:
        return self.request("GET", f"{self.agent_url}/intent")["document"]

    def raw_intent(self) -> tuple[dict[str, Any], str]:
        url = (
            f"{self.agent_dapr}/v1.0/state/agent-state/"
            f"{quote(self.info['intent_key'], safe='')}?consistency=strong"
        )
        response = self.client.get(url)
        response.raise_for_status()
        return response.json(), response.headers["ETag"]

    def save_intent(self, document: dict[str, Any], etag: str) -> None:
        self.request(
            "POST",
            f"{self.agent_dapr}/v1.0/state/agent-state",
            json=[
                {
                    "key": self.info["intent_key"],
                    "value": document,
                    "etag": etag,
                    "options": {"concurrency": "first-write", "consistency": "strong"},
                }
            ],
        )

    def subscribe(
        self,
        query_id: str = QUERY,
        *,
        operations: tuple[str, ...] = ("i",),
        instructions: str = "Record this independently admitted event.",
    ) -> dict[str, Any]:
        result = self.request(
            "POST",
            f"{self.agent_url}/subscriptions/{query_id}",
            json={"operations": list(operations), "instructions": instructions},
        )
        assert result["isError"] is False, result
        return self.intent()["intents"][query_id]

    def unsubscribe(self, query_id: str = QUERY) -> None:
        result = self.request("DELETE", f"{self.agent_url}/subscriptions/{query_id}")
        assert result["isError"] is False, result

    def rules(self) -> list[dict[str, Any]]:
        return self.request(
            "GET",
            f"{self.agent_dapr}/v1.0/invoke/drasi-router.drasi-integration/method/admin/rules",
        )["rules"]

    def publish(
        self,
        sequence: int,
        *,
        query_id: str = QUERY,
        added: list[dict[str, Any]] | None = None,
        updated: list[dict[str, Any]] | None = None,
        deleted: list[dict[str, Any]] | None = None,
    ) -> None:
        self.request(
            "POST",
            f"{self.router_dapr}/v1.0/publish/drasi-inbound/{query_id}-results",
            json={
                "kind": "change",
                "queryId": query_id,
                "sequence": sequence,
                "sourceTimeMs": 1_700_000_000_000,
                "addedResults": added or [],
                "updatedResults": updated or [],
                "deletedResults": deleted or [],
            },
        )

    def publish_delivery(
        self,
        delivery: dict[str, Any] | bytes,
        *,
        content_type: str = "application/json",
    ) -> str:
        topic = self.info["inbox"]
        body = (
            delivery if isinstance(delivery, bytes) else json.dumps(delivery).encode()
        )
        self.request(
            "POST",
            f"{self.router_dapr}/v1.0/publish/agent-bus/{topic}",
            content=body,
            headers={"Content-Type": content_type},
        )
        return self.stream(topic)[-1][0]

    def delivered(self, count: int = 1) -> list[tuple[str, dict[str, Any]]]:
        wait_for(
            lambda: len(self.stream(self.info["inbox"])) >= count,
            f"{count} routed inbox entries",
        )
        return self.stream(self.info["inbox"])

    def completed(self, count: int = 1) -> None:
        wait_for(
            lambda: len(self.evidence()["records"]) >= count,
            f"{count} ordinary action tool executions",
            timeout=90,
        )
        accepted = [
            entry["instance_id"]
            for entry in self.scheduling()
            if entry["phase"] == "accepted"
        ]
        assert accepted, self.scheduling()
        for instance_id in set(accepted):
            wait_for(
                lambda: (
                    self.request("GET", f"{self.agent_url}/workflows/{instance_id}")[
                        "status"
                    ]
                    == "COMPLETED"
                ),
                f"terminal workflow {instance_id}",
                timeout=90,
            )

    def restart(self, *services: str) -> None:
        self.compose("restart", "--timeout", "20", *services)
        if "router" in services:
            wait_for(
                lambda: self.client.get(f"{self.router_url}/readyz").status_code == 200,
                "router restart readiness",
                timeout=90,
            )
        if "agent" in services:
            self.info = wait_for(
                lambda: self.request("GET", f"{self.agent_url}/ready"),
                "agent restart preparation",
                timeout=90,
            )
