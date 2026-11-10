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

"""Opt-in, credential-free integration fixtures with precise project ownership."""

from __future__ import annotations

import sys
import hashlib
import json
from collections.abc import Iterator
from email.parser import BytesParser
from pathlib import Path
from types import ModuleType
from uuid import uuid4
from zipfile import ZipFile

import pytest

from .runtime import HERE, Images, Runtime, command, command_environment


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-drasi-managed",
        action="store_true",
        help="Build and run the isolated real-Dapr Drasi integration suite.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    if session.config.getoption("--run-drasi-managed") and (
        "dapr" in sys.modules and not isinstance(sys.modules["dapr"], ModuleType)
    ):
        raise pytest.UsageError(
            "Core conftest.py mocks the Dapr SDK. Run this suite separately with "
            "--confcutdir=tests/integration/drasi_managed."
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--run-drasi-managed"):
        for item in items:
            if Path(item.path).is_relative_to(HERE) and item.get_closest_marker(
                "integration"
            ):
                item.add_marker(pytest.mark.skip(reason="Use --run-drasi-managed."))


@pytest.fixture(scope="session")
def images(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Images]:
    directory = tmp_path_factory.mktemp("drasi-build")
    wheels = directory / "wheels"
    command(
        ["uv", "build", str(HERE.parents[2]), "--wheel", "--out-dir", str(wheels)],
        environment=command_environment(),
        timeout=300,
    )
    (wheel,) = wheels.glob("dapr_agents-*.whl")
    with ZipFile(wheel) as archive:
        (metadata_path,) = (
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
    core_version = metadata["Version"]
    if metadata["Name"] != "dapr-agents" or not core_version:
        raise RuntimeError("The local core wheel has invalid package metadata.")
    (directory / "source-build.json").write_text(
        json.dumps(
            {
                "wheel": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "core_version": core_version,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tag = f"drasi-i2-{uuid4().hex}"
    images = Images(
        agent=f"{tag}-agent:local",
        router=f"{tag}-router:local",
        core_version=core_version,
    )
    builder = Runtime(directory, images)
    try:
        builder.compose("config", "--quiet")
        builder.compose("build", "--quiet", "router", "model", timeout=1200)
        yield images
    finally:
        try:
            for image in (images.agent, images.router):
                existing = command(
                    ["docker", "image", "ls", "--quiet", image],
                    environment=builder.environment,
                )
                if existing:
                    command(
                        ["docker", "image", "rm", image],
                        environment=builder.environment,
                    )
        finally:
            builder.client.close()


@pytest.fixture
def runtime(tmp_path: Path, images: Images) -> Iterator[Runtime]:
    runtime = Runtime(tmp_path, images)
    try:
        runtime.start()
        yield runtime
    finally:
        try:
            runtime.capture()
        finally:
            runtime.close()
