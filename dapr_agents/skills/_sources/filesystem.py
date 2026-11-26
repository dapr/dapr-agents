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

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml
from pydantic import ValidationError

from dapr_agents.skills.errors import (
    SkillAccessError,
    SkillCatalogError,
    SkillError,
    SkillMalformedError,
    SkillNotFoundError,
    SkillSizeLimitError,
    SkillUnsafePathError,
)
from dapr_agents.skills.models import (
    MAX_INSTRUCTIONS_BYTES,
    MAX_RESOURCE_BYTES,
    Skill,
    SkillMetadata,
    SkillResource,
    _validate_skill_path,
)

logger = logging.getLogger(__name__)

_MAX_DISCOVERY_DEPTH = 16
_MAX_DISCOVERY_DIRECTORIES = 32768
_MAX_DIRECTORY_ENTRIES = 8192
_MAX_DISCOVERED_SKILLS = 4096
_MAX_RESOURCE_FILES = 4096
_MAX_RESOURCE_DEPTH = 16
_MAX_TOTAL_RESOURCE_FILES = 32768
_MAX_RESOURCE_PATH_BYTES = 4096
_MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
_MAX_FRONTMATTER_DEPTH = 16
_MAX_FRONTMATTER_NODES = 4096
_DEFAULT_DISCOVERY_DEPTH = 3
_INVALID_FILE_DESCRIPTOR = -1
_SKILL_FILE_NAME = "SKILL.md"
_SCRIPTS_DIRECTORY_NAME = "scripts"
_FRONTMATTER_OPENING = b"---"
_FRONTMATTER_DELIMITERS = frozenset((_FRONTMATTER_OPENING, b"..."))
_UTF8_ENCODING = "utf-8"
_BINARY_READ_MODE = "rb"
_DEFAULT_MEDIA_TYPE = "application/octet-stream"
_SNAPSHOT_VERSION = 1
_SNAPSHOT_ID_PREFIX = "sha256:"
_SNAPSHOT_ID_HEX_LENGTH = hashlib.sha256().digest_size * 2
_SNAPSHOT_VERSION_KEY = "version"
_SNAPSHOT_SKILLS_KEY = "skills"
_SNAPSHOT_ROOTS_KEY = "roots"
_SNAPSHOT_CONFIGURATION_KEY = "configuration"
_SNAPSHOT_MAX_DEPTH_KEY = "max_depth"
_SNAPSHOT_MAX_INSTRUCTIONS_KEY = "max_instructions_bytes"
_SNAPSHOT_MAX_RESOURCE_KEY = "max_resource_bytes"
_SNAPSHOT_ROOT_KEY = "root"
_SNAPSHOT_DIRECTORY_KEY = "directory"
_SNAPSHOT_METADATA_KEY = "metadata"
_SNAPSHOT_SKILL_FILE_KEY = "skill_file"
_SNAPSHOT_RESOURCES_KEY = "resources"
_STANDARD_FRONTMATTER_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata"}
)


@dataclass(frozen=True)
class _FileSignature:
    """Identity and size information for one regular filesystem file."""

    resolved_path: str
    device: int
    inode: int
    size: int
    modified_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_path": self.resolved_path,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "modified_ns": self.modified_ns,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "_FileSignature":
        if not isinstance(value, dict):
            raise ValueError("invalid filesystem signature")
        resolved_path = value.get("resolved_path")
        fields = ("device", "inode", "size", "modified_ns")
        if not isinstance(resolved_path, str) or not Path(resolved_path).is_absolute():
            raise ValueError("invalid filesystem signature")
        if any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] < 0
            for field in fields
        ):
            raise ValueError("invalid filesystem signature")
        return cls(resolved_path, *(value[field] for field in fields))


@dataclass(frozen=True)
class _DiscoveredSkill:
    """Filesystem location, metadata, and immutable inventory for one skill."""

    root: Path
    directory: Path
    metadata: SkillMetadata
    skill_file: _FileSignature
    resources: tuple[tuple[str, _FileSignature], ...]


class FileSystemSkillSource:
    """Discover Agent Skills from explicitly configured filesystem roots.

    Discovery creates a canonical, metadata-only snapshot manifest for each
    skill and carries its digest through durable catalog state; later reads
    verify the same paths and metadata-based file signatures before opening any
    bytes. Resources remain lazy, and detected mutations fail closed. The
    metadata-only manifest neither retains historical bytes nor provides
    cryptographic content integrity; use immutable or versioned artifacts when
    exact byte consistency is required.
    """

    def __init__(
        self,
        skill_paths: Sequence[str | Path],
        *,
        max_depth: int = _DEFAULT_DISCOVERY_DEPTH,
        max_instructions_bytes: int = MAX_INSTRUCTIONS_BYTES,
        max_resource_bytes: int = MAX_RESOURCE_BYTES,
        source_id: str | None = None,
    ) -> None:
        if isinstance(skill_paths, (str, Path)):
            raise TypeError("skill_paths must be a non-empty sequence")
        if not isinstance(skill_paths, Sequence) or not skill_paths:
            raise ValueError("skill_paths must be a non-empty sequence")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise TypeError("max_depth must be an integer")
        if not 0 <= max_depth <= _MAX_DISCOVERY_DEPTH:
            raise ValueError(f"max_depth must be between 0 and {_MAX_DISCOVERY_DEPTH}")
        if not 0 < max_instructions_bytes <= MAX_INSTRUCTIONS_BYTES:
            raise ValueError(
                "max_instructions_bytes must be between 1 and the core instruction limit"
            )
        if not 0 < max_resource_bytes <= MAX_RESOURCE_BYTES:
            raise ValueError(
                "max_resource_bytes must be between 1 and the core resource limit"
            )
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise TypeError("source_id must be a non-empty string")

        roots: list[Path] = []
        for value in skill_paths:
            if not isinstance(value, (str, Path)):
                raise TypeError("skill_paths must contain only strings or Path values")
            root = Path(value).expanduser().resolve(strict=False)
            if root not in roots:
                roots.append(root)

        self.skill_paths = tuple(roots)
        self.max_depth = max_depth
        self.max_instructions_bytes = max_instructions_bytes
        self.max_resource_bytes = max_resource_bytes
        self.source_id = source_id
        self._snapshot_records: dict[str, dict[str, _DiscoveredSkill]] = {}

    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        """Resolve the current filesystem version or reopen a persisted one."""
        try:
            if snapshot_id is None:
                snapshot_id = await asyncio.to_thread(self._discover_sync)
            else:
                await asyncio.to_thread(self._records_from_snapshot, snapshot_id)
            return snapshot_id
        except SkillError as exc:
            logger.warning("Filesystem snapshot resolution failed: %s", exc)
            raise
        except OSError as exc:
            logger.error("Filesystem snapshot resolution failed", exc_info=True)
            raise SkillAccessError() from exc
        except Exception as exc:  # noqa: BLE001 - redact parser details
            logger.error("Filesystem snapshot resolution failed", exc_info=True)
            # Discovery failures are source errors; persisted-ID failures mean
            # the durable catalog cannot be reopened safely.
            if snapshot_id is None:
                raise SkillMalformedError() from exc
            raise SkillCatalogError() from exc

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        """Return metadata from a pinned filesystem snapshot."""
        try:
            records = await self._records_for_snapshot(snapshot_id)
            return [record.metadata for record in records.values()]
        except SkillError as exc:
            logger.warning("Filesystem skill discovery failed: %s", exc)
            raise
        except OSError as exc:
            logger.error("Filesystem skill discovery failed", exc_info=True)
            raise SkillAccessError() from exc
        except Exception as exc:  # noqa: BLE001 - redact parser details
            logger.error("Filesystem skill discovery failed", exc_info=True)
            raise SkillMalformedError() from exc

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        """Load instructions and resource names from a pinned snapshot."""
        try:
            record = await self._record_for_snapshot(name, snapshot_id)
            return await asyncio.to_thread(self._load_record_sync, record)
        except SkillError as exc:
            logger.warning("Filesystem skill activation failed: %s", exc)
            raise
        except OSError as exc:
            logger.error("Filesystem skill activation failed: %s", exc, exc_info=True)
            raise SkillAccessError(skill_name=name) from exc
        except ValidationError as exc:
            logger.error(
                "Filesystem skill activation returned invalid data", exc_info=True
            )
            raise SkillMalformedError(skill_name=name) from exc
        except Exception as exc:  # noqa: BLE001 - safe public category
            logger.error("Filesystem skill activation failed", exc_info=True)
            raise SkillMalformedError(skill_name=name) from exc

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        """Read one bounded resource from a pinned snapshot."""
        _validate_resource_path(name, path)
        try:
            record = await self._record_for_snapshot(name, snapshot_id)
            return await asyncio.to_thread(self._read_resource_sync, record, path)
        except SkillError as exc:
            logger.warning("Filesystem skill resource read failed: %s", exc)
            raise
        except FileNotFoundError as exc:
            raise SkillNotFoundError(skill_name=name, path=path) from exc
        except OSError as exc:
            logger.error(
                "Filesystem skill resource read failed: %s", exc, exc_info=True
            )
            raise SkillAccessError(skill_name=name, path=path) from exc
        except ValidationError as exc:
            logger.error(
                "Filesystem skill resource returned invalid data", exc_info=True
            )
            raise SkillMalformedError(skill_name=name, path=path) from exc

    async def _records_for_snapshot(
        self, snapshot_id: str | None
    ) -> dict[str, _DiscoveredSkill]:
        """Resolve a snapshot and return every validated skill record."""
        snapshot_id = await self.resolve_snapshot(snapshot_id=snapshot_id)
        return await asyncio.to_thread(self._records_from_snapshot, snapshot_id)

    async def _record_for_snapshot(
        self, name: str, snapshot_id: str | None
    ) -> _DiscoveredSkill:
        """Resolve a snapshot and return one validated skill record."""
        snapshot_id = await self.resolve_snapshot(snapshot_id=snapshot_id)
        return await asyncio.to_thread(self._record_from_snapshot, snapshot_id, name)

    def _discover_sync(self) -> str:
        """Discover skills and return the digest of their snapshot manifest."""
        records = self._discover_records_sync()
        manifest = self._encode_snapshot(records.values())
        snapshot_id = self._snapshot_id(manifest)
        self._snapshot_records[snapshot_id] = records
        return snapshot_id

    def _discover_records_sync(self) -> dict[str, _DiscoveredSkill]:
        """Discover skills and retain their metadata-only snapshot records."""
        discovered: list[_DiscoveredSkill] = []
        names: set[str] = set()
        total_resources = 0

        for root in self.skill_paths:
            if not root.exists() or not root.is_dir():
                raise SkillAccessError(path=str(root))
            root = root.resolve(strict=True)

            for directory in self._skill_directories(root, root, 0, set()):
                skill_file = self._validated_skill_file(directory, root)
                metadata = self._read_frontmatter(skill_file, root, directory.name)

                if metadata.name != directory.name:
                    raise SkillMalformedError(skill_name=metadata.name)
                if metadata.name in names:
                    raise SkillCatalogError(skill_name=metadata.name)
                names.add(metadata.name)

                resources = self._resource_inventory(directory, root, metadata.name)
                total_resources += len(resources)
                if total_resources > _MAX_TOTAL_RESOURCE_FILES:
                    raise SkillCatalogError(skill_name=metadata.name)
                discovered.append(
                    _DiscoveredSkill(
                        root=root,
                        directory=directory,
                        metadata=metadata,
                        skill_file=self._file_signature(skill_file),
                        resources=resources,
                    )
                )
                if len(discovered) > _MAX_DISCOVERED_SKILLS:
                    raise SkillCatalogError(skill_name=metadata.name)

        return {record.metadata.name: record for record in discovered}

    def _skill_directories(
        self,
        directory: Path,
        root: Path,
        depth: int,
        visited: set[Path],
    ) -> Iterator[Path]:
        """Yield bounded, in-root directories containing a ``SKILL.md`` file."""
        resolved_directory = directory.resolve(strict=True)

        # Resolve before descending so symlink cycles cannot evade the visit cap.
        if resolved_directory in visited:
            return
        if len(visited) >= _MAX_DISCOVERY_DIRECTORIES:
            raise SkillCatalogError(path=str(directory))
        visited.add(resolved_directory)

        skill_file = directory / _SKILL_FILE_NAME
        if skill_file.is_file():
            yield directory
            return
        if depth >= self.max_depth:
            return

        try:
            entries: list[Path] = []
            for entry in directory.iterdir():
                if len(entries) >= _MAX_DIRECTORY_ENTRIES:
                    raise SkillCatalogError(path=str(directory))
                entries.append(entry)
        except SkillError:
            raise
        except OSError as exc:
            raise SkillAccessError(path=str(directory)) from exc
        if len(entries) > _MAX_DIRECTORY_ENTRIES:
            raise SkillCatalogError(path=str(directory))

        for entry in sorted(entries, key=lambda item: item.name):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError as exc:
                raise SkillAccessError(path=str(entry)) from exc
            if not resolved.is_relative_to(root):
                raise SkillUnsafePathError(path=str(entry))

            yield from self._skill_directories(entry, root, depth + 1, visited)

    def _validated_skill_file(self, directory: Path, root: Path) -> Path:
        """Resolve ``SKILL.md`` and reject links outside its skill and root."""
        skill_directory = directory.resolve(strict=True)
        skill_file = directory / _SKILL_FILE_NAME
        resolved = skill_file.resolve(strict=True)
        if (
            not skill_directory.is_relative_to(root)
            or not resolved.is_relative_to(skill_directory)
            or not resolved.is_file()
        ):
            raise SkillUnsafePathError(path=_SKILL_FILE_NAME)
        return resolved

    def _read_frontmatter(
        self, skill_file: Path, root: Path, skill_name: str
    ) -> SkillMetadata:
        """Read and parse a bounded ``SKILL.md`` through a root-relative fd."""
        file_descriptor = _INVALID_FILE_DESCRIPTOR
        try:
            # Open relative to the configured root; do not trust a path lookup
            # that could be redirected between validation and reading.
            file_descriptor, _ = self._open_relative_path(
                root, skill_file, skill_name, _SKILL_FILE_NAME
            )
            with os.fdopen(file_descriptor, _BINARY_READ_MODE) as document:
                file_descriptor = _INVALID_FILE_DESCRIPTOR
                content = document.read(self.max_instructions_bytes + 1)

            if len(content) > self.max_instructions_bytes:
                raise SkillSizeLimitError(skill_name=skill_name, path=_SKILL_FILE_NAME)
            return self._parse_skill_metadata(content, skill_name)
        except SkillError:
            raise
        except OSError as exc:
            raise SkillAccessError(
                skill_name=skill_name, path=_SKILL_FILE_NAME
            ) from exc
        finally:
            if file_descriptor != _INVALID_FILE_DESCRIPTOR:
                os.close(file_descriptor)

    @staticmethod
    def _parse_skill_metadata(content: bytes, skill_name: str) -> SkillMetadata:
        try:
            first_line, frontmatter = _frontmatter_parts(content)

            if first_line.strip() != _FRONTMATTER_OPENING:
                raise SkillMalformedError(skill_name=skill_name, path=_SKILL_FILE_NAME)

            raw = yaml.safe_load(frontmatter.decode(_UTF8_ENCODING))
            _validate_frontmatter_value(raw)
        except SkillError:
            raise
        except Exception as exc:  # noqa: BLE001 - redact parser details
            raise SkillMalformedError(
                skill_name=skill_name, path=_SKILL_FILE_NAME
            ) from exc
        return _metadata_from_frontmatter(raw, skill_name)

    def _load_record_sync(self, record: _DiscoveredSkill) -> Skill:
        """Revalidate a snapshot record and materialize its lazy resource map."""
        self._validate_snapshot_record(record, include_resources=True)

        content = self._read_bounded_file(
            record.root, record.skill_file, record.metadata.name
        )
        metadata = self._parse_skill_metadata(content, record.metadata.name)

        if metadata.name != record.directory.name:
            raise SkillMalformedError(skill_name=record.metadata.name)
        instructions = _frontmatter_body(content)

        resources = {
            path: SkillResource(
                path=path,
                content=b"",
                media_type=_media_type(path),
            )
            for path, _ in record.resources
        }
        return Skill(
            **metadata.model_dump(),
            instructions=instructions,
            resources=resources,
        )

    def _read_resource_sync(self, record: _DiscoveredSkill, path: str) -> SkillResource:
        """Revalidate and read one resource recorded in the immutable inventory."""
        # Validate the recorded inventory before resolving the requested path.
        self._validate_snapshot_record(record, path=path)

        skill_directory = record.directory.resolve(strict=True)
        candidate = record.directory.joinpath(*path.split("/"))
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise SkillNotFoundError(
                skill_name=record.metadata.name, path=path
            ) from exc
        if not resolved_candidate.is_relative_to(
            skill_directory
        ) or not resolved_candidate.is_relative_to(record.root):
            raise SkillUnsafePathError(skill_name=record.metadata.name, path=path)

        skill_file = Path(record.skill_file.resolved_path)
        if (
            resolved_candidate == skill_file
            or _SCRIPTS_DIRECTORY_NAME
            in resolved_candidate.relative_to(skill_directory).parts
        ):
            raise SkillUnsafePathError(skill_name=record.metadata.name, path=path)

        if not resolved_candidate.is_file():
            raise SkillNotFoundError(skill_name=record.metadata.name, path=path)

        expected = dict(record.resources).get(path)
        if expected is None:
            raise SkillAccessError(skill_name=record.metadata.name, path=path)
        if resolved_candidate != Path(expected.resolved_path):
            raise SkillAccessError(skill_name=record.metadata.name, path=path)

        content = self._read_snapshot_bytes(
            record.root,
            expected,
            record.metadata.name,
            path,
            self.max_resource_bytes,
        )
        return SkillResource(path=path, content=content, media_type=_media_type(path))

    def _read_bounded_file(
        self, root: Path, signature: _FileSignature, skill_name: str
    ) -> bytes:
        """Read bounded bytes from a signed snapshot file."""
        return self._read_snapshot_bytes(
            root, signature, skill_name, _SKILL_FILE_NAME, self.max_instructions_bytes
        )

    def _read_snapshot_bytes(
        self,
        root: Path,
        signature: _FileSignature,
        skill_name: str,
        path: str,
        max_bytes: int,
    ) -> bytes:
        """Read a validated snapshot file through a descriptor."""
        file_descriptor, _ = self._open_snapshot_file(root, signature, skill_name, path)

        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or not self._stat_matches(
                signature, file_stat
            ):
                raise SkillAccessError(skill_name=skill_name, path=path)

            with os.fdopen(file_descriptor, _BINARY_READ_MODE) as snapshot_file:
                file_descriptor = _INVALID_FILE_DESCRIPTOR
                content = snapshot_file.read(max_bytes + 1)
        finally:
            if file_descriptor != _INVALID_FILE_DESCRIPTOR:
                os.close(file_descriptor)

        if len(content) > max_bytes:
            raise SkillSizeLimitError(skill_name=skill_name, path=path)
        return content

    def _resource_inventory(
        self, directory: Path, root: Path, skill_name: str
    ) -> tuple[tuple[str, _FileSignature], ...]:
        """Build a sorted, bounded inventory of readable non-script resources."""
        paths: list[tuple[str, _FileSignature]] = []
        pending = [directory]
        visited: set[Path] = set()

        skill_directory = directory.resolve(strict=True)
        skill_file = self._validated_skill_file(directory, root)
        while pending:
            current = pending.pop()
            resolved_current = current.resolve(strict=True)
            if resolved_current in visited:
                continue
            if len(visited) >= _MAX_DISCOVERY_DIRECTORIES:
                raise SkillCatalogError(skill_name=skill_name)
            visited.add(resolved_current)
            if not resolved_current.is_relative_to(skill_directory):
                raise SkillUnsafePathError(skill_name=skill_name)

            entries: list[Path] = []
            for entry in current.iterdir():
                if len(entries) >= _MAX_DIRECTORY_ENTRIES:
                    raise SkillCatalogError(skill_name=skill_name)
                entries.append(entry)
            for entry in sorted(entries, key=lambda item: item.name):
                relative = entry.relative_to(directory).as_posix()
                parts = relative.split("/")
                if (
                    len(parts) > _MAX_RESOURCE_DEPTH
                    or len(relative.encode(_UTF8_ENCODING)) > _MAX_RESOURCE_PATH_BYTES
                ):
                    raise SkillCatalogError(skill_name=skill_name, path=relative)
                if any(part.startswith(".") for part in parts):
                    continue
                first_part, *_ = parts
                if (
                    first_part == _SCRIPTS_DIRECTORY_NAME
                    or relative == _SKILL_FILE_NAME
                ):
                    continue
                resolved = entry.resolve(strict=True)
                if not resolved.is_relative_to(
                    skill_directory
                ) or not resolved.is_relative_to(root):
                    raise SkillUnsafePathError(skill_name=skill_name, path=relative)
                if resolved == skill_file:
                    raise SkillUnsafePathError(skill_name=skill_name, path=relative)
                if entry.is_dir():
                    pending.append(entry)
                elif entry.is_file():
                    paths.append((relative, self._file_signature(resolved)))
                    if len(paths) > _MAX_RESOURCE_FILES:
                        raise SkillCatalogError(skill_name=skill_name)
        return tuple(sorted(paths))

    def _validate_snapshot_record(
        self,
        record: _DiscoveredSkill,
        *,
        path: str | None = None,
        include_resources: bool = False,
    ) -> None:
        """Ensure the live filesystem still matches a recorded snapshot record."""
        try:
            skill_directory = record.directory.resolve(strict=True)
            if not skill_directory.is_relative_to(record.root):
                raise SkillUnsafePathError(skill_name=record.metadata.name, path=path)
            current_skill_file = (record.directory / _SKILL_FILE_NAME).resolve(
                strict=True
            )
            if current_skill_file != Path(record.skill_file.resolved_path):
                raise SkillAccessError(
                    skill_name=record.metadata.name, path=_SKILL_FILE_NAME
                )

            self._assert_signature(record.skill_file)

            if include_resources:
                for resource_path, signature in record.resources:
                    self._assert_signature(signature)
                    if not Path(signature.resolved_path).is_relative_to(
                        skill_directory
                    ):
                        raise SkillUnsafePathError(
                            skill_name=record.metadata.name, path=resource_path
                        )
        except SkillError:
            raise
        except (FileNotFoundError, OSError) as exc:
            raise SkillAccessError(skill_name=record.metadata.name, path=path) from exc

    def _open_snapshot_file(
        self,
        root: Path,
        signature: _FileSignature,
        skill_name: str,
        path: str,
    ) -> tuple[int, Path]:
        """Open a signed file relative to its configured root."""
        return self._open_relative_path(
            root, Path(signature.resolved_path), skill_name, path
        )

    def _open_relative_path(
        self, root: Path, target: Path, skill_name: str, path: str
    ) -> tuple[int, Path]:
        """Open a root-relative path without following symlinks during traversal."""
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise SkillUnsafePathError(skill_name=skill_name, path=path) from exc
        if not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise SkillUnsafePathError(skill_name=skill_name, path=path)

        # Walk directory descriptors with O_NOFOLLOW to close symlink/TOCTOU
        # races between the snapshot checks and the final file open.
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

        file_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        *directory_parts, file_name = relative.parts

        directory_descriptor = os.open(root, directory_flags)
        try:
            for component in directory_parts:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=directory_descriptor
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            file_descriptor = os.open(
                file_name, file_flags, dir_fd=directory_descriptor
            )
        finally:
            os.close(directory_descriptor)

        return file_descriptor, root.joinpath(*directory_parts, file_name)

    @staticmethod
    def _assert_signature(signature: _FileSignature) -> None:
        """Raise if a live file no longer matches its recorded signature."""
        path = Path(signature.resolved_path)
        current = path.stat()
        if not stat.S_ISREG(current.st_mode) or not FileSystemSkillSource._stat_matches(
            signature, current
        ):
            raise SkillAccessError(path=str(path))

    @staticmethod
    def _stat_matches(expected: _FileSignature, file_stat: os.stat_result) -> bool:
        return (
            expected.device == file_stat.st_dev
            and expected.inode == file_stat.st_ino
            and expected.size == file_stat.st_size
            and expected.modified_ns == file_stat.st_mtime_ns
        )

    @staticmethod
    def _file_signature(path: Path) -> _FileSignature:
        """Capture the identity and size attributes used for snapshot validation."""
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()

        if not stat.S_ISREG(file_stat.st_mode):
            raise SkillUnsafePathError(path=str(path))
        return _FileSignature(
            str(resolved),
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
        )

    @staticmethod
    def _snapshot_id(manifest: str) -> str:
        """Return the stable digest ID for a canonical snapshot manifest."""
        digest = hashlib.sha256(manifest.encode(_UTF8_ENCODING)).hexdigest()
        return f"{_SNAPSHOT_ID_PREFIX}{digest}"

    def _encode_snapshot(self, discovered: Iterable[_DiscoveredSkill]) -> str:
        """Serialize discovered records into a bounded snapshot manifest."""
        payload = {
            _SNAPSHOT_VERSION_KEY: _SNAPSHOT_VERSION,
            _SNAPSHOT_ROOTS_KEY: [str(root) for root in self.skill_paths],
            _SNAPSHOT_CONFIGURATION_KEY: {
                _SNAPSHOT_MAX_DEPTH_KEY: self.max_depth,
                _SNAPSHOT_MAX_INSTRUCTIONS_KEY: self.max_instructions_bytes,
                _SNAPSHOT_MAX_RESOURCE_KEY: self.max_resource_bytes,
            },
            _SNAPSHOT_SKILLS_KEY: {
                record.metadata.name: {
                    _SNAPSHOT_ROOT_KEY: str(record.root),
                    _SNAPSHOT_DIRECTORY_KEY: str(record.directory),
                    _SNAPSHOT_METADATA_KEY: record.metadata.model_dump(mode="json"),
                    _SNAPSHOT_SKILL_FILE_KEY: record.skill_file.to_dict(),
                    _SNAPSHOT_RESOURCES_KEY: {
                        path: signature.to_dict()
                        for path, signature in record.resources
                    },
                }
                for record in discovered
            },
        }
        snapshot = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        if len(snapshot.encode(_UTF8_ENCODING)) > _MAX_SNAPSHOT_BYTES:
            raise SkillCatalogError()
        return snapshot

    def _records_from_snapshot(self, snapshot_id: str) -> dict[str, _DiscoveredSkill]:
        """Reopen a filesystem snapshot identified by its digest."""
        if len(snapshot_id.encode(_UTF8_ENCODING)) > _MAX_SNAPSHOT_BYTES:
            raise SkillCatalogError()
        if not snapshot_id.startswith(_SNAPSHOT_ID_PREFIX):
            raise SkillCatalogError()

        digest = snapshot_id.removeprefix(_SNAPSHOT_ID_PREFIX)
        if len(digest) != _SNAPSHOT_ID_HEX_LENGTH:
            raise SkillCatalogError()
        try:
            int(digest, 16)
        except ValueError as exc:
            raise SkillCatalogError() from exc

        cached_records = self._snapshot_records.get(snapshot_id)
        if cached_records is not None:
            return cached_records

        try:
            records = self._discover_records_sync()
        except SkillError as exc:
            raise SkillCatalogError() from exc
        manifest = self._encode_snapshot(records.values())
        if self._snapshot_id(manifest) != snapshot_id:
            raise SkillCatalogError()
        self._snapshot_records[snapshot_id] = records
        return records

    def _record_from_snapshot(self, snapshot_id: str, name: str) -> _DiscoveredSkill:
        """Return one named skill from a validated persisted snapshot."""
        records = self._records_from_snapshot(snapshot_id)
        try:
            return records[name]
        except KeyError as exc:
            raise SkillNotFoundError(skill_name=name) from exc


def _frontmatter_parts(content: bytes) -> tuple[bytes, bytes]:
    """Split a document into its opening line and frontmatter bytes."""
    # Only the first frontmatter delimiter is positional; the next delimiter
    # closes the header and leaves the instruction body untouched.
    lines = content.splitlines(keepends=True)
    if not lines:
        raise ValueError("empty skill document")
    first_line, *body_lines = lines
    closing = next(
        (
            index
            for index, line in enumerate(body_lines)
            if line.strip() in _FRONTMATTER_DELIMITERS
        ),
        None,
    )
    if closing is None:
        raise ValueError("unclosed frontmatter")
    return first_line, b"".join(body_lines[:closing])


def _frontmatter_body(content: bytes) -> str:
    """Return the UTF-8 instruction body after the frontmatter delimiter."""
    lines = content.splitlines(keepends=True)
    if not lines:
        raise ValueError("empty skill document")
    _, *body_lines = lines
    closing = next(
        index
        for index, line in enumerate(body_lines)
        if line.strip() in _FRONTMATTER_DELIMITERS
    )
    return b"".join(body_lines[closing + 1 :]).decode(_UTF8_ENCODING).lstrip("\r\n")


@dataclass
class _FrontmatterValidationState:
    """Shared node counter for recursive frontmatter validation."""

    nodes: int = 0


def _validate_frontmatter_value(
    value: Any,
    *,
    depth: int = 0,
    state: _FrontmatterValidationState | None = None,
) -> None:
    """Reject frontmatter trees exceeding the parser's depth or node limits."""
    if state is None:
        state = _FrontmatterValidationState()
    state.nodes += 1
    if state.nodes > _MAX_FRONTMATTER_NODES or depth > _MAX_FRONTMATTER_DEPTH:
        raise ValueError("frontmatter is too complex")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_frontmatter_value(key, depth=depth + 1, state=state)
            _validate_frontmatter_value(item, depth=depth + 1, state=state)
    elif isinstance(value, list):
        for item in value:
            _validate_frontmatter_value(item, depth=depth + 1, state=state)


def _metadata_from_frontmatter(raw: Any, skill_name: str) -> SkillMetadata:
    """Map standard and extension frontmatter fields to skill metadata."""
    if not isinstance(raw, dict):
        raise SkillMalformedError(skill_name=skill_name, path=_SKILL_FILE_NAME)
    try:
        standard = {key: raw[key] for key in _STANDARD_FRONTMATTER_FIELDS if key in raw}
        extra = {
            key: value
            for key, value in raw.items()
            if key not in _STANDARD_FRONTMATTER_FIELDS
        }
        return SkillMetadata(**standard, extra_frontmatter=extra)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SkillMalformedError(skill_name=skill_name, path=_SKILL_FILE_NAME) from exc


def _validate_resource_path(name: str, path: str) -> None:
    """Reject resource paths that escape the skill or target hidden content."""
    try:
        _validate_skill_path(path)
    except (TypeError, ValueError) as exc:
        raise SkillUnsafePathError(skill_name=name, path=path) from exc
    parts = path.split("/")
    first_part, *_ = parts
    if (
        any(part.startswith(".") for part in parts)
        or path == _SKILL_FILE_NAME
        or first_part == _SCRIPTS_DIRECTORY_NAME
    ):
        raise SkillUnsafePathError(skill_name=name, path=path)


def _media_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or _DEFAULT_MEDIA_TYPE


__all__ = ["FileSystemSkillSource"]
