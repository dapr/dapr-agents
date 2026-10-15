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

"""Checks on ``.github/dependabot.yml``."""

from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / ".github" / "dependabot.yml"
PYTHON_MANIFESTS = ("pyproject.toml", "requirements.txt", "uv.lock")


def _updates() -> List[Dict[str, Any]]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["updates"]


def _dirs(entry: Dict[str, Any]) -> List[str]:
    if "directories" in entry:
        return list(entry["directories"])
    return [entry["directory"]]


def _python_manifest_dirs() -> Set[str]:
    found: Set[str] = set()
    for name in PYTHON_MANIFESTS:
        for path in PROJECT_ROOT.rglob(name):
            rel = path.parent.relative_to(PROJECT_ROOT)
            if any(part.startswith(".") for part in rel.parts):
                continue
            found.add("/" + rel.as_posix() if rel.parts else "/")
    return found


def test_every_python_manifest_is_covered() -> None:
    covered: Set[str] = set()
    for entry in _updates():
        if entry["package-ecosystem"] in ("uv", "pip"):
            covered.update(_dirs(entry))
    missing = _python_manifest_dirs() - covered
    assert not missing, f"Python manifests not covered by Dependabot: {missing}"


def test_uv_entries_only_where_a_lock_exists() -> None:
    for entry in _updates():
        if entry["package-ecosystem"] != "uv":
            continue
        for d in _dirs(entry):
            assert (PROJECT_ROOT / d.lstrip("/") / "uv.lock").is_file(), d


def test_root_uv_groups_minor_and_patch_only() -> None:
    root = [e for e in _updates() if e["package-ecosystem"] == "uv" and "/" in _dirs(e)]
    assert len(root) == 1
    groups = root[0]["groups"]
    assert len(groups) == 1
    (group,) = groups.values()
    assert group["patterns"] == ["*"]
    assert sorted(group["update-types"]) == ["minor", "patch"]


def test_every_entry_keeps_a_schedule() -> None:
    for entry in _updates():
        assert entry["schedule"]["interval"] == "daily"
