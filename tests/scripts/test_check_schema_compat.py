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

"""Tests for ``scripts/check_schema_compat.py``."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_schema_compat.py"
LATEST_SCHEMA = PROJECT_ROOT / "schemas" / "agent-metadata" / "latest.json"


def _load_script_module():
    """Load the standalone script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("check_schema_compat", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = _load_script_module()

OLD_SCHEMA: Dict[str, Any] = {
    "properties": {"a": {}, "b": {}},
    "required": ["a"],
    "$defs": {
        "Thing": {"properties": {"x": {}, "y": {}}, "required": ["x"]},
    },
}


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
    )


def _write_schema(path: Path, schema: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(schema))
    return path


class TestCheckCompat:
    def test_identical_schemas_are_compatible(self):
        assert compat.check_compat(old=OLD_SCHEMA, new=OLD_SCHEMA) == []

    def test_removed_root_property_is_breaking(self):
        new = {**OLD_SCHEMA, "properties": {"a": {}}}
        issues = compat.check_compat(old=OLD_SCHEMA, new=new)
        assert issues == ["Removed property `b` from `(root)`"]

    def test_removed_defs_property_is_breaking(self):
        new = {
            **OLD_SCHEMA,
            "$defs": {"Thing": {"properties": {"x": {}}, "required": ["x"]}},
        }
        issues = compat.check_compat(old=OLD_SCHEMA, new=new)
        assert issues == ["Removed property `y` from `Thing`"]

    def test_removed_definition_reports_all_its_properties(self):
        new = {k: v for k, v in OLD_SCHEMA.items() if k != "$defs"}
        issues = compat.check_compat(old=OLD_SCHEMA, new=new)
        assert issues == [
            "Removed property `x` from `Thing`",
            "Removed property `y` from `Thing`",
        ]

    def test_new_required_field_unknown_to_old_schema_is_breaking(self):
        new = {
            **OLD_SCHEMA,
            "properties": {"a": {}, "b": {}, "c": {}},
            "required": ["a", "c"],
        }
        issues = compat.check_compat(old=OLD_SCHEMA, new=new)
        assert issues == [
            "New required field `c` in `(root)` (did not exist in previous version)"
        ]

    def test_promoting_existing_optional_property_is_not_breaking(self):
        new = {**OLD_SCHEMA, "required": ["a", "b"]}
        assert compat.check_compat(old=OLD_SCHEMA, new=new) == []


class TestCli:
    def test_breaking_change_renders_report_and_exits_zero(self, tmp_path):
        old = _write_schema(tmp_path / "old.json", OLD_SCHEMA)
        new = _write_schema(
            tmp_path / "new.json", {**OLD_SCHEMA, "properties": {"a": {}}}
        )
        result = _run_cli("--old", str(old), "--new", str(new))
        assert result.returncode == 0
        assert "### Breaking Metadata Schema Changes" in result.stdout
        assert "Removed property `b` from `(root)`" in result.stdout

    def test_compatible_schemas_report_no_changes(self, tmp_path):
        old = _write_schema(tmp_path / "old.json", OLD_SCHEMA)
        new = _write_schema(tmp_path / "new.json", OLD_SCHEMA)
        result = _run_cli("--old", str(old), "--new", str(new))
        assert result.returncode == 0
        assert "No breaking metadata schema changes detected." in result.stdout

    def test_missing_baseline_file_exits_zero(self, tmp_path):
        result = _run_cli("--old", str(tmp_path / "absent.json"))
        assert result.returncode == 0
        assert "Baseline file not found" in result.stdout

    def test_missing_candidate_file_exits_zero(self, tmp_path):
        result = _run_cli(
            "--old", str(LATEST_SCHEMA), "--new", str(tmp_path / "absent.json")
        )
        assert result.returncode == 0
        assert "Candidate file not found" in result.stdout

    def test_new_defaults_to_repo_latest_schema(self):
        result = _run_cli("--old", str(LATEST_SCHEMA))
        assert result.returncode == 0
        assert "No breaking metadata schema changes detected." in result.stdout

    def test_no_arguments_uses_index_fallback_and_exits_zero(self):
        result = _run_cli()
        assert result.returncode == 0
        assert result.stdout.strip()
