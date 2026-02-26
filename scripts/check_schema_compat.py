"""Check for breaking changes between the latest and previous metadata schema.

Compares ``latest.json`` against the most recent *previous* versioned schema
file in ``schemas/agent-metadata/``.

Always exits 0 — this script is informational and never blocks CI.
Output is a markdown report written to stdout.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas" / "agent-metadata"
LATEST_FILE = SCHEMAS_DIR / "latest.json"
INDEX_FILE = SCHEMAS_DIR / "index.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _previous_version_file() -> Path | None:
    """Return the path to the second-most-recent versioned schema, or None."""
    index = _load_json(INDEX_FILE)
    versions: List[str] = index.get("available_versions", [])
    # available_versions is sorted descending (e.g. ["v0.13.0", "v0.12.0"])
    if len(versions) < 2:
        return None
    previous = versions[1]  # second entry is the previous version
    return SCHEMAS_DIR / f"{previous}.json"


def _collect_properties(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Return a mapping of definition name -> set of property keys."""
    props: Dict[str, Set[str]] = {}

    # Top-level properties
    if "properties" in schema:
        props["(root)"] = set(schema["properties"].keys())

    # $defs sub-schemas
    for name, defn in schema.get("$defs", {}).items():
        if "properties" in defn:
            props[name] = set(defn["properties"].keys())

    return props


def _collect_required(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Return a mapping of definition name -> set of required field names."""
    reqs: Dict[str, Set[str]] = {}

    if "required" in schema:
        reqs["(root)"] = set(schema["required"])

    for name, defn in schema.get("$defs", {}).items():
        if "required" in defn:
            reqs[name] = set(defn["required"])

    return reqs


def check_compat(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Return a list of breaking-change descriptions (empty = compatible)."""
    issues: List[str] = []

    old_props = _collect_properties(old)
    new_props = _collect_properties(new)

    # Check for removed properties
    for defn_name, old_keys in old_props.items():
        new_keys = new_props.get(defn_name, set())
        removed = old_keys - new_keys
        for key in sorted(removed):
            issues.append(f"Removed property `{key}` from `{defn_name}`")

    old_reqs = _collect_required(old)
    new_reqs = _collect_required(new)

    # Check for new required fields that didn't exist in the old schema
    for defn_name, new_req_fields in new_reqs.items():
        old_req_fields = old_reqs.get(defn_name, set())
        old_all_props = old_props.get(defn_name, set())
        added_reqs = new_req_fields - old_req_fields
        for key in sorted(added_reqs):
            if key not in old_all_props:
                issues.append(
                    f"New required field `{key}` in `{defn_name}` "
                    f"(did not exist in previous version)"
                )

    return issues


def main() -> None:
    prev_file = _previous_version_file()
    if prev_file is None or not prev_file.exists():
        print("No previous schema version found — skipping compatibility check.")
        sys.exit(0)

    latest = _load_json(LATEST_FILE)
    previous = _load_json(prev_file)

    issues = check_compat(old=previous, new=latest)

    if not issues:
        print("No breaking metadata schema changes detected.")
    else:
        print("### Breaking Metadata Schema Changes\n")
        for issue in issues:
            print(f"- {issue}")

    sys.exit(0)


if __name__ == "__main__":
    main()
