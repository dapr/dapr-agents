"""Golden-file tests for AgentMetadataSchema.

Ensures the Pydantic model stays in sync with the committed schema files.
"""

import json
from difflib import unified_diff
from pathlib import Path

import pytest

from dapr_agents.agents.configs import AgentMetadataSchema

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas" / "agent-metadata"
LATEST_FILE = SCHEMAS_DIR / "latest.json"
INDEX_FILE = SCHEMAS_DIR / "index.json"


def _current_schema() -> dict:
    """Generate schema from the model using the version in latest.json."""
    with open(LATEST_FILE) as f:
        on_disk = json.load(f)
    version = on_disk.get("version", "0.0.0")
    return AgentMetadataSchema.export_json_schema(version)


@pytest.mark.schema
def test_schema_matches_golden_file():
    """Fail if the Pydantic model produces a schema different from latest.json.

    This catches "drift" — model changes that weren't followed by
    ``python scripts/generate_metadata_schema.py``.
    """
    with open(LATEST_FILE) as f:
        on_disk = json.load(f)

    from_model = _current_schema()

    on_disk_text = json.dumps(on_disk, indent=2, sort_keys=True) + "\n"
    from_model_text = json.dumps(from_model, indent=2, sort_keys=True) + "\n"

    if on_disk_text != from_model_text:
        diff = "".join(
            unified_diff(
                on_disk_text.splitlines(keepends=True),
                from_model_text.splitlines(keepends=True),
                fromfile="schemas/agent-metadata/latest.json (on disk)",
                tofile="AgentMetadataSchema.export_json_schema() (from model)",
            )
        )
        pytest.fail(
            f"Schema drift detected! latest.json does not match the model.\n\n"
            f"{diff}\n\n"
            f"Run 'python scripts/generate_metadata_schema.py' to regenerate."
        )


@pytest.mark.schema
def test_index_lists_latest_version():
    """Sanity-check that index.json is consistent with latest.json."""
    with open(LATEST_FILE) as f:
        latest = json.load(f)
    with open(INDEX_FILE) as f:
        index = json.load(f)

    latest_version = latest.get("version")
    assert latest_version, "latest.json is missing a 'version' field"

    assert index["current_version"] == latest_version, (
        f"index.json current_version ({index['current_version']}) "
        f"!= latest.json version ({latest_version})"
    )

    expected_entry = f"v{latest_version}"
    assert expected_entry in index["available_versions"], (
        f"{expected_entry} not found in index.json available_versions: "
        f"{index['available_versions']}"
    )
