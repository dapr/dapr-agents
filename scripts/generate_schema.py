import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

from dapr_agents.agents.configs import AgentMetadataSchema


def get_auto_version() -> str:
    """Get current package version automatically."""
    try:
        return version("dapr-ext-agent_core")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def generate_schema(output_dir: Path, schema_version: Optional[str] = None):
    """
    Generate versioned schema files.

    Args:
        output_dir: Directory to output schema files
        schema_version: Specific version to use. If None, auto-detects from package.
    """
    # Use provided version or auto-detect
    current_version = schema_version or get_auto_version()

    print(f"Generating schema for version: {current_version}")
    schema_dir = output_dir / "agent-metadata"

    # Export schema
    schema: dict[Any, Any] = AgentMetadataSchema.export_json_schema(current_version)

    # Write versioned file
    version_file = schema_dir / f"v{current_version}.json"
    with open(version_file, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Generated {version_file}")

    # Write latest.json
    latest_file = schema_dir / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Generated {latest_file}")

    # Write index with all versions
    index: dict[Any, Any] = {
        "current_version": current_version,
        "schema_url": f"https://raw.githubusercontent.com/dapr/dapr-agents/main/schemas/agent-metadata/v{current_version}.json",
        "available_versions": sorted(
            [f.stem for f in schema_dir.glob("v*.json")], reverse=True
        ),
    }

    index_file = schema_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)
    print(f"✓ Generated {index_file}")
    print(f"\nSchema generation complete for version {current_version}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate JSON schema files for agent metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect version from installed package
  python scripts/generate_schema.py

  # Generate schema for specific version
  python scripts/generate_schema.py --version 1.0.0

  # Generate for pre-release
  python scripts/generate_schema.py --version 1.1.0-rc1

  # Custom output directory
  python scripts/generate_schema.py --version 1.0.0 --output ./custom-schemas
        """,
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=None,
        help="Specific version to use for schema generation. If not provided, auto-detects from installed package.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for schemas. Defaults to 'schemas' in repo root.",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        schemas_dir = args.output
    else:
        repo_root = Path(__file__).parent.parent
        schemas_dir = repo_root / "schemas"

    # Generate schemas
    generate_schema(schemas_dir, schema_version=args.version)


if __name__ == "__main__":
    main()
