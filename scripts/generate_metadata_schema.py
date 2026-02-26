import argparse
import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional

from dapr_agents.agents.configs import AgentMetadataSchema

logger = logging.getLogger(__name__)

# Constants
PACKAGE_NAME = "dapr-agents"
DEV_VERSION = "0.0.0.dev0"
SCHEMA_SUBDIR = "agent-metadata"
LATEST_FILENAME = "latest.json"
INDEX_FILENAME = "index.json"
VERSION_FILE_PREFIX = "v"
SCHEMA_BASE_URL = (
    "https://raw.githubusercontent.com/dapr/dapr-agents/main/schemas/agent-metadata"
)


def get_auto_version() -> str:
    """Get current package version automatically."""
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        logger.warning(
            "Package '%s' not found; falling back to dev version '%s'",
            PACKAGE_NAME,
            DEV_VERSION,
        )
        return DEV_VERSION


def generate_schema(output_dir: Path, schema_version: Optional[str] = None) -> None:
    """
    Generate versioned schema files.

    Args:
        output_dir: Directory to output schema files.
        schema_version: Specific version to use. If None, auto-detects from package.
    """
    current_version = schema_version or get_auto_version()

    logger.info("Generating schema for version: %s", current_version)
    schema_dir = output_dir / SCHEMA_SUBDIR
    schema_dir.mkdir(parents=True, exist_ok=True)

    # Export schema
    schema: Dict[str, Any] = AgentMetadataSchema.export_json_schema(current_version)

    # Write versioned file
    version_file = schema_dir / f"{VERSION_FILE_PREFIX}{current_version}.json"
    try:
        with open(version_file, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info("Generated %s", version_file)
    except OSError as exc:
        logger.error("Failed to write %s: %s", version_file, exc)
        raise

    # Write latest.json
    latest_file = schema_dir / LATEST_FILENAME
    try:
        with open(latest_file, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info("Generated %s", latest_file)
    except OSError as exc:
        logger.error("Failed to write %s: %s", latest_file, exc)
        raise

    # Write index with all versions
    index: Dict[str, Any] = {
        "current_version": current_version,
        "schema_url": f"{SCHEMA_BASE_URL}/{VERSION_FILE_PREFIX}{current_version}.json",
        "available_versions": sorted(
            [f.stem for f in schema_dir.glob(f"{VERSION_FILE_PREFIX}*.json")],
            reverse=True,
        ),
    }

    index_file = schema_dir / INDEX_FILENAME
    try:
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        logger.info("Generated %s", index_file)
    except OSError as exc:
        logger.error("Failed to write %s: %s", index_file, exc)
        raise

    logger.info("Schema generation complete for version %s", current_version)


def main() -> None:
    """Main entry point with CLI argument parsing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate JSON schema files for agent metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect version from installed package
  python scripts/generate_metadata_schema.py

  # Generate schema for specific version
  python scripts/generate_metadata_schema.py --version 1.0.0

  # Generate for pre-release
  python scripts/generate_metadata_schema.py --version 1.1.0-rc1

  # Custom output directory
  python scripts/generate_metadata_schema.py --version 1.0.0 --output ./custom-schemas
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

    generate_schema(schemas_dir, schema_version=args.version)


if __name__ == "__main__":
    main()
