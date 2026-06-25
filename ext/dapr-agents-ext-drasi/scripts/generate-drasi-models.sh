#!/usr/bin/env bash
set -euo pipefail

EXT_DIR="."
SCHEMA_DIR="$EXT_DIR/schemas/output-unpacked"
OUT_DIR="$EXT_DIR/dapr_agents/ext/drasi/schemas/unpacked"

# Remove output directory to avoid stale generated models
rm -rf "$OUT_DIR"

# Ensure development dependencies are installed
uv sync --package dapr-agents-ext-drasi --group dev

# Generate Pydantic models from JSON Schema files and place them into `$OUT_DIR`
uv run datamodel-codegen \
  --input-file-type jsonschema \
  --input "$SCHEMA_DIR" \
  --formatters isort black \
  --use-annotated \
  --output-model-type pydantic_v2.BaseModel \
  --output "$OUT_DIR"