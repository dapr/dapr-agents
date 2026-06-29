#!/usr/bin/env bash
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