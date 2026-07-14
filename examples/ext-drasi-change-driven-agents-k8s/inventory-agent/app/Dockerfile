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

# ------------------------------------------
# Build stage
# ------------------------------------------
FROM python:3.13-slim-bookworm AS build

# Copy uv binary from distroless image
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install project dependencies only (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=ext/dapr-agents-ext-drasi,target=ext/dapr-agents-ext-drasi,rw \
    uv sync --frozen --no-install-project --package ext-drasi-change-driven-agents-k8s

COPY uv.lock pyproject.toml /app
COPY dapr_agents /app/dapr_agents
COPY ext/dapr-agents-ext-drasi /app/ext/dapr-agents-ext-drasi

COPY examples/ext-drasi-change-driven-agents-k8s/inventory-agent/app /app

# Build wheel and install it into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --package ext-drasi-change-driven-agents-k8s

# ------------------------------------------
# Runtime stage
# ------------------------------------------
FROM python:3.13-slim-bookworm

WORKDIR /app

COPY --from=build /app/.venv .venv
COPY --from=build /app/agent.py .
COPY --from=build /app/app.py .

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "app.py"]