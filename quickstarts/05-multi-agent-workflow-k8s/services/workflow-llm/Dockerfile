FROM astral/uv:python3.13-bookworm-slim@sha256:ae4e4c788a13a5a57c35740d1f648c0701936bf4e546b7e197d1210691977b6d AS build

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

ENV UV_NO_DEV=1
ENV UV_PYTHON_DOWNLOADS=0

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --package 05-multi-agent-workflows
COPY uv.lock pyproject.toml /app/
COPY dapr_agents /app/dapr_agents
COPY quickstarts/05-multi-agent-workflows/services/workflow-llm/app.py /app/app.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --package 05-multi-agent-workflows

FROM python:3.13-slim-bookworm@sha256:97e9392d12279f8c180eb80f0c7c0f3dfe5650f0f2573f7ad770aea58f75ed12

COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app
ENTRYPOINT ["/app/.venv/bin/python", "app.py"]
