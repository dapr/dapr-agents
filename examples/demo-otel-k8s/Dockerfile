FROM python:3.12-slim

WORKDIR /app

# Copy the full dapr-agents source for local editable install
COPY . /src/dapr-agents

# Install from local source in editable mode
RUN pip install --no-cache-dir -e /src/dapr-agents

# Copy agent source
COPY examples/demo-otel-k8s/agents/ /app/agents/

# Default: override CMD in each deployment manifest
CMD ["python", "-m", "agents.instantiation.app"]
