FROM debian:12-slim@sha256:1209d8fd77def86ceb6663deef7956481cc6c14a25e1e64daec12c0ceffcc19d AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-venv gcc libpython3-dev && \
    python3.11 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Build the virtualenv as a separate step: Only re-execute this step when requirements.txt changes
FROM build AS build-venv
COPY ./05-multi-agent-workflow-dapr-workflows/requirements.txt /requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r /requirements.txt

# Copy the virtualenv into a distroless image
FROM gcr.io/distroless/python3-debian12@sha256:7729c7979e1097d2b60fffb0af904a59e16af241b7c8eef4c613013ef1504ff8  
COPY --from=build-venv /venv /venv
COPY ./05-multi-agent-workflow-dapr-workflows/services/workflow-llm/app.py /app/app.py
WORKDIR /app

ENTRYPOINT ["/venv/bin/python3", "app.py"]
