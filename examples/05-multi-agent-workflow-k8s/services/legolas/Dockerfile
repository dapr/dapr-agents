FROM debian:12-slim@sha256:e899040a73d36e2b36fa33216943539d9957cba8172b858097c2cabcdb20a3e2 AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-venv gcc libpython3-dev && \
    python3.11 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

FROM build AS build-venv
COPY ./quickstarts/05-multi-agent-workflows/requirements.txt /requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r /requirements.txt

FROM gcr.io/distroless/python3-debian12@sha256:8ce6bba3f793ba7d834467dfe18983c42f9b223604970273e9e3a22b1891fc27
COPY --from=build-venv /venv /venv
COPY ./quickstarts/05-multi-agent-workflows/services/legolas/app.py /app/app.py
WORKDIR /app
ENTRYPOINT ["/venv/bin/python3", "app.py"]
