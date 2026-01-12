FROM debian:12-slim@sha256:d5d3f9c23164ea16f31852f95bd5959aad1c5e854332fe00f7b3a20fcc9f635c AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-venv gcc libpython3-dev && \
    python3.11 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

FROM build AS build-venv
COPY ./quickstarts/05-multi-agent-workflows/requirements.txt /requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r /requirements.txt

FROM gcr.io/distroless/python3-debian12@sha256:2971dbfa56eb4f2e33e94709579783406c1e7165b2d92edaa5657da5e8040617
COPY --from=build-venv /venv /venv
COPY ./quickstarts/05-multi-agent-workflows/services/legolas/app.py /app/app.py
WORKDIR /app
ENTRYPOINT ["/venv/bin/python3", "app.py"]
