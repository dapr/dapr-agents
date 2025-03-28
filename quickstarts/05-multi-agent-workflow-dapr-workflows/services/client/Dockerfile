FROM python:3.10 AS builder

WORKDIR /app
COPY ./requirements.txt ./services/client/k8s_http_client.py ./


RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "http_client.py"]

