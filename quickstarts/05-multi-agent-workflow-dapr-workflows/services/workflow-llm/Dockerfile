FROM python:3.10 AS builder

WORKDIR /app
COPY ./requirements.txt ./services/workflow-llm/app.py ./


RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "app.py"]

