from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

import requests

BASE_URL = os.getenv("WORKFLOW_BASE_URL", "http://localhost:8004")
ENTRY_PATH = os.getenv("WORKFLOW_ENTRY_PATH", "/run")
TASK_PROMPT = os.getenv(
    "WORKFLOW_TASK", "How do we get everyone to Mordor? We all need to help!"
)
MAX_ATTEMPTS = int(os.getenv("WORKFLOW_MAX_ATTEMPTS", "10"))
RETRY_DELAY_SECONDS = int(os.getenv("WORKFLOW_RETRY_DELAY_SECONDS", "5"))


def build_workflow_url() -> str:
    base = BASE_URL.rstrip("/")
    path = ENTRY_PATH if ENTRY_PATH.startswith("/") else f"/{ENTRY_PATH}"
    return f"{base}{path}"


def start_workflow() -> None:
    workflow_url = build_workflow_url()
    payload: Dict[str, Any] = {"task": TASK_PROMPT}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}: POST {workflow_url}")
        try:
            response = requests.post(workflow_url, json=payload, timeout=60)
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            print(f"Request failed: {exc}")
        else:
            if response.status_code in (200, 202):
                print("Workflow scheduled successfully!")
                try:
                    body = response.json()
                except ValueError:
                    body = {}

                instance_id = body.get("instance_id")
                status_url = body.get("status_url")
                if instance_id:
                    print(f"Instance ID: {instance_id}")
                if status_url:
                    print(f"Status URL: {status_url}")
                else:
                    print("Check Phoenix UI or Dapr workflow queries for progress.")
                return

            print(
                f"Received status code {response.status_code}: {response.text.strip()}"
            )

        if attempt < MAX_ATTEMPTS:
            print(f"Waiting {RETRY_DELAY_SECONDS}s before retrying...")
            time.sleep(RETRY_DELAY_SECONDS)

    print("Workflow app did not respond in time.")
    sys.exit(1)


if __name__ == "__main__":
    start_workflow()
    sys.exit(0)
