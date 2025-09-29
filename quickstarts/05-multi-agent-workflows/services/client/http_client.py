#!/usr/bin/env python3
import requests
import time
import sys


if __name__ == "__main__":
    status_url = "http://localhost:8004/status"
    healthy = False
    for attempt in range(1, 11):
        try:
            print(f"Attempt {attempt}...")
            response = requests.get(status_url, timeout=5)

            if response.status_code == 200:
                print("Workflow app is healthy!")
                healthy = True
                break
            else:
                print(f"Received status code {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        attempt += 1
        print("Waiting 5s seconds before next health checkattempt...")
        time.sleep(5)

    if not healthy:
        print("Workflow app is not healthy!")
        sys.exit(1)

    workflow_url = "http://localhost:8004/start-workflow"
    task_payload = {"task": "How to get to Mordor? We all need to help!"}

    try:
        print("Starting workflow...")
        response = requests.post(workflow_url, json=task_payload, timeout=120)

        if response.status_code == 202:
            print("Workflow started successfully!")
            print("Check Phoenix UI for workflow progress.")
            sys.exit(0)
        else:
            print(f"Received status code {response.status_code}: {response.text}")
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)
