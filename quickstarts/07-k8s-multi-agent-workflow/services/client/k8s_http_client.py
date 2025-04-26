#!/usr/bin/env python3
import sys
import requests
import time


def call_trigger_job(task):
    task_payload = {"task": task}
    base_url = "http://localhost:8004"
    workflow_url = f"{base_url}/start-workflow"
    status_url = f"{base_url}/status"

    for attempt in range(1, 11):
        try:
            print(f"Attempt {attempt}...")
            response = requests.get(status_url, timeout=5)

            if response.status_code == 200:
                print("Workflow app is healthy!")
                break
            else:
                print(f"Received status code {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        attempt += 1
        print("Waiting 5s seconds before next health checkattempt...")
        time.sleep(5)

    for attempt in range(1, 11):
        try:
            print(f"Attempt {attempt}...")
            response = requests.post(workflow_url, json=task_payload, timeout=5)

            if response.status_code == 202:
                print("Workflow started successfully!")
                sys.exit(0)
            else:
                print(f"Received status code {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        attempt += 1
        print("Waiting 1s seconds before next attempt...")
        time.sleep(1)

    print("Maximum attempts (10) reached without success.")

    print("Failed to get successful response")
    sys.exit(1)


if __name__ == "__main__":
    task = "How to get to Mordor? We all need to help!"
    call_trigger_job(task)
