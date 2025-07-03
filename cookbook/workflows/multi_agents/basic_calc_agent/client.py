#!/usr/bin/env python3
import json
import sys
import time
from dapr.clients import DaprClient

# Default Pub/Sub component
PUBSUB_NAME = "pubsub"


def main(orchestrator_topic, max_attempts=10, retry_delay=1):
    """
    Publishes a task to a specified Dapr Pub/Sub topic with retries.

    Args:
        orchestrator_topic (str): The name of the orchestrator topic.
        max_attempts (int): Maximum number of retry attempts.
        retry_delay (int): Delay in seconds between attempts.
    """
    task_message = {
        "task": "What is 1 + 1?",
    }

    time.sleep(5)

    attempt = 1

    while attempt <= max_attempts:
        try:
            print(
                f"📢 Attempt {attempt}: Publishing to topic '{orchestrator_topic}'..."
            )

            with DaprClient() as client:
                client.publish_event(
                    pubsub_name=PUBSUB_NAME,
                    topic_name=orchestrator_topic,
                    data=json.dumps(task_message),
                    data_content_type="application/json",
                    publish_metadata={
                        "cloudevent.type": "TriggerAction",
                    },
                )

            print(f"✅ Successfully published request to '{orchestrator_topic}'")
            sys.exit(0)

        except Exception as e:
            print(f"❌ Request failed: {e}")

        attempt += 1
        print(f"⏳ Waiting {retry_delay}s before next attempt...")
        time.sleep(retry_delay)

    print(f"❌ Maximum attempts ({max_attempts}) reached without success.")
    sys.exit(1)


if __name__ == "__main__":
    orchestrator_topic = "LLMOrchestrator"

    main(orchestrator_topic)
