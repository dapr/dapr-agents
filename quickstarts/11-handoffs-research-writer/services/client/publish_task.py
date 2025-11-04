#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import sys
import uuid
from typing import Any, Dict

from dapr.clients import DaprClient

logger = logging.getLogger("research.publisher")

PUBSUB_NAME = os.getenv("PUBSUB_NAME", "messagepubsub")
TOPIC_NAME = os.getenv("TOPIC_NAME", "research.requests")
CONTENT_TYPE = os.getenv("CONTENT_TYPE", "application/json")
CLIENT_SOURCE = os.getenv("CLIENT_SOURCE", "quickstarts.research.client")
RAW_DATA = os.getenv("RAW_DATA")

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "8"))
INITIAL_DELAY = float(os.getenv("INITIAL_DELAY", "0.5"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))
JITTER_FRAC = float(os.getenv("JITTER_FRAC", "0.2"))
STARTUP_DELAY = float(os.getenv("STARTUP_DELAY", "1.0"))


def _build_payload(task_text: str | None) -> Dict[str, Any]:
    if RAW_DATA:
        try:
            payload = json.loads(RAW_DATA)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid RAW_DATA JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("RAW_DATA must decode to a JSON object.")
    else:
        if not task_text:
            raise ValueError("Task text is required when RAW_DATA is not provided.")
        payload = {"task": task_text}

    payload.setdefault("_message_metadata", {})
    metadata = dict(payload["_message_metadata"])
    metadata.setdefault("source", CLIENT_SOURCE)
    metadata.setdefault("request_id", str(uuid.uuid4()))
    payload["_message_metadata"] = metadata
    return payload


async def _backoff_sleep(delay: float) -> float:
    jitter = delay * JITTER_FRAC
    actual = max(0.0, delay + random.uniform(-jitter, jitter))
    if actual:
        await asyncio.sleep(actual)
    return delay * BACKOFF_FACTOR


async def publish_with_retry(client: DaprClient, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    delay = INITIAL_DELAY

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "Publish attempt %d → %s/%s", attempt, PUBSUB_NAME, TOPIC_NAME
            )
            client.publish_event(
                pubsub_name=PUBSUB_NAME,
                topic_name=TOPIC_NAME,
                data=body,
                data_content_type=CONTENT_TYPE,
            )
            logger.info("Published request successfully.")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Publish attempt %d failed: %s", attempt, exc)
            if attempt == MAX_ATTEMPTS:
                raise
            logger.info("Retrying in ~%.2fs …", delay)
            delay = await _backoff_sleep(delay)


async def main() -> int:
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        task_text = " ".join(sys.argv[1:])
    elif not RAW_DATA:
        print("Usage: python publish_task.py \"<research topic>\"")
        return 1
    else:
        task_text = None

    payload = _build_payload(task_text)
    logger.info("Payload: %s", payload)

    if STARTUP_DELAY > 0:
        await asyncio.sleep(STARTUP_DELAY)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: _stop())
        signal.signal(signal.SIGTERM, lambda *_: _stop())

    try:
        with DaprClient() as client:
            await publish_with_retry(client, payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to publish after %d attempts: %s", MAX_ATTEMPTS, exc)
        return 1

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=0.2)
    except asyncio.TimeoutError:
        pass
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)
