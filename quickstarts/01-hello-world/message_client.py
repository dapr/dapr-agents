#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from typing import Any, Dict

from dapr.clients import DaprClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a TriggerAction to the TravelBuddy agent topic."
    )
    parser.add_argument(
        "--pubsub",
        default="messagepubsub",
        help="Dapr pub/sub component name (default: messagepubsub).",
    )
    parser.add_argument(
        "--topic",
        default="travel.requests",
        help="Topic to publish to (default: travel.requests).",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="I want to find flights to Paris",
        help="Simple task string to send (ignored if --payload is set).",
    )
    parser.add_argument(
        "--payload",
        help="Full JSON payload to publish. When set, overrides the positional task.",
    )
    parser.add_argument(
        "--content-type",
        default="application/json",
        help="Content type for the published event (default: application/json).",
    )
    parser.add_argument(
        "--source",
        default="quickstarts.travelbuddy.client",
        help="Metadata source identifier attached to the payload.",
    )
    parser.add_argument(
        "--cloudevent-type",
        help="Optional CloudEvent type metadata (sets cloudevent.type).",
    )
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.payload:
        try:
            data = json.loads(args.payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--payload must be valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("--payload must decode to a JSON object.")
        payload = dict(data)
    else:
        payload = {"task": args.task}

    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("source", args.source)
    metadata.setdefault("request_id", str(uuid.uuid4()))
    payload["metadata"] = metadata
    return payload


def publish(args: argparse.Namespace) -> None:
    payload = build_payload(args)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    logging.info(
        "Publishing to %s / %s with payload: %s",
        args.pubsub,
        args.topic,
        payload,
    )

    metadata = (
        {"cloudevent.type": args.cloudevent_type} if args.cloudevent_type else None
    )

    with DaprClient() as client:
        client.publish_event(
            pubsub_name=args.pubsub,
            topic_name=args.topic,
            data=body,
            data_content_type=args.content_type,
            publish_metadata=metadata,
        )

    logging.info("Publish succeeded.")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        args = parse_args()
        publish(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to publish message: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
