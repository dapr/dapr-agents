from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
from typing import Any, Dict, List

from dapr.clients import DaprClient

# ---------------------------
# Env defaults (CLI can override)
# ---------------------------
PUBSUB_NAME = os.getenv("PUBSUB_NAME", "messagepubsub")

# Named orchestrator topics (used when --orchestrator is provided)
# These orchestrators coordinate multiple agents in the Fellowship
RANDOM_TOPIC_DEFAULT = os.getenv(
    "RANDOM_TOPIC", "fellowship.orchestrator.random.requests"
)
ROUNDROBIN_TOPIC_DEFAULT = os.getenv(
    "ROUNDROBIN_TOPIC", "fellowship.orchestrator.roundrobin.requests"
)
LLM_TOPIC_DEFAULT = os.getenv("LLM_TOPIC", "llm.orchestrator.requests")

# Individual agent topics (can be used with --topic for direct agent messaging)
# Frodo: fellowship.frodo.requests
# Sam: fellowship.sam.requests
# Gandalf: fellowship.gandalf.requests
# Legolas: fellowship.legolas.requests

# Legacy single-topic env (still honored if you pass --topic without a value)
ORCHESTRATOR_TOPIC_ENV = os.getenv("ORCHESTRATOR_TOPIC")

RAW_DATA = os.getenv("RAW_DATA")
TASK_TEXT_DEFAULT = os.getenv(
    "TASK_TEXT",
    "Set the next step for the journey to Mordor. Consider safety, supplies, and stealth. Start from the Shire to Bree.",
)

CONTENT_TYPE = os.getenv("CONTENT_TYPE", "application/json")
CLOUDEVENT_TYPE_DEFAULT = os.getenv("CLOUDEVENT_TYPE", "TriggerAction")

PUBLISH_ONCE_DEFAULT = os.getenv("PUBLISH_ONCE", "true").lower() in {"1", "true", "yes"}
INTERVAL_SEC_DEFAULT = float(os.getenv("INTERVAL_SEC", "0"))
MAX_ATTEMPTS_DEFAULT = int(os.getenv("MAX_ATTEMPTS", "8"))
INITIAL_DELAY_DEFAULT = float(os.getenv("INITIAL_DELAY", "0.5"))
BACKOFF_FACTOR_DEFAULT = float(os.getenv("BACKOFF_FACTOR", "2.0"))
JITTER_FRAC_DEFAULT = float(os.getenv("JITTER_FRAC", "0.2"))
STARTUP_DELAY_DEFAULT = float(os.getenv("STARTUP_DELAY", "5.0"))

logger = logging.getLogger("orchestrator_publisher")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish TriggerAction messages to orchestrator(s)."
    )
    # Orchestrator selection (mutually exclusive with explicit topics)
    p.add_argument(
        "--orchestrator",
        choices=["random", "roundrobin", "llm"],
        help="Route to a named orchestrator topic (or both).",
    )
    p.add_argument(
        "--topic",
        action="append",
        help="Explicit topic name (can be repeated). If provided, --orchestrator is ignored.",
    )

    # Message/data knobs
    p.add_argument("--task", help="Task text to send (ignored if --raw JSON is used).")
    p.add_argument("--raw", help="Raw JSON object to send as the event data.")
    p.add_argument(
        "--ce-type", default=CLOUDEVENT_TYPE_DEFAULT, help="CloudEvent type to set."
    )
    p.add_argument("--content-type", default=CONTENT_TYPE, help="Content type.")

    # Behavior
    p.add_argument("--pubsub", default=PUBSUB_NAME, help="Dapr pubsub name.")
    p.add_argument(
        "--once",
        dest="publish_once",
        action="store_true",
        help="Publish once and exit.",
    )
    p.add_argument(
        "--loop",
        dest="publish_once",
        action="store_false",
        help="Publish periodically.",
    )
    p.set_defaults(publish_once=PUBLISH_ONCE_DEFAULT)

    p.add_argument(
        "--interval",
        type=float,
        default=INTERVAL_SEC_DEFAULT,
        help="Interval seconds when looping.",
    )
    p.add_argument(
        "--startup-delay",
        type=float,
        default=STARTUP_DELAY_DEFAULT,
        help="Initial delay seconds.",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS_DEFAULT,
        help="Max retry attempts.",
    )
    p.add_argument(
        "--initial-delay",
        type=float,
        default=INITIAL_DELAY_DEFAULT,
        help="Initial backoff seconds.",
    )
    p.add_argument(
        "--backoff-factor",
        type=float,
        default=BACKOFF_FACTOR_DEFAULT,
        help="Backoff multiplier.",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=JITTER_FRAC_DEFAULT,
        help="± jitter fraction on backoff.",
    )

    # Topic defaults (so you can override per run)
    p.add_argument(
        "--random-topic",
        default=RANDOM_TOPIC_DEFAULT,
        help="Topic for random orchestrator.",
    )
    p.add_argument(
        "--roundrobin-topic",
        default=ROUNDROBIN_TOPIC_DEFAULT,
        help="Topic for round-robin orchestrator.",
    )
    p.add_argument(
        "--llm-topic", default=LLM_TOPIC_DEFAULT, help="Topic for LLM orchestrator."
    )

    return p.parse_args()


async def _backoff_sleep(delay: float, jitter: float, factor: float) -> float:
    actual = max(0.0, delay * (1 + random.uniform(-jitter, jitter)))
    if actual:
        await asyncio.sleep(actual)
    return delay * factor


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.raw or RAW_DATA:
        raw = args.raw or RAW_DATA
        try:
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid --raw/RAW_DATA JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("--raw/RAW_DATA must be a JSON object")
        return data
    return {"task": args.task or TASK_TEXT_DEFAULT}


def _encode_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _resolve_topics(args: argparse.Namespace) -> List[str]:
    # If explicit topics were passed, use them directly
    if args.topic:
        return args.topic

    # Otherwise use orchestrator selection
    if args.orchestrator == "random":
        return [args.random_topic]
    if args.orchestrator == "roundrobin":
        return [args.roundrobin_topic]
    if args.orchestrator == "llm":
        return [args.llm_topic]

    # Fallbacks:
    # 1) Legacy single-topic env if set
    if ORCHESTRATOR_TOPIC_ENV:
        return [ORCHESTRATOR_TOPIC_ENV]

    # 2) Default to random
    return [args.random_topic]


async def publish_once(
    client: DaprClient,
    payload: Dict[str, Any],
    *,
    pubsub_name: str,
    topics: List[str],
    content_type: str,
    ce_type: str | None,
    max_attempts: int,
    initial_delay: float,
    jitter: float,
    backoff_factor: float,
) -> None:
    delay = initial_delay
    body = _encode_payload(payload)

    for attempt in range(1, max_attempts + 1):
        try:
            for topic in topics:
                logger.info("publish attempt %d → %s/%s", attempt, pubsub_name, topic)
                client.publish_event(
                    pubsub_name=pubsub_name,
                    topic_name=topic,
                    data=body,
                    data_content_type=content_type,
                    publish_metadata=(
                        {"cloudevent.type": ce_type} if ce_type else None
                    ),
                )
                logger.info("published successfully to %s", topic)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish failed: %s", exc)
            if attempt == max_attempts:
                raise
            logger.info("retrying in ~%.2fs …", delay)
            delay = await _backoff_sleep(delay, jitter, backoff_factor)


async def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    stop_event = asyncio.Event()

    # Signal-aware shutdown
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        import signal as _signal

        _signal.signal(_signal.SIGINT, lambda *_: _stop())
        _signal.signal(_signal.SIGTERM, lambda *_: _stop())

    # Optional warmup
    if args.startup_delay > 0:
        await asyncio.sleep(args.startup_delay)

    topics = _resolve_topics(args)
    payload = _build_payload(args)
    logger.info("payload: %s", payload)
    logger.info("topics: %s", topics)

    try:
        with DaprClient() as client:
            if args.publish_once:
                await publish_once(
                    client,
                    payload,
                    pubsub_name=args.pubsub,
                    topics=topics,
                    content_type=args.content_type,
                    ce_type=args.ce_type,
                    max_attempts=args.max_attempts,
                    initial_delay=args.initial_delay,
                    jitter=args.jitter,
                    backoff_factor=args.backoff_factor,
                )
                await asyncio.sleep(0.2)
                return 0

            # periodic mode
            if args.interval <= 0:
                logger.error("--interval must be > 0 when --loop is used")
                return 2

            logger.info("starting periodic publisher every %.2fs", args.interval)
            while not stop_event.is_set():
                try:
                    payload = _build_payload(args)
                    await publish_once(
                        client,
                        payload,
                        pubsub_name=args.pubsub,
                        topics=topics,
                        content_type=args.content_type,
                        ce_type=args.ce_type,
                        max_attempts=args.max_attempts,
                        initial_delay=args.initial_delay,
                        jitter=args.jitter,
                        backoff_factor=args.backoff_factor,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "giving up after %d attempts: %s", args.max_attempts, exc
                    )

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=args.interval)
                except asyncio.TimeoutError:
                    pass

            logger.info("shutdown requested; exiting")
            return 0

    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
