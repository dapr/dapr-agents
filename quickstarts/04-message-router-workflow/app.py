from __future__ import annotations

import asyncio
import logging
import signal

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dotenv import load_dotenv
from workflow import (
    blog_workflow,
    create_outline,
    write_post,
)

from dapr_agents.workflow.utils.registration import register_message_routes

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _wait_for_shutdown() -> None:
    """Block until Ctrl+C or SIGTERM."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _set_stop(*_: object) -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _set_stop)
        loop.add_signal_handler(signal.SIGTERM, _set_stop)
    except NotImplementedError:
        # Windows fallback
        signal.signal(signal.SIGINT, lambda *_: _set_stop())
        signal.signal(signal.SIGTERM, lambda *_: _set_stop())

    await stop.wait()


async def main() -> None:
    runtime = wf.WorkflowRuntime()

    runtime.register_workflow(blog_workflow)
    runtime.register_activity(create_outline)
    runtime.register_activity(write_post)

    runtime.start()

    try:
        with DaprClient() as client:
            # Wire streaming subscriptions for our router(s)
            closers = register_message_routes(
                targets=[blog_workflow],
                dapr_client=client,
            )

            try:
                await _wait_for_shutdown()
            finally:
                for close in closers:
                    try:
                        close()
                    except Exception:
                        logger.exception("Error while closing subscription")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
