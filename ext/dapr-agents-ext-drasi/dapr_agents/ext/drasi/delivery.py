#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Streaming inbox delivery and scheduling for agent-managed Drasi events."""

from __future__ import annotations

import json
import logging
from threading import Event, Lock, Thread, current_thread

from dapr.clients import DaprClient
from dapr.clients.grpc._response import TopicEventResponse, TopicEventResponseStatus
from dapr.clients.grpc.subscription import Subscription
from dapr.common.pubsub.subscription import (
    StreamCancelledError,
    StreamInactiveError,
    SubscriptionMessage,
)
from dapr.ext.workflow import DaprWorkflowClient
from grpc import RpcError, StatusCode

from ._interfaces import AdmissionHandler
from ._models import Discard, Poison, ResolvedDrasiConfig, Retry, SchedulingInput

logger = logging.getLogger(__name__)

_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class DrasiDeliveryError(Exception):
    """The inbox consumer could not start, remain active, or close cleanly."""


class DrasiInbox:
    """Owned dynamic-inbox consumer with retryable, resource-specific cleanup."""

    def __init__(
        self,
        *,
        config: ResolvedDrasiConfig,
        admission: AdmissionHandler,
        dapr_client: DaprClient,
        workflow_client: DaprWorkflowClient,
    ) -> None:
        self._config = config
        self._admission = admission
        self._dapr_client = dapr_client
        self._workflow_client = workflow_client
        try:
            self._subscription: Subscription = dapr_client.subscribe(
                pubsub_name=config.pubsub_name,
                topic=config.scope.inbox_topic,
                dead_letter_topic=config.scope.dead_letter_topic,
            )
        except Exception as error:
            logger.error("Drasi inbox subscription failed (%s).", type(error).__name__)
            raise DrasiDeliveryError(
                "Could not subscribe to the Drasi inbox."
            ) from None

        self._stopped = Event()
        self._subscription_lock = Lock()
        self._lifecycle_lock = Lock()
        self._transport_closed = False
        self._subscription_closed = False
        self._closed = False
        try:
            self._thread = Thread(
                target=self._consume,
                name="drasi-inbox",
                daemon=True,
            )
            self._thread.start()
        except Exception as error:
            self._stopped.set()
            logger.error(
                "Drasi inbox consumer startup failed (%s).", type(error).__name__
            )
            self._close_subscription()
            raise DrasiDeliveryError(
                "Could not start the Drasi inbox consumer."
            ) from None

    @property
    def is_stopped(self) -> bool:
        """Return whether the consumer thread can no longer use its dependencies."""
        return not self._thread.is_alive()

    @property
    def is_closed(self) -> bool:
        """Return whether all owned inbox resources are confirmed closed."""
        return self._closed

    def _close_subscription(self) -> None:
        with self._subscription_lock:
            try:
                self._subscription.close()
            except Exception as error:
                logger.error(
                    "Drasi inbox stream close failed (%s).", type(error).__name__
                )
                raise DrasiDeliveryError(
                    "Could not close the Drasi inbox stream."
                ) from None
            self._subscription_closed = True

    def _consume(self) -> None:
        try:
            for message in self._subscription:
                if self._stopped.is_set():
                    break
                if message is None:
                    continue
                response = _handle_message(
                    message,
                    admission=self._admission,
                    workflow_client=self._workflow_client,
                    workflow_name=self._config.workflow_name,
                )
                if self._stopped.is_set():
                    break
                self._subscription.respond(message, response.status)
            if not self._stopped.is_set():
                logger.error("Drasi inbox stream ended unexpectedly.")
        except (StreamCancelledError, StreamInactiveError) as error:
            if not self._stopped.is_set():
                logger.error("Drasi inbox stream stopped (%s).", type(error).__name__)
        except Exception as error:
            if not self._stopped.is_set():
                logger.error("Drasi inbox consumer failed (%s).", type(error).__name__)
        finally:
            try:
                self._close_subscription()
            except DrasiDeliveryError:
                logger.error("Drasi consumer exited without confirmed stream cleanup.")

    def close(self) -> None:
        """Stop intake and retry only the inbox resources still unresolved."""
        with self._lifecycle_lock:
            if self._closed:
                return

            self._stopped.set()
            errors: list[Exception] = []
            if not self._transport_closed:
                try:
                    self._dapr_client.close()
                except Exception as error:
                    logger.error(
                        "Drasi inbox Dapr client close failed (%s).",
                        type(error).__name__,
                    )
                    errors.append(
                        DrasiDeliveryError(
                            "Could not close the Drasi inbox Dapr client."
                        )
                    )
                else:
                    self._transport_closed = True

            if not self._subscription_closed:
                try:
                    self._close_subscription()
                except DrasiDeliveryError as error:
                    errors.append(error)

            if current_thread() is self._thread:
                logger.error("Drasi inbox consumer cannot join its own thread.")
                errors.append(
                    DrasiDeliveryError("The Drasi inbox consumer cannot close itself.")
                )
            else:
                self._thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
                if self._thread.is_alive():
                    logger.error(
                        "Drasi inbox consumer did not stop before its shutdown deadline."
                    )
                    errors.append(
                        DrasiDeliveryError(
                            "The Drasi inbox consumer did not stop in time."
                        )
                    )

            self._closed = (
                self._transport_closed
                and self._subscription_closed
                and not self._thread.is_alive()
            )
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("Could not close the Drasi inbox.", errors)

    def __call__(self) -> None:
        """Close the inbox for compatibility with existing private callers."""
        self.close()


def _handle_message(
    message: SubscriptionMessage,
    *,
    admission: AdmissionHandler,
    workflow_client: DaprWorkflowClient,
    workflow_name: str,
) -> TopicEventResponse:
    try:
        media_type = message.data_content_type().split(";", 1)[0].strip().lower()
        if media_type != "application/json" and not (
            media_type.startswith("application/") and media_type.endswith("+json")
        ):
            logger.warning(
                "Drasi inbox rejected non-JSON data; requesting dead letter."
            )
            return TopicEventResponse(TopicEventResponseStatus.drop)
        # The SDK has already separated CloudEvent metadata from these data bytes.
        data = json.loads(message.raw_data().decode("utf-8"))
    except (ValueError, RecursionError):
        logger.warning(
            "Drasi inbox could not decode JSON data; requesting dead letter."
        )
        return TopicEventResponse(TopicEventResponseStatus.drop)

    try:
        result = admission.admit(data)
    except Exception as error:
        logger.error(
            "Drasi inbox admission failed (%s); requesting retry.",
            type(error).__name__,
        )
        return TopicEventResponse(TopicEventResponseStatus.retry)

    if isinstance(result, Discard):
        return TopicEventResponse(TopicEventResponseStatus.success)
    if isinstance(result, Retry):
        return TopicEventResponse(TopicEventResponseStatus.retry)
    if isinstance(result, Poison):
        logger.warning(
            "Drasi inbox admission rejected delivery; requesting dead letter."
        )
        return TopicEventResponse(TopicEventResponseStatus.drop)
    if not isinstance(result, SchedulingInput):
        logger.error(
            "Drasi inbox received an invalid admission result; requesting retry."
        )
        return TopicEventResponse(TopicEventResponseStatus.retry)

    try:
        instance_id = workflow_client.schedule_new_workflow(
            workflow_name,
            input={"task": result.task},
            instance_id=result.instance_id,
        )
    except RpcError as error:
        code = getattr(error, "code", None)
        if callable(code) and code() == StatusCode.ALREADY_EXISTS:
            return TopicEventResponse(TopicEventResponseStatus.success)
        logger.warning("Drasi workflow scheduling failed over gRPC; requesting retry.")
        return TopicEventResponse(TopicEventResponseStatus.retry)
    except Exception as error:
        logger.error(
            "Drasi workflow scheduling failed (%s); requesting retry.",
            type(error).__name__,
        )
        return TopicEventResponse(TopicEventResponseStatus.retry)

    if instance_id != result.instance_id:
        logger.error(
            "Drasi scheduler returned an unexpected instance ID; requesting retry."
        )
        return TopicEventResponse(TopicEventResponseStatus.retry)
    return TopicEventResponse(TopicEventResponseStatus.success)


def subscribe_drasi_inbox(
    *,
    config: ResolvedDrasiConfig,
    admission: AdmissionHandler,
    dapr_client: DaprClient,
    workflow_client: DaprWorkflowClient,
) -> DrasiInbox:
    """Start an owned dynamic-inbox consumer.

    Dapr routes DROP responses to the configured dead-letter topic. RETRY
    exhaustion and dead-letter publication failures remain governed by the
    runtime/broker policies, not by an application-local delivery queue.
    """
    return DrasiInbox(
        config=config,
        admission=admission,
        dapr_client=dapr_client,
        workflow_client=workflow_client,
    )
