from __future__ import annotations

import logging
from typing import Optional, Sequence

from dapr_agents.agents.configs import WorkflowGrpcOptions

logger = logging.getLogger(__name__)


# This is a copy of the original get_grpc_channel function in durabletask.internal.shared at
# https://github.com/dapr/durabletask-python/blob/7070cb07d07978d079f8c099743ee4a66ae70e05/durabletask/internal/shared.py#L30C1-L61C19
# but with my option overrides applied above.
def apply_grpc_options(options: Optional[WorkflowGrpcOptions]) -> None:
    """
    Patch Durable Task's gRPC channel factory with custom message size limits.

    Durable Task (and therefore Dapr Workflows) creates its gRPC channels via
    ``durabletask.internal.shared.get_grpc_channel``.  This helper monkey patches
    that factory so that subsequent runtime/client instances honour the provided
    ``grpc.max_send_message_length`` / ``grpc.max_receive_message_length`` values.

    Users can set either or both options; any non-None value will be applied.
    """
    if not options:
        return
    # Early return if neither option is set
    if (
        options.max_send_message_length is None
        and options.max_receive_message_length is None
    ):
        return

    try:
        import grpc
        from durabletask.internal import shared
    except ImportError as exc:
        logger.error(
            "Failed to import grpc/durabletask for channel configuration: %s", exc
        )
        raise

    grpc_options = []
    if options.max_send_message_length:
        grpc_options.append(
            ("grpc.max_send_message_length", options.max_send_message_length)
        )
    if options.max_receive_message_length:
        grpc_options.append(
            ("grpc.max_receive_message_length", options.max_receive_message_length)
        )

    def get_grpc_channel_with_options(
        host_address: Optional[str],
        secure_channel: bool = False,
        interceptors: Optional[Sequence["grpc.ClientInterceptor"]] = None,
    ):
        if host_address is None:
            host_address = shared.get_default_host_address()

        for protocol in getattr(shared, "SECURE_PROTOCOLS", []):
            if host_address.lower().startswith(protocol):
                secure_channel = True
                host_address = host_address[len(protocol) :]
                break

        for protocol in getattr(shared, "INSECURE_PROTOCOLS", []):
            if host_address.lower().startswith(protocol):
                secure_channel = False
                host_address = host_address[len(protocol) :]
                break

        if secure_channel:
            credentials = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel(
                host_address, credentials, options=grpc_options
            )
        else:
            channel = grpc.insecure_channel(host_address, options=grpc_options)

        if interceptors:
            channel = grpc.intercept_channel(channel, *interceptors)

        return channel

    shared.get_grpc_channel = get_grpc_channel_with_options
    logger.debug(
        "Applied gRPC options to durabletask channel factory: %s", dict(grpc_options)
    )
