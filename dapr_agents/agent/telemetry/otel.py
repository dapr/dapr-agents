from logging import Logger
from typing import Any, Union

import functools
import logging

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


class DaprAgentsOTel:
    """
    OpenTelemetry configuration for Dapr agents.
    """

    def __init__(self, service_name: str = "", otlp_endpoint: str = ""):
        # Configure OpenTelemetry
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint

        self.setup_resources()

    def setup_resources(self):
        """
        Set up the resource for OpenTelemetry.
        """

        self._resource = Resource.create(
            attributes={
                SERVICE_NAME: str(self.service_name),
            }
        )

    def create_and_instrument_meter_provider(
        self,
        otlp_endpoint: str = "",
    ) -> MeterProvider:
        """
        Returns a `MeterProvider` that is configured to export metrics using the `PeriodicExportingMetricReader`
        which means that metrics are exported periodically in the background. The interval can be set by
        the environment variable `OTEL_METRIC_EXPORT_INTERVAL`. The default value is 60000ms (1 minute).

        Also sets the global OpenTelemetry meter provider to the returned meter provider.
        """

        # Ensure the endpoint is set correctly
        endpoint = self._endpoint_validator(
            endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
            telemetry_type="metrics",
        )

        metric_exporter = OTLPMetricExporter(endpoint=str(endpoint))
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(
            resource=self._resource, metric_readers=[metric_reader]
        )
        set_meter_provider(meter_provider)
        return meter_provider

    def create_and_instrument_tracer_provider(
        self,
        otlp_endpoint: str = "",
    ) -> TracerProvider:
        """
        Returns a `TracerProvider` that is configured to export traces using the `BatchSpanProcessor`
        which means that traces are exported in batches. The batch size can be set by
        the environment variable `OTEL_TRACES_EXPORT_BATCH_SIZE`. The default value is 512.
        Also sets the global OpenTelemetry tracer provider to the returned tracer provider.
        """

        # Ensure the endpoint is set correctly
        endpoint = self._endpoint_validator(
            endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
            telemetry_type="traces",
        )

        trace_exporter = OTLPSpanExporter(endpoint=str(endpoint))
        tracer_processor = BatchSpanProcessor(trace_exporter)
        tracer_provider = TracerProvider(resource=self._resource)
        tracer_provider.add_span_processor(tracer_processor)
        set_tracer_provider(tracer_provider)
        return tracer_provider

    def create_and_instrument_logging_provider(
        self,
        logger: Logger,
        otlp_endpoint: str = "",
    ) -> LoggerProvider:
        """
        Returns a `LoggingProvider` that is configured to export logs using the `BatchLogProcessor`
        which means that logs are exported in batches. The batch size can be set by
        the environment variable `OTEL_LOGS_EXPORT_BATCH_SIZE`. The default value is 512.
        Also sets the global OpenTelemetry logging provider to the returned logging provider.
        """

        # Ensure the endpoint is set correctly
        endpoint = self._endpoint_validator(
            endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
            telemetry_type="logs",
        )

        log_exporter = OTLPLogExporter(endpoint=str(endpoint))
        logging_provider = LoggerProvider(resource=self._resource)
        logging_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logging_provider)

        handler = LoggingHandler(logger_provider=logging_provider)
        logger.addHandler(handler)
        return logging_provider

    def _endpoint_validator(
        self,
        endpoint: str,
        telemetry_type: str,
    ) -> Union[str | Exception]:
        """
        Validates the endpoint and method.
        """

        if endpoint == "":
            raise ValueError(
                "OTLP endpoint must be set either in the environment variable OTEL_EXPORTER_OTLP_ENDPOINT or in the constructor."
            )
        if endpoint.startswith("https://"):
            raise NotImplementedError(
                "OTLP over HTTPS is not supported. Please use HTTP."
            )

        endpoint = (
            endpoint
            if endpoint.endswith(f"/v1/{telemetry_type}")
            else f"{endpoint}/v1/{telemetry_type}"
        )
        endpoint = endpoint if endpoint.startswith("http://") else f"http://{endpoint}"

        return endpoint


_propagator = TraceContextTextMapPropagator()


def async_span_decorator(name):
    """Decorator that creates an OpenTelemetry span for an async method."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            tracer = getattr(self, "_tracer", None)
            if not tracer:
                return await func(self, *args, **kwargs)

            # Extract OpenTelemetry context from kwargs if available
            otel_context = kwargs.pop("otel_context", None)

            ctx = None
            if otel_context:
                try:
                    # Convert Dapr-serialized context back to carrier format
                    carrier = {
                        "traceparent": otel_context.get("traceparent", ""),
                        "tracestate": otel_context.get("tracestate", ""),
                    }

                    # Extract context from carrier
                    ctx = _propagator.extract(carrier=carrier)
                    logging.debug(
                        f"Restored context for span '{name}': trace={carrier.get('traceparent', '')}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to extract context for span '{name}': {e}")

            # Start span with context
            with tracer.start_as_current_span(name, context=ctx) as span:
                span.set_attribute("function.name", func.__name__)
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def span_decorator(name):
    """Decorator that creates an OpenTelemetry span for a synchronous method."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            tracer = getattr(self, "_tracer", None)
            if not tracer:
                return func(self, *args, **kwargs)

            # Extract OpenTelemetry context from kwargs if available
            otel_context = kwargs.pop("otel_context", None)

            ctx = None
            if otel_context:
                try:
                    # Convert Dapr-serialized context back to carrier format
                    carrier = {
                        "traceparent": otel_context.get("traceparent", ""),
                        "tracestate": otel_context.get("tracestate", ""),
                    }

                    # Extract context from carrier
                    ctx = _propagator.extract(carrier=carrier)
                    logging.debug(
                        f"Restored context for span '{name}': trace={carrier.get('traceparent', '')}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to extract context for span '{name}': {e}")

            # Start span with context
            with tracer.start_as_current_span(name, context=ctx) as span:
                span.set_attribute("function.name", func.__name__)
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def extract_otel_context() -> dict[str, Any]:
    """
    Extract current OpenTelemetry context for cross-boundary propagation.
    Returns a format that can be properly serialized by Dapr workflows.
    """
    carrier: dict[str, Any] = {}
    _propagator.inject(carrier)

    span = trace.get_current_span()
    ctx = span.get_span_context()

    # Always extract these values regardless of condition
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    flags = "01" if ctx.trace_flags.sampled else "00"

    # If the propagator didn't inject a traceparent, create it manually
    if "traceparent" not in carrier and span and span.is_recording():
        carrier["traceparent"] = f"00-{trace_id}-{span_id}-{flags}"

    # Ensure tracestate exists in the carrier (default to empty string if not present)
    if "tracestate" not in carrier:
        carrier["tracestate"] = ""

    # Add the extra fields directly to the carrier
    carrier["trace_id"] = trace_id
    carrier["span_id"] = span_id
    carrier["trace_flags"] = flags

    return carrier
