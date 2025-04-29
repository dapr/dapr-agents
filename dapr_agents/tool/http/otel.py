from pydantic import Field
from logging import Logger
from typing import Optional
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


class DaprAgentOTel:
    """
    OpenTelemetry configuration for Dapr agents.
    """

    service_name: Optional[str] = Field(
        default="", description="Service name for Agent."
    )

    otlp_endpoint: Optional[str] = Field(
        default="localhost:4317",
        description="OTLP endpoint for OpenTelemetry. Defaults to localhost:4317.",
    )

    def __init__(self):
        # Configure OpenTelemetry
        self.setup_resources()

    def setup_resources(self):
        """
        Set up the resource for OpenTelemetry.
        """

        self._resource = Resource.create(
            attributes={
                SERVICE_NAME: str(self.service_name).capitalize(),
            }
        )

    def create_and_instrument_meter_provider(
        self,
    ) -> MeterProvider:
        """
        Returns a `MeterProvider` that is configured to export metrics using the `PeriodicExportingMetricReader`
        which means that metrics are exported periodically in the background. The interval can be set by
        the environment variable `OTEL_METRIC_EXPORT_INTERVAL`. The default value is 60000ms (1 minute).

        Also sets the global OpenTelemetry meter provider to the returned meter provider.
        """

        metric_exporter = OTLPMetricExporter(endpoint=self.otlp_endpoint)
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(
            resource=self._resource, metric_readers=[metric_reader]
        )
        set_meter_provider(meter_provider)
        return meter_provider

    def create_and_instrument_tracer_provider(
        self,
    ) -> TracerProvider:
        """
        Returns a `TracerProvider` that is configured to export traces using the `BatchSpanProcessor`
        which means that traces are exported in batches. The batch size can be set by
        the environment variable `OTEL_TRACES_EXPORT_BATCH_SIZE`. The default value is 512.
        Also sets the global OpenTelemetry tracer provider to the returned tracer provider.
        """

        trace_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
        tracer_processor = BatchSpanProcessor(trace_exporter)
        tracer_provider = TracerProvider(resource=self._resource)
        tracer_provider.add_span_processor(tracer_processor)
        set_tracer_provider(tracer_provider)
        return tracer_provider

    def create_and_instrument_logging_provider(
        self,
        logger: Logger,
    ) -> LoggerProvider:
        """
        Returns a `LoggingProvider` that is configured to export logs using the `BatchLogProcessor`
        which means that logs are exported in batches. The batch size can be set by
        the environment variable `OTEL_LOGS_EXPORT_BATCH_SIZE`. The default value is 512.
        Also sets the global OpenTelemetry logging provider to the returned logging provider.
        """

        log_exporter = OTLPLogExporter(endpoint=self.otlp_endpoint)
        logging_provider = LoggerProvider(resource=self._resource)
        logging_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logging_provider)

        handler = LoggingHandler(logger_provider=logging_provider)
        logger.addHandler(handler)
        return logging_provider
