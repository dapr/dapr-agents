from fastapi import HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from cloudevents.http.conversion import from_http
from cloudevents.http.event import CloudEvent
from dapr_agents.service.fastapi import DaprFastAPIServer
from dapr_agents.workflow import WorkflowApp
from pydantic import Field
from typing import Any
import asyncio
import logging

logger = logging.getLogger(__name__)

class WorkflowAppService(DaprFastAPIServer, WorkflowApp):
    """
    Abstract base class for agentic workflows, providing a template for common workflow operations.
    """

    workflow_name: str = Field(default=None, init=False, description="The main workflow name for this service.")

    def model_post_init(self, __context: Any) -> None:
        """
        Configure workflows and initialize AgentService and WorkflowApp.
        """

        super().model_post_init(__context)

        # Register API Routes
        self.app.add_api_route("/RunWorkflow", self.run_workflow_from_request, methods=["POST"])
        self.app.add_api_route("/RaiseWorkflowEvent", self.raise_workflow_event_from_request, methods=["POST"])
    
    async def run_workflow_from_request(self, request: Request) -> JSONResponse:
        """
        Run a workflow instance triggered by an incoming HTTP request.
        Handles both CloudEvents and plain JSON input, with background monitoring.

        Args:
            request (Request): The incoming request containing input data for the workflow.

        Returns:
            Response: Accepted or error response based on event processing.
        """
        try:
            # Extract headers and body
            headers = request.headers
            body = await request.body()

            # Attempt to parse as CloudEvent
            try:
                event: CloudEvent = from_http(dict(headers), body)
                input_data = event.data
            except Exception:
                # Fallback to plain JSON
                data = await request.json()
                input_data = data

            logger.info(f"Starting '{self.workflow_name}' from request with input: {input_data}")

            # Start the workflow
            instance_id = self.run_workflow(workflow=self.workflow_name, input=input_data)

            # Schedule background monitoring
            asyncio.create_task(self.monitor_workflow_completion(instance_id))

            # Respond with the workflow instance ID immediately
            return JSONResponse(
                content={"message": "Workflow initiated successfully.", "workflow_instance_id": instance_id},
                status_code=status.HTTP_202_ACCEPTED,
            )

        except Exception as e:
            logging.error(f"Error starting workflow: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": "Failed to start workflow", "details": str(e)},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
    
    async def raise_workflow_event_from_request(self, request: Request) -> Response:
        """
        Handles an API request or pub/sub message to trigger a workflow event.

        This method processes the incoming CloudEvent or HTTP request, extracts metadata
        (workflow_instance_id, event_name), and triggers the corresponding event for a running workflow instance.

        Args:
            request (Request): The incoming request containing event details.

        Returns:
            Response: Success or error response based on event processing.
        """
        try:
            # Parse the incoming CloudEvent
            body = await request.body()
            headers = request.headers
            event: CloudEvent = from_http(dict(headers), body)

            # Extract essential metadata from headers
            workflow_instance_id = headers.get("workflow_instance_id")
            event_name = headers.get("event_name")

            if not workflow_instance_id:
                logger.warning("Workflow event missing 'workflow_instance_id'.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Header 'workflow_instance_id' is required."
                )

            if not event_name:
                logger.warning("Workflow event missing 'event_name'.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Header 'event_name' is required."
                )

            # Extract event details
            source = event.get("source")
            event_data = event.data

            if not event_data:
                logger.warning("Event data is empty or missing.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Event data is required."
                )

            logger.info(f"Processing event '{event_name}' for workflow '{workflow_instance_id}' from '{source}'.")

            # Trigger the workflow event, passing all event data as `data`
            self.raise_workflow_event(instance_id=workflow_instance_id, event_name=event_name, data=event_data)

            return JSONResponse(content={"message": "Workflow event triggered successfully."}, status_code=status.HTTP_200_OK)

        except HTTPException as e:
            logger.error(f"Error processing workflow event: {e.detail}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error processing workflow event: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing workflow event: {str(e)}"
            )