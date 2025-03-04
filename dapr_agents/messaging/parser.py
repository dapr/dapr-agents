from typing import Optional, Tuple, Type
from dapr_agents.types.message import EventMessageMetadata
from pydantic import BaseModel, ValidationError
from cloudevents.http.event import CloudEvent
from cloudevents.http import from_http
from fastapi import HTTPException, Request
import logging

logger = logging.getLogger(__name__)

async def parse_cloudevent(request: Request, model: Optional[Type[BaseModel]] = None) -> Tuple[BaseModel, EventMessageMetadata]:
    """
    Parses and validates an incoming CloudEvent request.
    Returns the validated message and metadata.

    Args:
        request (Request): The incoming request.
        model (Optional[Type[BaseModel]]): The Pydantic model to validate the message payload.

    Returns:
        Tuple[BaseModel, EventMessageMetadata]: The validated message and metadata.

    Raises:
        HTTPException: If the CloudEvent is invalid or message validation fails.
    """
    try:
        # Parse the CloudEvent
        logger.debug("Parsing CloudEvent request...")
        body = await request.body()
        headers = request.headers
        event: CloudEvent = from_http(dict(headers), body)
    except Exception as e:
        logger.error(f"Failed to parse CloudEvent: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CloudEvent: {str(e)}")

    # Extract metadata
    metadata = EventMessageMetadata(
        id=event.get("id"),
        datacontenttype=event.get("datacontenttype"),
        pubsubname=event.get("pubsubname"),
        source=event.get("source"),
        specversion=event.get("specversion"),
        time=event.get("time"),
        topic=event.get("topic"),
        traceid=event.get("traceid"),
        traceparent=event.get("traceparent"),
        type=event.get("type"),
        tracestate=event.get("tracestate"),
        headers=dict(headers),
    )

    logger.debug(f"Extracted CloudEvent metadata: {metadata}")

    # Validate message payload
    if model:
        try:
            logger.debug(f"Validating payload with model '{model.__name__}'...")
            message = model(**event.data)
        except ValidationError as ve:
            logger.error(f"Message validation failed for model '{model.__name__}': {ve}")
            raise HTTPException(status_code=422, detail=f"Message validation failed: {ve}")
    else:
        raise HTTPException(status_code=500, detail="Message validation failed: No Pydantic model provided.")
    
    # Return the validated message, and metadata
    logger.debug(f"Message successfully parsed and validated: {message}")
    
    return message, metadata
