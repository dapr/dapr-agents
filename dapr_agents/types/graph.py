from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Optional, List


class Node(BaseModel):
    id: Any = Field(description="The unique identifier of the node.")
    label: str = Field(description="The primary label or type of the node.")
    properties: Dict[str, Any] = Field(
        description="A dictionary of properties associated with the node."
    )
    additional_labels: Optional[List[str]] = Field(
        default=[],
        description="Additional labels or categories associated with the node.",
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional embedding vector for the node, useful for vector-based similarity searches.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "1",
                "label": "Person",
                "properties": {"name": "Alice", "age": 30},
                "additional_labels": ["Employee"],
                "embedding": [0.1, 0.2, 0.3],
            }
        }
    )


class Relationship(BaseModel):
    source_node_id: Any = Field(
        description="The unique identifier of the source node in the relationship."
    )
    target_node_id: Any = Field(
        description="The unique identifier of the target node in the relationship."
    )
    type: str = Field(
        description="The type or label of the relationship, describing the connection between nodes."
    )
    properties: Optional[Dict[str, Any]] = Field(
        default={}, description="Optional properties associated with the relationship."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_node_id": "1",
                "target_node_id": "2",
                "type": "FRIEND",
                "properties": {"since": 2020},
            }
        }
    )
