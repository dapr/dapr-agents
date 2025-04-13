from dapr_agents.storage.graphstores import GraphStoreBase
from dapr_agents.storage.graphstores.neo4j.client import Neo4jClient
from dapr_agents.storage.graphstores.neo4j.utils import value_sanitize, get_current_time
from dapr_agents.types import Node, Relationship
from pydantic import BaseModel, ValidationError, Field
from typing import Any, Dict, Optional, List, Literal
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStoreBase):
    """
    Neo4j-based graph store implementation using Pydantic.
    """

    uri: str = Field(..., description="The URI of the Neo4j database.")
    user: str = Field(..., description="The username for authentication.")
    password: str = Field(..., description="The password for authentication.")
    database: str = Field(default="neo4j", description="The Neo4j database to use.")
    sanitize: bool = Field(default=True, description="Whether to sanitize the results.")
    timeout: int = Field(default=30, description="Query timeout in seconds.")
    graph_schema: Dict[str, Any] = Field(
        default_factory=dict, description="Schema of the graph structure."
    )

    # Client initialized in model_post_init, not during regular initialization
    client: Optional[Neo4jClient] = Field(
        default=None,
        init=False,
        description="Client for interacting with the Neo4j database.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set up the Neo4j client after model instantiation.
        """
        self.client = Neo4jClient(
            uri=self.uri, user=self.user, password=self.password, database=self.database
        )
        logger.info(f"Neo4jGraphStore initialized with database {self.database}")

        # Complete post-initialization
        super().model_post_init(__context)

    def batch_execute(
        self, query: str, data: List[Dict[str, Any]], batch_size: int = 1000
    ) -> None:
        """
        Execute a Cypher query in batches.

        Args:
            query (str): The Cypher query to execute.
            data (List[Dict[str, Any]]): The data to pass to the query.
            batch_size (int): The size of each batch. Defaults to 1000.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        from neo4j.exceptions import Neo4jError

        total_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            try:
                with self.client.driver.session(
                    database=self.client.database
                ) as session:
                    # Pass the correct parameter name
                    session.run(query, {"data": batch})
                    logger.info(
                        "Processed batch %d/%d", i // batch_size + 1, total_batches
                    )
            except Neo4jError as e:
                logger.error("Batch execution failed: %s", str(e))
                raise ValueError(f"Batch execution failed: {str(e)}")

    def add_node(self, node: Node) -> None:
        """
        Add a single node to the Neo4j database.

        Args:
            node (Node): The node to add.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        # Encapsulate single node in a list and call `add_nodes`
        self.add_nodes([node])

    def add_nodes(self, nodes: List[Node], batch_size: int = 1000) -> None:
        """
        Add multiple nodes to the Neo4j database in batches, supporting different labels.
        Handles cases where vector support is not available.

        Args:
            nodes (List[Node]): A list of nodes to add.
            batch_size (int): The size of each batch. Defaults to 1000.

        Raises:
            ValueError: If there is an issue with the query execution.
        """

        # Group nodes by their labels
        nodes_by_label = defaultdict(list)
        for node in nodes:
            nodes_by_label[node.label].append(node)

        for label, grouped_nodes in nodes_by_label.items():
            query = f"""
            UNWIND $data AS node
            MERGE (n:`{label}` {{id: node.id}})
            ON CREATE SET n.createdAt = node.current_time
            SET n.updatedAt = node.current_time, n += apoc.map.clean(node.properties, [], [])
            WITH n, node.additional_labels AS additional_labels, node.embedding AS embedding
            CALL apoc.create.addLabels(n, additional_labels)
            YIELD node AS labeled_node
            WITH labeled_node AS n, embedding
            CALL apoc.do.when(
                embedding IS NOT NULL,
                'CALL db.create.setNodeVectorProperty(n, "embedding", $embedding) YIELD node RETURN node',
                'RETURN n',
                {{n: n, embedding: embedding}}
            ) YIELD value AS final_node
            RETURN final_node
            """

            # Prepare data for batch processing
            current_time = get_current_time()
            data = [
                {**n.model_dump(), "current_time": current_time} for n in grouped_nodes
            ]

            # Execute in batches for the current label
            try:
                self.batch_execute(query, data, batch_size)
                logger.info(f"Nodes with label `{label}` added successfully.")
            except ValueError as e:
                logger.error(f"Failed to add nodes with label `{label}`: {str(e)}")
                raise

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Create a single relationship between two nodes in the Neo4j database.

        Args:
            relationship (Relationship): The relationship to create.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        # Encapsulate the single relationship in a list and delegate
        self.add_relationships([relationship])

    def add_relationships(
        self, relationships: List[Relationship], batch_size: int = 1000
    ) -> None:
        """
        Create multiple relationships between nodes in the Neo4j database in batches.

        Args:
            relationships (List[Relationship]): A list of relationships to create.
            batch_size (int): The size of each batch. Defaults to 1000.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        # Group relationships by their types
        relationships_by_type = defaultdict(list)
        for relationship in relationships:
            relationships_by_type[relationship.type].append(relationship)

        # Process each relationship type separately
        for rel_type, rel_group in relationships_by_type.items():
            query = f"""
            UNWIND $data AS rel
            MATCH (a {{id: rel.source_node_id}}), (b {{id: rel.target_node_id}})
            MERGE (a)-[r:`{rel_type}`]->(b)
            ON CREATE SET r.createdAt = rel.current_time
            SET r.updatedAt = rel.current_time, r += rel.properties
            RETURN r
            """

            # Prepare data for batch processing
            current_time = get_current_time()
            data = [
                {**rel.model_dump(), "current_time": current_time} for rel in rel_group
            ]

            # Execute in batches for the current relationship type
            try:
                self.batch_execute(query, data, batch_size)
                logger.info(f"Relationships of type `{rel_type}` added successfully.")
            except ValueError as e:
                logger.error(
                    f"Failed to add relationships of type `{rel_type}`: {str(e)}"
                )
                raise

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        sanitize: Optional[bool] = None,
        pagination_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Neo4j database and optionally sanitize or paginate the results.

        Args:
            query (str): The Cypher query to execute.
            params (Dict[str, Any], optional): Parameters for the query. Defaults to None.
            sanitize (bool, optional): Whether to sanitize the results. Defaults to class-level setting.
            pagination_limit (int, optional): Limit the number of results for pagination. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query results.

        Raises:
            ValueError: If there is a syntax error in the Cypher query.
            Neo4jError: If any other Neo4j-related error occurs.
        """
        from neo4j import Query
        from neo4j.exceptions import Neo4jError, CypherSyntaxError
        import time

        params = params or {}
        sanitize = sanitize if sanitize is not None else self.sanitize
        start_time = time.time()

        try:
            with self.client.driver.session(database=self.client.database) as session:
                # Add pagination support if a limit is provided
                if pagination_limit:
                    query = f"{query} LIMIT {pagination_limit}"

                result = session.run(
                    Query(text=query, timeout=self.timeout), parameters=params
                )
                json_data = [record.data() for record in result]

                # Optional sanitization of results
                if sanitize:
                    json_data = [value_sanitize(el) for el in json_data]

                execution_time = time.time() - start_time
                logger.info(
                    "Query executed successfully: %s | Time: %.2f seconds | Results: %d",
                    query,
                    execution_time,
                    len(json_data),
                )
                return json_data

        except CypherSyntaxError as e:
            logger.error("Syntax error in Cypher query: %s | Query: %s", str(e), query)
            raise ValueError(f"Syntax error in Cypher query: {str(e)}")
        except Neo4jError as e:
            logger.error("Neo4j error: %s | Query: %s", str(e), query)
            raise ValueError(f"Neo4j error: {str(e)}")

    def reset(self):
        """
        Reset the Neo4j database by deleting all nodes and relationships.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        from neo4j.exceptions import Neo4jError

        try:
            with self.client.driver.session() as session:
                session.run("CALL apoc.schema.assert({}, {})")
                session.run(
                    "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DETACH DELETE n', {batchSize:1000, iterateList:true})"
                )
                logger.info("Database reset successfully")
        except Neo4jError as e:
            logger.error("Failed to reset database: %s", str(e))
            raise ValueError(f"Failed to reset database: {str(e)}")

    def refresh_schema(self) -> None:
        """
        Refresh the database schema, including node properties, relationship properties, constraints, and indexes.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        from neo4j.exceptions import Neo4jError

        try:
            # Define queries as constants for reusability
            NODE_PROPERTIES_QUERY = """
            CALL apoc.meta.data()
            YIELD label, property, type
            WHERE type <> 'RELATIONSHIP'
            RETURN label, collect({property: property, type: type}) AS properties
            """
            RELATIONSHIP_PROPERTIES_QUERY = """
            CALL apoc.meta.data()
            YIELD label, property, type
            WHERE type = 'RELATIONSHIP'
            RETURN label, collect({property: property, type: type}) AS properties
            """
            INDEXES_QUERY = """
            CALL apoc.schema.nodes()
            YIELD label, properties, type, size, valuesSelectivity
            WHERE type = 'RANGE'
            RETURN *, size * valuesSelectivity AS distinctValues
            """

            # Execute queries
            logger.debug("Refreshing node properties...")
            node_properties = self.query(NODE_PROPERTIES_QUERY)

            logger.debug("Refreshing relationship properties...")
            relationship_properties = self.query(RELATIONSHIP_PROPERTIES_QUERY)

            logger.debug("Refreshing constraints...")
            constraints = self.query("SHOW CONSTRAINTS")

            logger.debug("Refreshing indexes...")
            indexes = self.query(INDEXES_QUERY)

            # Transform query results into schema dictionary
            self.graph_schema = {
                "node_props": {
                    record.get("label"): record.get("properties", [])
                    for record in (node_properties or [])
                },
                "rel_props": {
                    record.get("label"): record.get("properties", [])
                    for record in (relationship_properties or [])
                },
                "constraints": constraints or [],
                "indexes": indexes or [],
            }

            logger.info("Schema refreshed successfully")

        except Neo4jError as e:
            logger.error("Failed to refresh schema: %s", str(e))
            raise ValueError(f"Failed to refresh schema: {str(e)}")

        except Exception as e:
            logger.error("Unexpected error while refreshing schema: %s", str(e))
            raise ValueError(f"Unexpected error while refreshing schema: {str(e)}")

    def get_schema(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get the schema of the Neo4jGraph store.

        Args:
            refresh (bool): Whether to refresh the schema before returning it. Defaults to False.

        Returns:
            Dict[str, Any]: The schema of the Neo4jGraph store.
        """
        if not self.graph_schema or refresh:
            self.refresh_schema()
        return self.graph_schema

    def validate_schema(self, expected_schema: BaseModel) -> bool:
        """
        Validate the current graph schema against an expected Pydantic schema model.

        Args:
            expected_schema (Type[BaseModel]): The Pydantic schema model to validate against.

        Returns:
            bool: True if schema matches, False otherwise.
        """
        # Retrieve the current schema
        current_schema = self.get_schema()

        try:
            # Attempt to initialize the expected schema with the current schema
            validated_schema = expected_schema(**current_schema)
            logger.info("Schema validation passed: %s", validated_schema)
            return True
        except ValidationError as e:
            # Handle and log validation errors
            logger.error("Schema validation failed due to validation errors:")
            for error in e.errors():
                logger.error(f"Field: {error['loc']}, Error: {error['msg']}")
            return False

    def create_vector_index(
        self,
        label: str,
        property: str,
        dimensions: int,
        similarity_function: Literal["cosine", "dot", "euclidean"] = "cosine",
    ) -> None:
        """
        Creates a vector index for a specified label and property in the Neo4j database.

        Args:
            label (str): The label of the nodes to index (non-empty).
            property (str): The property of the nodes to index (non-empty).
            dimensions (int): The number of dimensions of the vector.
            similarity_function (Literal): The similarity function to use ('cosine', 'dot', 'euclidean').

        Raises:
            ValueError: If there is an issue with the query execution or invalid arguments.
        """
        from neo4j.exceptions import Neo4jError

        # Ensure label and property are non-empty strings
        if not all([label, property]):
            raise ValueError("Both `label` and `property` must be non-empty strings.")

        # Ensure dimensions is valid
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("`dimensions` must be a positive integer.")

        # Construct index name and query
        index_name = f"{label.lower()}_{property}_vector_index"
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON (n.{property})
        OPTIONS {{
            indexConfig: {{
                'vector.dimensions': {dimensions},
                'vector.similarity_function': '{similarity_function}'
            }}
        }}
        """

        try:
            with self.client.driver.session(database=self.database) as session:
                session.run(query)
                logger.info(
                    "Vector index `%s` for label `%s` on property `%s` created successfully.",
                    index_name,
                    label,
                    property,
                )

            # Optionally update graph schema
            if "indexes" in self.graph_schema:
                self.graph_schema["indexes"].append(
                    {
                        "label": label,
                        "property": property,
                        "dimensions": dimensions,
                        "similarity_function": similarity_function,
                    }
                )

        except Neo4jError as e:
            logger.error("Failed to create vector index: %s", str(e))
            raise ValueError(f"Failed to create vector index: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error during vector index creation: %s", str(e))
            raise ValueError(f"Unexpected error: {str(e)}")
