from pydantic import BaseModel, Field
from typing import Optional, Any
import os
import logging

logger = logging.getLogger(__name__)


class Neo4jClient(BaseModel):
    """
    Client for interacting with a Neo4j database.
    Handles connection initialization, closing, and basic testing of connectivity.
    """

    uri: str = Field(
        default=None,
        description="The URI of the Neo4j database. Defaults to the 'NEO4J_URI' environment variable.",
    )
    user: str = Field(
        default=None,
        description="The username for Neo4j authentication. Defaults to the 'NEO4J_USERNAME' environment variable.",
    )
    password: str = Field(
        default=None,
        description="The password for Neo4j authentication. Defaults to the 'NEO4J_PASSWORD' environment variable.",
    )
    database: str = Field(
        default="neo4j", description="The default database to use. Defaults to 'neo4j'."
    )
    driver: Optional[Any] = Field(
        default=None,
        init=False,
        description="The Neo4j driver instance for database operations. Initialized in 'model_post_init'.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization logic to handle dynamic imports and environment variable defaults.
        """
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "The 'neo4j' package is required but not installed. Install it with 'pip install neo4j'."
            ) from e

        # Handle environment variable defaults
        self.uri = self.uri or os.getenv("NEO4J_URI")
        self.user = self.user or os.getenv("NEO4J_USERNAME")
        self.password = self.password or os.getenv("NEO4J_PASSWORD")

        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Missing required connection parameters (uri, user, password). Set them as environment variables or pass explicitly."
            )

        # Initialize the Neo4j driver
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            logger.info("Successfully created the driver for URI: %s", self.uri)
        except Exception as e:
            logger.error("Failed to create the driver: %s", str(e))
            raise ValueError(f"Failed to initialize the Neo4j driver: {str(e)}")

        # Complete post-initialization
        super().model_post_init(__context)

    def close(self) -> None:
        """
        Closes the Neo4j driver connection.
        """
        if self.driver is not None:
            self.driver.close()
            logger.info("Neo4j driver connection closed")

    def test_connection(self) -> bool:
        """
        Tests the connection to the Neo4j database.

        Returns:
            bool: True if the connection is successful, False otherwise.

        Raises:
            ValueError: If there is an error testing the connection.
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    "CALL dbms.components() YIELD name, versions, edition"
                )
                record = result.single()
                if record:
                    logger.info(
                        "Connected to %s version %s (%s edition)",
                        record["name"],
                        record["versions"][0],
                        record["edition"],
                    )
                    return True
                else:
                    logger.warning("No record found during the connection test")
                    return False
        except Exception as e:
            logger.error("Error testing connection: %s", str(e))
            raise ValueError(f"Error testing connection: {str(e)}")
