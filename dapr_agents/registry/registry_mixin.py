"""Registry mixin for agent registration functionality."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class RegistryMixin(ABC):
    """
    Mixin providing registry functionality for agents and workflows.
    
    This mixin provides a unified interface for registering agents with the
    Dapr Agents registry. Classes using this mixin must implement
    _build_agent_metadata() to provide their specific metadata structure.
    
    Note: Classes using this mixin should declare _registry as PrivateAttr
    or the mixin will create it lazily on first access.
    """

    @abstractmethod
    def _build_agent_metadata(self) -> Optional[Any]:
        """
        Build agent metadata for registry.
        
        Must be implemented by subclasses to provide their specific
        metadata structure (typically returns AgentMetadata instance).
        
        Returns:
            Optional[AgentMetadata]: Metadata instance or None if registration
                should be skipped
        """
        pass

    def _get_registry_instance(
        self, 
        store_name: str, 
        store_key: str,
        dapr_client: Any
    ) -> Any:
        """
        Get or create a Registry instance.
        
        Args:
            store_name: Name of the state store
            store_key: Key for the registry data
            dapr_client: Dapr client instance
            
        Returns:
            Registry instance
        """
        from dapr_agents.registry.registry import Registry
        
        # Get existing registry attribute, or initialize it
        registry = getattr(self, "_registry", None)
        
        # Create new registry if needed or if config changed
        if (
            registry is None
            or registry.store_name != store_name
            or registry.store_key != store_key
            or registry.client is not dapr_client
        ):
            registry = Registry(
                client=dapr_client,
                store_name=store_name,
                store_key=store_key,
            )
            # Store it back (works with both PrivateAttr and regular attributes)
            object.__setattr__(self, "_registry", registry)
        
        return registry

    def register_agent(
        self,
        *,
        store_name: str,
        store_key: str = "agent_registry",
        agent_name: str,
        agent_metadata: dict,
        agent_identity: Optional[str] = None,
    ) -> None:
        """
        Register agent with the registry.
        
        This method provides a unified interface for agent registration.
        It creates/reuses a Registry instance and delegates to it.
        
        Args:
            store_name: Name of the state store
            store_key: Key for registry data (default: "agent_registry")
            agent_name: Name of the agent to register
            agent_metadata: Metadata dict to store
            agent_identity: Unique identity for this agent instance
        """
        # Get Dapr client - different attribute names in different classes
        dapr_client = getattr(self, "_dapr_client", None)
        if dapr_client is None:
            logger.debug(
                f"Agent '{agent_name}' does not have Dapr client, skipping registration"
            )
            return

        # Get or create registry instance
        registry = self._get_registry_instance(store_name, store_key, dapr_client)
        
        # Use identity if provided, otherwise use object id
        identity = agent_identity or str(id(self))
        
        # Register with the registry
        registry.register_agent(
            agent_name=agent_name,
            agent_metadata=agent_metadata,
            agent_identity=identity,
        )

    def _register_agent_metadata(self) -> None:
        """
        Build and register agent metadata.
        
        This is a convenience method that builds metadata and registers it.
        Called during agent initialization.
        """
        from dapr.clients import DaprClient
        from dapr_agents.registry.registry import Registry
        
        metadata = self._build_agent_metadata()
        if metadata is None:
            return

        # Get agent registry config (clean - no fallback needed)
        registry_config = getattr(self, "_agent_registry_config", None)
        if registry_config is None or registry_config.store is None:
            logger.debug("No agent registry config or store configured, skipping registration")
            return

        # Get agent name
        agent_name = getattr(self, "name", None)
        if agent_name is None:
            logger.warning("Agent has no name, cannot register")
            return

        # Get store name from registry config
        store_name = registry_config.store.store_name

        # Create temporary DaprClient and register
        try:
            with DaprClient() as client:
                registry = Registry(
                    client=client,
                    store_name=store_name,
                    store_key="agents-registry",
                )
                registry.register_agent(
                    agent_name=agent_name,
                    agent_metadata=metadata.model_dump_for_registry(),
                    agent_identity=str(id(self)),
                )
                logger.info(f"Registered agent '{agent_name}' in agent registry")
        except Exception as exc:
            logger.warning(
                f"Failed to register agent '{agent_name}' in agent registry: {exc}"
            )

