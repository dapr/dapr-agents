"""Component configuration management for different testing scenarios."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Handle imports for both module and direct execution
try:
    from .scenario_manager import DevelopmentScenario
except ImportError:
    # Direct execution - add parent directory to path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from scenario_manager import DevelopmentScenario

logger = logging.getLogger(__name__)


class ComponentManager:
    """Manages Dapr component configurations for different testing scenarios."""

    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent.parent
        self.components_root = self.workspace_root / "components"
        self.test_components_root = self.workspace_root / "tests" / "components"

    def get_components_for_scenario(self, scenario: DevelopmentScenario) -> str:
        """Get component directory for development scenario."""
        scenario_map = {
            DevelopmentScenario.LOCAL_FULL: self.test_components_root / "local_dev",
            DevelopmentScenario.LOCAL_PARTIAL: self.test_components_root
            / "partial_dev",
            DevelopmentScenario.AGENT_ONLY: self.test_components_root / "production",
            DevelopmentScenario.PRODUCTION: self.test_components_root / "production",
        }

        component_dir = scenario_map[scenario]

        # Create directory if it doesn't exist
        component_dir.mkdir(parents=True, exist_ok=True)

        return str(component_dir)

    def validate_components_for_scenario(
        self, scenario: DevelopmentScenario
    ) -> Dict[str, any]:
        """Validate that components are compatible with scenario."""
        component_dir = Path(self.get_components_for_scenario(scenario))

        validation_result = {
            "scenario": scenario.value,
            "component_dir": str(component_dir),
            "components_found": [],
            "components_valid": [],
            "components_invalid": [],
            "errors": [],
            "warnings": [],
        }

        if not component_dir.exists():
            validation_result["errors"].append(
                f"Component directory does not exist: {component_dir}"
            )
            return validation_result

        # Find all YAML component files
        yaml_files = list(component_dir.glob("*.yaml")) + list(
            component_dir.glob("*.yml")
        )

        for yaml_file in yaml_files:
            component_name = yaml_file.stem
            validation_result["components_found"].append(component_name)

            try:
                # Validate YAML syntax
                with open(yaml_file, "r") as f:
                    component_config = yaml.safe_load(f)

                # Basic component validation
                if self._validate_component_config(component_config, component_name):
                    validation_result["components_valid"].append(component_name)
                else:
                    validation_result["components_invalid"].append(component_name)
                    validation_result["warnings"].append(
                        f"Component {component_name} may have configuration issues"
                    )

            except yaml.YAMLError as e:
                validation_result["components_invalid"].append(component_name)
                validation_result["errors"].append(
                    f"YAML syntax error in {component_name}: {e}"
                )
            except Exception as e:
                validation_result["components_invalid"].append(component_name)
                validation_result["errors"].append(
                    f"Error validating {component_name}: {e}"
                )

        return validation_result

    def _validate_component_config(self, config: Dict, component_name: str) -> bool:
        """Validate basic component configuration structure."""
        required_fields = ["apiVersion", "kind", "metadata", "spec"]

        for field in required_fields:
            if field not in config:
                logger.warning(
                    f"Component {component_name} missing required field: {field}"
                )
                return False

        # Check component type
        if config.get("kind") != "Component":
            logger.warning(
                f"Component {component_name} has unexpected kind: {config.get('kind')}"
            )
            return False

        # Check metadata
        metadata = config.get("metadata", {})
        if "name" not in metadata:
            logger.warning(f"Component {component_name} missing metadata.name")
            return False

        # Check spec
        spec = config.get("spec", {})
        if "type" not in spec:
            logger.warning(f"Component {component_name} missing spec.type")
            return False

        return True

    def create_scenario_components(
        self, scenario: DevelopmentScenario, temp_dir: Optional[str] = None
    ) -> str:
        """Create component configs for specific scenario."""
        if temp_dir:
            target_dir = Path(temp_dir)
        else:
            target_dir = Path(self.get_components_for_scenario(scenario))

        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy base components from main components directory
        base_components = self._get_base_components()

        for component_name, component_config in base_components.items():
            # Modify component config based on scenario
            modified_config = self._modify_component_for_scenario(
                component_config, scenario
            )

            # Write component file
            component_file = target_dir / f"{component_name}.yaml"
            with open(component_file, "w") as f:
                yaml.dump(modified_config, f, default_flow_style=False)

            logger.info(
                f"Created component {component_name} for scenario {scenario.value}"
            )

        return str(target_dir)

    def _get_base_components(self) -> Dict[str, Dict]:
        """Get base component configurations from main components directory."""
        base_components = {}

        if not self.components_root.exists():
            logger.warning(
                f"Base components directory not found: {self.components_root}"
            )
            return base_components

        # Find all YAML files in components directory
        yaml_files = list(self.components_root.glob("*.yaml")) + list(
            self.components_root.glob("*.yml")
        )

        for yaml_file in yaml_files:
            # Skip disabled components
            if yaml_file.name.endswith(".disabled"):
                continue

            try:
                with open(yaml_file, "r") as f:
                    config = yaml.safe_load(f)

                component_name = yaml_file.stem
                base_components[component_name] = config

            except Exception as e:
                logger.warning(f"Could not load base component {yaml_file.name}: {e}")

        return base_components

    def _modify_component_for_scenario(
        self, config: Dict, scenario: DevelopmentScenario
    ) -> Dict:
        """Modify component configuration based on scenario."""
        # Make a deep copy to avoid modifying original
        import copy

        modified_config = copy.deepcopy(config)

        # Scenario-specific modifications
        if scenario == DevelopmentScenario.LOCAL_FULL:
            # For local full development, may need to point to local builds
            self._apply_local_full_modifications(modified_config)

        elif scenario == DevelopmentScenario.LOCAL_PARTIAL:
            # For partial local development, mix of local and released
            self._apply_local_partial_modifications(modified_config)

        elif scenario in [
            DevelopmentScenario.AGENT_ONLY,
            DevelopmentScenario.PRODUCTION,
        ]:
            # For production-like scenarios, ensure stable configurations
            self._apply_production_modifications(modified_config)

        return modified_config

    def _apply_local_full_modifications(self, config: Dict):
        """Apply modifications for local full development scenario."""
        # Add development-specific metadata
        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["labels"] = config["metadata"].get("labels", {})
        config["metadata"]["labels"]["scenario"] = "local_full"

        # Modify spec for local development if needed
        spec = config.get("spec", {})

        # For conversation components, might need different endpoints
        if spec.get("type", "").startswith("conversation."):
            # Add development-specific configuration
            metadata = spec.get("metadata", [])

            # Add timeout for development
            timeout_found = any(item.get("name") == "timeout" for item in metadata)
            if not timeout_found:
                metadata.append({"name": "timeout", "value": "60s"})

            spec["metadata"] = metadata

    def _apply_local_partial_modifications(self, config: Dict):
        """Apply modifications for local partial development scenario."""
        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["labels"] = config["metadata"].get("labels", {})
        config["metadata"]["labels"]["scenario"] = "local_partial"

    def _apply_production_modifications(self, config: Dict):
        """Apply modifications for production scenarios."""
        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["labels"] = config["metadata"].get("labels", {})
        config["metadata"]["labels"]["scenario"] = "production"

        # Ensure production-ready timeouts and configurations
        spec = config.get("spec", {})
        if spec.get("type", "").startswith("conversation."):
            metadata = spec.get("metadata", [])

            # Set production timeout
            for item in metadata:
                if item.get("name") == "timeout":
                    item["value"] = "30s"  # Shorter timeout for production
                    break
            else:
                metadata.append({"name": "timeout", "value": "30s"})

            spec["metadata"] = metadata

    def get_minimal_config(self, provider: str) -> Optional[str]:
        """Get minimal component config for provider."""
        minimal_configs = {
            "echo": {
                "apiVersion": "dapr.io/v1alpha1",
                "kind": "Component",
                "metadata": {"name": "echo-tools", "labels": {"scenario": "minimal"}},
                "spec": {
                    "type": "conversation.echo",
                    "version": "v1",
                    "metadata": [{"name": "timeout", "value": "30s"}],
                },
            },
            "anthropic": {
                "apiVersion": "dapr.io/v1alpha1",
                "kind": "Component",
                "metadata": {"name": "anthropic", "labels": {"scenario": "minimal"}},
                "spec": {
                    "type": "conversation.anthropic",
                    "version": "v1",
                    "metadata": [
                        {
                            "name": "apiKey",
                            "secretKeyRef": {
                                "name": "anthropic-secret",
                                "key": "api-key",
                            },
                        },
                        {"name": "timeout", "value": "30s"},
                    ],
                },
            },
        }

        config = minimal_configs.get(provider)
        if config:
            return yaml.dump(config, default_flow_style=False)

        return None

    def list_available_components(
        self, scenario: Optional[DevelopmentScenario] = None
    ) -> List[str]:
        """List available components for scenario."""
        if scenario:
            component_dir = Path(self.get_components_for_scenario(scenario))
        else:
            component_dir = self.components_root

        if not component_dir.exists():
            return []

        components = []
        yaml_files = list(component_dir.glob("*.yaml")) + list(
            component_dir.glob("*.yml")
        )

        for yaml_file in yaml_files:
            if not yaml_file.name.endswith(".disabled"):
                components.append(yaml_file.stem)

        return sorted(components)

    def get_component_summary(self) -> str:
        """Get formatted component summary."""
        summary = "Component Configuration Summary:\n"

        # Base components
        base_components = self.list_available_components()
        summary += f"\nBase Components ({len(base_components)}):\n"
        for component in base_components:
            summary += f"  - {component}\n"

        # Scenario-specific components
        for scenario in DevelopmentScenario:
            scenario_components = self.list_available_components(scenario)
            summary += (
                f"\n{scenario.value.title()} Scenario ({len(scenario_components)}):\n"
            )
            for component in scenario_components:
                summary += f"  - {component}\n"

        return summary


if __name__ == "__main__":
    # CLI interface for component management
    import argparse

    parser = argparse.ArgumentParser(description="Manage Dapr components for testing")
    parser.add_argument("--validate", help="Validate components for scenario")
    parser.add_argument("--create", help="Create components for scenario")
    parser.add_argument("--list", action="store_true", help="List available components")
    parser.add_argument("--summary", action="store_true", help="Show component summary")
    args = parser.parse_args()

    manager = ComponentManager()

    if args.validate:
        try:
            scenario = DevelopmentScenario(args.validate)
            result = manager.validate_components_for_scenario(scenario)

            print(f"Validation for {scenario.value}:")
            print(f"  Components found: {len(result['components_found'])}")
            print(f"  Valid: {len(result['components_valid'])}")
            print(f"  Invalid: {len(result['components_invalid'])}")

            if result["errors"]:
                print("\nErrors:")
                for error in result["errors"]:
                    print(f"  ✗ {error}")

            if result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"  ⚠ {warning}")

        except ValueError:
            print(f"Invalid scenario: {args.validate}")
            print(f"Valid scenarios: {[s.value for s in DevelopmentScenario]}")

    elif args.create:
        try:
            scenario = DevelopmentScenario(args.create)
            target_dir = manager.create_scenario_components(scenario)
            print(f"Created components for scenario: {scenario.value}")
            print(f"Target directory: {target_dir}")

        except ValueError:
            print(f"Invalid scenario: {args.create}")
            print(f"Valid scenarios: {[s.value for s in DevelopmentScenario]}")

    elif args.list:
        print("Available Components:")
        components = manager.list_available_components()
        for component in components:
            print(f"  - {component}")

    elif args.summary:
        print(manager.get_component_summary())

    else:
        print("Use --help for available options")
