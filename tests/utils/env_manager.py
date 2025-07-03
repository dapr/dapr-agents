"""Environment management for testing scenarios."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from dotenv import load_dotenv
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


class EnvManager:
    """Manages environment setup and validation for different testing scenarios."""

    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent.parent
        self.env_file = self.workspace_root / ".env"
        self.test_env_file = self.workspace_root / "tests" / ".env.test"

    def load_env_file(self) -> bool:
        """Load environment variables from .env file."""
        env_loaded = False

        # Set faster Dapr health check timeout for development
        if not os.getenv("DAPR_HEALTH_TIMEOUT"):
            os.environ["DAPR_HEALTH_TIMEOUT"] = "5"  # 5 seconds instead of 60

        # Try to load from main .env file first
        if self.env_file.exists():
            load_dotenv(self.env_file)
            env_loaded = True
            logger.info(f"Loaded environment from {self.env_file}")

        # Also try to load test-specific env file
        if self.test_env_file.exists():
            load_dotenv(self.test_env_file)
            env_loaded = True
            logger.info(f"Loaded test environment from {self.test_env_file}")

        if not env_loaded:
            logger.warning(
                "No .env file found. Using system environment variables only."
            )

        return env_loaded

    def detect_development_setup(self) -> Dict[str, bool]:
        """Detect which repositories are in local development."""
        workspace_parent = self.workspace_root.parent

        repos = {
            "dapr": workspace_parent / "dapr",
            "python_sdk": workspace_parent / "python-sdk",
            "components_contrib": workspace_parent / "components-contrib",
            "dapr_agents": self.workspace_root,  # Always local
        }

        development_status = {}
        for repo_name, repo_path in repos.items():
            if repo_name == "dapr_agents":
                development_status[repo_name] = True
            else:
                development_status[repo_name] = self._is_repo_in_development(repo_path)

        return development_status

    def _is_repo_in_development(self, repo_path: Path) -> bool:
        """Check if a repository appears to be in active development."""
        if not repo_path.exists():
            return False

        # Check if it's a git repository
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return False

        # Check for common development indicators
        indicators = [
            repo_path / "Makefile",
            repo_path / "go.mod",  # For Dapr and components-contrib
            repo_path / "setup.py",  # For Python SDK
            repo_path / "pyproject.toml",  # For Python projects
        ]

        return any(indicator.exists() for indicator in indicators)

    def setup_for_scenario(self, scenario: DevelopmentScenario) -> Dict[str, any]:
        """Setup environment for specific development scenario."""
        setup_result = {
            "scenario": scenario.value,
            "env_loaded": self.load_env_file(),
            "api_keys": self.check_api_keys(),
            "development_repos": self.detect_development_setup(),
            "requirements_met": True,
            "warnings": [],
            "errors": [],
        }

        # Scenario-specific setup
        if scenario == DevelopmentScenario.LOCAL_FULL:
            self._setup_local_full_scenario(setup_result)
        elif scenario == DevelopmentScenario.LOCAL_PARTIAL:
            self._setup_local_partial_scenario(setup_result)
        elif scenario == DevelopmentScenario.AGENT_ONLY:
            self._setup_agent_only_scenario(setup_result)
        elif scenario == DevelopmentScenario.PRODUCTION:
            self._setup_production_scenario(setup_result)

        return setup_result

    def _setup_local_full_scenario(self, setup_result: Dict):
        """Setup for full local development scenario."""
        dev_repos = setup_result["development_repos"]

        # Check that all repos are available for local development
        required_repos = ["dapr", "python_sdk", "components_contrib"]
        for repo in required_repos:
            if not dev_repos.get(repo, False):
                setup_result["errors"].append(
                    f"{repo} repository not found or not in development"
                )
                setup_result["requirements_met"] = False

        # Check for build requirements
        build_requirements = self.check_build_requirements(
            DevelopmentScenario.LOCAL_FULL
        )
        if not all(build_requirements.values()):
            setup_result["warnings"].append("Some build requirements may not be met")

    def _setup_local_partial_scenario(self, setup_result: Dict):
        """Setup for partial local development scenario."""
        dev_repos = setup_result["development_repos"]

        # Check that Python SDK is available
        if not dev_repos.get("python_sdk", False):
            setup_result["warnings"].append("Python SDK not found in local development")

        # Check for released Dapr runtime
        if not self._check_released_dapr():
            setup_result["errors"].append("Released Dapr runtime not found")
            setup_result["requirements_met"] = False

    def _setup_agent_only_scenario(self, setup_result: Dict):
        """Setup for agent-only development scenario."""
        # Check for released dependencies
        if not self._check_released_dapr():
            setup_result["errors"].append("Released Dapr runtime not found")
            setup_result["requirements_met"] = False

        if not self._check_released_python_sdk():
            setup_result["errors"].append("Released Python SDK not found")
            setup_result["requirements_met"] = False

    def _setup_production_scenario(self, setup_result: Dict):
        """Setup for production-like scenario."""
        # All dependencies should be released versions
        if not self._check_released_dapr():
            setup_result["errors"].append("Released Dapr runtime not found")
            setup_result["requirements_met"] = False

        if not self._check_released_python_sdk():
            setup_result["errors"].append("Released Python SDK not found")
            setup_result["requirements_met"] = False

    def check_api_keys(self, providers: Optional[List[str]] = None) -> Dict[str, bool]:
        """Check which API keys are available."""
        if providers is None:
            providers = ["openai", "anthropic", "gemini", "azure_openai"]

        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
        }

        available_keys = {}
        for provider in providers:
            env_var = api_key_map.get(provider, f"{provider.upper()}_API_KEY")
            available_keys[provider] = bool(os.getenv(env_var))

        return available_keys

    def get_makefile_targets(self) -> List[str]:
        """Get available Makefile targets for current scenario."""
        makefile_path = self.workspace_root / "Makefile"
        if not makefile_path.exists():
            return []

        targets = []
        try:
            content = makefile_path.read_text()
            for line in content.split("\n"):
                if (
                    ":" in line
                    and not line.startswith("\t")
                    and not line.startswith("#")
                ):
                    target = line.split(":")[0].strip()
                    if target and not target.startswith("."):
                        targets.append(target)
        except Exception as e:
            logger.warning(f"Could not parse Makefile: {e}")

        return sorted(targets)

    def check_build_requirements(
        self, scenario: DevelopmentScenario
    ) -> Dict[str, bool]:
        """Check if build requirements are met for scenario."""
        requirements = {}

        if scenario in [
            DevelopmentScenario.LOCAL_FULL,
            DevelopmentScenario.LOCAL_PARTIAL,
        ]:
            # Check for Go (needed for Dapr and components-contrib)
            requirements["go"] = self._check_go_available()

            # Check for Python development tools
            requirements["python_dev"] = self._check_python_dev_tools()

            # Check for Make
            requirements["make"] = self._check_make_available()

        if scenario == DevelopmentScenario.LOCAL_FULL:
            # Additional requirements for full local development
            requirements["docker"] = self._check_docker_available()
            requirements["git"] = self._check_git_available()

        return requirements

    def _check_released_dapr(self) -> bool:
        """Check if released Dapr runtime is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["dapr", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_released_python_sdk(self) -> bool:
        """Check if released Python SDK is available."""
        try:
            import dapr

            return True
        except ImportError:
            return False

    def _check_go_available(self) -> bool:
        """Check if Go is available."""
        try:
            import subprocess

            result = subprocess.run(["go", "version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _check_python_dev_tools(self) -> bool:
        """Check if Python development tools are available."""
        try:
            # Check for pip and basic dev tools without importing setuptools
            import pip

            # Just check if we can import these basic modules
            import sys
            import subprocess

            # Try to run pip as a simple check
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_make_available(self) -> bool:
        """Check if Make is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["make", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_git_available(self) -> bool:
        """Check if Git is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def create_test_env_template(self) -> None:
        """Create a template .env.test file."""
        template_content = """# Test Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here

# Test Configuration
TEST_TIMEOUT=30
TEST_LOG_LEVEL=INFO

# Dapr Configuration
DAPR_HTTP_ENDPOINT=http://localhost:3500
DAPR_GRPC_ENDPOINT=localhost:50001
"""

        if not self.test_env_file.exists():
            self.test_env_file.write_text(template_content)
            logger.info(f"Created test environment template at {self.test_env_file}")

    def get_environment_summary(self) -> str:
        """Get formatted environment summary."""
        self.load_env_file()

        summary = "Environment Summary:\n"

        # Development repositories
        dev_repos = self.detect_development_setup()
        summary += "\nDevelopment Repositories:\n"
        for repo, is_dev in dev_repos.items():
            status = "LOCAL DEV" if is_dev else "RELEASED"
            summary += f"  {repo}: {status}\n"

        # API Keys
        api_keys = self.check_api_keys()
        summary += "\nAPI Keys:\n"
        for provider, available in api_keys.items():
            status = "✓" if available else "✗"
            summary += f"  {status} {provider}\n"

        # Makefile targets
        targets = self.get_makefile_targets()
        if targets:
            summary += f"\nAvailable Makefile Targets: {', '.join(targets[:5])}"
            if len(targets) > 5:
                summary += f" (and {len(targets) - 5} more)"

        return summary


if __name__ == "__main__":
    # CLI interface for environment management
    import argparse

    parser = argparse.ArgumentParser(description="Manage test environment")
    parser.add_argument(
        "--summary", action="store_true", help="Show environment summary"
    )
    parser.add_argument(
        "--create-template", action="store_true", help="Create test env template"
    )
    parser.add_argument("--check-keys", action="store_true", help="Check API keys")
    args = parser.parse_args()

    manager = EnvManager()

    if args.create_template:
        manager.create_test_env_template()
        print("Test environment template created.")

    elif args.check_keys:
        keys = manager.check_api_keys()
        print("API Key Status:")
        for provider, available in keys.items():
            status = "✓ Available" if available else "✗ Missing"
            print(f"  {provider}: {status}")

    elif args.summary:
        print(manager.get_environment_summary())

    else:
        print("Use --help for available options")
