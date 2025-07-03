"""Version management and compatibility checking."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages version detection and compatibility checking."""

    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent.parent

    def get_dapr_version(self) -> Optional[str]:
        """Get current Dapr runtime version."""
        try:
            result = subprocess.run(
                ["dapr", "--version"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                # Parse version from output like "CLI version: 1.12.0"
                for line in result.stdout.split("\n"):
                    if "CLI version:" in line:
                        version_match = re.search(r"(\d+\.\d+\.\d+)", line)
                        if version_match:
                            return version_match.group(1)

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return None

    def get_python_sdk_version(self) -> Optional[str]:
        """Get current Python SDK version."""
        try:
            import dapr

            return getattr(dapr, "__version__", None)
        except ImportError:
            return None

    def get_agents_version(self) -> Optional[str]:
        """Get current dapr-agents version."""
        try:
            # Try to read from pyproject.toml
            pyproject_path = self.workspace_root / "pyproject.toml"
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    return version_match.group(1)

            # Try to read from setup.py or __init__.py
            init_path = self.workspace_root / "dapr_agents" / "__init__.py"
            if init_path.exists():
                content = init_path.read_text()
                version_match = re.search(
                    r'__version__\s*=\s*["\']([^"\']+)["\']', content
                )
                if version_match:
                    return version_match.group(1)

            return "dev"  # Default for development

        except Exception as e:
            logger.warning(f"Could not determine agents version: {e}")
            return "dev"

    def get_python_version(self) -> str:
        """Get current Python version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def get_all_versions(self) -> Dict[str, Optional[str]]:
        """Get all component versions."""
        return {
            "python": self.get_python_version(),
            "dapr": self.get_dapr_version(),
            "python_sdk": self.get_python_sdk_version(),
            "dapr_agents": self.get_agents_version(),
        }

    def check_compatibility(self, versions: Dict[str, str]) -> Dict[str, bool]:
        """Check version compatibility matrix."""
        compatibility = {}

        # Python version compatibility
        python_version = versions.get("python", self.get_python_version())
        python_major_minor = ".".join(python_version.split(".")[:2])
        compatibility["python_supported"] = python_major_minor in [
            "3.8",
            "3.9",
            "3.10",
            "3.11",
            "3.12",
        ]

        # Dapr runtime compatibility
        dapr_version = versions.get("dapr")
        if dapr_version:
            dapr_major_minor = ".".join(dapr_version.split(".")[:2])
            compatibility["dapr_supported"] = self._is_dapr_version_supported(
                dapr_major_minor
            )
        else:
            compatibility["dapr_supported"] = False

        # Python SDK compatibility
        sdk_version = versions.get("python_sdk")
        if sdk_version:
            compatibility["sdk_supported"] = self._is_sdk_version_supported(sdk_version)
        else:
            compatibility["sdk_supported"] = False

        # Cross-component compatibility
        if dapr_version and sdk_version:
            compatibility["dapr_sdk_compatible"] = self._check_dapr_sdk_compatibility(
                dapr_version, sdk_version
            )
        else:
            compatibility["dapr_sdk_compatible"] = False

        return compatibility

    def _is_dapr_version_supported(self, version: str) -> bool:
        """Check if Dapr version is supported."""
        supported_versions = ["1.11", "1.12", "1.13", "1.14"]
        return version in supported_versions

    def _is_sdk_version_supported(self, version: str) -> bool:
        """Check if Python SDK version is supported."""
        # For now, accept any version that looks like a semantic version
        return bool(re.match(r"\d+\.\d+\.\d+", version))

    def _check_dapr_sdk_compatibility(
        self, dapr_version: str, sdk_version: str
    ) -> bool:
        """Check if Dapr runtime and Python SDK versions are compatible."""
        # Generally, SDK and runtime should have matching major.minor versions
        dapr_major_minor = ".".join(dapr_version.split(".")[:2])
        sdk_major_minor = ".".join(sdk_version.split(".")[:2])

        return dapr_major_minor == sdk_major_minor

    def get_test_matrix(self) -> List[Dict[str, str]]:
        """Get list of version combinations to test."""
        # Define test matrix for different scenarios
        matrix = [
            # Current stable versions
            {
                "name": "stable",
                "dapr": "1.12.0",
                "python_sdk": "1.12.0",
                "python": "3.11",
            },
            # Latest versions
            {
                "name": "latest",
                "dapr": "1.13.0",
                "python_sdk": "1.13.0",
                "python": "3.12",
            },
            # Minimum supported versions
            {
                "name": "minimum",
                "dapr": "1.11.0",
                "python_sdk": "1.11.0",
                "python": "3.8",
            },
        ]

        return matrix

    def validate_test_environment(self) -> Dict[str, any]:
        """Validate current test environment versions."""
        versions = self.get_all_versions()
        compatibility = self.check_compatibility(versions)

        validation_result = {
            "versions": versions,
            "compatibility": compatibility,
            "all_compatible": all(compatibility.values()),
            "warnings": [],
            "errors": [],
        }

        # Add warnings for version issues
        if not compatibility.get("python_supported", False):
            validation_result["warnings"].append(
                f"Python version {versions['python']} may not be fully supported"
            )

        if not compatibility.get("dapr_supported", False):
            validation_result["errors"].append(
                f"Dapr version {versions['dapr']} is not supported"
            )

        if not compatibility.get("sdk_supported", False):
            validation_result["errors"].append(
                f"Python SDK version {versions['python_sdk']} is not supported"
            )

        if not compatibility.get("dapr_sdk_compatible", False):
            validation_result["warnings"].append(
                f"Dapr runtime ({versions['dapr']}) and Python SDK ({versions['python_sdk']}) versions may be incompatible"
            )

        return validation_result

    def get_version_info_summary(self) -> str:
        """Get formatted version information summary."""
        versions = self.get_all_versions()
        compatibility = self.check_compatibility(versions)

        summary = "Version Information:\n"
        summary += f"  Python: {versions['python']}\n"
        summary += f"  Dapr Runtime: {versions['dapr'] or 'Not found'}\n"
        summary += f"  Python SDK: {versions['python_sdk'] or 'Not found'}\n"
        summary += f"  Dapr Agents: {versions['dapr_agents']}\n"

        summary += "\nCompatibility:\n"
        for check, result in compatibility.items():
            status = "✓" if result else "✗"
            summary += f"  {status} {check.replace('_', ' ').title()}\n"

        return summary


if __name__ == "__main__":
    # CLI interface for version checking
    import argparse

    parser = argparse.ArgumentParser(description="Check version compatibility")
    parser.add_argument("--matrix", action="store_true", help="Show test matrix")
    parser.add_argument(
        "--validate", action="store_true", help="Validate current environment"
    )
    args = parser.parse_args()

    manager = VersionManager()

    if args.matrix:
        print("Test Matrix:")
        matrix = manager.get_test_matrix()
        for config in matrix:
            print(f"\n{config['name'].upper()}:")
            for key, value in config.items():
                if key != "name":
                    print(f"  {key}: {value}")

    elif args.validate:
        validation = manager.validate_test_environment()
        print("Environment Validation:")
        print(
            f"Overall Status: {'✓ PASS' if validation['all_compatible'] else '✗ ISSUES FOUND'}"
        )

        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  ⚠ {warning}")

        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  ✗ {error}")

    else:
        print(manager.get_version_info_summary())
