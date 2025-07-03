#!/usr/bin/env python3
"""Test runner for Dapr Agents with scenario support."""

import argparse
import subprocess
import sys
from pathlib import Path
from utils import ScenarioManager, EnvManager, VersionManager

# Add the tests directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Dapr Agents tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast tests only (no slow/api_key tests)",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--scenario-info", action="store_true", help="Show scenario information"
    )
    parser.add_argument(
        "--validate-env", action="store_true", help="Validate environment"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Show scenario information
    if args.scenario_info:
        print("🔍 Development Scenario Information")
        print("=" * 50)

        scenario_manager = ScenarioManager()
        scenario = scenario_manager.detect_scenario()

        print(f"Detected Scenario: {scenario.value}")
        print(f"Description: {scenario_manager.get_scenario_description(scenario)}")

        print("\nRepository Status:")
        repos = scenario_manager._detect_development_repos()
        for repo, is_dev in repos.items():
            status = "🔧 LOCAL DEVELOPMENT" if is_dev else "📦 RELEASED/STABLE"
            print(f"  {repo}: {status}")

        print("\nComponent Directory:")
        component_dir = scenario_manager.get_component_config_dir(scenario)
        print(f"  {component_dir}")

        return

    # Validate environment
    if args.validate_env:
        print("🔍 Environment Validation")
        print("=" * 50)

        version_manager = VersionManager()
        validation = version_manager.validate_test_environment()

        print(
            f"Overall Status: {'✅ PASS' if validation['all_compatible'] else '❌ ISSUES FOUND'}"
        )

        print("\nVersions:")
        for component, version in validation["versions"].items():
            print(f"  {component}: {version or 'Not found'}")

        print("\nCompatibility:")
        for check, result in validation["compatibility"].items():
            status = "✅" if result else "❌"
            print(f"  {status} {check.replace('_', ' ').title()}")

        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  ⚠️  {warning}")

        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  ❌ {error}")

        env_manager = EnvManager()
        print("\nAPI Keys:")
        api_keys = env_manager.check_api_keys()
        for provider, available in api_keys.items():
            status = "✅" if available else "❌"
            print(f"  {status} {provider}")

        return

    # Default to showing help if no specific test type selected
    if not any([args.unit, args.integration, args.fast, args.all]):
        print("🧪 Dapr Agents Test Runner")
        print("=" * 50)
        print("Please specify test type:")
        print("  --unit          Run unit tests only")
        print("  --integration   Run integration tests only")
        print("  --fast          Run fast tests (no slow/api_key tests)")
        print("  --all           Run all tests")
        print("  --scenario-info Show development scenario info")
        print("  --validate-env  Validate test environment")
        print("")
        print("Examples:")
        print("  python tests/run_tests.py --fast")
        print("  python tests/run_tests.py --unit --verbose")
        print("  python tests/run_tests.py --scenario-info")
        return

    # Build pytest command
    cmd = ["pytest"]

    if args.verbose:
        cmd.append("-v")

    # Add test paths and markers
    if args.unit:
        cmd.extend(["tests/unit/", "-m", "not slow"])
    elif args.integration:
        cmd.extend(["tests/integration/", "-m", "not requires_api_key"])
    elif args.fast:
        cmd.extend(["tests/", "-m", "not slow and not requires_api_key"])
    elif args.all:
        cmd.append("tests/")

    # Run tests
    success = run_command(cmd, "Running tests")

    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
