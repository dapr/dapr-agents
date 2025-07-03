#!/usr/bin/env python3

"""
Quickstart Validation Tool

Validates all quickstart directories, optionally with isolated virtual environments.
This replaces the complex bash logic that was in the Makefile.

Usage:
    python tools/validate_quickstarts.py [--isolated] [--quickstart <name>]

Options:
    --isolated    Create isolated virtual environments for each quickstart
    --quickstart  Validate only a specific quickstart
    --help        Show this help message
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import yaml
import ast


class QuickstartValidator:
    """Handles validation of quickstart directories."""

    def __init__(self, use_isolated_envs: bool = False):
        self.use_isolated_envs = use_isolated_envs
        self.project_root = Path(__file__).parent.parent
        self.quickstarts_dir = self.project_root / "quickstarts"

        if not self.quickstarts_dir.exists():
            raise FileNotFoundError(
                f"Quickstarts directory not found: {self.quickstarts_dir}"
            )

    def discover_quickstarts(self) -> List[str]:
        """Discover all quickstart directories."""
        quickstarts = []
        for item in self.quickstarts_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                quickstarts.append(item.name)
        return sorted(quickstarts)

    def _validate_readme(self, quickstart_path: Path) -> bool:
        """Check for README.md and basic headers."""
        readme_path = quickstart_path / "README.md"
        if not readme_path.exists():
            print("❌ Missing README.md")
            return False
        print("✅ README.md exists")

        with open(readme_path, "r", encoding="utf-8") as f:
            if "# " not in f.read():
                print("⚠️  README.md might be missing proper headers")
            else:
                print("✅ README.md has headers")
        return True

    def _validate_python_syntax(self, quickstart_path: Path) -> bool:
        """Validate syntax of all Python files."""
        python_files = list(quickstart_path.rglob("*.py"))
        if not python_files:
            print("⚠️  No Python files found")
            return True

        print("🐍 Checking Python syntax...")
        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    ast.parse(f.read())
            except Exception as e:
                print(
                    f"❌ Python syntax error in: {py_file.relative_to(self.project_root)}"
                )
                print(f"   {e}")
                return False
        print("✅ All Python files have valid syntax")
        return True

    def _validate_requirements(self, quickstart_path: Path) -> bool:
        """Validate requirements.txt if it exists."""
        requirements_path = quickstart_path / "requirements.txt"
        if not requirements_path.exists():
            print("⚠️  No requirements.txt found")
            return True

        print("📦 Checking requirements.txt...")
        if requirements_path.stat().st_size == 0:
            print("⚠️  requirements.txt is empty")
            return True

        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(("#", "-")):
                    # A very basic check to see if a line looks like a package
                    if not line[0].isalpha():
                        print(
                            f"❌ requirements.txt format appears invalid at line: {line}"
                        )
                        return False
        print("✅ requirements.txt format looks valid")
        return True

    def _validate_dapr_components(self, quickstart_path: Path) -> bool:
        """Validate Dapr component YAML files."""
        components_dir = quickstart_path / "components"
        if not components_dir.exists():
            print("⚠️  No components directory found")
            return True

        print("🔧 Checking Dapr components...")
        yaml_files = list(components_dir.glob("*.yaml")) + list(
            components_dir.glob("*.yml")
        )
        if not yaml_files:
            print("⚠️  No component YAML files found in components directory")
            return True

        print("✅ Found Dapr component files")
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
            except Exception as e:
                print(
                    f"❌ YAML syntax error in: {yaml_file.relative_to(self.project_root)}"
                )
                print(f"   {e}")
                return False
        print("✅ All YAML files have valid syntax")
        return True

    def _run_all_validations(self, quickstart_path: Path) -> bool:
        """Run all individual validation checks for a quickstart."""
        checks = [
            self._validate_readme,
            self._validate_python_syntax,
            self._validate_requirements,
            self._validate_dapr_components,
        ]

        for check in checks:
            if not check(quickstart_path):
                return False
        return True

    def validate_quickstart(self, quickstart_name: str) -> bool:
        """Validate a single quickstart directory."""
        quickstart_path = self.quickstarts_dir / quickstart_name

        if not quickstart_path.exists():
            print(f"❌ Quickstart directory not found: {quickstart_name}")
            return False

        print(f"\n=== Validating {quickstart_name} ===")

        if self.use_isolated_envs and (quickstart_path / "requirements.txt").exists():
            return self._validate_with_isolated_env(quickstart_path)
        else:
            return self._validate_with_current_env(quickstart_path)

    def _validate_with_isolated_env(self, quickstart_path: Path) -> bool:
        """Validate quickstart with an isolated virtual environment."""
        venv_path = quickstart_path / ".venv"

        try:
            print(f"Creating virtual environment for {quickstart_path.name}...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                check=True,
                capture_output=True,
            )

            # Determine the activate script path
            if sys.platform == "win32":
                python_executable = venv_path / "Scripts" / "python"
            else:
                python_executable = venv_path / "bin" / "python"

            print(f"Installing requirements for {quickstart_path.name}...")
            subprocess.run(
                [
                    str(python_executable),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.txt",
                ],
                cwd=quickstart_path,
                check=True,
                capture_output=True,
            )

            print(f"Running validation for {quickstart_path.name} in isolated env...")
            success = self._run_all_validations(quickstart_path)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during isolated environment validation: {e}")
            success = False

        finally:
            # Clean up virtual environment
            if venv_path.exists():
                print(f"Cleaning up virtual environment for {quickstart_path.name}...")
                if sys.platform == "win32":
                    subprocess.run(
                        ["rmdir", "/s", "/q", str(venv_path)],
                        shell=True,
                        capture_output=True,
                    )
                else:
                    subprocess.run(["rm", "-rf", str(venv_path)], capture_output=True)

        return success

    def _validate_with_current_env(self, quickstart_path: Path) -> bool:
        """Validate quickstart with the current environment."""
        print(f"Running validation for {quickstart_path.name} in current env...")
        return self._run_all_validations(quickstart_path)

    def validate_all(self, specific_quickstart: Optional[str] = None) -> bool:
        """Validate all quickstarts or a specific one."""
        if specific_quickstart:
            quickstarts = [specific_quickstart]
        else:
            quickstarts = self.discover_quickstarts()

        print(f"🧪 Validating {len(quickstarts)} quickstart(s)...")
        if self.use_isolated_envs:
            print("📦 Using isolated virtual environments")
        else:
            print("🔄 Using current environment")

        failed_quickstarts = []

        for quickstart in quickstarts:
            try:
                success = self.validate_quickstart(quickstart)
                if success:
                    print(f"✅ {quickstart} validated successfully")
                else:
                    print(f"❌ {quickstart} validation failed")
                    failed_quickstarts.append(quickstart)

                # Brief pause between validations
                if len(quickstarts) > 1:
                    time.sleep(1)

            except KeyboardInterrupt:
                print("⚠️  Validation interrupted by user")
                return False
            except Exception as e:
                print(f"❌ Unexpected error validating {quickstart}: {e}")
                failed_quickstarts.append(quickstart)

        # Summary
        if failed_quickstarts:
            print(f"\n❌ Validation failed for: {', '.join(failed_quickstarts)}")
            return False
        else:
            print("🎉 All quickstart validations completed successfully!")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Dapr Agents quickstart directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/validate_quickstarts.py
    python tools/validate_quickstarts.py --isolated
    python tools/validate_quickstarts.py --quickstart 01-hello-world
    python tools/validate_quickstarts.py --isolated --quickstart 01-hello-world
        """,
    )

    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Create isolated virtual environments for quickstarts with requirements.txt",
    )

    parser.add_argument(
        "--quickstart",
        type=str,
        help="Validate only the specified quickstart directory",
    )

    args = parser.parse_args()

    try:
        validator = QuickstartValidator(use_isolated_envs=args.isolated)
        success = validator.validate_all(specific_quickstart=args.quickstart)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
