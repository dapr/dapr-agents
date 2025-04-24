from __future__ import annotations
import shutil
from pathlib import Path
from typing import List, Optional, Set
from functools import lru_cache
from enum import Enum


class PackageManagerType(str, Enum):
    """Types of package managers that can be detected."""

    PIP = "pip"
    POETRY = "poetry"
    PIPENV = "pipenv"
    CONDA = "conda"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    BUN = "bun"
    CARGO = "cargo"
    GO = "go"
    MAVEN = "maven"
    GRADLE = "gradle"
    COMPOSER = "composer"
    UNKNOWN = "unknown"


class ProjectType(str, Enum):
    """Types of projects that can be detected."""

    PYTHON = "python"
    NODE = "node"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    PHP = "php"
    UNKNOWN = "unknown"


class PackageManager:
    """Information about a package manager and its commands."""

    def __init__(
        self,
        name: PackageManagerType,
        project_type: ProjectType,
        install_cmd: str,
        add_cmd: Optional[str] = None,
        remove_cmd: Optional[str] = None,
        update_cmd: Optional[str] = None,
        markers: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a package manager.

        Args:
            name: Package manager identifier.
            project_type: Type of project this manager serves.
            install_cmd: Command to install project dependencies.
            add_cmd: Command to add a single package.
            remove_cmd: Command to remove a package.
            update_cmd: Command to update packages.
            markers: Filenames indicating this manager in a project.
        """
        self.name = name
        self.project_type = project_type
        self.install_cmd = install_cmd
        self.add_cmd = add_cmd or f"{name.value} install"
        self.remove_cmd = remove_cmd or f"{name.value} remove"
        self.update_cmd = update_cmd or f"{name.value} update"
        self.markers = markers or []

    def __str__(self) -> str:
        return self.name.value


# Known package managers
PACKAGE_MANAGERS: dict[PackageManagerType, PackageManager] = {
    # Python package managers
    PackageManagerType.PIP: PackageManager(
        name=PackageManagerType.PIP,
        project_type=ProjectType.PYTHON,
        install_cmd="pip install -r requirements.txt",
        add_cmd="pip install",
        remove_cmd="pip uninstall",
        update_cmd="pip install --upgrade",
        markers=["requirements.txt", "setup.py", "setup.cfg"],
    ),
    PackageManagerType.POETRY: PackageManager(
        name=PackageManagerType.POETRY,
        project_type=ProjectType.PYTHON,
        install_cmd="poetry install",
        add_cmd="poetry add",
        remove_cmd="poetry remove",
        update_cmd="poetry update",
        markers=["pyproject.toml", "poetry.lock"],
    ),
    PackageManagerType.PIPENV: PackageManager(
        name=PackageManagerType.PIPENV,
        project_type=ProjectType.PYTHON,
        install_cmd="pipenv install",
        add_cmd="pipenv install",
        remove_cmd="pipenv uninstall",
        update_cmd="pipenv update",
        markers=["Pipfile", "Pipfile.lock"],
    ),
    PackageManagerType.CONDA: PackageManager(
        name=PackageManagerType.CONDA,
        project_type=ProjectType.PYTHON,
        install_cmd="conda env update -f environment.yml",
        add_cmd="conda install",
        remove_cmd="conda remove",
        update_cmd="conda update",
        markers=["environment.yml", "environment.yaml"],
    ),
    # JavaScript package managers
    PackageManagerType.NPM: PackageManager(
        name=PackageManagerType.NPM,
        project_type=ProjectType.NODE,
        install_cmd="npm install",
        add_cmd="npm install",
        remove_cmd="npm uninstall",
        update_cmd="npm update",
        markers=["package.json", "package-lock.json"],
    ),
    PackageManagerType.YARN: PackageManager(
        name=PackageManagerType.YARN,
        project_type=ProjectType.NODE,
        install_cmd="yarn install",
        add_cmd="yarn add",
        remove_cmd="yarn remove",
        update_cmd="yarn upgrade",
        markers=["package.json", "yarn.lock"],
    ),
    PackageManagerType.PNPM: PackageManager(
        name=PackageManagerType.PNPM,
        project_type=ProjectType.NODE,
        install_cmd="pnpm install",
        add_cmd="pnpm add",
        remove_cmd="pnpm remove",
        update_cmd="pnpm update",
        markers=["package.json", "pnpm-lock.yaml"],
    ),
    PackageManagerType.BUN: PackageManager(
        name=PackageManagerType.BUN,
        project_type=ProjectType.NODE,
        install_cmd="bun install",
        add_cmd="bun add",
        remove_cmd="bun remove",
        update_cmd="bun update",
        markers=["package.json", "bun.lockb"],
    ),
}


@lru_cache(maxsize=None)
def is_installed(name: str) -> bool:
    """
    Check if a given command exists on PATH.

    Args:
        name: Command name to check.

    Returns:
        True if the command is available, False otherwise.
    """
    return shutil.which(name) is not None


@lru_cache(maxsize=None)
def detect_package_managers(directory: str) -> List[PackageManager]:
    """
    Detect all installed package managers by looking for marker files.

    Args:
        directory: Path to the project root.

    Returns:
        A list of PackageManager instances found in the directory.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    found: List[PackageManager] = []
    for pm in PACKAGE_MANAGERS.values():
        for marker in pm.markers:
            if (dir_path / marker).exists() and is_installed(pm.name.value):
                found.append(pm)
                break
    return found


@lru_cache(maxsize=None)
def get_primary_package_manager(directory: str) -> Optional[PackageManager]:
    """
    Determine the primary package manager using lockfile heuristics.

    Args:
        directory: Path to the project root.

    Returns:
        The chosen PackageManager or None if none detected.
    """
    managers = detect_package_managers(directory)
    if not managers:
        return None
    if len(managers) == 1:
        return managers[0]

    dir_path = Path(directory)
    # Prefer lockfiles over others
    lock_priority = [
        (PackageManagerType.POETRY, "poetry.lock"),
        (PackageManagerType.PIPENV, "Pipfile.lock"),
        (PackageManagerType.PNPM, "pnpm-lock.yaml"),
        (PackageManagerType.YARN, "yarn.lock"),
        (PackageManagerType.BUN, "bun.lockb"),
        (PackageManagerType.NPM, "package-lock.json"),
    ]
    for pm_type, lock in lock_priority:
        if (dir_path / lock).exists() and PACKAGE_MANAGERS.get(pm_type) in managers:
            return PACKAGE_MANAGERS[pm_type]

    return managers[0]


def get_install_command(directory: str) -> Optional[str]:
    """
    Get the shell command to install project dependencies.

    Args:
        directory: Path to the project root.

    Returns:
        A shell command string or None if no manager detected.
    """
    pm = get_primary_package_manager(directory)
    return pm.install_cmd if pm else None


def get_add_command(directory: str, package: str, dev: bool = False) -> Optional[str]:
    """
    Get the shell command to add a package to the project.

    Args:
        directory: Path to the project root.
        package: Package name to add.
        dev: Whether to add as a development dependency.

    Returns:
        A shell command string or None if no manager detected.
    """
    pm = get_primary_package_manager(directory)
    if not pm:
        return None
    base = pm.add_cmd
    if dev and pm.name in {
        PackageManagerType.PIP,
        PackageManagerType.POETRY,
        PackageManagerType.NPM,
        PackageManagerType.YARN,
        PackageManagerType.PNPM,
        PackageManagerType.BUN,
        PackageManagerType.COMPOSER,
    }:
        flag = (
            "--dev"
            if pm.name in {PackageManagerType.PIP, PackageManagerType.POETRY}
            else "--save-dev"
        )
        return f"{base} {package} {flag}"
    return f"{base} {package}"


def get_project_type(directory: str) -> ProjectType:
    """
    Infer project type from the primary package manager or file extensions.

    Args:
        directory: Path to the project root.

    Returns:
        The detected ProjectType.
    """
    pm = get_primary_package_manager(directory)
    if pm:
        return pm.project_type

    # Fallback by extension scanning
    exts: Set[str] = set()
    for path in Path(directory).rglob("*"):
        if path.is_file():
            exts.add(path.suffix.lower())
            if len(exts) > 50:
                break
    if ".py" in exts:
        return ProjectType.PYTHON
    if {".js", ".ts"} & exts:
        return ProjectType.NODE
    if ".rs" in exts:
        return ProjectType.RUST
    if ".go" in exts:
        return ProjectType.GO
    if ".java" in exts:
        return ProjectType.JAVA
    if ".php" in exts:
        return ProjectType.PHP
    return ProjectType.UNKNOWN
