import os
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import pytest

logger = logging.getLogger(__name__)


def setup_quickstart_venv(quickstart_dir: Path, project_root: Path) -> Path:
    """
    Setup a virtual environment for a quickstart so as to better match end user setup.

    Creates a venv in the quickstart directory and installs from requirements.txt.
    So, if we are testing locally then we can update the requirements.txt to use the editable install from the project root.

    Args:
        quickstart_dir: Path to the quickstart directory
        project_root: Path to the project root (for editable install if needed)

    Returns:
        Path to the venv Python executable
    """
    # Each quickstart has its own directory, so venv is already unique per quickstart
    # Since pytest-xdist runs one test file per worker, and each quickstart has
    # a unique directory, then we know that each venv path is unique and will not conflict.
    # The venv name is set to ephemeral_test_venv to make it clear these are temporary test venvs.
    venv_path = quickstart_dir / "ephemeral_test_venv"
    if not venv_path.exists():
        logger.info(f"Creating venv in {quickstart_dir}")
        result = subprocess.run(
            ["python3", "-m", "venv", str(venv_path)],
            cwd=quickstart_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create venv: {result.stderr}\n{result.stdout}"
            )

    venv_python = venv_path / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_path / "Scripts" / "python.exe"
    if not venv_python.exists():
        raise RuntimeError(f"Venv Python not found at {venv_python}")

    requirements_file = quickstart_dir / "requirements.txt"
    # Skip installation if already done (for parallel execution)
    installed_marker = venv_path / ".installed"
    if installed_marker.exists():
        logger.info(f"Dependencies already installed for {quickstart_dir}")
    else:
        if requirements_file.exists():
            # Install dependencies from requirements.txt first
            logger.info(f"Installing dependencies from {requirements_file} using uv")
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "-r",
                    str(requirements_file),
                    "--python",
                    str(venv_python),
                ],
                cwd=quickstart_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install requirements: {result.stderr}\n{result.stdout}"
                )

            # Override with editable dapr-agents from current repo changes (for PR testing)
            # This ensures we test against the current repo changes, so we test local changes before release
            logger.info(
                f"Installing editable dapr-agents from {project_root} to override requirements.txt"
            )
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "-e",
                    str(project_root),
                    "--python",
                    str(venv_python),
                ],
                cwd=quickstart_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install editable dapr-agents: {result.stderr}\n{result.stdout}"
                )
        else:
            # No requirements.txt - install editable dapr-agents for testing using uv
            logger.info(
                "No requirements.txt found, installing editable dapr-agents using uv"
            )
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "-e",
                    str(project_root),
                    "--python",
                    str(venv_python),
                ],
                cwd=quickstart_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install dapr-agents: {result.stderr}\n{result.stdout}"
                )

        # Mark as installed so if we're running in parallel, we don't reinstall the dependencies.
        installed_marker.touch()

    return venv_python


def run_quickstart_script(
    script_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: int = 60,
    use_dapr: bool = False,
    app_id: Optional[str] = None,
    resources_path: Optional[Path] = None,
    dapr_http_port: int = 3500,
    create_venv: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a quickstart script and return the result.

    This function mimics the user's setup process:
    1. Creates a venv in the quickstart directory (like users do)
    2. Installs from requirements.txt (but uses local editable install for testing)
    3. Runs the script from that venv

    If USE_EXISTING_VENV=true then we will use the system Python venv (current venv or system) instead of creating a new venv.

    Args:
        script_path: Path to the Python script to run
        cwd: Working directory (defaults to script's parent directory)
        env: Environment variables to set
        timeout: Timeout in seconds
        use_dapr: Whether to run with `dapr run`
        app_id: Dapr app ID (required if use_dapr=True)
        resources_path: Path to Dapr resources/components (defaults to cwd/components)
        dapr_http_port: Dapr HTTP port (defaults to 3500)
        create_venv: Whether to create and use an ephemeral test venv (defaults to True).
                  Set to False or USE_EXISTING_VENV=true to use system Python instead.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    cwd_path = cwd or script_path.parent
    quickstart_dir = cwd_path
    project_root = (
        quickstart_dir.parent.parent
        if "quickstarts" in str(quickstart_dir)
        else quickstart_dir.parent
    )

    # Setup venv if requested
    # For local dev, set USE_EXISTING_VENV=true to skip venv creation and use system Python
    # Or pass create_venv=False to use system Python
    should_create_venv = create_venv and os.getenv(
        "USE_EXISTING_VENV", ""
    ).lower() not in ("true", "1")

    venv_python = None
    python_cmd = "python"  # Default fallback
    if should_create_venv:
        venv_python = setup_quickstart_venv(quickstart_dir, project_root)
        if venv_python.exists():
            python_cmd = str(venv_python)
    else:
        logger.info(
            f"Using existing venv/system Python for {quickstart_dir} (create_venv=False or USE_EXISTING_VENV=true)"
        )
        python_cmd = "python"

    if use_dapr:
        if not app_id:
            raise ValueError("app_id is required when use_dapr=True")
        if not resources_path:
            resources_path = cwd_path / "components"

        # Check if components directory exists and resolve env vars if needed
        project_root_path = (
            cwd_path.parent.parent
            if "quickstarts" in str(cwd_path)
            else cwd_path.parent
        )
        resolve_script = project_root_path / "quickstarts" / "resolve_env_templates.py"
        if resolve_script.exists() and resources_path.exists():
            resolve_python = (
                str(venv_python) if venv_python and venv_python.exists() else "python"
            )
            resolve_result = subprocess.run(
                [resolve_python, str(resolve_script), str(resources_path)],
                cwd=cwd_path,
                env=full_env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if resolve_result.returncode == 0:
                resolved_path = resolve_result.stdout.strip()
                if resolved_path and Path(resolved_path).exists():
                    resources_path = Path(resolved_path)

        # Build dapr run command
        cmd = [
            "dapr",
            "run",
            "--app-id",
            app_id,
            "--dapr-http-port",
            str(dapr_http_port),
            "--resources-path",
            str(resources_path),
            "--",
            python_cmd,
            str(script_path),
        ]
    else:
        # Use venv python if available, otherwise system python
        cmd = [python_cmd, str(script_path)]

    result = subprocess.run(
        cmd,
        cwd=cwd_path,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script_path} failed with return code {result.returncode}.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    return result


@pytest.fixture(scope="session", autouse=True)
def cleanup_quickstart_venvs(request):
    """
    Cleanup ephemeral test venvs after all tests complete.

    Note: Venvs are created in each quickstart directory as `ephemeral_test_venv`.
    Example: `quickstarts/01-hello-world/ephemeral_test_venv`
    These are cleaned up after tests complete, unless USE_EXISTING_VENV is set.
    """
    yield

    # Skip cleanup if using existing venv (for local dev)
    if os.getenv("USE_EXISTING_VENV", "").lower() in ("true", "1"):
        logger.info("Skipping ephemeral test venv cleanup (USE_EXISTING_VENV=true)")
        return

    # Cleanup ephemeral test venvs
    # Get quickstarts directory from project root (parent conftest fixture path)
    project_root = Path(__file__).parent.parent.parent.parent
    quickstarts_dir = project_root / "quickstarts"

    if quickstarts_dir.exists():
        logger.info("Cleaning up ephemeral test venvs...")
        for quickstart_dir in quickstarts_dir.iterdir():
            if quickstart_dir.is_dir():
                venv_path = quickstart_dir / "ephemeral_test_venv"
                if venv_path.exists():
                    logger.info(f"Removing ephemeral test venv: {venv_path}")
                    try:
                        shutil.rmtree(venv_path)
                    except OSError as e:
                        logger.warning(f"Failed to remove {venv_path}: {e}")
