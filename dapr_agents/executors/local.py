from dapr_agents.executors import CodeExecutorBase
from dapr_agents.types.executor import ExecutionRequest, ExecutionResult
from typing import List, Union, Any, Callable
from pydantic import Field
from pathlib import Path
import asyncio
import venv
import logging
import hashlib
import inspect
import time
import ast

logger = logging.getLogger(__name__)

class LocalCodeExecutor(CodeExecutorBase):
    """Executes code locally in an optimized virtual environment with caching, 
    user-defined functions, and enhanced security.

    Supports Python and shell execution with real-time logging, 
    efficient dependency management, and reduced file I/O.
    """

    cache_dir: Path = Field(default_factory=lambda: Path.cwd() / ".dapr_agents_cached_envs", description="Directory for cached virtual environments and execution artifacts.")
    user_functions: List[Callable] = Field(default_factory=list, description="List of user-defined functions available during execution.")
    cleanup_threshold: int = Field(default=604800, description="Time (in seconds) before cached virtual environments are considered for cleanup.")

    _env_lock = asyncio.Lock()

    def model_post_init(self, __context: Any) -> None:
        """Ensures the cache directory is created after model initialization."""
        super().model_post_init(__context)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache directory set.")
        logger.debug(f"{self.cache_dir}")

    async def execute(self, request: Union[ExecutionRequest, dict]) -> List[ExecutionResult]:
        """Executes Python or shell code securely in a persistent virtual environment with caching and real-time logging.

        Args:
            request (Union[ExecutionRequest, dict]): The execution request containing code snippets.

        Returns:
            List[ExecutionResult]: A list of execution results for each snippet.
        """
        if isinstance(request, dict):
            request = ExecutionRequest(**request)

        self.validate_snippets(request.snippets)
        results = []
    
        for snippet in request.snippets:
            start_time = time.time()

            if snippet.language == "python":
                required_packages = self._extract_imports(snippet.code)
                logger.info(f"Packages Required: {required_packages}")
                venv_path = await self._get_or_create_cached_env(required_packages)

                # Load user-defined functions dynamically in memory
                function_code = "\n".join(inspect.getsource(f) for f in self.user_functions) if self.user_functions else ""
                exec_script = f"{function_code}\n{snippet.code}" if function_code else snippet.code

                python_executable = venv_path / "bin" / "python3"
                command = [str(python_executable), "-c", exec_script]
            else:
                command = ["sh", "-c", snippet.code]

            logger.info("Executing command")
            logger.debug(f"{' '.join(command)}")

            try:
                # Start subprocess execution with explicit timeout
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    close_fds=True
                )

                # Wait for completion with timeout enforcement
                stdout_output, stderr_output = await asyncio.wait_for(process.communicate(), timeout=request.timeout)

                status = "success" if process.returncode == 0 else "error"
                execution_time = time.time() - start_time

                logger.info(f"Execution completed in {execution_time:.2f} seconds.")
                if stderr_output:
                    logger.error(f"STDERR: {stderr_output.decode()}")

                results.append(ExecutionResult(
                    status=status,
                    output=stdout_output.decode(),
                    exit_code=process.returncode
                ))

            except asyncio.TimeoutError:
                process.terminate()  # Ensure subprocess is killed if it times out
                results.append(ExecutionResult(status="error", output="Execution timed out", exit_code=1))
            except Exception as e:
                results.append(ExecutionResult(status="error", output=str(e), exit_code=1))

        return results

    def _extract_imports(self, code: str) -> List[str]:
        """Parses a Python script and extracts top-level module imports.

        Args:
            code (str): The Python code snippet to analyze.

        Returns:
            List[str]: A list of imported module names found in the code.

        Raises:
            SyntaxError: If the code has invalid syntax and cannot be parsed.
        """
        try:
            parsed_code = ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error while parsing code: {e}")
            return []

        modules = set()
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split('.')[0])  # Get the top-level package
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split('.')[0])

        return list(modules)

    async def _get_missing_packages(self, packages: List[str], env_path: Path) -> List[str]:
        """Determines which packages are missing inside a given virtual environment.

        Args:
            packages (List[str]): A list of package names to check.
            env_path (Path): Path to the virtual environment.

        Returns:
            List[str]: A list of packages that are missing from the virtual environment.
        """
        python_bin = env_path / "bin" / "python3"

        async def check_package(pkg):
            process = await asyncio.create_subprocess_exec(
                str(python_bin), "-c", f"import {pkg}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            return pkg if process.returncode != 0 else None  # Return package name if missing

        tasks = [check_package(pkg) for pkg in packages]
        results = await asyncio.gather(*tasks)

        return [pkg for pkg in results if pkg]  # Filter out installed packages


    async def _get_or_create_cached_env(self, dependencies: List[str]) -> Path:
        """Creates or retrieves a cached virtual environment based on dependencies.

        This function checks if a suitable cached virtual environment exists.
        If it does not, it creates a new one and installs missing dependencies.

        Args:
            dependencies (List[str]): List of required package names.

        Returns:
            Path: Path to the virtual environment directory.

        Raises:
            RuntimeError: If virtual environment creation or package installation fails.
        """
        async with self._env_lock:
            env_hash = hashlib.md5(",".join(sorted(dependencies)).encode()).hexdigest()
            env_path = self.cache_dir / f"env_{env_hash}"

            if env_path.exists():
                logger.info("Reusing cached virtual environment.")
            else:
                logger.info("Setting up a new virtual environment.")
                try:
                    venv.create(str(env_path), with_pip=True)
                except Exception as e:
                    logger.error(f"Failed to create virtual environment: {e}")
                    raise RuntimeError(f"Virtual environment creation failed: {e}")

            # Identify missing packages
            missing_packages = await self._get_missing_packages(dependencies, env_path)

            if missing_packages:
                await self._install_missing_packages(missing_packages, env_path)

            return env_path


    async def _install_missing_packages(self, packages: List[str], env_dir: Path):
        """Installs missing Python packages inside the virtual environment.

        Args:
            packages (List[str]): A list of package names to install.
            env_dir (Path): Path to the virtual environment where packages should be installed.

        Raises:
            RuntimeError: If the package installation process fails.
        """
        if not packages:
            return

        python_bin = env_dir / "bin" / "python3"
        command = [str(python_bin), "-m", "pip", "install", *packages]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,  # Suppresses stdout since it's not used
            stderr=asyncio.subprocess.PIPE,
            close_fds=True
        )
        _, stderr = await process.communicate()  # Capture only stderr

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"Package installation failed: {error_msg}")
            raise RuntimeError(f"Package installation failed: {error_msg}")

        logger.info(f"Installed dependencies: {', '.join(packages)}")