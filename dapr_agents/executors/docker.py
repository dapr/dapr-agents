from dapr_agents.types.executor import ExecutionRequest, ExecutionResult
from typing import List, Any, Optional, Union, Literal
from dapr_agents.executors import CodeExecutorBase
from pydantic import Field
import tempfile
import logging
import asyncio
import shutil
import ast
import os

logger = logging.getLogger(__name__)


class DockerCodeExecutor(CodeExecutorBase):
    """Executes code securely inside a persistent Docker container with dynamic volume updates."""

    image: Optional[str] = Field(
        "python:3.9", description="Docker image used for execution."
    )
    container_name: Optional[str] = Field(
        "dapr_agents_code_executor", description="Name of the Docker container."
    )
    disable_network_access: bool = Field(
        default=True, description="Disable network access inside the container."
    )
    execution_timeout: int = Field(
        default=60, description="Max execution time (seconds)."
    )
    execution_mode: str = Field(
        "detached", description="Execution mode: 'interactive' or 'detached'."
    )
    restart_policy: str = Field(
        "no", description="Container restart policy: 'no', 'on-failure', 'always'."
    )
    max_memory: str = Field("500m", description="Max memory for execution.")
    cpu_quota: int = Field(50000, description="CPU quota limit.")
    runtime: Optional[str] = Field(
        default=None, description="Container runtime (e.g., 'nvidia')."
    )
    auto_remove: bool = Field(
        default=False, description="Keep container running to reuse it."
    )
    auto_cleanup: bool = Field(
        default=False,
        description="Automatically clean up the workspace after execution.",
    )
    volume_access_mode: Literal["ro", "rw"] = Field(
        default="ro", description="Access mode for the workspace volume."
    )
    host_workspace: Optional[str] = Field(
        default=None,
        description="Custom workspace on host. If None, defaults to system temp dir.",
    )

    docker_client: Optional[Any] = Field(
        default=None, init=False, description="Docker client instance."
    )
    execution_container: Optional[Any] = Field(
        default=None, init=False, description="Persistent Docker container."
    )
    container_workspace: Optional[str] = Field(
        default="/workspace", init=False, description="Mounted workspace in container."
    )

    def model_post_init(self, __context: Any) -> None:
        """Initializes the Docker client and ensures a reusable execution container is ready."""
        try:
            from docker import DockerClient
            from docker.errors import DockerException
        except ImportError as e:
            raise ImportError(
                "Install 'docker' package with 'pip install docker'."
            ) from e

        try:
            self.docker_client: DockerClient = DockerClient.from_env()
        except DockerException as e:
            raise RuntimeError("Docker not running or unreachable.") from e

        # Validate or Set the Host Workspace
        if self.host_workspace:
            self.host_workspace = os.path.abspath(
                self.host_workspace
            )  # Ensure absolute path
        else:
            self.host_workspace = os.path.join(
                tempfile.gettempdir(), "dapr_agents_executor_workspace"
            )

        # Ensure the directory exists
        os.makedirs(self.host_workspace, exist_ok=True)

        # Log the workspace path for visibility
        logger.info(f"Using host workspace: {self.host_workspace}")

        self.ensure_container()

        super().model_post_init(__context)

    def ensure_container(self) -> None:
        """Ensures that the execution container exists. If not, it creates and starts one."""
        try:
            from docker.errors import NotFound
        except ImportError as e:
            raise ImportError(
                "Install 'docker' package with 'pip install docker'."
            ) from e

        try:
            self.execution_container = self.docker_client.containers.get(
                self.container_name
            )
            logger.info(f"Reusing existing container: {self.container_name}")
        except NotFound:
            logger.info(f"Creating a new container: {self.container_name}")
            self.create_container()
            self.execution_container.start()
            logger.info(f"Started container: {self.container_name}")

    def create_container(self) -> None:
        """Creates a reusable Docker container."""
        try:
            from docker.errors import DockerException, APIError
        except ImportError as e:
            raise ImportError(
                "Install 'docker' package with 'pip install docker'."
            ) from e
        try:
            self.execution_container = self.docker_client.containers.create(
                self.image,
                name=self.container_name,
                command="/bin/sh -c 'while true; do sleep 30; done'",
                detach=True,
                stdin_open=True,
                tty=(self.execution_mode == "interactive"),
                auto_remove=False,
                network_disabled=self.disable_network_access,
                mem_limit=self.max_memory,
                cpu_quota=self.cpu_quota,
                security_opt=["no-new-privileges"],
                restart_policy={"Name": self.restart_policy},
                runtime=self.runtime,
                working_dir=self.container_workspace,
                volumes={
                    self.host_workspace: {
                        "bind": self.container_workspace,
                        "mode": self.volume_access_mode,
                    }
                },
            )
        except (DockerException, APIError) as e:
            logger.error(f"Failed to create the execution container: {str(e)}")
            raise RuntimeError(
                f"Failed to create the execution container: {str(e)}"
            ) from e

    async def execute(
        self, request: Union[ExecutionRequest, dict]
    ) -> List[ExecutionResult]:
        """
        Executes code inside the persistent Docker container.
        The code is written to a shared volume instead of stopping & starting the container.

        Args:
            request (Union[ExecutionRequest, dict]): The execution request containing code snippets.

        Returns:
            List[ExecutionResult]: A list of execution results.
        """
        if isinstance(request, dict):
            request = ExecutionRequest(**request)

        self.validate_snippets(request.snippets)
        results = []

        try:
            for snippet in request.snippets:
                if snippet.language == "python":
                    required_packages = self._extract_imports(snippet.code)
                    if required_packages:
                        logger.info(
                            f"Installing missing dependencies: {required_packages}"
                        )
                        await self._install_missing_packages(required_packages)

                script_filename = f"script.{snippet.language}"
                script_path_host = os.path.join(self.host_workspace, script_filename)
                script_path_container = f"{self.container_workspace}/{script_filename}"

                # Write the script dynamically
                with open(script_path_host, "w", encoding="utf-8") as script_file:
                    script_file.write(snippet.code)

                cmd = (
                    f"timeout {self.execution_timeout} python3 {script_path_container}"
                    if snippet.language == "python"
                    else f"timeout {self.execution_timeout} sh {script_path_container}"
                )

                # Run command dynamically inside the running container
                exec_result = await asyncio.to_thread(
                    self.execution_container.exec_run, cmd
                )

                exit_code = exec_result.exit_code
                logs = exec_result.output.decode("utf-8", errors="ignore").strip()
                status = "success" if exit_code == 0 else "error"

                results.append(
                    ExecutionResult(status=status, output=logs, exit_code=exit_code)
                )

        except Exception as e:
            logs = self.get_container_logs()
            logger.error(f"Execution error: {str(e)}\nContainer logs:\n{logs}")
            results.append(ExecutionResult(status="error", output=str(e), exit_code=1))

        finally:
            if self.auto_cleanup:
                if os.path.exists(self.host_workspace):
                    shutil.rmtree(self.host_workspace, ignore_errors=True)
                    logger.info(
                        f"Temporary workspace {self.host_workspace} cleaned up."
                    )

            if self.auto_remove:
                self.execution_container.stop()
                logger.info(f"Container {self.execution_container.id} stopped.")

        return results

    def _extract_imports(self, code: str) -> List[str]:
        """
        Parses a Python script and extracts the top-level imported modules.

        Args:
            code (str): The Python code to analyze.

        Returns:
            List[str]: A list of unique top-level module names imported in the script.

        Raises:
            SyntaxError: If the provided code is not valid Python, an error is logged,
                        and an empty list is returned.
        """
        try:
            parsed_code = ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return []

        modules = set()
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])

        return list(modules)

    async def _install_missing_packages(self, packages: List[str]) -> None:
        """
        Installs missing Python dependencies inside the execution container.

        Args:
            packages (List[str]): A list of package names to install.

        Raises:
            RuntimeError: If the package installation fails.
        """
        if not packages:
            return

        command = f"python3 -m pip install {' '.join(packages)}"
        result = await asyncio.to_thread(self.execution_container.exec_run, command)

        if result.exit_code != 0:
            error_msg = result.output.decode().strip()
            logger.error(f"Dependency installation failed: {error_msg}")
            raise RuntimeError(f"Dependency installation failed: {error_msg}")

        logger.info(f"Dependencies installed: {', '.join(packages)}")

    def get_container_logs(self) -> str:
        """
        Retrieves and returns the logs from the execution container.

        Returns:
            str: The container logs as a string.

        Raises:
            Exception: If log retrieval fails, an error message is logged.
        """
        try:
            logs = self.execution_container.logs(stdout=True, stderr=True).decode(
                "utf-8"
            )
            return logs
        except Exception as e:
            logger.error(f"Failed to retrieve container logs: {str(e)}")
            return ""
