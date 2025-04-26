"""Local executor that runs Python or shell snippets in cached virtual-envs."""

import asyncio
import ast
import hashlib
import inspect
import logging
import time
import venv
from pathlib import Path
from typing import Any, Callable, List, Sequence, Union

from pydantic import Field, PrivateAttr

from dapr_agents.executors import CodeExecutorBase
from dapr_agents.executors.sandbox import detect_backend, wrap_command, SandboxType
from dapr_agents.executors.utils.package_manager import (
    get_install_command,
    get_project_type,
)
from dapr_agents.types.executor import ExecutionRequest, ExecutionResult

logger = logging.getLogger(__name__)


class LocalCodeExecutor(CodeExecutorBase):
    """
    Run snippets locally with **optional OS-level sandboxing** and
    per-snippet virtual-env caching.
    """

    cache_dir: Path = Field(
        default_factory=lambda: Path.cwd() / ".dapr_agents_cached_envs",
        description="Directory that stores cached virtual environments.",
    )
    user_functions: List[Callable] = Field(
        default_factory=list,
        description="Functions whose source is prepended to every Python snippet.",
    )
    sandbox: SandboxType = Field(
        default="auto",
        description="'seatbelt' | 'firejail' | 'none' | 'auto' (best available)",
    )
    writable_paths: List[Path] = Field(
        default_factory=list,
        description="Extra paths the sandboxed process may write to.",
    )
    cleanup_threshold: int = Field(
        default=604_800,  # one week
        description="Seconds before a cached venv is considered stale.",
    )

    _env_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _bootstrapped_root: Path | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        """Create ``cache_dir`` after pydantic instantiation."""
        super().model_post_init(__context)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("venv cache directory: %s", self.cache_dir)

    async def execute(
        self, request: Union[ExecutionRequest, dict]
    ) -> List[ExecutionResult]:
        """
        Run the snippets in *request* and return their results.

        Args:
            request: ``ExecutionRequest`` instance or a raw mapping that can
                be unpacked into one.

        Returns:
            A list with one ``ExecutionResult`` for every snippet in the
            original request.
        """
        if isinstance(request, dict):
            request = ExecutionRequest(**request)

        await self._bootstrap_project()
        self.validate_snippets(request.snippets)

        #  Resolve sandbox once
        eff_backend: SandboxType = (
            detect_backend() if self.sandbox == "auto" else self.sandbox
        )
        if eff_backend != "none":
            logger.info(
                "Sandbox backend enabled: %s%s",
                eff_backend,
                f" (writable: {', '.join(map(str, self.writable_paths))})"
                if self.writable_paths
                else "",
            )
        else:
            logger.info("Sandbox disabled - running commands directly.")

        # Main loop
        results: list[ExecutionResult] = []
        for snip_idx, snippet in enumerate(request.snippets, start=1):
            start = time.perf_counter()

            # Assemble the *raw* command
            if snippet.language == "python":
                env = await self._prepare_python_env(snippet.code)
                python_bin = env / "bin" / "python3"
                prelude = "\n".join(inspect.getsource(fn) for fn in self.user_functions)
                script = f"{prelude}\n{snippet.code}" if prelude else snippet.code
                raw_cmd: Sequence[str] = [str(python_bin), "-c", script]
            else:
                raw_cmd = ["sh", "-c", snippet.code]

            # Wrap for sandbox
            final_cmd = wrap_command(raw_cmd, eff_backend, self.writable_paths)
            logger.debug(
                "Snippet %s - launch command: %s",
                snip_idx,
                " ".join(final_cmd),
            )

            # Run it
            snip_timeout = getattr(snippet, "timeout", request.timeout)
            results.append(await self._run_subprocess(final_cmd, snip_timeout))

            logger.info(
                "Snippet %s finished in %.3fs",
                snip_idx,
                time.perf_counter() - start,
            )

        return results

    async def _bootstrap_project(self) -> None:
        """Install top-level dependencies once per executor instance."""
        cwd = Path.cwd().resolve()
        if self._bootstrapped_root == cwd:
            return

        install_cmd = get_install_command(str(cwd))
        if install_cmd:
            logger.info(
                "bootstrapping %s project with '%s'",
                get_project_type(str(cwd)).value,
                install_cmd,
            )

            proc = await asyncio.create_subprocess_shell(
                install_cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
            if proc.returncode:
                logger.warning(
                    "bootstrap failed (%d): %s", proc.returncode, err.decode().strip()
                )

        self._bootstrapped_root = cwd

    async def _prepare_python_env(self, code: str) -> Path:
        """
        Ensure a virtual-env exists that satisfies *code* imports.

        Args:
            code: User-supplied Python source.

        Returns:
            Path to the virtual-env directory.
        """
        imports = self._extract_imports(code)
        env = await self._get_or_create_cached_env(imports)
        missing = await self._get_missing_packages(imports, env)
        if missing:
            await self._install_missing_packages(missing, env)
        return env

    @staticmethod
    def _extract_imports(code: str) -> List[str]:
        """
        Return all top-level imported module names in *code*.

        Args:
            code: Python source to scan.

        Returns:
            Unique list of first-segment module names.

        Raises:
            SyntaxError: If *code* cannot be parsed.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.error("cannot parse user code, assuming no imports")
            return []

        names = {
            alias.name.partition(".")[0]
            for node in ast.walk(tree)
            for alias in getattr(node, "names", [])
            if isinstance(node, (ast.Import, ast.ImportFrom))
        }
        if any(
            isinstance(node, ast.ImportFrom) and node.module for node in ast.walk(tree)
        ):
            names |= {
                node.module.partition(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
        return sorted(names)

    async def _get_missing_packages(
        self, packages: List[str], env_path: Path
    ) -> List[str]:
        """
        Identify which *packages* are not importable from *env_path*.

        Args:
            packages: Candidate import names.
            env_path: Path to the virtual-env.

        Returns:
            Subset of *packages* that need installation.
        """
        python = env_path / "bin" / "python3"

        async def probe(pkg: str) -> str | None:
            proc = await asyncio.create_subprocess_exec(
                str(python),
                "- <<PY\nimport importlib.util, sys;"
                f"sys.exit(importlib.util.find_spec('{pkg}') is None)\nPY",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return pkg if proc.returncode else None

        missing = await asyncio.gather(*(probe(p) for p in packages))
        return [m for m in missing if m]

    async def _get_or_create_cached_env(self, deps: List[str]) -> Path:
        """
        Return a cached venv path keyed by the sorted list *deps*.

        Args:
            deps: Import names required by user code.

        Returns:
            Path to the virtual-env directory.

        Raises:
            RuntimeError: If venv creation fails.
        """
        digest = hashlib.sha1(",".join(sorted(deps)).encode()).hexdigest()
        env_path = self.cache_dir / f"env_{digest}"

        async with self._env_lock:
            if env_path.exists():
                logger.info("Reusing cached virtual environment.")
            else:
                try:
                    venv.create(env_path, with_pip=True)
                    logger.info("Created a new virtual environment")
                    logger.debug("venv %s created", env_path)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("virtual-env creation failed") from exc
        return env_path

    async def _install_missing_packages(
        self, packages: List[str], env_dir: Path
    ) -> None:
        """
        ``pip install`` *packages* inside *env_dir*.

        Args:
            packages: Package names to install.
            env_dir: Target virtual-env directory.

        Raises:
            RuntimeError: If installation returns non-zero exit code.
        """
        python = env_dir / "bin" / "python3"
        cmd = [str(python), "-m", "pip", "install", *packages]
        logger.info("Installing %s", ", ".join(packages))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            msg = err.decode().strip()
            logger.error("pip install failed: %s", msg)
            raise RuntimeError(msg)
        logger.debug("Installed %d package(s)", len(packages))

    async def _run_subprocess(
        self, cmd: Sequence[str], timeout: int
    ) -> ExecutionResult:
        """
        Run *cmd* with *timeout* seconds.

        Args:
            cmd: Command list to execute.
            timeout: Maximum runtime in seconds.

        Returns:
            ``ExecutionResult`` with captured output.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
            status = "success" if proc.returncode == 0 else "error"
            if err:
                logger.debug("stderr: %s", err.decode().strip())
            return ExecutionResult(
                status=status, output=out.decode(), exit_code=proc.returncode
            )

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecutionResult(
                status="error", output="execution timed out", exit_code=1
            )

        except Exception as exc:
            return ExecutionResult(status="error", output=str(exc), exit_code=1)
