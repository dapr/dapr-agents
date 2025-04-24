"""Light-weight cross-platform sandbox helpers."""

import platform
import shutil
from pathlib import Path
from typing import List, Literal, Sequence

SandboxType = Literal["none", "seatbelt", "firejail", "auto"]

_READ_ONLY_SEATBELT_POLICY = r"""
(version 1)

; ---------------- default = deny everything -----------------
(deny default)

; ---------------- read-only FS access -----------------------
(allow file-read*)

; ---------------- minimal process mgmt ----------------------
(allow process-exec)
(allow process-fork)
(allow signal (target self))

; ---------------- write-only to /dev/null -------------------
(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; ---------------- harmless sysctls --------------------------
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.version")
  (sysctl-name-prefix "hw.perflevel")
)
"""


def detect_backend() -> SandboxType:  # noqa: D401
    """Return the best-effort sandbox backend for the current host."""
    system = platform.system()
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    if system == "Linux" and shutil.which("firejail"):
        return "firejail"
    return "none"


def _seatbelt_cmd(cmd: Sequence[str], writable_paths: List[Path]) -> List[str]:
    """
    Construct a **macOS seatbelt** command line.

    The resulting list can be passed directly to `asyncio.create_subprocess_exec`.
    It launches the target *cmd* under **sandbox-exec** with an
    *initially-read-only* profile; every directory in *writable_paths* is added
    as an explicit “write-allowed sub-path”.

    Args:
        cmd:
            The *raw* command (program + args) that should run inside the sandbox.
        writable_paths:
            Absolute paths that the child process must be able to modify
            (e.g. a temporary working directory).
            Each entry becomes a param `-D WR<i>=<path>` and a corresponding
            ``file-write*`` rule in the generated profile.

    Returns:
        list[str]
            A fully-assembled ``sandbox-exec`` invocation:
            ``['sandbox-exec', '-p', <profile>, …, '--', *cmd]``.
    """
    policy = _READ_ONLY_SEATBELT_POLICY
    params: list[str] = []

    if writable_paths:
        # Build parameter substitutions and the matching `(allow file-write*)` stanza.
        write_terms: list[str] = []
        for idx, path in enumerate(writable_paths):
            param = f"WR{idx}"
            params.extend(["-D", f"{param}={path}"])
            write_terms.append(f'(subpath (param "{param}"))')

        policy += f"\n(allow file-write*\n  {' '.join(write_terms)}\n)"

    return [
        "sandbox-exec",
        "-p",
        policy,
        *params,
        "--",
        *cmd,
    ]


def _firejail_cmd(cmd: Sequence[str], writable_paths: List[Path]) -> List[str]:
    """
    Build a **Firejail** command line (Linux only).

    The wrapper enables seccomp, disables sound and networking, and whitelists
    the provided *writable_paths* so the child process can persist data there.

    Args:
        cmd:
            The command (program + args) to execute.
        writable_paths:
            Directories that must remain writable inside the Firejail sandbox.

    Returns:
        list[str]
            A Firejail-prefixed command suitable for
            ``asyncio.create_subprocess_exec``.

    Raises:
        ValueError
            If *writable_paths* contains non-absolute paths.
    """
    for p in writable_paths:
        if not p.is_absolute():
            raise ValueError(f"Firejail whitelist paths must be absolute: {p}")

    rw_flags = sum([["--whitelist", str(p)] for p in writable_paths], [])
    return [
        "firejail",
        "--quiet",  # suppress banner
        "--seccomp",  # enable seccomp filter
        "--nosound",
        "--net=none",
        *rw_flags,
        "--",
        *cmd,
    ]


def wrap_command(
    cmd: Sequence[str],
    backend: SandboxType,
    writable_paths: List[Path] | None = None,
) -> List[str]:
    """
    Produce a sandbox-wrapped command according to *backend*.

    This is the single public helper used by the executors: it hides the
    platform-specific details of **seatbelt** and **Firejail** while providing
    a graceful fallback to “no sandbox”.

    Args:
        cmd:
            The raw command (program + args) to execute.
        backend:
            One of ``'seatbelt'``, ``'firejail'``, ``'none'`` or ``'auto'``.
            When ``'auto'`` is supplied the caller should already have resolved the
            platform with :func:`detect_backend`; the value is treated as ``'none'``.
        writable_paths:
            Extra directories that must remain writable inside the sandbox.
            Ignored when *backend* is ``'none'`` / ``'auto'``.

    Returns:
        list[str]
            The command list ready for ``asyncio.create_subprocess_exec``.
            If sandboxing is disabled, this is simply ``list(cmd)``.

    Raises:
        ValueError
            If an unrecognised *backend* value is given.
    """
    if backend in ("none", "auto"):
        return list(cmd)

    writable_paths = writable_paths or []

    if backend == "seatbelt":
        return _seatbelt_cmd(cmd, writable_paths)

    if backend == "firejail":
        return _firejail_cmd(cmd, writable_paths)

    raise ValueError(f"Unknown sandbox backend: {backend!r}")
