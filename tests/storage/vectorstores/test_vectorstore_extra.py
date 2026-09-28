#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Checks that `pip install dapr-agents[vectorstore]` covers the vector stores.

The vector-store modules import their backends lazily (inside a function or a
``try/except ImportError`` block), so a missing entry in the ``vectorstore``
extra is only noticed when a user constructs the store. These tests read the
modules' source and ``pyproject.toml``/``uv.lock`` from disk; they need no
network and do not import the optional backends.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, Iterator, List, Set

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
VECTORSTORES_DIR: Path = REPO_ROOT / "dapr_agents" / "storage" / "vectorstores"
EXTRA_NAME: str = "vectorstore"
# PEP 508: the distribution name is the leading run of letters, digits, ".-_".
REQUIREMENT_NAME: re.Pattern[str] = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Top-level module name -> distribution name on PyPI. A new lazily imported
# module must be added here, which makes the author decide which package
# provides it.
MODULE_TO_DISTRIBUTION: Dict[str, str] = {
    "chromadb": "chromadb",
    "redisvl": "redisvl",
    "redis": "redis",
    "psycopg": "psycopg",
    "psycopg_pool": "psycopg-pool",
    "pgvector": "pgvector",
}

# Modules deliberately left out of the extra. The Postgres store tells users to
# install its driver separately (`pip install 'psycopg[binary,pool]' pgvector`).
NOT_IN_EXTRA: Set[str] = {"psycopg", "psycopg_pool", "pgvector"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        names: List[ast.expr] = []
        if isinstance(handler.type, ast.Tuple):
            names = list(handler.type.elts)
        elif handler.type is not None:
            names = [handler.type]
        for name in names:
            if isinstance(name, ast.Name) and name.id in (
                "ImportError",
                "ModuleNotFoundError",
            ):
                return True
    return False


def _lazy_import_roots(tree: ast.AST) -> Iterator[str]:
    """Yield top-level module names imported inside a function or guarded try."""

    def visit(node: ast.AST, lazy: bool) -> Iterator[str]:
        if lazy and isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif lazy and isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]
        for child in ast.iter_child_nodes(node):
            child_lazy = lazy or isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            if isinstance(node, ast.Try) and child in node.body:
                child_lazy = child_lazy or _catches_import_error(node)
            yield from visit(child, child_lazy)

    yield from visit(tree, False)


def _lazy_third_party_modules() -> Set[str]:
    modules: Set[str] = set()
    for path in sorted(VECTORSTORES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules.update(_lazy_import_roots(tree))
    return {
        m for m in modules if m not in sys.stdlib_module_names and m != "dapr_agents"
    }


def _extra_requirements() -> Set[str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    entries: List[str] = pyproject["project"]["optional-dependencies"][EXTRA_NAME]
    names: Set[str] = set()
    for entry in entries:
        match = REQUIREMENT_NAME.match(entry)
        assert match, f"Cannot parse requirement {entry!r}"
        names.add(_normalize(match.group(1)))
    return names


def _locked_dependency_graph() -> Dict[str, Set[str]]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text("utf-8"))
    graph: Dict[str, Set[str]] = {}
    for package in lock.get("package", []):
        deps = {_normalize(dep["name"]) for dep in package.get("dependencies", [])}
        graph.setdefault(_normalize(package["name"]), set()).update(deps)
    return graph


def _closure(roots: Set[str], graph: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack: List[str] = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph.get(name, set()))
    return seen


def test_lazy_imports_are_found() -> None:
    modules = _lazy_third_party_modules()
    assert {"chromadb", "redisvl", "redis"} <= modules


def test_every_lazy_import_has_a_known_distribution() -> None:
    unknown = _lazy_third_party_modules() - MODULE_TO_DISTRIBUTION.keys()
    assert not unknown, (
        f"Add {sorted(unknown)} to MODULE_TO_DISTRIBUTION and to the "
        f"'{EXTRA_NAME}' extra in pyproject.toml (or to NOT_IN_EXTRA with a reason)."
    )


@pytest.mark.parametrize("module", ["redisvl", "chromadb"])
def test_extra_lists_backend_directly(module: str) -> None:
    assert _normalize(MODULE_TO_DISTRIBUTION[module]) in _extra_requirements()


def test_extra_installs_every_lazily_imported_backend() -> None:
    installed = _closure(_extra_requirements(), _locked_dependency_graph())
    missing = sorted(
        module
        for module in _lazy_third_party_modules() - NOT_IN_EXTRA
        if _normalize(MODULE_TO_DISTRIBUTION.get(module, module)) not in installed
    )
    assert not missing, (
        f"The '{EXTRA_NAME}' extra does not install {missing}, which "
        f"dapr_agents/storage/vectorstores imports lazily."
    )
