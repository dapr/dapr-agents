from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int

    @staticmethod
    def parse(version: str) -> "Version":
        """Parse a semver-like string 'MAJOR.MINOR.PATCH' ignoring pre-release/build.

        Non-numeric or missing parts default to 0; extra parts are ignored.
        Examples: '1.16.0', '1.16', '1' -> (1,16,0)/(1,0,0).
        """
        core = version.split("-")[0].split("+")[0]
        parts = core.split(".")
        nums: List[int] = []
        for i in range(3):
            try:
                nums.append(int(parts[i]))
            except Exception:
                nums.append(0)
        return Version(nums[0], nums[1], nums[2])


def _parse_constraint_token(token: str) -> Tuple[str, Version]:
    token = token.strip()
    ops = ["<=", ">=", "==", "!=", "<", ">"]
    for op in ops:
        if token.startswith(op):
            return op, Version.parse(token[len(op) :].strip())
    # default to == if no operator present
    return "==", Version.parse(token)


def _satisfies(version: Version, op: str, bound: Version) -> bool:
    if op == "==":
        return version == bound
    if op == "!=":
        return version != bound
    if op == ">":
        return version > bound
    if op == ">=":
        return version >= bound
    if op == "<":
        return version < bound
    if op == "<=":
        return version <= bound
    raise ValueError(f"Unknown operator: {op}")


def is_version_supported(version: str, constraints: str) -> bool:
    """Return True if the given version satisfies the constraints.

    Constraints syntax:
      - Comma-separated items are ANDed: ">=1.16.0, <2.0.0"
      - Use '||' for OR groups: ">=1.16.0, <2.0.0 || ==0.0.0"
      - Each token supports operators: ==, !=, >=, <=, >, <
      - Missing operator defaults to ==
    """
    if version == "edge":
        return True
    v = Version.parse(version)
    for group in constraints.split("||"):
        group = group.strip()
        if not group:
            continue
        tokens = [t for t in (tok.strip() for tok in group.split(",")) if t]
        if not tokens:
            continue
        if all(_satisfies(v, *_parse_constraint_token(tok)) for tok in tokens):
            return True
    return False
