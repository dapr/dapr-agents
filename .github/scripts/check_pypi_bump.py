# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version


def fetch_latest(package: str) -> str:
    """Highest non-yanked version on PyPI"""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        releases = json.load(response)["releases"]
    parsed: list[tuple[Version, str]] = []
    for version, files in releases.items():
        if not files or all(file.get("yanked") for file in files):
            continue
        try:
            parsed.append((Version(version), version))
        except InvalidVersion:
            continue
    return max(parsed)[1]


def main(packages: list[str]) -> None:
    pyproject_text = Path("pyproject.toml").read_text()
    deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
    pins = dict(spec.split("==", 1) for spec in deps if "==" in spec)

    missing = [pkg for pkg in packages if pkg not in pins]
    if missing:
        print(f"::error::missing '==' pins in pyproject.toml for: {missing}")
        sys.exit(1)

    versions_current = {pkg: pins[pkg] for pkg in packages}
    if len({Version(v) for v in versions_current.values()}) != 1:
        print(
            f"::notice::current pins out of sync in pyproject.toml: {versions_current}"
        )
        return
    version_current = next(iter(versions_current.values()))

    versions_pypi = {pkg: fetch_latest(pkg) for pkg in packages}
    if len({Version(v) for v in versions_pypi.values()}) != 1:
        print(f"::notice::packages out of sync on PyPI: {versions_pypi}")
        return
    version_latest = next(iter(versions_pypi.values()))

    if Version(version_latest) <= Version(version_current):
        print(f"Already at {version_current}")
        return

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a") as output:
        output.write(
            f"should_bump=true\ncurrent={version_current}\nlatest={version_latest}\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])