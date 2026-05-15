import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path


def fetch_latest(package: str) -> str:
    """Do not bump until the packages are available already"""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["info"]["version"]


def to_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def main(packages: list[str]) -> None:
    pyproject_text = Path("pyproject.toml").read_text()
    deps = tomllib.loads(pyproject_text)["project"]["dependencies"]
    pins = dict(spec.split("==", 1) for spec in deps if "==" in spec)
    version_current = pins[packages[0]]

    versions_pypi = {pkg: fetch_latest(pkg) for pkg in packages}
    unique = set(versions_pypi.values())
    if len(unique) != 1:
        print(f"::notice::packages out of sync on PyPI: {versions_pypi}")
        return

    version_latest = unique.pop()
    if to_tuple(version_latest) <= to_tuple(version_current):
        print(f"Already at {version_current}")
        return

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a") as output:
        output.write(
            f"should_bump=true\ncurrent={version_current}\nlatest={version_latest}\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
