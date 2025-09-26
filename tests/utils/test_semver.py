import pytest
from pathlib import Path
from importlib.machinery import SourceFileLoader

_SEMVER_PATH = (
    Path(__file__).resolve().parents[2] / "dapr_agents" / "utils" / "semver.py"
)
semver = SourceFileLoader("_semver_test_module", str(_SEMVER_PATH)).load_module()
is_version_supported = semver.is_version_supported
Version = semver.Version


@pytest.mark.parametrize(
    "version,constraint,expected",
    [
        ("1.2.3", "==1.2.3", True),
        ("1.2.3", "!=1.2.3", False),
        ("1.2.3", ">1.2.2", True),
        ("1.2.3", ">=1.2.3", True),
        ("1.2.3", "<1.2.4", True),
        ("1.2.3", "<=1.2.3", True),
        # implicit equals
        ("1.2.3", "1.2.3", True),
        ("1.2.3", "1.2.4", False),
    ],
)
def test_basic_operators(version: str, constraint: str, expected: bool) -> None:
    assert is_version_supported(version, constraint) is expected


def test_and_constraints() -> None:
    assert is_version_supported("1.16.1", ">=1.16.0, <2.0.0") is True
    assert is_version_supported("2.0.0", ">=1.16.0, <2.0.0") is False


def test_or_constraints() -> None:
    constraints = ">=2.0.0, <3.0.0 || ==1.16.1"
    assert is_version_supported("1.16.1", constraints) is True
    assert is_version_supported("2.1.0", constraints) is True
    assert is_version_supported("1.16.0", constraints) is False


def test_whitespace_and_formatting() -> None:
    assert is_version_supported("1.16.1", "  >=1.16.0 ,   <2.0.0  ") is True


def test_prerelease_ignored_for_core_comparison() -> None:
    # Our parser ignores pre-release/build and compares core MAJOR.MINOR.PATCH
    # So 1.16.0-rc.1 is treated as 1.16.0
    assert is_version_supported("1.16.0-rc.1", ">=1.16.0, <2.0.0") is True
    assert is_version_supported("1.16.0-rc.1+build.5", "==1.16.0") is True


def test_version_parse_defaults_missing_parts_to_zero() -> None:
    assert Version.parse("1").major == 1
    assert Version.parse("1").minor == 0
    assert Version.parse("1").patch == 0
    assert Version.parse("1.2").major == 1
    assert Version.parse("1.2").minor == 2
    assert Version.parse("1.2").patch == 0


def test_edge_version_always_satisfies_constraints() -> None:
    """Edge version should always satisfy any constraint."""
    assert is_version_supported("edge", "==1.2.3") is True
    assert is_version_supported("edge", "!=1.2.3") is True
    assert is_version_supported("edge", ">1.2.2") is True
    assert is_version_supported("edge", ">=1.2.3") is True
    assert is_version_supported("edge", "<1.2.4") is True
    assert is_version_supported("edge", "<=1.2.3") is True
    assert is_version_supported("edge", ">=1.16.0, <2.0.0") is True
    assert is_version_supported("edge", ">=2.0.0, <3.0.0 || ==1.16.1") is True
    assert is_version_supported("edge", "==0.0.0") is True
    assert is_version_supported("edge", ">999.999.999") is True
    assert is_version_supported("edge", "<0.0.1") is True


def test_edge_version_case_sensitivity() -> None:
    """Edge version should be case sensitive."""
    assert is_version_supported("edge", "==1.2.3") is True
    assert is_version_supported("Edge", "==1.2.3") is False
    assert is_version_supported("EDGE", "==1.2.3") is False
