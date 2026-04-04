"""Smoke tests for the project bootstrap."""

from skating_aerial_alignment import __version__


def test_package_version_is_defined() -> None:
    """The package exposes a semantic version string."""

    assert __version__ == "0.1.0"
