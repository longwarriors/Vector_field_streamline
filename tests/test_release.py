"""Release metadata stays consistent across source, lockfile, and documentation."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from vectorviz import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_declared_versions_match_runtime_version() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(
        package for package in lockfile["package"] if package["name"] == "vector-field-streamline"
    )

    assert pyproject["project"]["version"] == __version__
    assert locked_project["version"] == __version__


def test_changelog_latest_release_matches_runtime_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    releases = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.MULTILINE)

    assert releases
    assert releases[0] == __version__


def test_documented_health_version_matches_runtime_version() -> None:
    api_reference = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    documented_versions = re.findall(r'"version": "(\d+\.\d+\.\d+)"', api_reference)

    assert documented_versions == [__version__]
