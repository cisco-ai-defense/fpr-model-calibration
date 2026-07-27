# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for repository release-policy enforcement."""

from pathlib import Path

import pytest

from scripts.check_release_policy import (
    PolicyError,
    parse_changelog,
    parse_release_impact,
    validate_main_ancestry,
    validate_pull_request_state,
    validate_repository,
)

ROOT = Path(__file__).resolve().parents[1]


def _changelog(version: str = "0.1.0") -> str:
    return f"""# Changelog

## [Unreleased]

## [{version}] - 2026-07-08

### Added

- Initial public behavior.
"""


def _write_release_files(root: Path, *, version: str = "0.1.0") -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "fprcal"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(_changelog(version), encoding="utf-8")


def test_repository_release_metadata_is_valid() -> None:
    """The checked-in version and changelog must satisfy the policy."""
    validate_repository(ROOT)


@pytest.mark.parametrize("impact", ["patch", "minor", "breaking", "none", "release"])
def test_parse_release_impact_accepts_each_option(impact: str) -> None:
    """Each documented release-impact option is recognized."""
    body = f"- [x] `{impact}`: selected\n"
    assert parse_release_impact(body) == impact


def test_parse_release_impact_rejects_multiple_options() -> None:
    """A pull request cannot claim conflicting release impacts."""
    with pytest.raises(PolicyError, match="exactly one"):
        parse_release_impact("- [x] `patch`\n- [x] `none`\n")


def test_pull_request_template_matches_release_impact_parser() -> None:
    """The checked-in template produces the value enforced by CI."""
    template_path = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
    body = template_path.read_text(encoding="utf-8").replace("- [ ] `none`:", "- [x] `none`:")
    assert parse_release_impact(body) == "none"


def test_validate_repository_rejects_version_mismatch(tmp_path: Path) -> None:
    """The package version must identify the newest finalized release."""
    _write_release_files(tmp_path, version="0.2.0")
    (tmp_path / "CHANGELOG.md").write_text(_changelog("0.1.0"), encoding="utf-8")

    with pytest.raises(PolicyError, match="must match"):
        validate_repository(tmp_path)


def test_validate_repository_rejects_release_tag_mismatch(tmp_path: Path) -> None:
    """A GitHub release tag must identify the built package version."""
    _write_release_files(tmp_path)

    with pytest.raises(PolicyError, match="does not match"):
        validate_repository(tmp_path, release_tag="v0.2.0")


def test_parse_changelog_rejects_empty_release() -> None:
    """A finalized release cannot omit installed-package changes."""
    changelog = """# Changelog

## [Unreleased]

## [0.1.0] - 2026-07-08
"""
    with pytest.raises(PolicyError, match="public change entry"):
        parse_changelog(changelog)


def test_current_commit_is_eligible_main_ancestry() -> None:
    """The ancestry check accepts a release commit contained in its target ref."""
    validate_main_ancestry(ROOT, "HEAD")


def test_main_ancestry_rejects_invalid_ref() -> None:
    """The ancestry check rejects ref text outside the safe Git syntax subset."""
    with pytest.raises(PolicyError, match="Invalid main ref"):
        validate_main_ancestry(ROOT, "HEAD;false")


def test_behavior_change_requires_unreleased_entry() -> None:
    """Bug and feature pull requests must change the unreleased section."""
    with pytest.raises(PolicyError, match=r"update CHANGELOG\.md"):
        validate_pull_request_state(
            impact="patch",
            base_version="0.1.0",
            current_version="0.1.0",
            changed_files={"src/fprcal/calibration.py"},
            base_unreleased="",
            current_unreleased="",
        )


def test_no_release_change_rejects_version_bump() -> None:
    """Maintenance changes cannot consume a public package version."""
    with pytest.raises(PolicyError, match="must not bump"):
        validate_pull_request_state(
            impact="none",
            base_version="0.1.0",
            current_version="0.1.1",
            changed_files={"README.md"},
            base_unreleased="",
            current_unreleased="",
        )


def test_release_requires_existing_unreleased_change() -> None:
    """A version bump cannot manufacture an empty release."""
    with pytest.raises(PolicyError, match="existing unreleased"):
        validate_pull_request_state(
            impact="release",
            base_version="0.1.0",
            current_version="0.1.1",
            changed_files={"CHANGELOG.md", "pyproject.toml", "uv.lock"},
            base_unreleased="",
            current_unreleased="",
        )


def test_release_accepts_increasing_version_with_existing_change() -> None:
    """A release may finalize an accumulated public change."""
    validate_pull_request_state(
        impact="release",
        base_version="0.1.0",
        current_version="0.2.0",
        changed_files={"CHANGELOG.md", "pyproject.toml", "uv.lock"},
        base_unreleased="### Added\n\n- Add a public capability.",
        current_unreleased="",
    )
