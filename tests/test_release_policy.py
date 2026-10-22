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

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import check_release_policy
from scripts.check_release_policy import (
    PolicyError,
    parse_changelog,
    parse_release_impact,
    validate_main_ancestry,
    validate_pull_request,
    validate_pull_request_state,
    validate_repository,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "## [0.1.0] - 2026-07-08\n\n### Added\n\n- Initial public behavior.\n"
ENTRY = "### Fixed\n\n- Fix a public behavior."
PASSED = "Release policy checks passed."


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


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "-c",
            "user.name=Release Policy Test",
            "-c",
            "user.email=release-policy@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _event(root: Path, body: str | None) -> Path:
    path = root / "event.json"
    path.write_text(json.dumps({"pull_request": {"body": body}}), encoding="utf-8")
    return path


def _validate_state(
    impact: str,
    *,
    base: str = "0.1.0",
    current: str = "0.1.0",
    changed: tuple[str, ...] = (),
    base_unreleased: str = "",
    current_unreleased: str = "",
) -> None:
    validate_pull_request_state(
        impact=impact,
        base_version=base,
        current_version=current,
        changed_files=set(changed),
        base_unreleased=base_unreleased,
        current_unreleased=current_unreleased,
    )


@pytest.fixture
def release_repo(tmp_path: Path) -> tuple[Path, str]:
    """A two-commit repository whose second commit changes only the README."""
    _write_release_files(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "Release metadata")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "README.md").write_text("Documentation\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "Documentation")
    return tmp_path, base_sha


def test_repository_release_metadata_is_valid() -> None:
    """The checked-in version and changelog must satisfy the policy."""
    validate_repository(ROOT)


@pytest.mark.parametrize("impact", ["patch", "minor", "breaking", "none", "release"])
def test_parse_release_impact_accepts_each_option(impact: str) -> None:
    """Each documented release-impact option is recognized."""
    body = f"- [x] `{impact}`: selected\n"
    assert parse_release_impact(body) == impact


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("- [x] `patch`\n- [x] `none`\n", id="two-checked"),
        pytest.param("- [ ] `patch`\n- [ ] `none`\n", id="none-checked"),
    ],
)
def test_parse_release_impact_requires_exactly_one_option(body: str) -> None:
    """A pull request must claim exactly one release impact."""
    with pytest.raises(PolicyError, match="exactly one"):
        parse_release_impact(body)


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


@pytest.mark.parametrize(
    ("version_toml", "match"),
    [
        pytest.param('"0.1"', "final X.Y.Z", id="two-part"),
        pytest.param('"0.1.0rc1"', "final X.Y.Z", id="pre-release"),
        pytest.param("1", "must be a string", id="not-a-string"),
    ],
)
def test_validate_repository_rejects_malformed_version(
    tmp_path: Path, version_toml: str, match: str
) -> None:
    """project.version must be a final three-part version string."""
    _write_release_files(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "fprcal"\nversion = {version_toml}\n', encoding="utf-8"
    )

    with pytest.raises(PolicyError, match=match):
        validate_repository(tmp_path)


@pytest.mark.parametrize(
    ("changelog", "match"),
    [
        pytest.param("# Changelog\n", r"must contain an \[Unreleased\]", id="no-unreleased"),
        pytest.param(RELEASE + "\n## [Unreleased]\n", "must be the first", id="unreleased-second"),
        pytest.param("## [Unreleased]\n\n## [Unreleased]\n\n" + RELEASE, "exactly one", id="two"),
        pytest.param("## [Unreleased]\n", "at least one finalized release", id="no-release"),
        pytest.param(
            "## [Unreleased]\n\n## [0.1.0]\n\n### Added\n\n- Change.\n",
            "Release heading",
            id="heading-without-date",
        ),
        pytest.param(
            "## [Unreleased]\n\n### Misc\n\n- Change.\n\n" + RELEASE,
            "unsupported changelog category",
            id="unknown-category",
        ),
        pytest.param(
            "## [Unreleased]\n\n- Change.\n\n" + RELEASE,
            "outside an allowed category",
            id="entry-without-category",
        ),
        pytest.param(
            "## [Unreleased]\n\n## [0.1.0] - 2026-07-08\n",
            "public change entry",
            id="empty-release",
        ),
        pytest.param(
            "## [Unreleased]\n\n" + RELEASE + "\n" + RELEASE,
            "duplicate release version",
            id="duplicate-version",
        ),
        pytest.param(
            "## [Unreleased]\n\n" + RELEASE + "\n" + RELEASE.replace("0.1.0", "0.2.0"),
            "descending version order",
            id="ascending-order",
        ),
    ],
)
def test_parse_changelog_rejects_invalid_structure(changelog: str, match: str) -> None:
    """The changelog must keep one leading [Unreleased] and valid, ordered releases."""
    with pytest.raises(PolicyError, match=match):
        parse_changelog(changelog)


@pytest.mark.parametrize(
    ("state", "match"),
    [
        pytest.param(
            {
                "impact": "patch",
                "current": "0.1.1",
                "changed": ("CHANGELOG.md",),
                "current_unreleased": ENTRY,
            },
            "must not bump",
            id="patch-bumps-version",
        ),
        pytest.param(
            {"impact": "patch", "changed": ("src/fprcal/calibration.py",)},
            r"update CHANGELOG\.md",
            id="patch-without-changelog",
        ),
        pytest.param(
            {"impact": "minor", "changed": ("CHANGELOG.md",), "current_unreleased": "### Added"},
            "public change entry",
            id="minor-without-entry",
        ),
        pytest.param(
            {"impact": "none", "current": "0.1.1", "changed": ("README.md",)},
            "must not bump",
            id="none-bumps-version",
        ),
        pytest.param(
            {"impact": "release", "changed": ("CHANGELOG.md",), "base_unreleased": ENTRY},
            "must bump project.version",
            id="release-without-bump",
        ),
        pytest.param(
            {
                "impact": "release",
                "current": "0.2.0",
                "changed": ("pyproject.toml",),
                "base_unreleased": ENTRY,
            },
            r"must finalize CHANGELOG\.md",
            id="release-without-changelog",
        ),
        pytest.param(
            {
                "impact": "release",
                "base": "0.2.0",
                "current": "0.1.0",
                "changed": ("CHANGELOG.md",),
                "base_unreleased": ENTRY,
            },
            "must increase",
            id="release-lowers-version",
        ),
        pytest.param(
            {
                "impact": "release",
                "current": "0.1.1",
                "changed": ("CHANGELOG.md", "pyproject.toml", "uv.lock"),
            },
            "existing unreleased",
            id="release-without-unreleased-change",
        ),
        pytest.param({"impact": "major"}, "Unknown release impact", id="unknown-impact"),
    ],
)
def test_pull_request_state_rejects_policy_violations(state: dict[str, Any], match: str) -> None:
    """Each release impact enforces its version and changelog obligations."""
    with pytest.raises(PolicyError, match=match):
        _validate_state(**state)


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(
            {
                "impact": "patch",
                "changed": ("CHANGELOG.md", "src/fprcal/calibration.py"),
                "current_unreleased": ENTRY,
            },
            id="patch-with-entry",
        ),
        pytest.param({"impact": "none", "changed": ("uv.lock",)}, id="none"),
        pytest.param(
            {
                "impact": "release",
                "current": "0.2.0",
                "changed": ("CHANGELOG.md", "pyproject.toml", "uv.lock"),
                "base_unreleased": "### Added\n\n- Add a public capability.",
            },
            id="release",
        ),
    ],
)
def test_pull_request_state_accepts_valid_changes(state: dict[str, Any]) -> None:
    """Valid patch, maintenance, and release pull requests pass."""
    _validate_state(**state)


def test_pull_request_check_accepts_documentation_change(release_repo: tuple[Path, str]) -> None:
    """A no-release pull request passes against its Git base."""
    root, base_sha = release_repo
    validate_pull_request(
        root,
        base_sha=base_sha,
        event_path=_event(root, "- [x] `none`: documentation"),
        current_changelog=validate_repository(root),
    )


def test_pull_request_check_reads_changes_from_git(release_repo: tuple[Path, str]) -> None:
    """The check sees the committed diff, so a patch claim without a changelog fails."""
    root, base_sha = release_repo
    with pytest.raises(PolicyError, match=r"update CHANGELOG\.md"):
        validate_pull_request(
            root,
            base_sha=base_sha,
            event_path=_event(root, "- [x] `patch`: fix"),
            current_changelog=validate_repository(root),
        )


@pytest.mark.parametrize(
    ("shorten", "body", "match"),
    [
        pytest.param(True, "- [x] `none`", "full lowercase Git SHA", id="abbreviated-base"),
        pytest.param(False, None, "body is required", id="missing-body"),
    ],
)
def test_pull_request_check_rejects_unusable_inputs(
    release_repo: tuple[Path, str], shorten: bool, body: str | None, match: str
) -> None:
    """The base must be a full SHA and the pull request must carry a body."""
    root, base_sha = release_repo
    with pytest.raises(PolicyError, match=match):
        validate_pull_request(
            root,
            base_sha=base_sha[:12] if shorten else base_sha,
            event_path=_event(root, body),
            current_changelog=validate_repository(root),
        )


def test_main_ancestry_accepts_commit_contained_in_target(release_repo: tuple[Path, str]) -> None:
    """The ancestry check accepts a release commit contained in its target ref."""
    root, _ = release_repo
    validate_main_ancestry(root, "HEAD")


def test_main_ancestry_rejects_commit_outside_target(release_repo: tuple[Path, str]) -> None:
    """A release commit that the target ref does not contain is rejected."""
    root, base_sha = release_repo
    with pytest.raises(PolicyError, match="not an ancestor"):
        validate_main_ancestry(root, base_sha)


def test_main_ancestry_rejects_invalid_ref() -> None:
    """The ancestry check rejects ref text outside the safe Git syntax subset."""
    with pytest.raises(PolicyError, match="Invalid main ref"):
        validate_main_ancestry(ROOT, "HEAD;false")


def test_main_ancestry_surfaces_git_errors(release_repo: tuple[Path, str]) -> None:
    """An unknown ref is a Git failure, not a silent pass."""
    root, _ = release_repo
    with pytest.raises(subprocess.CalledProcessError):
        validate_main_ancestry(root, "refs/heads/missing")


@pytest.mark.parametrize("mode", ["repository", "release", "pull-request"])
def test_cli_runs_requested_checks(
    release_repo: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    """The command line runs the repository, release, and pull-request checks CI uses."""
    root, base_sha = release_repo
    options = {
        "repository": [],
        "release": ["--release-tag", "v0.1.0", "--main-ref", "HEAD"],
        "pull-request": ["--pr-base", base_sha, "--event-path", str(_event(root, "- [x] `none`"))],
    }[mode]
    monkeypatch.setattr(sys, "argv", ["check_release_policy.py", "--root", str(root), *options])

    check_release_policy.main()

    assert capsys.readouterr().out.strip() == PASSED


@pytest.mark.parametrize(
    ("options", "match"),
    [
        pytest.param(["--pr-base", "0" * 40], "supplied together", id="pr-base-alone"),
        pytest.param(["--main-ref", "HEAD"], "requires --release-tag", id="main-ref-alone"),
    ],
)
def test_cli_rejects_incomplete_option_pairs(
    release_repo: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    match: str,
) -> None:
    """Options that only make sense together are rejected when given alone."""
    root, _ = release_repo
    monkeypatch.setattr(sys, "argv", ["check_release_policy.py", "--root", str(root), *options])
    with pytest.raises(PolicyError, match=match):
        check_release_policy.main()
