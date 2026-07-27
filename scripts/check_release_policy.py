"""Validate FPRCal version, changelog, pull request, and release invariants."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
RELEASE_HEADING_PATTERN = re.compile(
    r"^## \[(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\]"
    r" - (?P<released_on>\d{4}-\d{2}-\d{2})$"
)
IMPACT_PATTERN = re.compile(
    r"^- \[(?P<checked>[ xX])\] "
    r"`(?P<impact>patch|minor|breaking|none|release)`(?::|\s|$)",
    re.MULTILINE,
)
FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
PUBLIC_CHANGE_CATEGORIES = {
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
}


class PolicyError(RuntimeError):
    """A release-policy invariant was violated."""


@dataclass(frozen=True)
class Release:
    """One finalized changelog release."""

    version: str
    released_on: date
    body: str


@dataclass(frozen=True)
class Changelog:
    """Parsed release-relevant changelog state."""

    unreleased: str
    releases: tuple[Release, ...]


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse the project's restricted, final-release Semantic Version."""
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise PolicyError(f"Version {version!r} must use final X.Y.Z form.")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def project_version_from_text(pyproject_text: str) -> str:
    """Read and validate project.version from pyproject text."""
    project = tomllib.loads(pyproject_text)["project"]
    version = project["version"]
    if not isinstance(version, str):
        raise PolicyError("project.version must be a string.")
    parse_version(version)
    return version


def _validate_change_section(body: str, label: str, *, require_entry: bool) -> bool:
    current_category: str | None = None
    found_entry = False

    for line in body.splitlines():
        if line.startswith("### "):
            current_category = line.removeprefix("### ")
            if current_category not in PUBLIC_CHANGE_CATEGORIES:
                raise PolicyError(
                    f"{label} uses unsupported changelog category {current_category!r}."
                )
        elif line.startswith("- "):
            if current_category is None:
                raise PolicyError(f"{label} has a change entry outside an allowed category.")
            found_entry = True

    if require_entry and not found_entry:
        raise PolicyError(f"{label} must contain at least one public change entry.")
    return found_entry


def parse_changelog(changelog_text: str) -> Changelog:
    """Parse and validate changelog headings, categories, dates, and order."""
    lines = changelog_text.splitlines()
    section_starts = [index for index, line in enumerate(lines) if line.startswith("## [")]
    if not section_starts:
        raise PolicyError("CHANGELOG.md must contain an [Unreleased] section.")

    headings = [lines[index] for index in section_starts]
    if headings[0] != "## [Unreleased]":
        raise PolicyError("[Unreleased] must be the first changelog section.")
    if headings.count("## [Unreleased]") != 1:
        raise PolicyError("CHANGELOG.md must contain exactly one [Unreleased] section.")

    section_ends = [*section_starts[1:], len(lines)]
    unreleased = "\n".join(lines[section_starts[0] + 1 : section_ends[0]]).strip()
    _validate_change_section(unreleased, "[Unreleased]", require_entry=False)

    releases: list[Release] = []
    for start, end in zip(section_starts[1:], section_ends[1:], strict=True):
        heading = lines[start]
        match = RELEASE_HEADING_PATTERN.fullmatch(heading)
        if match is None:
            raise PolicyError(f"Release heading {heading!r} must use '## [X.Y.Z] - YYYY-MM-DD'.")
        version = match.group("version")
        released_on = date.fromisoformat(match.group("released_on"))
        body = "\n".join(lines[start + 1 : end]).strip()
        parse_version(version)
        _validate_change_section(body, f"[{version}]", require_entry=True)
        releases.append(Release(version=version, released_on=released_on, body=body))

    if not releases:
        raise PolicyError("CHANGELOG.md must contain at least one finalized release.")

    version_numbers = [parse_version(release.version) for release in releases]
    if len(set(version_numbers)) != len(version_numbers):
        raise PolicyError("CHANGELOG.md contains a duplicate release version.")
    adjacent_versions = zip(version_numbers, version_numbers[1:], strict=False)
    if any(older >= newer for newer, older in adjacent_versions):
        raise PolicyError("Changelog releases must appear in descending version order.")

    return Changelog(unreleased=unreleased, releases=tuple(releases))


def validate_repository(root: Path, *, release_tag: str | None = None) -> Changelog:
    """Validate repository release metadata and an optional publication tag."""
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    changelog_text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    project_version = project_version_from_text(pyproject_text)
    changelog = parse_changelog(changelog_text)

    if changelog.releases[0].version != project_version:
        raise PolicyError("project.version must match the newest finalized CHANGELOG.md release.")
    if release_tag is not None and release_tag != f"v{project_version}":
        raise PolicyError(
            f"Release tag {release_tag!r} does not match package version v{project_version}."
        )
    return changelog


def parse_release_impact(pull_request_body: str) -> str:
    """Return the single checked release-impact value from a pull request body."""
    checked = [
        match.group("impact")
        for match in IMPACT_PATTERN.finditer(pull_request_body)
        if match.group("checked").lower() == "x"
    ]
    if len(checked) != 1:
        raise PolicyError("Pull request body must select exactly one release-impact option.")
    return checked[0]


def validate_pull_request_state(
    *,
    impact: str,
    base_version: str,
    current_version: str,
    changed_files: set[str],
    base_unreleased: str,
    current_unreleased: str,
) -> None:
    """Validate release-impact consequences independently of Git transport."""
    version_changed = current_version != base_version
    changelog_changed = "CHANGELOG.md" in changed_files

    if impact in {"patch", "minor", "breaking"}:
        if version_changed:
            raise PolicyError("Ordinary behavior changes must not bump project.version.")
        if not changelog_changed or current_unreleased == base_unreleased:
            raise PolicyError(f"A {impact} change must update CHANGELOG.md under [Unreleased].")
        if not _validate_change_section(current_unreleased, "[Unreleased]", require_entry=False):
            raise PolicyError("[Unreleased] must contain a public change entry.")
        return

    if impact == "none":
        if version_changed:
            raise PolicyError("A no-release change must not bump project.version.")
        return

    if impact != "release":
        raise PolicyError(f"Unknown release impact {impact!r}.")
    if not version_changed:
        raise PolicyError("A release pull request must bump project.version.")
    if not changelog_changed:
        raise PolicyError("A release pull request must finalize CHANGELOG.md.")
    if parse_version(current_version) <= parse_version(base_version):
        raise PolicyError("A release pull request must increase project.version.")
    if not _validate_change_section(base_unreleased, "Base [Unreleased]", require_entry=False):
        raise PolicyError("A release requires an existing unreleased public change.")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", *arguments],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def validate_pull_request(
    root: Path,
    *,
    base_sha: str,
    event_path: Path,
    current_changelog: Changelog,
) -> None:
    """Validate a pull request against its immutable GitHub base SHA."""
    if FULL_SHA_PATTERN.fullmatch(base_sha) is None:
        raise PolicyError("Pull request base must be a full lowercase Git SHA.")

    event = json.loads(event_path.read_text(encoding="utf-8"))
    pull_request = event["pull_request"]
    body = pull_request["body"]
    if not isinstance(body, str):
        raise PolicyError("Pull request body is required for release classification.")

    changed_files = set(_git(root, "diff", "--name-only", f"{base_sha}...HEAD").splitlines())
    base_pyproject = _git(root, "show", f"{base_sha}:pyproject.toml")
    base_changelog_text = _git(root, "show", f"{base_sha}:CHANGELOG.md")
    base_unreleased = ""
    if "## [Unreleased]" in base_changelog_text:
        base_unreleased = parse_changelog(base_changelog_text).unreleased

    current_pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    validate_pull_request_state(
        impact=parse_release_impact(body),
        base_version=project_version_from_text(base_pyproject),
        current_version=project_version_from_text(current_pyproject),
        changed_files=changed_files,
        base_unreleased=base_unreleased,
        current_unreleased=current_changelog.unreleased,
    )


def validate_main_ancestry(root: Path, main_ref: str) -> None:
    """Require the release commit to be reachable from the configured main ref."""
    if GIT_REF_PATTERN.fullmatch(main_ref) is None:
        raise PolicyError(f"Invalid main ref {main_ref!r}.")
    result = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", "HEAD", main_ref],  # noqa: S607
        cwd=root,
        check=False,
    )
    if result.returncode == 1:
        raise PolicyError(f"Release commit is not an ancestor of {main_ref}.")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the script's parent repository).",
    )
    parser.add_argument("--release-tag", help="GitHub release tag to validate.")
    parser.add_argument("--main-ref", help="Main ref that must contain the release.")
    parser.add_argument("--pr-base", help="Pull request base commit SHA.")
    parser.add_argument(
        "--event-path",
        type=Path,
        help="GitHub pull request event JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run requested release-policy validations."""
    args = parse_args()
    root = args.root.resolve()
    changelog = validate_repository(root, release_tag=args.release_tag)

    if (args.pr_base is None) != (args.event_path is None):
        raise PolicyError("--pr-base and --event-path must be supplied together.")
    if args.pr_base is not None:
        validate_pull_request(
            root,
            base_sha=args.pr_base,
            event_path=args.event_path,
            current_changelog=changelog,
        )

    if args.main_ref is not None:
        if args.release_tag is None:
            raise PolicyError("--main-ref requires --release-tag.")
        validate_main_ancestry(root, args.main_ref)

    print("Release policy checks passed.")


if __name__ == "__main__":
    main()
