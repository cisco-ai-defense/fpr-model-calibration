# Releasing FPRCal

GitHub releases publish the package to PyPI through trusted publishing. The
workflow does not use a PyPI password or API token.

## Release policy

FPRCal uses three-part, PEP 440-compatible Semantic Versions. The public
contract includes exported Python functions, their observable results,
supported Python versions, runtime dependency compatibility, and published
package metadata.

Ordinary pull requests record release impact in `CHANGELOG.md` but do not
change `project.version`. Maintainers may batch several compatible changes into
one release. A release requires at least one installed-package change under
`[Unreleased]`, and the highest-impact entry determines the next version:

- Increment `PATCH` for backward-compatible bug fixes and security fixes.
- Increment `MINOR` for backward-compatible features and deprecations.
- Increment `MAJOR` for incompatible public changes after 1.0.0.
- Before 1.0.0, increment `MINOR` for an incompatible public change and reset
  `PATCH` to zero.

Documentation, tests, CI, development tooling, code formatting, development
lock updates, and refactors that preserve installed behavior do not justify a
version bump or PyPI release. Runtime dependency or package-metadata changes
do require a release when users need the new metadata; classify a
backward-compatible security or compatibility correction as `PATCH` and a
reduction in supported environments as breaking.

The project does not publish empty releases. It also does not publish
automatically after every merge. These rules follow [Semantic
Versioning](https://semver.org/), Python's [version-specifier
standard](https://packaging.python.org/en/latest/specifications/version-specifiers/),
and the `[Unreleased]` workflow from [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/).

## One-time setup

Before the first release, create a pending trusted publisher from the PyPI
account's **Publishing** page with these values:

- PyPI project name: `fprcal`
- GitHub owner: `cisco-ai-defense`
- GitHub repository: `fpr-model-calibration`
- Workflow: `publish.yml`
- Environment: `pypi`

Create a GitHub environment named `pypi`, assign at least one named maintainer
as a required reviewer, and require that maintainer's approval before
deployment. PyPI creates the project when the pending publisher uploads the
first release; configuring the publisher does not reserve the project name.

See PyPI's [pending publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for the current setup procedure.

## Prepare a release

1. Confirm that `[Unreleased]` contains at least one `Added`, `Changed`,
   `Deprecated`, `Removed`, `Fixed`, or `Security` entry from an installed
   package change.
2. Choose the next version from the highest-impact unreleased entry.
3. Move the unreleased entries under `## [X.Y.Z] - YYYY-MM-DD`, leave an empty
   `## [Unreleased]` heading at the top, and update the comparison links at the
   bottom of `CHANGELOG.md`.
4. Update `project.version` in `pyproject.toml`, then run `uv lock` so the
   package version in `uv.lock` matches.
5. Run the release checks from a clean checkout:

   ```bash
   uv sync --all-extras --locked
   uv run python scripts/check_release_policy.py
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check src tests scripts
   uv run pytest --cov=fprcal --cov-report=term-missing
   uv run pip-audit --skip-editable --cache-dir .uv-cache/pip-audit
   uv run licensecheck
   uv lock --check
   uv build
   uvx --from twine twine check --strict dist/*
   ```

6. Open a pull request, select the `release` impact, and request at least one
   peer reviewer. The policy check requires the previous revision to contain
   unreleased package changes and rejects an unchanged version.
7. Merge only after CI passes, a peer approves the pull request, and all review
   comments are resolved.

## Publish a release

Create a GitHub release from the merge commit using a tag that matches the
package version with a `v` prefix, such as `v0.1.0`. Release notes must describe
new functionality, bug fixes when applicable, and supported Python versions.
Exclude CI-only changes from the public notes.

Publishing the GitHub release starts `.github/workflows/publish.yml`, which
verifies the tag, changelog entry, package version, and ancestry from `main`;
builds and validates both distributions; waits for approval in the `pypi`
environment; and uploads them to PyPI through trusted publishing. After
publication, the workflow installs the exact public version on every supported
Python version and runs an import, version, fit, and prediction smoke test.

Confirm that all publish and verification jobs pass and that the release is
visible on the [FPRCal PyPI page](https://pypi.org/project/fprcal/) before
announcing it.

PyPI does not permit replacing files for an existing version. If publishing
fails after an upload, increment the version and create a new release rather
than reusing the tag.
