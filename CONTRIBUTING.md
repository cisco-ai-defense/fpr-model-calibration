# Contributing to FPRCal

All project interactions must follow the [Code of Conduct](/CODE_OF_CONDUCT.md).

## Report an issue

Search the [existing issues](https://github.com/cisco-ai-defense/fpr-model-calibration/issues)
before opening a report. A useful bug report identifies the FPRCal and Python
versions, includes a minimal reproduction, and states the expected and actual
behavior.

Do not disclose security vulnerabilities in a public issue. Follow the private
reporting process in [SECURITY.md](/SECURITY.md).

## Set up the development environment

FPRCal uses Python 3.12 or later and [`uv`](https://docs.astral.sh/uv/) for the
locked development environment.

```bash
git clone https://github.com/cisco-ai-defense/fpr-model-calibration.git
cd fpr-model-calibration
uv sync --all-extras --locked
```

## Validate a change

Run the same checks required by continuous integration before opening a pull
request:

```bash
uv run python scripts/check_release_policy.py
uv run ruff check .
uv run ruff format --check .
uv run ty check src tests scripts
uv run pytest --cov=fprcal --cov-report=term-missing
uv run pip-audit --skip-editable --cache-dir .uv-cache/pip-audit
uv run licensecheck
uv lock --check
```

Changes to packaging metadata or release automation must also pass distribution
validation:

```bash
uv build
uvx twine check --strict dist/*
```

## Open a pull request

Keep each pull request focused on one change. Describe the behavior and reason
for the change, link related issues, and include tests for affected public
behavior.

Select exactly one release-impact option in the pull request template:

- `patch` for a backward-compatible bug or security fix;
- `minor` for backward-compatible functionality or a deprecation;
- `breaking` for an incompatible public change;
- `none` for documentation, tests, CI, development tooling, dependency-lock
  maintenance, or an internal refactor that preserves installed behavior; or
- `release` only for the dedicated pull request that prepares a PyPI release.

A `patch`, `minor`, or `breaking` change must add a concise entry under
`[Unreleased]` in `CHANGELOG.md`. Do not change `project.version` in an ordinary
change pull request. The release pull request chooses one version for all
accumulated entries, which avoids competing version bumps when several changes
are in progress.

Every pull request requires an approving review from someone other than its
author before merge. Resolve review comments and required checks rather than
bypassing them. Release pull requests have additional requirements in
[RELEASING.md](/RELEASING.md).
