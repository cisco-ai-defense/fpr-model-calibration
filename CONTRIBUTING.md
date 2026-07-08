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
uv run ruff check .
uv run ruff format --check .
uv run ty check src tests
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
behavior. Update user documentation and `CHANGELOG.md` when a change affects
the public API or release behavior.

Every pull request requires an approving review from someone other than its
author before merge. Resolve review comments and required checks rather than
bypassing them. Release pull requests have additional requirements in
[RELEASING.md](/RELEASING.md).
