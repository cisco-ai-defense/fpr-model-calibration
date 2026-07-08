# Releasing FPRCal

GitHub releases publish the package to PyPI through trusted publishing. The
workflow does not use a PyPI password or API token.

## One-time setup

Before the first release, create a pending trusted publisher from the PyPI
account's **Publishing** page with these values:

- PyPI project name: `fprcal`
- GitHub owner: `cisco-ai-defense`
- GitHub repository: `fpr-model-calibration`
- Workflow: `publish.yml`
- Environment: `pypi`

Create a GitHub environment named `pypi` and require a maintainer's approval
before deployment. PyPI creates the project when the pending publisher uploads
the first release; configuring the publisher does not reserve the project name.

See PyPI's [pending publisher documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for the current setup procedure.

## Prepare a release

1. Update `project.version` in `pyproject.toml` and finalize `CHANGELOG.md`.
2. Run the release checks from a clean checkout:

   ```bash
   uv sync --all-extras --locked
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check src tests
   uv run pytest --cov=fprcal --cov-report=term-missing
   uv run pip-audit --skip-editable
   uv build
   uvx --from twine twine check --strict dist/*
   ```

3. Open and merge the release pull request after CI passes.

## Publish a release

Create a GitHub release from the merge commit using a tag that matches the
package version with a `v` prefix, such as `v0.1.0`. Publishing the GitHub
release starts `.github/workflows/publish.yml`, which verifies the tag, builds
and validates both distributions, and uploads them to PyPI.

PyPI does not permit replacing files for an existing version. If publishing
fails after an upload, increment the version and create a new release rather
than reusing the tag.
