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

Create a GitHub environment named `pypi`, assign at least one named maintainer
as a required reviewer, and require that maintainer's approval before
deployment. PyPI creates the project when the pending publisher uploads the
first release; configuring the publisher does not reserve the project name.

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

3. Open the release pull request and request at least one peer reviewer.
4. Merge only after CI passes, a peer approves the pull request, and all review
   comments are resolved.

## Publish a release

Create a GitHub release from the merge commit using a tag that matches the
package version with a `v` prefix, such as `v0.1.0`. Release notes must describe
new functionality, bug fixes when applicable, and supported Python versions.
Exclude CI-only changes from the public notes.

Publishing the GitHub release starts `.github/workflows/publish.yml`, which
verifies the tag, builds and validates both distributions, waits for approval
in the `pypi` environment, and uploads them to PyPI. After publication, the
workflow installs the exact public version on every supported Python version
and runs an import, version, fit, and prediction smoke test.

Confirm that all publish and verification jobs pass and that the release is
visible on the [FPRCal PyPI page](https://pypi.org/project/fprcal/) before
announcing it.

PyPI does not permit replacing files for an existing version. If publishing
fails after an upload, increment the version and create a new release rather
than reusing the tag.
