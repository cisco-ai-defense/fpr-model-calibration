## Summary

Describe what changed and why.

## Related issue

Link the issue or explain why no issue is needed.

## Validation

List the commands or manual checks that verify the change.

## Release impact

Select exactly one option. See `RELEASING.md` for the versioning policy.

- [ ] `patch`: backward-compatible bug or security fix
- [ ] `minor`: backward-compatible feature or deprecation
- [ ] `breaking`: incompatible public behavior or supported-environment change
- [ ] `none`: no installed-package behavior change
- [ ] `release`: dedicated version-bump and changelog-finalization pull request

## Checklist

- [ ] I added or updated tests for affected behavior.
- [ ] I added a `CHANGELOG.md` entry under `[Unreleased]` for a `patch`, `minor`, or `breaking` change.
- [ ] I left `project.version` unchanged unless this is the dedicated release pull request.
- [ ] I ran the repository validation commands in `CONTRIBUTING.md`.
- [ ] I did not add secrets, credentials, customer data, or internal-only URLs.
