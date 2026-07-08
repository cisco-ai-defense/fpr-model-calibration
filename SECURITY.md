# Security Policies and Procedures

This document outlines security procedures and general policies for the
`fprcal` project.

- [Disclosing a security issue](#disclosing-a-security-issue)
- [Vulnerability management](#vulnerability-management)
- [Suggesting changes](#suggesting-changes)

## Disclosing a security issue

Do not disclose a suspected vulnerability in a public GitHub issue. Submit a
[private vulnerability report](https://github.com/cisco-ai-defense/fpr-model-calibration/security/advisories/new)
so the maintainers can discuss and remediate the issue without exposing it.
If GitHub private reporting is unavailable, send the report to the
[Cisco Open security contact](mailto:oss-security@cisco.com).

Here are some helpful details to include in your report:

- a detailed description of the issue
- the steps required to reproduce the issue
- versions of the project that may be affected by the issue
- if known, any mitigations for the issue

Maintainers will acknowledge the report, share the next steps, and may ask for
additional information needed to reproduce or assess the issue. Follow up at the
same address if you do not receive a response.

## Vulnerability management

When the maintainers receive a disclosure report, they will assign it to a
primary handler.

This person will coordinate the fix and release process, which involves the
following steps:

- confirming the issue
- determining affected versions of the project
- auditing code to find any potential similar problems
- preparing fixes for all releases under maintenance

## Suggesting changes

If you have suggestions on how this process could be improved please submit an
issue or pull request.
