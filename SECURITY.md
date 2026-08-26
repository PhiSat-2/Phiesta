# Security

Do not report credentials or access tokens in public issues.

If you discover a secret accidentally committed to this repository, revoke/rotate it first and then contact the repository maintainers privately so the history can be remediated.

Phiesta interacts with external authenticated services. Credentials should be provided through environment variables or interactive prompts and must never be committed to the repository.
