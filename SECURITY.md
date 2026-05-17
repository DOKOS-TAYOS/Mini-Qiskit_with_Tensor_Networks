# Security Policy

This is an educational repository for learning quantum circuits and tensor networks.
Please do not post credentials, private datasets, tokens, or unpublished research data in
issues, pull requests, notebooks, or screenshots.

## Supported Versions

The main branch is the supported version for security fixes.

## Reporting a Vulnerability

If you find a vulnerability, please report it privately using GitHub private
vulnerability reporting if it is enabled for this repository. If that option is not
available, contact the repository maintainer privately before opening a public issue.

Please include:

- A short description of the problem.
- The affected file, dependency, or workflow.
- Steps to reproduce the issue safely.
- Whether any secret, token, or private data may have been exposed.

## For Students and Contributors

- Use a virtual environment and install dependencies from `requirements.txt`.
- Never commit `.env` files, API tokens, private keys, or credentials.
- Avoid adding notebooks with sensitive outputs, hidden cells, or private data.
- Keep examples reproducible with public data and deterministic random seeds where possible.
