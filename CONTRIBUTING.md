# Contributing to RAGFlow MCP

Maintainers work directly on `main`. Keep one development branch in this
repository; do not create feature or release branches for routine work.

## Prepare and validate a change

```sh
git switch main
git pull --ff-only origin main
uv sync --locked --extra dev
```

Use Python 3.12 or newer. Keep functions focused, use type hints where they help,
and update the README when user-facing behavior changes.

Before committing, run the tests and the checks relevant to your change:

```sh
uv run --locked pytest
uv build
git diff --check
```

Check MCP startup when changing server behavior. Mocked tests do not replace a
live check of an affected RAGFlow or provider integration.

## Commit and push

Use SSH-signed commits; see [COMMIT_SIGNING.md](COMMIT_SIGNING.md) for setup.
Use a conventional prefix such as `fix:`, `docs:`, `test:`, or `chore:` and
describe the concrete change.

```sh
git add <changed-files>
git commit -S -m "fix: describe the change"
git push origin main
```

Review hosted test and security results after pushing. Tag releases from the
validated `main` commit. The tag triggers the release test/build workflow.
Dependabot may open temporary update branches; incorporate the useful updates,
then close superseded requests and remove their branches.

Contributors without write access can submit a pull request from a fork.
Describe the problem, the resulting behavior, and what you tested.

## Secrets and security reports

Do not commit API keys, tokens, passwords, or private configuration. Review
logs and examples before sharing them. If a credential reaches the remote,
rotate it; deleting it from the latest commit does not remove the exposure.

Report vulnerabilities privately to the maintainer rather than including
sensitive details in a public issue.
