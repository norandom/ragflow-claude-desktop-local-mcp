# Contributing to RAGFlow Claude MCP Server

Thanks for the interest. A few notes before you open a PR.

## Protected main branch

`main` is protected. Direct pushes are rejected — open a PR instead.

## Getting set up

1. Fork on GitHub and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ragflow-claude-desktop-local-mcp.git
   cd ragflow-claude-desktop-local-mcp
   ```

2. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/norandom/ragflow-claude-desktop-local-mcp.git
   ```

3. Set up commit signing. SSH-signed commits are required:
   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global commit.gpgsign true
   ```
   See [COMMIT_SIGNING.md](COMMIT_SIGNING.md) if you need the longer walkthrough.

## Working on a change

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

Commit messages follow the conventional-commit prefixes I happen to use:
`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. Not strictly enforced, but it keeps the log readable.

Push to your fork and open a PR against `main`.

## Pull request requirements

- All commits must be SSH-signed. Unsigned commits get rejected.
- TruffleHog scans every PR. Don't include API keys, tokens, passwords, or private URLs.
- Describe what the PR does, why, and what you tested.
- Update docs if you add or change a feature.

A simple PR description like this is fine:

```markdown
## What
Brief description of what this PR does.

## Why
Why it's needed.

## Tested
- [ ] Locally
- [ ] Added/updated tests
```

## Secrets

If you accidentally commit a secret:

1. Don't push.
2. Remove the secret from the code.
3. Amend the commit (`git commit --amend`).
4. If it's already pushed, ping the maintainers — rotating the secret matters more than the git history at that point.

For security vulnerabilities, email the maintainers rather than filing a public issue.

## Code style

Python 3.8+, PEP 8, type hints where they help. Keep functions focused. Add a docstring when the function is non-obvious.

## Testing

Before opening the PR: run the tests, confirm the MCP server still starts, and check that the feature you touched still works end to end.

## Questions

Search existing issues first, then file a new one if needed.

## License

By contributing, you agree your contribution is licensed under the same license as the project.
