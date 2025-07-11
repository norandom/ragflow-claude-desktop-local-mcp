# Contributing to RAGFlow Claude MCP Server

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🚨 Important: Protected Main Branch

The `main` branch is protected and requires pull requests for all changes. Direct pushes to `main` are not allowed.

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button on GitHub to create your own copy

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ragflow-claude-desktop-local-mcp.git
   cd ragflow-claude-desktop-local-mcp
   ```

3. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/norandom/ragflow-claude-desktop-local-mcp.git
   ```

4. **Set Up Commit Signing** (Required)
   ```bash
   # Configure SSH signing
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global commit.gpgsign true
   ```
   
   For detailed setup instructions, see [Commit Signing Guide](COMMIT_SIGNING.md).

## Development Workflow

### 1. Create a Feature Branch

Always create a new branch for your work:

```bash
# Update your main branch first
git checkout main
git pull upstream main

# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clear, concise commit messages
- Follow the existing code style
- Update documentation as needed
- Add tests if applicable

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature X"
# or
git commit -m "fix: resolve issue with Y"
```

Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for test additions/changes
- `chore:` for maintenance tasks

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" 
3. Ensure the base repository is `norandom/ragflow-claude-desktop-local-mcp` and base branch is `main`
4. Provide a clear title and description
5. Link any related issues

## Pull Request Guidelines

### PR Requirements

- ✅ **Signed Commits**: All commits MUST be signed with SSH keys
  - Configure signing before making commits
  - See [Commit Signing Guide](COMMIT_SIGNING.md) for setup
  - Unsigned commits will be rejected

- ✅ **No Secrets**: All PRs are automatically scanned by TruffleHog. Do not include:
  - API keys
  - Tokens
  - Passwords
  - Private URLs
  - Any sensitive information

- ✅ **Clear Description**: Include:
  - What the PR does
  - Why it's needed
  - Any breaking changes
  - Testing performed

- ✅ **Documentation**: Update relevant docs for new features

- ✅ **Code Quality**: Ensure your code:
  - Follows Python best practices
  - Is properly formatted
  - Has meaningful variable/function names
  - Includes comments for complex logic

### PR Template

When creating a PR, use this template:

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] I have tested this locally
- [ ] I have added tests (if applicable)

## Checklist
- [ ] My code follows the project style
- [ ] I have updated documentation
- [ ] I have checked for exposed secrets
- [ ] I have tested my changes
```

## Security

### Handling Secrets

Never commit secrets to the repository. If you accidentally commit a secret:

1. **Do NOT** push the commit
2. Remove the secret from your code
3. Use `git commit --amend` to update the commit
4. If already pushed, notify maintainers immediately

### Reporting Security Issues

For security vulnerabilities, please email the maintainers directly rather than creating a public issue.

## Code Style

- Use Python 3.8+ features appropriately
- Follow PEP 8 guidelines
- Use type hints where beneficial
- Keep functions focused and single-purpose
- Add docstrings to functions and classes

## Testing

Before submitting a PR:

1. Test your changes locally
2. Ensure existing functionality still works
3. Add tests for new features if applicable
4. Verify the MCP server starts correctly

## Questions or Issues?

- Check existing issues first
- Create a new issue for bugs or feature requests
- Join discussions in existing issues
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉