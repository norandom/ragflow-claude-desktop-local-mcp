# Commit Signing with SSH Keys

This guide explains how to set up and use SSH key signing for Git commits in this repository.

## Why Sign Commits?

Signing commits provides:
- **Authentication**: Proves that commits actually came from you
- **Integrity**: Ensures commits haven't been tampered with
- **Trust**: Builds confidence in the codebase's authenticity

## Prerequisites

- Git version 2.34 or later (check with `git --version`)
- An existing SSH key (or create one following the steps below)
- A GitHub account with your SSH key added

## Setup Instructions

### 1. Check Git Version

SSH commit signing requires Git 2.34 or later:

```bash
git --version
```

If your version is older, update Git first.

### 2. Generate an SSH Key (if needed)

If you don't have an SSH key, create one:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

For legacy systems that don't support Ed25519:
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

### 3. Add SSH Key to SSH Agent

```bash
# Start the ssh-agent
eval "$(ssh-agent -s)"

# Add your SSH private key
ssh-add ~/.ssh/id_ed25519
# or for RSA
ssh-add ~/.ssh/id_rsa
```

### 4. Add SSH Key to GitHub

1. Copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   # or for RSA
   cat ~/.ssh/id_rsa.pub
   ```

2. Go to GitHub → Settings → SSH and GPG keys
3. Click "New SSH key"
4. Choose "Authentication Key" or "Signing Key" (or add it twice for both)
5. Paste your public key and save

### 5. Configure Git for SSH Signing

Set up Git to use SSH for commit signing:

```bash
# Tell Git to use SSH for signing
git config --global gpg.format ssh

# Specify which SSH key to use for signing
git config --global user.signingkey ~/.ssh/id_ed25519.pub
# or for RSA
git config --global user.signingkey ~/.ssh/id_rsa.pub

# Enable commit signing by default (optional but recommended)
git config --global commit.gpgsign true

# Enable tag signing by default (optional)
git config --global tag.gpgsign true
```

### 6. Create Allowed Signers File

Git needs to know which SSH keys to trust. Create an allowed signers file:

```bash
# Create the file
touch ~/.ssh/allowed_signers

# Add your key to the file
echo "$(git config --get user.email) $(cat ~/.ssh/id_ed25519.pub)" >> ~/.ssh/allowed_signers
# or for RSA
echo "$(git config --get user.email) $(cat ~/.ssh/id_rsa.pub)" >> ~/.ssh/allowed_signers

# Tell Git where to find the file
git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers
```

## Usage

### Signing Individual Commits

If you didn't enable signing by default, sign individual commits with:

```bash
git commit -S -m "Your commit message"
```

### Verifying Signatures

To verify commit signatures:

```bash
# Show signature for the last commit
git log --show-signature -1

# Verify all commits in a range
git log --show-signature main..feature-branch

# Show commits with signature status
git log --pretty="format:%h %G? %aN  %s" -10
```

Signature status codes:
- `G`: Good (valid signature)
- `B`: Bad signature
- `U`: Good signature with unknown validity
- `X`: Good signature that has expired
- `Y`: Good signature made by an expired key
- `R`: Good signature made by a revoked key
- `E`: Signature cannot be checked (missing key)
- `N`: No signature

### Viewing Signatures on GitHub

GitHub automatically displays the signature status for all commits once your SSH public key is uploaded to your account:

- **Verified** badge: Commit is signed with a key associated with your GitHub account
- **Unverified**: Commit is signed but the key isn't associated with your account
- No badge: Commit is not signed

To enable this:
1. Add your SSH public key to GitHub (Settings → SSH and GPG keys)
2. Make sure to add it as a "Signing Key" (or add it for both authentication and signing)
3. GitHub will automatically verify all past and future commits signed with that key

## Troubleshooting

### "Error: unsupported value for gpg.format"

Your Git version is too old. Update to Git 2.34 or later.

### "Error: user.signingkey needs to be set"

You haven't configured which SSH key to use:
```bash
git config --global user.signingkey ~/.ssh/id_ed25519.pub
```

### Commits Not Showing as Verified on GitHub

1. Ensure your SSH key is added to GitHub as a signing key
2. Check that your Git email matches the one associated with your GitHub account:
   ```bash
   git config --get user.email
   ```
3. Make sure the SSH key in your Git config matches the one on GitHub

### Permission Denied When Signing

Ensure your SSH key is added to the SSH agent:
```bash
ssh-add -l  # List loaded keys
ssh-add ~/.ssh/id_ed25519  # Add your key if not listed
```

## Repository-Specific Configuration

To require signed commits for this repository only:

```bash
# Navigate to the repository
cd /path/to/ragflow-claude-desktop-local-mcp

# Set repository-specific signing
git config commit.gpgsign true
git config gpg.format ssh
git config user.signingkey ~/.ssh/id_ed25519.pub
```

## Additional Security

For maximum security, consider:

1. **Using a dedicated signing key**: Create a separate SSH key just for signing
2. **Hardware keys**: Use a hardware security key that supports SSH (like YubiKey)
3. **Key rotation**: Periodically rotate your signing keys
4. **Backup**: Keep secure backups of your private keys

## Resources

- [GitHub's SSH Commit Signing Documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
- [Git's Official Signing Documentation](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [SSH Key Generation Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)