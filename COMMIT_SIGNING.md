# Commit signing with SSH keys

How to sign Git commits with an SSH key for this repo.

## Why

Signed commits prove the commit really came from you and that nobody altered it after the fact. GitHub then shows a "Verified" badge next to it. That's the whole story.

## Prerequisites

- Git 2.34 or newer (`git --version`)
- An SSH key (or generate one — see below)
- A GitHub account with that key added

## Setup

### 1. Check your Git version

SSH commit signing needs Git 2.34+:

```bash
git --version
```

If you're older than that, update first.

### 2. Generate an SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

For older systems without Ed25519 support:
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

### 3. Add the key to your SSH agent

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519   # or ~/.ssh/id_rsa
```

### 4. Add the public key to GitHub

```bash
cat ~/.ssh/id_ed25519.pub   # or id_rsa.pub
```

On GitHub: Settings → SSH and GPG keys → New SSH key. Pick "Signing Key" (or add the same key twice, once for auth and once for signing). Paste, save.

### 5. Configure Git to sign with SSH

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
git config --global tag.gpgsign true   # optional
```

### 6. Allowed-signers file

Git also needs a list of keys it trusts when verifying signatures locally:

```bash
touch ~/.ssh/allowed_signers
echo "$(git config --get user.email) $(cat ~/.ssh/id_ed25519.pub)" >> ~/.ssh/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers
```

## Usage

If you skipped `commit.gpgsign true`, sign individual commits with `-S`:

```bash
git commit -S -m "Your commit message"
```

To check signatures locally:

```bash
git log --show-signature -1
git log --pretty="format:%h %G? %aN  %s" -10
```

The `%G?` codes:
- `G` — good signature
- `B` — bad signature
- `U` — good but unknown validity
- `X` — expired signature
- `Y` — expired key
- `R` — revoked key
- `E` — can't verify (missing key)
- `N` — unsigned

GitHub shows "Verified" automatically once your signing key is on your account.

## Troubleshooting

**`Error: unsupported value for gpg.format`** — Git is too old; upgrade to 2.34+.

**`Error: user.signingkey needs to be set`** — set it:
```bash
git config --global user.signingkey ~/.ssh/id_ed25519.pub
```

**Commits show as unverified on GitHub** — make sure (a) the key is uploaded as a *signing* key, (b) `git config --get user.email` matches your GitHub email, (c) the key path in your Git config matches the public key on GitHub.

**Permission denied when signing** — your key isn't loaded into the agent:
```bash
ssh-add -l
ssh-add ~/.ssh/id_ed25519
```

## Per-repo override

To require signing only for this repo:

```bash
cd /path/to/ragflow-claude-desktop-local-mcp
git config commit.gpgsign true
git config gpg.format ssh
git config user.signingkey ~/.ssh/id_ed25519.pub
```

## Other things worth doing

- Keep a separate signing-only key if you want to scope blast radius.
- A hardware key (YubiKey etc.) is nice if you're already using one.
- Back up your private key somewhere safe.

## Links

- [GitHub: SSH commit signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
- [Git: signing your work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)
- [GitHub: generating an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
