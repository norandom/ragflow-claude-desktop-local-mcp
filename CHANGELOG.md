# Changelog

## 0.2.1 - 2026-10-25

- Refresh the dependency lock: aiohttp 3.14.5, MCP 1.30.0, certifi
  2026.10.23, multidict 6.9.1, PyJWT 2.15.1, filelock 4.2.4, and
  yarl 1.25.1. Retain urllib3 2.8.0, already present on main.
  These updates include or supersede the dependency updates in PR #22.
- Raise the aiohttp and MCP minimum versions to their security maintenance
  releases. Keep the locked MCP dependency on 1.x for this patch release.
- Fix Bandit's SARIF generation/upload order and remove duplicate scans and
  uploads. Pin Bandit 1.9.4 with its SARIF extra and add manual dispatch.
- Update GitHub Actions and pin all workflow action references to commits.
  Configure Dependabot for both uv and GitHub Actions. Routine version PRs
  are disabled for the owner's main-only workflow; security alerts remain active.
- Enforce the checked-in lockfile in test CI and Dagger tests.
- Run release tests and builds directly with uv on Python 3.12. The previous
  Dagger release path failed while loading its Python SDK, before project tests.
  Manual releases now check out the requested tag rather than the default ref.
- Isolate DSPy settings in OpenRouter configuration tests, restoring the
  previously skipped missing-key fallback test.
- Rewrite the README around a minimal first retrieval, with Claude Desktop,
  Codex CLI, and Antigravity CLI setup, data-flow guidance, and troubleshooting.

### Security review (2026-10-25)

GitHub reported 30 open Dependabot alerts and no open code-scanning alerts.
Some alerts still refer to versions older than the current lockfile, including
PyJWT and urllib3. Recheck alert reconciliation after the maintenance PR merges;
alerts have not been dismissed as part of this work.

An independent pip-audit scan identified additional filelock vulnerabilities,
addressed by the update to 4.2.4. After updating, the only remaining unique
advisory is DiskCache **GHSA-w8v5-vhqr-4h9v / CVE-2025-69872**. The audit feed
returns that advisory twice. DiskCache 5.6.3 is required by DSPy, and no fixed
version is listed. DSPy enables disk caching by default: another user able to
write to its cache directory can cause code execution through pickle loading.
Keep the DSPy cache private to the account running this application; do not
share or import untrusted cache directories. This release does not resolve or
suppress that advisory. The failed Dependabot run 39884110845 specifically
reported `security_update_not_found` for DiskCache.

At the start of the audit, Bandit's repository workflow was disabled due to
inactivity. It is re-enabled as part of this release. The advanced CodeQL workflow is
manually disabled; GitHub default CodeQL scanning is active. Do not enable both
CodeQL configurations without first choosing which configuration to retain.

### Validation

- Python 3.12: 28 tests passed, none skipped; coverage is 49%.
- Bandit 1.9.4: no findings; SARIF generated successfully.
- actionlint: no workflow errors.
- Wheel and source distribution build successfully.
- pip-audit: only the unresolved DiskCache advisory described above remains.
- Live RAGFlow/LLM integration, GitHub-hosted execution of the changed workflows,
  and Gitea runner compatibility have not been verified locally. Updated Node
  actions require a compatible runner runtime.
