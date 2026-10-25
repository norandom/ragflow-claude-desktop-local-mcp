# RAGFlow MCP for Claude Desktop, Codex, and Antigravity

[![GitHub release](https://img.shields.io/github/v/release/norandom/ragflow-claude-desktop-local-mcp?sort=semver)](https://github.com/norandom/ragflow-claude-desktop-local-mcp/releases)
[![Tests](https://github.com/norandom/ragflow-claude-desktop-local-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/norandom/ragflow-claude-desktop-local-mcp/actions/workflows/tests.yml)

Search documents you already have in [RAGFlow](https://github.com/infiniflow/ragflow) from your AI client. This local MCP server retrieves document chunks and similarity scores, so you can ask questions across your knowledge bases without copying passages into each conversation.

Start with ordinary retrieval. You need a RAGFlow API key, but no separate OpenAI or OpenRouter key for this server unless you enable optional query deepening. Your AI client still needs its own account or model configuration.

I built this for my own R&D. Try it on a dataset whose contents you know before relying on it for research. The first check is concrete: can your client list that dataset and retrieve a passage you can verify in the original document?

## Before you start

You need:

- A reachable RAGFlow instance with an API key and at least one parsed, indexed document.
- Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Python 3.12 or newer. The quickstart uses Python 3.12, which is tested in CI.
- Claude Desktop, Codex CLI, Antigravity CLI (`agy`), or another client that can launch a local stdio MCP server.

This project connects to an existing RAGFlow installation. It does not install RAGFlow or upload and index your documents.

The MCP process runs on your computer. Queries go to your RAGFlow instance; retrieved content returns to your AI client and may be sent to that client's model provider. Optional DSPy deepening also sends queries and result summaries to the provider you configure below. Check those data paths before using sensitive documents.

## Install and configure

Run these commands in a terminal, including PowerShell on Windows:

```sh
git clone --branch v0.2.1 https://github.com/norandom/ragflow-claude-desktop-local-mcp.git
cd ragflow-claude-desktop-local-mcp
uv sync --locked --python 3.12
```

This checks out the release tag and installs its locked dependencies. A detached-HEAD message is normal for a release checkout.

Create `config.json` in the cloned directory with these two entries:

```json
{
  "RAGFLOW_BASE_URL": "http://your-ragflow-server:9380",
  "RAGFLOW_API_KEY": "replace-with-your-ragflow-api-key"
}
```

Use the base URL, without `/api/v1`. Use HTTPS when connecting across an untrusted network. `config.json` is ignored by Git, but contains a credential: keep it private. [config.json.sample](config.json.sample) lists optional settings; leave unused placeholder credentials out of your working configuration.

Environment variables override values in the file. `RAGFLOW_CONFIG_PATH` selects a different config file; otherwise the server reads `config.json` from its working directory. There is no built-in RAGFlow address or API key.

## Connect your client

Choose one client below. Replace `/absolute/path/to/ragflow-claude-desktop-local-mcp` with the directory you cloned. On Windows, `C:/Users/you/Source/ragflow-claude-desktop-local-mcp` works in these examples. Quote paths containing spaces.

The client starts the server on demand over stdio. You do not need to start a web server or open a local port.

### Claude Desktop

Open Settings, then Developer, then Edit Config. Add the `ragflow` entry to the existing `mcpServers` object in `claude_desktop_config.json`; preserve your other servers.

```json
{
  "mcpServers": {
    "ragflow": {
      "command": "uv",
      "args": [
        "run",
        "--locked",
        "--directory",
        "/absolute/path/to/ragflow-claude-desktop-local-mcp",
        "ragflow-claude-mcp"
      ]
    }
  }
}
```

Fully quit and reopen Claude Desktop. If it cannot find `uv`, replace `command` with its full executable path. Find it with `Get-Command uv` in PowerShell or `command -v uv` on macOS/Linux. See the [local MCP setup guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers) for client configuration and logs.

### Codex CLI

Register the local command:

```sh
codex mcp add ragflow -- uv run --locked --directory "/absolute/path/to/ragflow-claude-desktop-local-mcp" ragflow-claude-mcp
codex mcp get ragflow
```

Start a new `codex` session and use `/mcp` to check the connection. Codex stores MCP settings in `~/.codex/config.toml`; alternatively, add this table there instead of running the registration command:

```toml
[mcp_servers.ragflow]
command = "uv"
args = ["run", "--locked", "--directory", "/absolute/path/to/ragflow-claude-desktop-local-mcp", "ragflow-claude-mcp"]
```

To use a config file outside the checkout, add `--env "RAGFLOW_CONFIG_PATH=/absolute/path/to/config.json"` before the `--` in the registration command. See [OpenAI's MCP documentation](https://developers.openai.com/codex/mcp) for client settings.

### Antigravity CLI (`agy`)

Register the same stdio command:

```sh
agy mcp add --type stdio ragflow uv run --locked --directory "/absolute/path/to/ragflow-claude-desktop-local-mcp" ragflow-claude-mcp
agy mcp list
```

Start a new `agy` session and open `/mcp` to check the server. If it is disabled, run `agy mcp enable ragflow`.

For a config file outside the checkout, put `--env "RAGFLOW_CONFIG_PATH=/absolute/path/to/config.json"` before `ragflow` in the registration command. The `agy mcp add` options belong before the server name; the arguments after `uv` belong to uv. See [Antigravity's MCP documentation](https://antigravity.google/docs/mcp?tab=cli) for details.

### Other MCP clients

Use `uv` as the command and the same argument list shown for Claude Desktop. This server exposes stdio, not an HTTP MCP endpoint: do not enter your RAGFlow URL as an MCP server URL.

## Verify your first retrieval

In your client, ask:

```text
Use ragflow_list_datasets and show the available dataset names.
```

Then substitute an actual dataset name and a question whose answer you know:

```text
Use ragflow_retrieval_by_name with dataset_names ["My Knowledge Base"],
query "What does the document say about our backup schedule?",
page_size 5, deepening_level 0, and use_rerank false.
Show the returned passages and their document references before summarizing.
```

A working connection returns dataset names, then relevant chunks from the selected dataset. Compare a returned passage with its source. If the passage is missing or irrelevant, check indexing and retrieval settings before enabling deepening. A fluent answer alone does not establish that retrieval worked.

For a team trial, have someone familiar with the dataset check a few known-answer queries. Share the queries and source checks with a colleague who uses the same material before adding more datasets.

## Search options and tools

`ragflow_retrieval_by_name` is the usual starting point. It takes `dataset_names` (a list) and `query`. Use `ragflow_retrieval` with `dataset_ids` when you want to avoid ambiguity in name matching.

| Option | Default | When to change it |
| --- | --- | --- |
| `document_name` | No filter | Restrict retrieval to a document; names match fuzzily. |
| `page_size` | `10` | Request fewer chunks for a quick check or more for broader research. |
| `page` | `1` | Fetch another page. |
| `similarity_threshold` | `0.2` | Lower it to admit more results; raise it to filter weak matches. |
| `top_k` | `1024` | Change the candidate pool used for vector search. |
| `use_rerank` | `false` | Enable after checking your RAGFlow reranker configuration. |
| `deepening_level` | `0` | Try up to 1, 2, or 3 refinement passes after configuring DSPy. |

To search several datasets, pass multiple names, for example `["Finance Reports", "Legal Documents"]`. Use datasets with compatible embedding models. Dataset lookup is case-insensitive and supports partial names; use exact names or IDs when several are similar.

For a document-specific query, first list documents, then use a specific `document_name`. Fuzzy matching can select a different document than you intended; inspect the response metadata for matches.

| Tool | Required inputs | Purpose |
| --- | --- | --- |
| `ragflow_list_datasets` | None | List datasets accessible with your API key. |
| `ragflow_list_documents` | `dataset_id` | List documents in a dataset. |
| `ragflow_list_documents_by_name` | `dataset_name` | List documents using a dataset name. |
| `ragflow_get_chunks` | `dataset_id`, `document_id` | Read chunks for a document. |
| `ragflow_retrieval` | `dataset_ids`, `query` | Retrieve chunks by dataset IDs. |
| `ragflow_retrieval_by_name` | `dataset_names`, `query` | Retrieve chunks by dataset names. |
| `ragflow_list_sessions` | None | Inspect locally tracked chat sessions. |
| `ragflow_reset_session` | `dataset_id` | Reset a tracked session. |

## Optional configuration

### Query deepening with DSPy

Deepening asks an additional LLM to analyze retrieved results and rewrite the query. It adds latency and provider usage charges. Leave `deepening_level=0` for ordinary retrieval.

For OpenAI, add these entries to your `config.json`:

```json
{
  "PROVIDER": "openai",
  "DSPY_MODEL": "openai/gpt-4o-mini",
  "OPENAI_API_KEY": "replace-with-your-openai-api-key"
}
```

For OpenRouter, use:

```json
{
  "PROVIDER": "openrouter",
  "DSPY_MODEL": "openai/gpt-4o-mini",
  "OPENROUTER_API_KEY": "replace-with-your-openrouter-api-key"
}
```

These are additions to the RAGFlow settings, not replacement files. Restart the MCP server after changing configuration. `OPENROUTER_SITE_URL` and `OPENROUTER_SITE_NAME` are optional attribution settings.

Start with `deepening_level=1` and compare its results with the original query. Levels 1 through 3 set the maximum refinement passes; the loop can stop early. The implementation returns the last result set, not a merged union of all searches. Metadata records the refinement process. Configuration or refinement failures can fall back to ordinary retrieval, so check metadata and logs if deepening appears to have no effect.

### Cloudflare Zero Trust

If RAGFlow is behind Cloudflare Access, add both `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` from an authorized service token. The server sends the corresponding Access headers with API requests. Leave both unset for a direct RAGFlow connection.

### Reranking

`RAGFLOW_DEFAULT_RERANK` selects the reranking model when `use_rerank=true`. This project has encountered `UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol` with reranking on RAGFlow. That observation does not establish the status of every RAGFlow version. Leave reranking off for the first retrieval; if you see this error, use `use_rerank=false` and check the provider URL configured in RAGFlow.

## Troubleshooting

| Symptom | Check next |
| --- | --- |
| Client cannot start the server | Run `uv sync --locked --python 3.12` in the checkout. Verify the absolute checkout path and that the client can find `uv`. |
| Server waits silently in a terminal | It speaks MCP over stdio. Let your client launch it; this is not an interactive shell or HTTP server. |
| Missing configuration | Confirm `config.json` is in the checkout, or set an absolute `RAGFLOW_CONFIG_PATH`. Check environment overrides. |
| Connection error or 401 | Check the RAGFlow base URL, network access, and API key. The base URL must include `http://` or `https://`. |
| Cloudflare 403 | Check both Access service-token values and the application's Access policy. |
| Dataset not found | Call `ragflow_list_datasets`, then copy an exact name or use its ID. |
| Empty or irrelevant results | Check that documents are indexed. Try a known phrase and a lower `similarity_threshold`; inspect the selected dataset/document. |
| Deepening has no effect | Check the provider key and response metadata. Ordinary retrieval can still succeed when deepening fails. |

For a startup check outside your client:

```sh
uv run --locked --directory "/absolute/path/to/ragflow-claude-desktop-local-mcp" ragflow-claude-mcp
```

Errors go to stderr. A running process waiting for MCP input is expected; stop it with Ctrl+C. Review logs before sharing them: queries, document excerpts, and configuration details can be sensitive.

## Limits and security

DSPy depends on DiskCache 5.6.3, which has an unresolved pickle-deserialization advisory, [GHSA-w8v5-vhqr-4h9v](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v). DSPy enables disk caching by default. Keep its cache directory private to the account running this server; do not share or import caches from untrusted sources. No patched DiskCache release was listed during the v0.2.1 audit.

The v0.2.1 local test run passed 28 tests with 49% coverage. Those checks use mocks; they do not certify your RAGFlow version, document relevance, or model-provider setup. See [CHANGELOG.md](CHANGELOG.md) for the maintenance changes and audit findings.

## Update or remove

To update a release checkout, fetch tags, check out the desired tag, run `uv sync --locked --python 3.12`, then restart your client. Keep your private `config.json`. To roll back, check out the previous tag and sync again; dependency security fixes may be lost on rollback.

To disconnect, remove the `ragflow` entry from Claude Desktop, run `codex mcp remove ragflow`, or run `agy mcp remove ragflow`. Restart the client. Removing the client entry does not delete your RAGFlow data.

## Development

```sh
git switch main
git pull --ff-only origin main
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked pytest --cov=src/ragflow_claude_mcp --cov-report=term-missing
uv build
```

The release workflow tests and builds through Dagger. Maintainers work directly on `main` and keep no long-lived feature branches. Use SSH-signed commits as described in [CONTRIBUTING.md](CONTRIBUTING.md) and [COMMIT_SIGNING.md](COMMIT_SIGNING.md).
