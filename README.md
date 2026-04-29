# RAGFlow Claude MCP Server

A small Model Context Protocol (MCP) server that hooks Claude Desktop (and other MCP clients) up to a [RAGFlow](https://github.com/infiniflow/ragflow) instance. It exposes the RAGFlow REST API as a handful of tools so the LLM can query knowledge bases and pull document chunks into its context.

This is personal-use software I wrote for my own R&D. It's not bug-free and the code is not pretty. It works for what I need.

## What it does

- Direct retrieval: pulls raw document chunks with similarity scores from RAGFlow's `/retrieval` endpoint.
- Multi-KB search: a single query can hit several knowledge bases at once.
- DSPy query deepening: optional iterative query refinement (uses an LLM to analyse intermediate results and rewrite the query).
- ~~Reranking~~ — currently broken on the RAGFlow side, see [Known issues](#known-issues).
- Tunable result control: `page_size`, `similarity_threshold`, `top_k`, pagination.
- Document filter: limit results to one document inside a dataset (fuzzy name matching).
- Dataset lookup by name (case-insensitive, fuzzy) instead of by ID.
- Cloudflare Zero Trust authentication when your RAGFlow sits behind it.

## Installation

1. Clone:
   ```bash
   git clone https://github.com/norandom/ragflow-claude-desktop-local-mcp
   cd ragflow-claude-desktop-local-mcp
   ```

2. Install:
   ```bash
   # On macOS, install DSPy first to dodge build issues:
   pip install git+https://github.com/stanfordnlp/dspy.git

   uv install
   ```

3. Configure: copy the sample and fill in your RAGFlow details.
   ```bash
   cp config.json.sample config.json
   ```
   Keys:
   - `RAGFLOW_BASE_URL`: e.g. `http://your-ragflow-server:9380`
   - `RAGFLOW_API_KEY`: your RAGFlow API key
   - `RAGFLOW_DEFAULT_RERANK`: rerank model (default `rerank-multilingual-v3.0`)
   - `CF_ACCESS_CLIENT_ID` *(optional)*: Cloudflare Zero Trust service-token ID
   - `CF_ACCESS_CLIENT_SECRET` *(optional)*: Cloudflare Zero Trust service-token secret
   - `DSPY_MODEL`: DSPy LM (default `openai/gpt-4o-mini`)
   - `OPENAI_API_KEY`: needed for DSPy deepening

### Cloudflare Zero Trust

If your RAGFlow is behind Cloudflare Zero Trust, grab a service token from the dashboard and add it to `config.json`:

```json
{
  "CF_ACCESS_CLIENT_ID": "your-client-id.access",
  "CF_ACCESS_CLIENT_SECRET": "your-client-secret"
}
```

When both are set, every API request goes out with the `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers. No code change needed.

## Claude Desktop config

```json
{
  "mcpServers": {
    "ragflow": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/ragflow-claude-desktop-local-mcp",
        "ragflow-claude-mcp"
      ]
    }
  }
}
```

## Tools

### `ragflow_retrieval_by_name` (the one I use most)

Retrieve chunks across one or more datasets by name. Returns raw chunks with similarity scores.

Params:
- `dataset_names` (required) — list, e.g. `["BASF", "Quant Literature"]`
- `query` (required)
- `document_name` (optional) — restrict to one document; fuzzy match
- `top_k` (optional, default 1024) — vector candidates
- `similarity_threshold` (optional, default 0.2) — 0.0–1.0
- `page` (optional, default 1)
- `page_size` (optional, default 10)
- `use_rerank` (optional, default false) — currently broken upstream, see Known issues
- `deepening_level` (optional, default 0) — DSPy refinement, 0–3

### `ragflow_retrieval`

Same shape, but takes `dataset_ids: List[str]` instead of names.

### Multi-KB search

You can search across several knowledge bases in one call. Make sure they share an embedding model — mixing incompatible embeddings will tank the relevance scores.

```
Use ragflow_retrieval_by_name with dataset_names ["Finance Reports", "Legal Documents"] and query "Summarize the key financial risks and compliance requirements for new market entry."
```

### `ragflow_list_datasets`

Lists every knowledge base on your RAGFlow instance. No params. Walks all pages internally.

### `ragflow_list_documents`

Lists documents in a dataset. Walks all pages.
- `dataset_id` (required)

### `ragflow_get_chunks`

Returns chunks (with references) for one document.
- `dataset_id` (required)
- `document_id` (required)

### `ragflow_list_sessions`

Shows active chat sessions per dataset. No params.

### `ragflow_list_documents_by_name`

Lists documents in a dataset, looked up by name.
- `dataset_name` (required)

### `ragflow_reset_session`

Drops the chat session for a dataset.
- `dataset_id` (required)

## Tuning the retrieval

The retrieval tools take three knobs:

- `page_size` — chunks per page (default 10).
- `similarity_threshold` — drops chunks below this score (default 0.2).
- `top_k` — pool size for the vector search before filtering (default 1024).

Some starting points that work for me:

- Broader recall: `page_size=15`, `similarity_threshold=0.15`.
- Tight precision: `page_size=5`, `similarity_threshold=0.4`.
- Heavy research: `page_size=20`, `similarity_threshold=0.1`, `deepening_level=1`.
- Hard queries: `deepening_level=2`.
- Speed: keep `deepening_level=0` and skip rerank.

## Examples

Basic retrieval by name:

```
Use ragflow_retrieval_by_name with dataset_names ["BASF"] and query "What is BASF's latest income statement? Revenue, operating income, net income, and other key figures."
```

Restrict to one document:

```
Use ragflow_retrieval_by_name with dataset_names ["BASF"], document_name "annual_report_2023", and query "What were the key financial highlights for 2023?"
```

Document names match fuzzily — `"annual"` will hit `annual_report_2023.pdf` and `annual_report_2024.pdf`. When several match, the server picks the most recent and lists the alternatives in the response metadata.

DSPy deepening for a tricky query:

```
Use ragflow_retrieval_by_name with dataset_names ["Quant Literature"], query "what is a volatility clock", deepening_level 2.
```

Multi-page:

```
Use ragflow_retrieval_by_name with dataset_names ["BASF"], query "BASF business segments", page_size 10, page 2.
```

List what's available:

```
Use ragflow_list_datasets.
```

```
Use ragflow_list_documents_by_name with dataset_name "BASF".
```

Pull specific chunks:

```
Use ragflow_get_chunks with dataset_id "43066ee0599411f089787a39c10de57b" and document_id "d74a1c105a3311f09fc94a0fcd8b7722".
```

## Bigger prompts

Some examples of how I drive it from Claude Desktop.

Financial deep-dive:

```
Help me analyse BASF's recent financials.

1. Use ragflow_retrieval_by_name to search ["BASF"] for the latest income statement
   (revenue, operating income, net income). Use page_size 15,
   similarity_threshold 0.15, deepening_level 1.

2. Then run ragflow_retrieval_by_name again for the cash flow statement,
   page_size 10, similarity_threshold 0.2.

3. Finally look for year-over-year changes with page_size 12,
   similarity_threshold 0.18.
```

Multilingual research:

```
Use ragflow_retrieval_by_name with dataset_names ["BASF"],
query "Was sind die wichtigsten Geschäftsbereiche von BASF?",
deepening_level 2.
```

DSPy detects the query language and refines accordingly. I've used this for German, English, and mixed-language queries. It works as long as the underlying documents have content in those languages.

Document-filtered research:

```
1. Use ragflow_list_documents_by_name with dataset_name "BASF" to see what's in there.
2. Use ragflow_retrieval_by_name with dataset_names ["BASF"],
   document_name "sustainability_report", query "carbon neutrality goals",
   page_size 15, deepening_level 1.
3. Follow up with document_name "annual_report_2023" and
   query "environmental investments".
```

Cross-KB query:

```
Use ragflow_retrieval_by_name with dataset_names ["BASF", "Industry Reports"],
query "chemical industry sustainability benchmarks",
page_size 12, deepening_level 1.
```

## How DSPy deepening works

`deepening_level` runs an LLM-driven refinement loop on top of the retrieval:

- 0: no deepening (default).
- 1: one refinement pass.
- 2: two passes with gap analysis.
- 3: three+ passes plus result merging.

Each pass: do the search, summarise the top results, ask the LLM what's missing, generate a new query, run that. The response metadata includes the original query, every refined query, and the reasoning at each step.

DSPy needs:
- `DSPY_MODEL` — `openai/gpt-4o-mini` works fine
- `OPENAI_API_KEY`

## Reranking (currently broken)

When working, reranking replaces the vector cosine score with the rerank model's score (typically 10–30% better relevance in my experience). RAGFlow has a known bug right now where `use_rerank=true` produces:

> UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol

So leave `use_rerank=false` until the upstream issue is fixed. Standard vector retrieval works normally.

## How dataset lookup works

- Case-insensitive name matching.
- Fuzzy match for partial names.
- Datasets are cached for name lookup; cache misses trigger a refresh.
- If lookup fails, the error includes the available dataset names so you know what was actually there.

## Document matching

When you pass `document_name`:

- Exact match wins, then "starts with", then "contains", then partial.
- Among ties, the more recently updated document wins.
- Names containing `2024`, `2023`, `latest`, `current`, or `new` get a small score bonus.
- All matches are returned in the response metadata so you can re-issue with a more specific name.

## Error handling

Reasonable error messages for: API errors, missing datasets, unreachable RAGFlow, broken sessions, invalid input, and config problems. Sensitive values are redacted in logs.

## Environment variables

- `RAGFLOW_BASE_URL` — overrides the config file. Default in code: `http://192.168.122.93:9380` (which is my local instance).
- `RAGFLOW_API_KEY` — required.

## Development

Run the server directly:

```bash
uv run ragflow-claude-mcp
```

It listens on stdio, the way MCP servers do.

Dev deps:

```bash
uv install --extra dev
```

That gets pytest + the asyncio/mock/cov plugins.

Tests:

```bash
uv run pytest
uv run pytest --cov=src --cov-report=html --cov-report=term
uv run pytest tests/test_server.py
uv run pytest -v
```

Coverage is around 44% with 22/23 tests passing (one is skipped because of an intermittent CI flake). Tests cover server init, RAGFlow API integration, DSPy deepening, OpenAI/OpenRouter config branches, and config loading.

## Implementation notes

The retrieval API is the only RAGFlow surface the server actually relies on. No assistant/chat dependencies, no server-side prompt config — just chunks back. Easier to reason about, easier to debug.

## Troubleshooting

- "Dataset not found": run `ragflow_list_datasets` to see what's actually there.
- Connection errors: double-check `RAGFLOW_BASE_URL` and `RAGFLOW_API_KEY`.
- Server won't start: did `uv install` actually finish?
- Need raw chunks: that's `ragflow_retrieval_by_name` / `ragflow_retrieval`.
- Stuck session: `ragflow_list_sessions` then `ragflow_reset_session`.
- Cloudflare 403s: confirm `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` match an active service token on the Zero Trust app.

## Known issues

### Rerank is broken upstream

`use_rerank=true` errors out with `UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol`. This is a RAGFlow-side defect. Workaround: leave it off. I'm watching the RAGFlow repo for a fix.

## Contributing

PRs only — `main` is protected. Commits must be SSH-signed.

1. Fork.
2. `git checkout -b feature/your-thing`.
3. Make the change, write a clear commit message.
4. Push to your fork.
5. Open a PR against `main`.

PRs run TruffleHog automatically — don't include keys, tokens, or secrets. See [CONTRIBUTING.md](CONTRIBUTING.md) for the longer version.
