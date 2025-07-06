# RAGFlow Claude MCP Server

A Model Context Protocol (MCP) server that provides seamless integration between Claude Desktop and RAGFlow's REST API for knowledge base querying and document management.

## Features

- **Cross-Language Queries**: Built-in support for multilingual queries with English and German as defaults
- **Context7 Integration**: Leverages Context7 for up-to-date documentation and cross-language searches
- **Automatic Session Management**: Sessions are automatically created and reused per dataset
- **Dataset Name Lookup**: Query knowledge bases using familiar names instead of cryptic IDs
- **Fuzzy Matching**: Find datasets with partial name matches (case-insensitive)
- **Session Persistence**: Conversation context is maintained across queries
- **Enhanced Error Handling**: Clear error messages and dataset suggestions
- **Multiple Query Methods**: Support for both ID-based and name-based queries

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/norandom/ragflow-claude-desktop-local-mcp
   cd ragflow-claude-desktop-local-mcp
   ```

2. Install dependencies:
   ```bash
   uv install
   ```

3. Configure environment variables:
   ```bash
   export RAGFLOW_BASE_URL="http://your-ragflow-server:port"
   export RAGFLOW_API_KEY="your-api-key"
   ```

## Claude Desktop Configuration

Add the following to your Claude Desktop MCP configuration:

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
      ],
      "env": {
        "RAGFLOW_BASE_URL": "http://192.168.122.93:9380",
        "RAGFLOW_API_KEY": "your-ragflow-api-key"
      }
    }
  }
}
```

## Available Tools

### 1. `ragflow_query_by_name`
Query RAGFlow knowledge bases using dataset names (recommended for most users).

**Parameters:**
- `dataset_name` (required): Name of the knowledge base (e.g., "BASF", "Company Reports")
- `query` (required): Your question or search query
- `session_name` (optional): Custom session name for organization
- `languages` (optional): Array of language codes to search in (e.g., ["en", "de", "es"]). Defaults to ["en", "de"]

### 2. `ragflow_query`
Query RAGFlow knowledge bases using dataset IDs (for advanced users).

**Parameters:**
- `dataset_id` (required): Unique identifier of the dataset
- `query` (required): Your question or search query
- `session_name` (optional): Custom session name for organization
- `languages` (optional): Array of language codes to search in (e.g., ["en", "de", "es"]). Defaults to ["en", "de"]

### 3. `ragflow_list_datasets`
List all available knowledge bases in your RAGFlow instance.

**Parameters:** None

### 4. `ragflow_list_documents`
List documents within a specific dataset.

**Parameters:**
- `dataset_id` (required): ID of the dataset

### 5. `ragflow_get_chunks`
Get document chunks with references from a specific document.

**Parameters:**
- `dataset_id` (required): ID of the dataset
- `document_id` (required): ID of the document

### 6. `ragflow_list_sessions`
Show active chat sessions for all datasets.

**Parameters:** None

### 7. `ragflow_reset_session`
Reset/clear the chat session for a specific dataset when encountering session ownership issues.

**Parameters:**
- `dataset_id` (required): ID of the dataset to reset session for

## Usage Examples

### Basic Query by Dataset Name

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "basf-financial-analysis", and query "What is BASF's latest income statement? Please provide the revenue, operating income, net income, and other key financial figures."
```

### Cross-Language Query

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "multilingual-analysis", query "Was sind die Hauptgeschäftsbereiche von BASF?", and languages ["en", "de", "es"].
```

### Query with Custom Session Name

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "financial-analysis", and query "What were BASF's key financial metrics for the last fiscal year?"
```

### List Available Datasets

```
Please use the ragflow_list_datasets tool to show me all available knowledge bases.
```

### Follow-up Questions

Once you've queried a dataset, follow-up questions automatically use the same session:

```
Please use the ragflow_query_by_name tool with dataset_name "BASF" and query "What about their cash flow statement?"
```

### Reset Session (if encountering ownership issues)

```
Please use the ragflow_reset_session tool with dataset_id "43066ee0599411f089787a39c10de57b" to clear any problematic sessions, then retry your query.
```

## Sample Prompt for Claude Desktop

Here's a comprehensive prompt you can use in Claude Desktop:

```
I need to analyze BASF's financial performance. Please help me by:

1. First, use the ragflow_query_by_name tool to search the "BASF" knowledge base for their latest income statement. Ask for revenue, operating income, net income, and key financial figures.

2. Then, ask a follow-up question about their cash flow statement using the same dataset.

3. Finally, inquire about any significant changes in their financial performance compared to the previous year.

Please use session_name "basf-financial-analysis" for better organization.
```

### Enhanced Single Query Prompt

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "basf-financial-analysis", and query "What is BASF's latest income statement? Please provide detailed information about revenue, operating income, net income, gross margin, operating margin, and any other key financial metrics."
```

### Multilingual Research Query

```
Please use the ragflow_query_by_name tool with dataset_name "Company Research", session_name "multilingual-search", query "Find information about renewable energy investments and sustainability initiatives", and languages ["en", "de", "fr", "es"] to search across multiple languages and provide comprehensive results.
```

## Technical Details

### Cross-Language Features
- **Default Languages**: English (`en`) and German (`de`) are set as defaults
- **Context7 Integration**: Leverages Context7 for up-to-date multilingual documentation
- **Enhanced Prompts**: Queries automatically include instructions for multilingual search
- **Language Customization**: Optional `languages` parameter accepts custom language codes
- **Translation Support**: Provides translations and summaries when content is found in different languages

### Session Management
- Each dataset automatically gets a unique session created via RAGFlow's chat API
- Sessions are properly registered with RAGFlow to avoid ownership issues
- Automatic session recovery if ownership errors occur
- Sessions are reused for subsequent queries to maintain conversation context
- Session format: `{session_name}-{timestamp}` or `ragflow-session-{timestamp}`

### Dataset Lookup
- Case-insensitive name matching
- Fuzzy matching for partial names
- Automatic caching of dataset information
- Clear error messages with available dataset suggestions

### Error Handling
- Comprehensive error messages for API failures
- Dataset availability checks
- Automatic session ownership recovery
- Session reset capability for problematic sessions
- Network connectivity issues
- Graceful fallback for session creation failures

## Environment Variables

- `RAGFLOW_BASE_URL`: Base URL of your RAGFlow instance (default: http://192.168.122.93:9380)
- `RAGFLOW_API_KEY`: Your RAGFlow API key (required)

## Development

To run the server directly:

```bash
uv run ragflow-claude-mcp
```

The server will start and listen for MCP requests via stdio.

## Troubleshooting

1. **"Dataset not found" errors**: Use `ragflow_list_datasets` to see available datasets
2. **"You don't own the chat" errors**: The server now automatically handles this by creating new sessions and retrying. If issues persist, use `ragflow_reset_session` to manually clear problematic sessions.
3. **Session ownership issues**: Use `ragflow_reset_session` with the dataset ID to clear the session, then retry your query
4. **Connection errors**: Verify your `RAGFLOW_BASE_URL` and `RAGFLOW_API_KEY` are correct
5. **Server won't start**: Check that all dependencies are installed with `uv install`
6. **Persistent session errors**: Check `ragflow_list_sessions` to see active sessions and reset individual datasets as needed
