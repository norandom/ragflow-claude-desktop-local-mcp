# RAGFlow Claude MCP Server

A Model Context Protocol (MCP) server that provides seamless integration between Claude Desktop and RAGFlow's REST API for knowledge base querying and document management.

## Features

- **Enhanced Result Control**: Fine-tuned control over query results with `top_n` (default: 10) and `similarity_threshold` (default: 0.2) parameters
- **Cross-Language Queries**: Built-in support for multilingual queries with English and German as defaults
- **Context7 Integration**: Leverages Context7 for up-to-date documentation and cross-language searches
- **Automatic Session Management**: Sessions are automatically created and reused per dataset
- **Dataset Name Lookup**: Query knowledge bases using familiar names instead of cryptic IDs
- **Fuzzy Matching**: Find datasets with partial name matches (case-insensitive)
- **Session Persistence**: Conversation context is maintained across queries
- **Enhanced Error Handling**: Clear error messages and dataset suggestions
- **Multiple Query Methods**: Support for both ID-based and name-based queries
- **Improved Query Quality**: Increased default result limit from 1 to 10 chunks for better responses

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

3. Configure the server:
   Create a `config.json` file by copying the `config.json.sample` file:
   ```bash
   cp config.json.sample config.json
   ```
   Then, edit `config.json` with your RAGFlow server details:
   - `RAGFLOW_BASE_URL`: The URL of your RAGFlow instance (e.g., "http://your-ragflow-server:port").
   - `RAGFLOW_API_KEY`: Your RAGFlow API key.

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
      ]
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
- `top_n` (optional): Number of top chunks above similarity threshold to feed to LLM. Defaults to 10.
- `similarity_threshold` (optional): Minimum similarity score for chunks (0.0 to 1.0). Defaults to 0.2.

### 2. `ragflow_query`
Query RAGFlow knowledge bases using dataset IDs (for advanced users).

**Parameters:**
- `dataset_id` (required): Unique identifier of the dataset
- `query` (required): Your question or search query
- `session_name` (optional): Custom session name for organization
- `languages` (optional): Array of language codes to search in (e.g., ["en", "de", "es"]). Defaults to ["en", "de"]
- `top_n` (optional): Number of top chunks above similarity threshold to feed to LLM. Defaults to 10.
- `similarity_threshold` (optional): Minimum similarity score for chunks (0.0 to 1.0). Defaults to 0.2.

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

## Result Control and Filtering

### Enhanced Query Parameters
The server now supports fine-tuned control over query results:

- **`top_n`**: Controls the number of document chunks fed to the LLM (default: 10, previously 1)
- **`similarity_threshold`**: Filters chunks based on relevance score (default: 0.2, range: 0.0-1.0)

### Result Optimization Tips
- **For broader results**: Use `top_n=15` and `similarity_threshold=0.15`
- **For precise results**: Use `top_n=5` and `similarity_threshold=0.4`
- **For comprehensive analysis**: Use `top_n=20` and `similarity_threshold=0.1`

## Usage Examples

### Basic Query by Dataset Name

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "basf-financial-analysis", and query "What is BASF's latest income statement? Please provide the revenue, operating income, net income, and other key financial figures."
```

### Enhanced Query with Custom Result Limits

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "comprehensive-analysis", query "Analyze BASF's financial performance and business strategy", top_n 15, and similarity_threshold 0.15 for comprehensive results.
```

### Precise Query with High Similarity Threshold

```
Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "precise-search", query "What is BASF's exact revenue for Q4 2023?", top_n 5, and similarity_threshold 0.4 for highly relevant results only.
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

## Sample Prompts for Claude Desktop

### Comprehensive Financial Analysis
```
I need to analyze BASF's financial performance. Please help me by:

1. First, use the ragflow_query_by_name tool to search the "BASF" knowledge base for their latest income statement. Ask for revenue, operating income, net income, and key financial figures. Use top_n 15 and similarity_threshold 0.15 for comprehensive results.

2. Then, ask a follow-up question about their cash flow statement using the same dataset with top_n 10 and similarity_threshold 0.2.

3. Finally, inquire about any significant changes in their financial performance compared to the previous year using top_n 12 and similarity_threshold 0.18.

Please use session_name "basf-financial-analysis" for better organization.
```

### Precision Search Query
```
I need to find specific financial metrics for BASF. Please use the ragflow_query_by_name tool with dataset_name "BASF", session_name "precision-search", query "What were BASF's exact Q4 2023 revenue figures and operating margin?", top_n 5, and similarity_threshold 0.45 to get only the most relevant and precise results.
```

### Multi-Language Research
```
Please conduct a comprehensive multilingual search about BASF's sustainability initiatives. Use ragflow_query_by_name with dataset_name "BASF", session_name "sustainability-research", query "Find information about BASF's environmental sustainability programs, carbon reduction targets, and green chemistry initiatives", languages ["en", "de", "fr"], top_n 20, and similarity_threshold 0.12 for broad coverage across languages.
```

### Comparative Analysis
```
I need to compare BASF's performance across different business segments. Please use ragflow_query_by_name with dataset_name "BASF", session_name "segment-analysis", query "Compare the financial performance of BASF's different business segments including Chemicals, Materials, Industrial Solutions, Surface Technologies, Nutrition & Health, and Agricultural Solutions", top_n 18, and similarity_threshold 0.16 for comprehensive segment data.
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

### Result Control and Optimization
- **Enhanced Query Performance**: Default `top_n` increased from 1 to 10 chunks for more comprehensive responses
- **Similarity Filtering**: `similarity_threshold` parameter filters chunks based on relevance scores (0.0-1.0)
- **Customizable Result Limits**: Both parameters can be adjusted per query for optimal results
- **Backward Compatibility**: All existing queries continue to work with improved defaults

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
