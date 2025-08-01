# RAGFlow Claude MCP Server

A Model Context Protocol (MCP) server that provides seamless integration between tools like Claude Desktop and RAGFlow's REST API for knowledge base querying and document management.
It enriches the context of the LLMs. 

This is a persoal-use software, which I create for my own RnD. It's not bug-free, and certainly not high quality code. But it gets the job done. :) 

## Features

- **Direct Document Retrieval**: Access raw document chunks with similarity scores using RAGFlow's retrieval API
- **DSPy Query Deepening**: Intelligent query refinement using DSPy for iterative search improvement
~~- **Reranking Support**: Optional reranking for improved result quality (configurable on/off)~~ defect for now, working on it
- **Enhanced Result Control**: Default 10 results per query with configurable `page_size` and `similarity_threshold` parameters
- **Document Filtering**: Limit search results to specific documents within a dataset (supports fuzzy matching)
- **Dataset Name Lookup**: Query knowledge bases using familiar names instead of cryptic IDs
- **Fuzzy Matching**: Find datasets with partial name matches (case-insensitive)
- **Pagination Support**: Retrieve results in manageable batches with full pagination control
- **Source References**: Each chunk includes document ID, similarity scores, and highlighted matches
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
   # On macOS, install DSPy first to avoid build issues:
   pip install git+https://github.com/stanfordnlp/dspy.git
   
   # Then install all dependencies
   uv install
   ```

3. Configure the server:
   Create a `config.json` file by copying the `config.json.sample` file:
   ```bash
   cp config.json.sample config.json
   ```
   Then, edit `config.json` with your server details:
   - `RAGFLOW_BASE_URL`: The URL of your RAGFlow instance (e.g., "http://your-ragflow-server:port").
   - `RAGFLOW_API_KEY`: Your RAGFlow API key.
   - `RAGFLOW_DEFAULT_RERANK`: Default rerank model (default: "rerank-multilingual-v3.0").
   - `CF_ACCESS_CLIENT_ID`: (Optional) Cloudflare Zero Trust Access Client ID.
   - `CF_ACCESS_CLIENT_SECRET`: (Optional) Cloudflare Zero Trust Access Client Secret.
   - `DSPY_MODEL`: DSPy language model for query refinement (default: "openai/gpt-4o-mini").
   - `OPENAI_API_KEY`: OpenAI API key (required for DSPy query deepening).

### Cloudflare Zero Trust Configuration (Optional)

If your RAGFlow instance is protected by Cloudflare Zero Trust, you can configure authentication:

1. Obtain your Service Token credentials from your Cloudflare Zero Trust dashboard
2. Add the following to your `config.json`:
   ```json
   {
     "CF_ACCESS_CLIENT_ID": "your-client-id.access",
     "CF_ACCESS_CLIENT_SECRET": "your-client-secret",
     ...
   }
   ```

The server will automatically include the necessary `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers in all API requests when these credentials are configured.

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

### 1. `ragflow_retrieval_by_name` ⭐ **Recommended**
Retrieve document chunks by dataset name using the retrieval API. Returns raw chunks with similarity scores.

**Parameters:**
- `dataset_name` (required): Name of the dataset/knowledge base to search (e.g., "BASF")
- `query` (required): Search query or question
- `document_name` (optional): Name of document to filter results to specific document (supports partial matching)
- `top_k` (optional): Number of chunks for vector cosine computation. Defaults to 1024.
- `similarity_threshold` (optional): Minimum similarity score for chunks (0.0 to 1.0). Defaults to 0.2.
- `page` (optional): Page number for pagination. Defaults to 1.
- `page_size` (optional): Number of chunks per page. Defaults to 10.
- `use_rerank` (optional): Enable reranking for improved quality. Defaults to false. 
- `deepening_level` (optional): DSPy query refinement level (0-3). Defaults to 0.

### 2. `ragflow_retrieval`
Retrieve document chunks directly from RAGFlow datasets using the retrieval API. Returns raw chunks with similarity scores.

**Parameters:**
- `dataset_id` (required): ID of the dataset/knowledge base to search
- `query` (required): Search query or question
- `document_name` (optional): Name of document to filter results to specific document (supports partial matching)
- `top_k` (optional): Number of chunks for vector cosine computation. Defaults to 1024.
- `similarity_threshold` (optional): Minimum similarity score for chunks (0.0 to 1.0). Defaults to 0.2.
- `page` (optional): Page number for pagination. Defaults to 1.
- `page_size` (optional): Number of chunks per page. Defaults to 10.
- `use_rerank` (optional): Enable reranking for improved quality. Defaults to false.
- `deepening_level` (optional): DSPy query refinement level (0-3). Defaults to 0.

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

### 7. `ragflow_list_documents_by_name`
List documents in a dataset by dataset name.

**Parameters:**
- `dataset_name` (required): Name of the dataset/knowledge base to list documents from

### 8. `ragflow_reset_session`
Reset/clear the chat session for a specific dataset.

**Parameters:**
- `dataset_id` (required): ID of the dataset to reset session for

## Result Control and Filtering

### Enhanced Retrieval Parameters
The retrieval tools support fine-tuned control over query results:

- **`page_size`**: Number of chunks returned per page (default: 10, previously 30)
- **`similarity_threshold`**: Filters chunks based on relevance score (default: 0.2, range: 0.0-1.0)
- **`top_k`**: Number of chunks for vector computation (default: 1024)

### Result Optimization Tips
- **For broader results**: Use `page_size=15`, `similarity_threshold=0.15`, and `use_rerank=true`
- **For precise results**: Use `page_size=5`, `similarity_threshold=0.4`, and `use_rerank=true`
- **For comprehensive analysis**: Use `page_size=20`, `similarity_threshold=0.1`, `use_rerank=true`, and `deepening_level=1`
- **For complex queries**: Use `deepening_level=2` with `use_rerank=true` for intelligent refinement
- **For maximum quality**: Use `deepening_level=2`, `use_rerank=true`, and `similarity_threshold=0.3`
- **For speed**: Keep `use_rerank=false` and `deepening_level=0` (default behavior)

## Usage Examples

### Basic Retrieval by Dataset Name

```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF" and query "What is BASF's latest income statement? Please provide the revenue, operating income, net income, and other key financial figures."
```

### Document-Specific Search

```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", document_name "annual_report_2023", and query "What were the key financial highlights for 2023?" to search only within the specific annual report document.
```

**Note**: Document names support fuzzy matching - you can use partial names like "annual" to match "annual_report_2023.pdf". When multiple documents match (e.g., "annual" matches both "annual_report_2023.pdf" and "annual_report_2024.pdf"), the system uses the most recent document by default and provides information about all matches in the response metadata for user choice.

### Enhanced Retrieval with Reranking (defect)

```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "Analyze BASF's financial performance and business strategy", page_size 15, similarity_threshold 0.15, and use_rerank true for higher quality results.
```

### DSPy Query Deepening for Complex Queries

```
Please use the ragflow_retrieval_by_name tool with dataset_name "Quant Literature", query "what is a volatility clock", deepening_level 2, and use_rerank false for intelligent query refinement and best quality results.
```

### Precise Retrieval with High Similarity Threshold

```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "What is BASF's exact revenue for Q4 2023?", page_size 5, similarity_threshold 0.4, and use_rerank true for highly relevant results only.
```

### Multi-Page Retrieval

```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "BASF business segments", page_size 10, and page 2 to get the next set of results.
```

### List Available Datasets

```
Please use the ragflow_list_datasets tool to show me all available knowledge bases.
```

### Get Document Details

```
Please use the ragflow_list_documents tool with dataset_id "43066ee0599411f089787a39c10de57b" to see what BASF documents are available.
```

### List Documents by Dataset Name

```
Please use the ragflow_list_documents_by_name tool with dataset_name "BASF" to see all available documents in the BASF knowledge base.
```

### Get Specific Document Chunks

```
Please use the ragflow_get_chunks tool with dataset_id "43066ee0599411f089787a39c10de57b" and document_id "d74a1c105a3311f09fc94a0fcd8b7722" to get chunks from the BASF annual report.
```

## Sample Prompts for Claude Desktop

### Comprehensive Financial Analysis with Enhanced Features
```
I need to analyze BASF's financial performance. Please help me by:

1. First, use the ragflow_retrieval_by_name tool to search the "BASF" knowledge base for their latest income statement. Ask for revenue, operating income, net income, and key financial figures. Use page_size 15, similarity_threshold 0.15, use_rerank true, and deepening_level 1 for comprehensive and refined results.

2. Then, use the ragflow_retrieval_by_name tool again with query about their cash flow statement, page_size 10, similarity_threshold 0.2, and use_rerank true.

3. Finally, use the ragflow_retrieval_by_name tool to find information about significant changes in their financial performance compared to the previous year using page_size 12, similarity_threshold 0.18, and use_rerank true.
```

### Advanced Query with DSPy Deepening
```
I need to understand complex financial concepts. Please use the ragflow_retrieval_by_name tool with dataset_name "finance_kb", query "what is a volatility clock", deepening_level 2, use_rerank true, and page_size 10. This will use intelligent query refinement to find better results about this complex topic.
```

### Precision Search with Maximum Quality
```
I need to find specific financial metrics for BASF. Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "What were BASF's exact Q4 2023 revenue figures and operating margin?", page_size 5, similarity_threshold 0.45, and use_rerank true to get only the most relevant and precise results with highest quality ranking.
```

### Multi-Page Comprehensive Research with Reranking
```
Please conduct comprehensive research about BASF's sustainability initiatives. Use ragflow_retrieval_by_name with dataset_name "BASF", query "Find information about BASF's environmental sustainability programs, carbon reduction targets, and green chemistry initiatives", page_size 20, similarity_threshold 0.12, use_rerank true, and deepening_level 1 for broad coverage with intelligent refinement. If needed, use page 2 and page 3 for additional results.
```

### Comparative Analysis with Enhanced Quality
```
I need to compare BASF's performance across different business segments. Please use ragflow_retrieval_by_name with dataset_name "BASF", query "Compare the financial performance of BASF's different business segments including Chemicals, Materials, Industrial Solutions, Surface Technologies, Nutrition & Health, and Agricultural Solutions", page_size 18, similarity_threshold 0.16, use_rerank true, and deepening_level 1 for comprehensive segment data with enhanced quality.
```

### Enhanced Single Query with All Features
```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "What is BASF's latest income statement? Please provide detailed information about revenue, operating income, net income, gross margin, operating margin, and any other key financial metrics.", page_size 15, similarity_threshold 0.2, use_rerank true, and deepening_level 1 for the best possible results with intelligent query refinement and reranking.
```

### Document Exploration Workflow

```
Please help me explore the BASF dataset by:
1. First using ragflow_list_documents_by_name with dataset_name "BASF" to see available documents
2. Then using ragflow_retrieval_by_name with dataset_name "BASF", document_name "sustainability_report_2023", query "carbon reduction targets", page_size 10
3. Finally using ragflow_get_chunks for specific document analysis if needed
```

### Document-Filtered Research with Advanced Features

```
Please conduct targeted research on BASF's sustainability initiatives using document filtering:
1. First, use ragflow_list_documents_by_name with dataset_name "BASF" to see available documents
2. Then, use ragflow_retrieval_by_name with dataset_name "BASF", document_name "sustainability_report", query "carbon neutrality goals and timeline", page_size 15, similarity_threshold 0.2, use_rerank true, deepening_level 1
3. Follow up with ragflow_retrieval_by_name using document_name "annual_report_2023" and query "environmental investments and green chemistry initiatives"
```

### Handling Multiple Document Matches

```
When using document_name "annual", the system might find multiple matches:
- "annual_report_2024.pdf" (most recent, used by default)
- "annual_report_2023.pdf" 
- "annual_financial_summary_2023.pdf"

The response metadata will show all matches. To target a specific document, use more specific names like "annual_report_2023" or check the response metadata for exact document names.
```

## Technical Details

### DSPy Query Deepening
- **Intelligent Query Refinement**: Uses DSPy to analyze search results and generate improved queries
- **Iterative Improvement**: Deepening levels 1-3 perform multiple refinement cycles
- **Gap Analysis**: Identifies missing information in initial results and targets specific improvements
- **Query Evolution**: Tracks original query → refined queries → final results with full metadata
- **Multilingual Support**: Automatically handles queries in German, English, and other languages

### Multilingual Queries

The DSPy query deepening system supports multilingual queries and will intelligently handle different languages during refinement:

#### English Queries
```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "What are BASF's main business segments?", deepening_level 2, use_rerank false
```

#### German Queries
```
Please use the ragflow_retrieval_by_name tool with dataset_name "BASF", query "Was sind die wichtigsten Geschäftsbereiche von BASF?", deepening_level 2, use_rerank false
```

#### Mixed Language Research
```
Please use the ragflow_retrieval_by_name tool with dataset_name "Finance KB", query "Volatility clustering und Markteffizienz", deepening_level 2, use_rerank false
```

**Key Features:**
- **Automatic Language Detection**: DSPy automatically detects the query language
- **Intelligent Translation**: When needed, queries are refined in the most appropriate language
- **Cross-Language Results**: Can find relevant content regardless of document language
- **Enhanced with Deepening**: Use `deepening_level 1-3` for best multilingual performance

**Requirements:**
- `DSPY_MODEL`: "openai/gpt-4o-mini" (configured in `config.json`)
- `OPENAI_API_KEY`: Required for multilingual DSPy functionality

### Reranking Support
- **Optional Enhancement**: Reranking disabled by default for speed, enabled via `use_rerank: true`
- **Server-Side Configuration**: Uses `rerank-multilingual-v3.0` model configured in `config.json`
- **Quality Improvement**: Typically 10-30% better relevance scores when enabled
- **Performance Trade-off**: Significantly increases response time but improves result quality

### Result Control and Optimization
- **Enhanced Retrieval Performance**: Default `page_size` set to 10 chunks for optimal response size
- **Similarity Filtering**: `similarity_threshold` parameter filters chunks based on relevance scores (0.0-1.0)
- **Pagination Support**: Use `page` parameter to retrieve additional results beyond the first 10
- **Vector Search Control**: `top_k` parameter controls the number of chunks for vector computation (default: 1024)

### Retrieval Features  
- **Direct Document Access**: Raw document chunks with exact text from source files
- **Similarity Scoring**: Each chunk includes relevance scores for quality assessment
- **Source References**: Full document and chunk location information provided
- **Flexible Pagination**: Retrieve results in manageable batches with full control

### Session Management
- Session management tools are available for workflow compatibility
- `ragflow_list_sessions` shows active chat sessions  
- `ragflow_reset_session` clears problematic sessions
- **Note:** Retrieval tools (`ragflow_retrieval` and `ragflow_retrieval_by_name`) don't require session management

### Dataset Lookup
- Case-insensitive name matching
- Fuzzy matching for partial names
- Automatic caching of dataset information
- Clear error messages with available dataset suggestions

### Document Matching & Ranking
- **Intelligent Ranking**: Exact matches > starts with > contains > partial matches
- **Recency Priority**: Most recent documents (by update time) ranked higher
- **Multiple Match Handling**: When multiple documents match, uses best-ranked document and provides all options in response metadata
- **Smart Keywords**: Documents with "2024", "2023", "latest", "current", "new" get ranking boost
- **User Choice**: Response metadata includes all matching documents for user selection

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

### Running the Server

To run the server directly:

```bash
uv run ragflow-claude-mcp
```

The server will start and listen for MCP requests via stdio.

### Development Dependencies

For development and testing, install the optional development dependencies:

```bash
uv install --extra dev
```

This includes:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support  
- `pytest-mock` - Mocking utilities
- `pytest-cov` - Coverage reporting

### Running Tests

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
uv run pytest tests/test_server.py

# Run with verbose output
uv run pytest -v
```

### Test Coverage

The project maintains comprehensive test coverage for:
- Server initialization and configuration
- RAGFlow API integration
- DSPy query deepening functionality  
- OpenRouter and OpenAI configuration
- Error handling and edge cases
- Configuration file loading

Current coverage: **44%** with **22/23 tests passing** (1 test skipped due to intermittent CI issues).

### Code Quality

The test suite includes:
- Unit tests for core functionality
- Integration tests for external API calls
- Configuration validation tests
- Mock-based testing for external dependencies
- Async test support for asynchronous operations

## Implementation Notes

### RAGFlow API Approach
- **Primary Tools**: Use `ragflow_retrieval_by_name` and `ragflow_retrieval` for document search
- **Direct Access**: Retrieval API provides raw document chunks without LLM processing
- **Better Control**: Full pagination and similarity filtering control
- **No Dependencies**: Works independently of server-side configurations

## Troubleshooting

1. **"Dataset not found" errors**: Use `ragflow_list_datasets` to see available datasets
2. **Connection errors**: Verify your `RAGFLOW_BASE_URL` and `RAGFLOW_API_KEY` are correct
3. **Server won't start**: Check that all dependencies are installed with `uv install`
4. **Need raw document access**: Use `ragflow_retrieval_by_name` or `ragflow_retrieval` for direct document chunk access
5. **Session issues**: If using session tools, check `ragflow_list_sessions` and use `ragflow_reset_session` if needed

## Known Issues

### Rerank Functionality Protocol Error
- **Issue**: Using `use_rerank=true` parameter causes "UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol" error
- **Status**: Known defect mentioned by RAGFlow developers
- **Workaround**: Use `use_rerank=false` (default) for standard vector similarity retrieval
- **Impact**: Reranking feature currently unavailable, but standard retrieval works normally
- **Follow-up**: Monitor RAGFlow GitHub issues for resolution

## Contributing

We welcome contributions! However, please note that **all changes must be submitted via Pull Requests (PRs)** as the main branch is protected.

### How to Contribute

1. Fork the repository
2. Create a new branch for your feature/fix: `git checkout -b feature/your-feature-name`
3. Make your changes and commit them with clear, descriptive messages
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request against the main branch

### PR Requirements

- All PRs are automatically scanned for exposed credentials using TruffleHog
- Ensure your code doesn't contain any API keys, tokens, or secrets
- Include a clear description of what your PR does and why it's needed
- Update documentation if you're adding new features

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
