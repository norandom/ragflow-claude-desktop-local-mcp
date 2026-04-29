#!/usr/bin/env python3
"""RAGFlow MCP server — registers tools and dispatches calls to the RAGFlow client."""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from .client.ragflow import RAGFlowMCPServer
from .common.exceptions import (
    RAGFlowAPIError, RAGFlowConnectionError, ConfigurationError,
    ValidationError, DatasetNotFoundError,
)
from .common.validation import validate_dataset_id, redact_sensitive_data

# DSPy availability flag. Defined here so tests can patch it via
# `ragflow_claude_mcp.server.DSPY_AVAILABLE`. The client reads this attribute
# at call time (late import) so the patch propagates correctly.
try:
    import dspy  # noqa: F401
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Plain stderr logging. Avoid logging_config here so the import doesn't go circular.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("ragflow-mcp")


server = Server("ragflow-mcp")

# Lazily-built singleton RAGFlow client. Tests patch `_ragflow_client` directly.
_ragflow_client: Optional[RAGFlowMCPServer] = None


def load_config() -> Dict[str, Any]:
    """Load config.json from the cwd. Missing file is fine; bad JSON is fatal."""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ConfigurationError("config.json must contain a JSON object")

        return config
    except FileNotFoundError:
        logger.warning("config.json not found. Please create it based on config.json.sample")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding config.json: {e}. Please check its format.")
        raise ConfigurationError(f"Invalid JSON in config.json: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise ConfigurationError(f"Failed to load configuration: {e}")


def get_ragflow_client() -> RAGFlowMCPServer:
    """Lazy singleton. Reads config from env first, falls back to config.json."""
    global _ragflow_client
    if _ragflow_client is None:
        try:
            config = load_config()

            # Env overrides config file.
            RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", config.get("RAGFLOW_BASE_URL"))
            RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", config.get("RAGFLOW_API_KEY"))
            RAGFLOW_DEFAULT_RERANK = config.get("RAGFLOW_DEFAULT_RERANK", "rerank-multilingual-v3.0")

            CF_ACCESS_CLIENT_ID = os.getenv("CF_ACCESS_CLIENT_ID", config.get("CF_ACCESS_CLIENT_ID"))
            CF_ACCESS_CLIENT_SECRET = os.getenv("CF_ACCESS_CLIENT_SECRET", config.get("CF_ACCESS_CLIENT_SECRET"))

            if not RAGFLOW_BASE_URL or not RAGFLOW_API_KEY:
                raise ConfigurationError(
                    "RAGFLOW_BASE_URL and RAGFLOW_API_KEY must be set in config.json or environment variables"
                )

            if not RAGFLOW_BASE_URL.startswith(('http://', 'https://')):
                raise ConfigurationError("RAGFLOW_BASE_URL must start with http:// or https://")

            # Cheap sanity check — we don't actually know the real API key format.
            if len(RAGFLOW_API_KEY.strip()) < 10:
                raise ConfigurationError("RAGFLOW_API_KEY appears to be invalid (too short)")

            _ragflow_client = RAGFlowMCPServer(
                RAGFLOW_BASE_URL,
                RAGFLOW_API_KEY,
                RAGFLOW_DEFAULT_RERANK,
                CF_ACCESS_CLIENT_ID,
                CF_ACCESS_CLIENT_SECRET,
            )
            logger.info("RAGFlow client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAGFlow client: {redact_sensitive_data(str(e))}")
            raise

    return _ragflow_client


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="ragflow_list_datasets",
            description="List all available datasets/knowledge bases in RAGFlow",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="ragflow_list_documents",
            description="List documents in a specific dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to list documents from"
                    }
                },
                "required": ["dataset_id"]
            }
        ),
        types.Tool(
            name="ragflow_get_chunks",
            description="Get chunks with references from a specific document",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset"
                    },
                    "document_id": {
                        "type": "string",
                        "description": "ID of the document to get chunks from"
                    }
                },
                "required": ["dataset_id", "document_id"]
            }
        ),
        types.Tool(
            name="ragflow_list_sessions",
            description="List active chat sessions for all datasets",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="ragflow_reset_session",
            description="Reset/clear the chat session for a specific dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset to reset session for"
                    }
                },
                "required": ["dataset_id"]
            }
        ),
        types.Tool(
            name="ragflow_retrieval",
            description="Retrieve document chunks directly from RAGFlow datasets using the retrieval API. Returns raw chunks with similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of IDs of the datasets/knowledge bases to search"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or question"
                    },
                    "document_name": {
                        "type": "string",
                        "description": "Optional document name to filter results to specific document"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks for vector cosine computation. Defaults to 1024."
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score for chunks (0.0 to 1.0). Defaults to 0.2."
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for pagination. Defaults to 1."
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of chunks per page. Defaults to 10."
                    },
                    "use_rerank": {
                        "type": "boolean",
                        "description": "Whether to enable reranking for better result quality. Default: false (uses vector similarity only)."
                    },
                    "deepening_level": {
                        "type": "integer",
                        "description": "Level of DSPy query refinement (0-3). 0=none, 1=basic refinement, 2=gap analysis, 3=full optimization. Default: 0",
                        "minimum": 0,
                        "maximum": 3
                    }
                },
                "required": ["dataset_ids", "query"]
            }
        ),
        types.Tool(
            name="ragflow_retrieval_by_name",
            description="Retrieve document chunks by dataset names using the retrieval API. Returns raw chunks with similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of names of the datasets/knowledge bases to search (e.g., ['BASF', 'Legal'])"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or question"
                    },
                    "document_name": {
                        "type": "string",
                        "description": "Optional document name to filter results to specific document"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks for vector cosine computation. Defaults to 1024."
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score for chunks (0.0 to 1.0). Defaults to 0.2."
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for pagination. Defaults to 1."
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of chunks per page. Defaults to 10."
                    },
                    "use_rerank": {
                        "type": "boolean",
                        "description": "Whether to enable reranking for better result quality. Default: false (uses vector similarity only)."
                    },
                    "deepening_level": {
                        "type": "integer",
                        "description": "Level of DSPy query refinement (0-3). 0=none, 1=basic refinement, 2=gap analysis, 3=full optimization. Default: 0",
                        "minimum": 0,
                        "maximum": 3
                    }
                },
                "required": ["dataset_names", "query"]
            }
        ),
        types.Tool(
            name="ragflow_list_documents_by_name",
            description="List documents in a dataset by dataset name",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset/knowledge base to list documents from"
                    }
                },
                "required": ["dataset_name"]
            }
        )
    ]


def _handle_tool_error(e: Exception, tool_name: str) -> List[types.TextContent]:
    """Map exceptions to MCP TextContent error responses, with severity-aware logging."""
    if isinstance(e, ValidationError):
        logger.warning(f"Validation error in {tool_name}: {e}")
        return [types.TextContent(type="text", text=f"Validation Error: {str(e)}")]
    elif isinstance(e, DatasetNotFoundError):
        logger.warning(f"Dataset not found in {tool_name}: {e}")
        return [types.TextContent(type="text", text=str(e))]
    elif isinstance(e, (RAGFlowAPIError, RAGFlowConnectionError)):
        logger.error(f"RAGFlow API error in {tool_name}: {e}")
        return [types.TextContent(type="text", text=f"API Error: {str(e)}")]
    elif isinstance(e, ConfigurationError):
        logger.error(f"Configuration error in {tool_name}: {e}")
        return [types.TextContent(type="text", text=f"Configuration Error: {str(e)}")]
    else:
        logger.error(f"Unexpected error in {tool_name}: {redact_sensitive_data(str(e))}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def _handle_retrieval_tool(
    ragflow_client: RAGFlowMCPServer,
    dataset_ids: List[str],
    arguments: Dict[str, Any],
    include_dataset_info: bool = False,
) -> Dict[str, Any]:
    """Shared body for the retrieval and retrieval_by_name tool handlers."""
    deepening_level = arguments.get("deepening_level", 0)

    # Reranking is broken upstream. Only allow it through if the caller
    # explicitly passed True (some agents pass truthy values by accident).
    use_rerank_param = arguments.get("use_rerank", False)
    use_rerank = use_rerank_param is True

    retrieval_args = {
        "top_k": arguments.get("top_k", 1024),
        "similarity_threshold": arguments.get("similarity_threshold", 0.2),
        "page": arguments.get("page", 1),
        "page_size": arguments.get("page_size", 10),
        "use_rerank": use_rerank,
        "document_name": arguments.get("document_name"),
    }

    if deepening_level > 0:
        result = await ragflow_client.retrieval_with_deepening(
            dataset_ids=dataset_ids,
            query=arguments["query"],
            deepening_level=deepening_level,
            **retrieval_args,
        )
    else:
        result = await ragflow_client.retrieval_query(
            dataset_ids=dataset_ids,
            query=arguments["query"],
            **retrieval_args,
        )

    if include_dataset_info:
        dataset_names = arguments.get("dataset_names", [])
        return {
            "datasets_found": [
                {"name": name, "id": id}
                for name, id in zip(dataset_names, dataset_ids)
            ],
            "retrieval_result": result,
        }

    return result


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """MCP tool dispatcher. Routes by name and converts exceptions to error TextContent."""
    try:
        ragflow_client = get_ragflow_client()

        if name == "ragflow_list_datasets":
            result = await ragflow_client.list_datasets()
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "ragflow_list_documents":
            result = await ragflow_client.list_documents(arguments["dataset_id"])
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "ragflow_get_chunks":
            result = await ragflow_client.get_document_chunks(
                dataset_id=arguments["dataset_id"],
                document_id=arguments["document_id"],
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "ragflow_list_sessions":
            sessions = {
                dataset_id: {
                    "chat_id": chat_id,
                    "dataset_id": dataset_id,
                }
                for dataset_id, chat_id in ragflow_client.active_sessions.items()
            }
            return [types.TextContent(type="text", text=json.dumps(sessions, indent=2))]

        elif name == "ragflow_retrieval":
            result = await _handle_retrieval_tool(
                ragflow_client, arguments["dataset_ids"], arguments, include_dataset_info=False,
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "ragflow_retrieval_by_name":
            dataset_names = arguments["dataset_names"]
            try:
                dataset_ids = await ragflow_client.find_datasets_by_names(dataset_names)
            except DatasetNotFoundError:
                # Try refreshing cache once.
                logger.info(f"One or more datasets in '{dataset_names}' not found, refreshing cache...")
                await ragflow_client.list_datasets()
                dataset_ids = await ragflow_client.find_datasets_by_names(dataset_names)

            result = await _handle_retrieval_tool(
                ragflow_client, dataset_ids, arguments, include_dataset_info=True,
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "ragflow_list_documents_by_name":
            dataset_name = arguments["dataset_name"]
            try:
                dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            except DatasetNotFoundError:
                logger.info(f"Dataset '{dataset_name}' not found, refreshing cache...")
                await ragflow_client.list_datasets()
                dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)

            result = await ragflow_client.list_documents(dataset_id)

            response_data = {
                "dataset_found": {
                    "name": dataset_name,
                    "id": dataset_id,
                },
                "documents": result,
            }
            return [types.TextContent(type="text", text=json.dumps(response_data, indent=2))]

        elif name == "ragflow_reset_session":
            dataset_id = validate_dataset_id(arguments["dataset_id"])
            if dataset_id in ragflow_client.active_sessions:
                old_session = ragflow_client.active_sessions[dataset_id]
                del ragflow_client.active_sessions[dataset_id]
                return [types.TextContent(type="text", text=f"Session reset for dataset {dataset_id[:10]}.... Old session: {old_session}")]
            else:
                return [types.TextContent(type="text", text=f"No active session found for dataset {dataset_id[:10]}...")]

        else:
            raise ValidationError(f"Unknown tool: {name}")

    except Exception as e:
        return _handle_tool_error(e, name)


def main():
    """Entry point. Runs the MCP server and tears down the HTTP session on exit."""
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available - query deepening will be disabled")

    async def run_server():
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="ragflow-mcp",
                        server_version="1.0.0",
                        capabilities=types.ServerCapabilities(
                            tools=types.ToolsCapability(listChanged=True),
                            logging=types.LoggingCapability(),
                        ),
                    ),
                )
        finally:
            global _ragflow_client
            if _ragflow_client:
                await _ragflow_client.close_session()
                logger.info("RAGFlow client session closed")

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {redact_sensitive_data(str(e))}")
        raise


if __name__ == "__main__":
    main()
