#!/usr/bin/env python3
"""
RAGFlow MCP Server
A Model Context Protocol server that provides tools to interact with RAGFlow's retrieval API.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import aiohttp
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragflow-mcp")

class RAGFlowMCPServer:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.active_sessions = {}  # dataset_id -> chat_id mapping
        self.dataset_cache = {}  # Cache for dataset list
        
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to RAGFlow API"""
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"RAGFlow API error {response.status}: {error_text}")
            except Exception as e:
                logger.error(f"Request failed: {e}")
                raise



    async def retrieval_query(self, dataset_id: str, query: str, top_k: int = 1024, similarity_threshold: float = 0.2, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """Query RAGFlow using dedicated retrieval endpoint for direct document access"""
        endpoint = "/api/v1/retrieval"
        data = {
            "question": query,
            "dataset_ids": [dataset_id],
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "page": page,
            "page_size": page_size
        }
        
        try:
            result = await self._make_request("POST", endpoint, data)
            return result
        except Exception as e:
            logger.error(f"Retrieval query failed: {e}")
            raise

    async def list_datasets(self) -> Dict[str, Any]:
        """List available datasets/knowledge bases"""
        endpoint = "/api/v1/datasets"
        result = await self._make_request("GET", endpoint)
        
        # Cache datasets for name lookup
        if result.get("code") == 0 and "data" in result:
            self.dataset_cache = {
                dataset["name"].lower(): dataset["id"]
                for dataset in result["data"]
            }
        
        return result

    async def find_dataset_by_name(self, name: str) -> Optional[str]:
        """Find dataset ID by name (case-insensitive)"""
        # Refresh cache if empty
        if not self.dataset_cache:
            await self.list_datasets()
        
        # Exact match first
        name_lower = name.lower()
        if name_lower in self.dataset_cache:
            return self.dataset_cache[name_lower]
        
        # Fuzzy match - find datasets containing the name
        matches = [
            dataset_id for dataset_name, dataset_id in self.dataset_cache.items()
            if name_lower in dataset_name or dataset_name in name_lower
        ]
        
        return matches[0] if matches else None

    async def upload_documents(self, dataset_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """Upload documents to a dataset"""
        endpoint = f"/api/v1/datasets/{dataset_id}/documents"
        # Note: This requires multipart/form-data, simplified for MCP
        return {"message": "Upload endpoint available but requires file handling"}

    async def list_documents(self, dataset_id: str) -> Dict[str, Any]:
        """List documents in a dataset"""
        endpoint = f"/api/v1/datasets/{dataset_id}/documents"
        return await self._make_request("GET", endpoint)

    async def get_document_chunks(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """Get chunks from a specific document"""
        endpoint = f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
        return await self._make_request("GET", endpoint)

# Initialize the MCP server
server = Server("ragflow-mcp")

# Load configuration from config.json
def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error("config.json not found. Please create it based on config.json.sample")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding config.json. Please check its format.")
        return {}

config = load_config()

# RAGFlow configuration
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", config.get("RAGFLOW_BASE_URL"))
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", config.get("RAGFLOW_API_KEY"))

if not RAGFLOW_BASE_URL or not RAGFLOW_API_KEY:
    raise ValueError("RAGFLOW_BASE_URL and RAGFLOW_API_KEY must be set in config.json or environment variables")

ragflow_client = RAGFlowMCPServer(RAGFLOW_BASE_URL, RAGFLOW_API_KEY)

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools"""
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
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset/knowledge base to search"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or question"
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
                    }
                },
                "required": ["dataset_id", "query"]
            }
        ),
        types.Tool(
            name="ragflow_retrieval_by_name",
            description="Retrieve document chunks by dataset name using the retrieval API. Returns raw chunks with similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset/knowledge base to search (e.g., 'BASF')"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or question"
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
                    }
                },
                "required": ["dataset_name", "query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool calls"""
    try:
            
        if name == "ragflow_list_datasets":
            result = await ragflow_client.list_datasets()
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_list_documents":
            result = await ragflow_client.list_documents(arguments["dataset_id"])
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_get_chunks":
            result = await ragflow_client.get_document_chunks(
                dataset_id=arguments["dataset_id"],
                document_id=arguments["document_id"]
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_list_sessions":
            sessions = {
                dataset_id: {
                    "chat_id": chat_id,
                    "dataset_id": dataset_id
                }
                for dataset_id, chat_id in ragflow_client.active_sessions.items()
            }
            return [types.TextContent(type="text", text=json.dumps(sessions, indent=2))]
            
        elif name == "ragflow_retrieval":
            result = await ragflow_client.retrieval_query(
                dataset_id=arguments["dataset_id"],
                query=arguments["query"],
                top_k=arguments.get("top_k", 1024),
                similarity_threshold=arguments.get("similarity_threshold", 0.2),
                page=arguments.get("page", 1),
                page_size=arguments.get("page_size", 10)
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_retrieval_by_name":
            dataset_name = arguments["dataset_name"]
            dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            
            if not dataset_id:
                available_datasets = list(ragflow_client.dataset_cache.keys()) if ragflow_client.dataset_cache else []
                error_msg = f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}"
                return [types.TextContent(type="text", text=error_msg)]
            
            result = await ragflow_client.retrieval_query(
                dataset_id=dataset_id,
                query=arguments["query"],
                top_k=arguments.get("top_k", 1024),
                similarity_threshold=arguments.get("similarity_threshold", 0.2),
                page=arguments.get("page", 1),
                page_size=arguments.get("page_size", 10)
            )
            
            # Include dataset info in response
            response_data = {
                "dataset_found": {
                    "name": dataset_name,
                    "id": dataset_id
                },
                "retrieval_result": result
            }
            return [types.TextContent(type="text", text=json.dumps(response_data, indent=2))]
            
            
        elif name == "ragflow_reset_session":
            dataset_id = arguments["dataset_id"]
            if dataset_id in ragflow_client.active_sessions:
                old_session = ragflow_client.active_sessions[dataset_id]
                del ragflow_client.active_sessions[dataset_id]
                return [types.TextContent(type="text", text=f"Session reset for dataset {dataset_id}. Old session: {old_session}")]
            else:
                return [types.TextContent(type="text", text=f"No active session found for dataset {dataset_id}")]
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

def main():
    """Run the MCP server"""
    async def run_server():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ragflow-mcp",
                    server_version="1.0.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability(listChanged=True),
                        logging=types.LoggingCapability()
                    )
                )
            )
    
    asyncio.run(run_server())

if __name__ == "__main__":
    main()