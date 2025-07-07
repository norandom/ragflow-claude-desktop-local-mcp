#!/usr/bin/env python3
"""
RAGFlow MCP Server
A Model Context Protocol server that provides tools to interact with RAGFlow's REST API.
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

    async def create_chat_session(self, dataset_id: str, session_name: Optional[str] = None) -> str:
        """Create a new chat session in RAGFlow"""
        # Generate unique chat_id
        timestamp = int(time.time())
        if session_name:
            chat_id = f"{session_name}-{timestamp}"
        else:
            chat_id = f"ragflow-session-{timestamp}"
        
        # Create the chat session via RAGFlow API
        endpoint = "/api/v1/chats"
        data = {
            "name": chat_id,
            "dataset_ids": [dataset_id]
        }
        
        try:
            result = await self._make_request("POST", endpoint, data)
            if result.get("code") == 0:
                # Use the chat ID returned by RAGFlow
                actual_chat_id = result.get("data", {}).get("id", chat_id)
                self.active_sessions[dataset_id] = actual_chat_id
                logger.info(f"Created chat session: {actual_chat_id} for dataset: {dataset_id}")
                return actual_chat_id
            else:
                logger.error(f"Failed to create chat session: {result}")
                # Fall back to generated ID
                self.active_sessions[dataset_id] = chat_id
                return chat_id
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            # Fall back to generated ID
            self.active_sessions[dataset_id] = chat_id
            return chat_id

    def get_or_create_session(self, dataset_id: str, session_name: Optional[str] = None) -> str:
        """Get existing session or generate new one for a dataset"""
        if dataset_id in self.active_sessions:
            return self.active_sessions[dataset_id]
        
        # Generate unique chat_id (will be properly created during query)
        timestamp = int(time.time())
        if session_name:
            chat_id = f"{session_name}-{timestamp}"
        else:
            chat_id = f"ragflow-session-{timestamp}"
        
        self.active_sessions[dataset_id] = chat_id
        return chat_id

    async def query_ragflow(self, dataset_id: str, query: str, session_name: Optional[str] = None, stream: bool = False, languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query RAGFlow using chat completions endpoint with cross-language support"""
        # Default to English and German if no languages specified
        if languages is None:
            languages = ["en", "de"]
        
        # Check if we have an existing session
        if dataset_id not in self.active_sessions:
            # Create a new chat session
            chat_id = await self.create_chat_session(dataset_id, session_name)
        else:
            chat_id = self.active_sessions[dataset_id]
        
        # Enhance query with cross-language instructions
        enhanced_query = f"""Please search for information in multiple languages (primarily English and German) and provide a comprehensive answer. 

Languages to consider: {', '.join(languages)}

Original query: {query}

Please provide responses that include information found in any of the specified languages. When relevant content is found in different languages, provide translations or summaries as appropriate. Use context7 for up-to-date documentation and cross-language queries when possible."""
        
        endpoint = f"/api/v1/chats_openai/{chat_id}/chat/completions"
        data = {
            "model": "ragflow",
            "messages": [{"role": "user", "content": enhanced_query}],
            "stream": stream,
            "dataset_id": dataset_id
        }
        
        try:
            result = await self._make_request("POST", endpoint, data)
            
            # If we get a session ownership error, try creating a new session
            if result.get("code") == 102 and "don't own the chat" in result.get("message", ""):
                logger.info(f"Session ownership issue, creating new session for dataset {dataset_id}")
                # Remove the old session
                if dataset_id in self.active_sessions:
                    del self.active_sessions[dataset_id]
                # Create a new session
                chat_id = await self.create_chat_session(dataset_id, session_name)
                # Retry the query
                endpoint = f"/api/v1/chats_openai/{chat_id}/chat/completions"
                data["dataset_id"] = dataset_id  # Ensure dataset_id is included
                result = await self._make_request("POST", endpoint, data)
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
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
            name="ragflow_query",
            description="Query RAGFlow knowledge base and get answers with references. Sessions are automatically managed per dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the dataset/knowledge base to query"
                    },
                    "query": {
                        "type": "string",
                        "description": "Question or query to ask RAGFlow"
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Optional session name for the chat (e.g., 'basf-financial-analysis'). If not provided, a default session will be used."
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of language codes to search in (e.g., ['en', 'de']). Defaults to ['en', 'de'] for English and German."
                    }
                },
                "required": ["dataset_id", "query"]
            }
        ),
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
            name="ragflow_query_by_name",
            description="Query RAGFlow knowledge base by dataset name instead of ID. Perfect for when you only know the dataset name like 'BASF'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset/knowledge base to query (e.g., 'BASF', 'Company Reports')"
                    },
                    "query": {
                        "type": "string",
                        "description": "Question or query to ask RAGFlow"
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Optional session name for the chat (e.g., 'basf-financial-analysis')"
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of language codes to search in (e.g., ['en', 'de']). Defaults to ['en', 'de'] for English and German."
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
        if name == "ragflow_query":
            result = await ragflow_client.query_ragflow(
                dataset_id=arguments["dataset_id"],
                query=arguments["query"],
                session_name=arguments.get("session_name"),
                languages=arguments.get("languages")
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_list_datasets":
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
            
        elif name == "ragflow_query_by_name":
            dataset_name = arguments["dataset_name"]
            dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            
            if not dataset_id:
                available_datasets = list(ragflow_client.dataset_cache.keys()) if ragflow_client.dataset_cache else []
                error_msg = f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}"
                return [types.TextContent(type="text", text=error_msg)]
            
            result = await ragflow_client.query_ragflow(
                dataset_id=dataset_id,
                query=arguments["query"],
                session_name=arguments.get("session_name"),
                languages=arguments.get("languages")
            )
            
            # Include dataset info in response
            response_data = {
                "dataset_found": {
                    "name": dataset_name,
                    "id": dataset_id
                },
                "query_result": result
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