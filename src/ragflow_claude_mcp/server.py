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
from typing import Any, Dict, List, Optional, Union
import aiohttp
from mcp.server import Server
from mcp.server.models import InitializationOptions

# Local imports
from .exceptions import (
    RAGFlowAPIError, RAGFlowConnectionError, ConfigurationError,
    ValidationError, DatasetNotFoundError, DocumentNotFoundError,
    DSPyConfigurationError
)
from .validation import (
    validate_dataset_id, validate_document_id, validate_query,
    validate_dataset_name, validate_document_name, validate_pagination_params,
    validate_similarity_threshold, validate_top_k, validate_deepening_level,
    redact_sensitive_data
)
from .cache import DatasetCache
from .logging_config import setup_logging, get_logger, log_info, log_warning, log_error, log_debug

# DSPy integration
try:
    import dspy
    from .dspy_deepening import get_deepener
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Will setup logger after configuration below
import mcp.server.stdio
import mcp.types as types

# Configure logging - simple setup to avoid import issues
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("ragflow-mcp")

class RAGFlowMCPServer:
    def __init__(self, base_url: str, api_key: str, default_rerank: Optional[str] = None, 
                 cf_access_client_id: Optional[str] = None, cf_access_client_secret: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.default_rerank = default_rerank
        self.cf_access_client_id = cf_access_client_id
        self.cf_access_client_secret = cf_access_client_secret
        
        # Build headers
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Add Cloudflare Zero Trust headers if provided
        if cf_access_client_id and cf_access_client_secret:
            self.headers['CF-Access-Client-Id'] = cf_access_client_id
            self.headers['CF-Access-Client-Secret'] = cf_access_client_secret
            logger.info("Cloudflare Zero Trust authentication enabled")
        
        self.active_sessions: Dict[str, str] = {}  # dataset_id -> chat_id mapping
        self.dataset_cache = DatasetCache()  # Enhanced cache with TTL
        self._session: Optional[aiohttp.ClientSession] = None  # Reusable session
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'RAGFlow-MCP-Server/1.0.0'}
            )
        return self._session
    
    async def close_session(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to RAGFlow API with improved error handling."""
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()
        
        try:
            async with session.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        raise RAGFlowAPIError(
                            f"Invalid JSON response from RAGFlow API",
                            status_code=response.status,
                            response_text=response_text[:500],
                            details={'json_error': str(e)}
                        )
                else:
                    raise RAGFlowAPIError(
                        f"RAGFlow API error: {response.reason}",
                        status_code=response.status,
                        response_text=response_text[:500]
                    )
                    
        except aiohttp.ClientError as e:
            logger.error(f"Connection error to RAGFlow API: {e}")
            raise RAGFlowConnectionError(f"Failed to connect to RAGFlow API: {str(e)}")
        except Exception as e:
            if isinstance(e, (RAGFlowAPIError, RAGFlowConnectionError)):
                raise
            logger.error(f"Unexpected error in API request: {e}")
            raise RAGFlowAPIError(f"Unexpected API error: {str(e)}")



    async def retrieval_query(
        self, 
        dataset_id: str, 
        query: str, 
        top_k: int = 1024, 
        similarity_threshold: float = 0.2, 
        page: int = 1, 
        page_size: int = 10, 
        use_rerank: bool = False, 
        document_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query RAGFlow using dedicated retrieval endpoint for direct document access
        
        Args:
            dataset_id: ID of the dataset to search
            query: Search query
            top_k: Number of chunks for vector cosine computation
            similarity_threshold: Minimum similarity score (0.0-1.0)
            page: Page number for pagination
            page_size: Number of chunks per page
            use_rerank: Whether to enable reranking. Default False (uses vector similarity only).
            document_name: Optional document name to filter results to specific document
            
        Returns:
            Dictionary containing search results
            
        Raises:
            ValidationError: If input parameters are invalid
            RAGFlowAPIError: If the API request fails
        """
        # Validate inputs
        dataset_id = validate_dataset_id(dataset_id)
        query = validate_query(query)
        
        if document_name:
            document_name = validate_document_name(document_name)
            
        pagination = validate_pagination_params(page, page_size)
        page = pagination.get('page', page)
        page_size = pagination.get('page_size', page_size)
        
        similarity_threshold = validate_similarity_threshold(similarity_threshold) or 0.2
        top_k = validate_top_k(top_k) or 1024
        endpoint = "/api/v1/retrieval"
        data = {
            "question": query,
            "dataset_ids": [dataset_id],
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "page": page,
            "page_size": page_size
        }
        
        logger.info(f"Executing retrieval query for dataset {dataset_id[:10]}... with {len(query)} char query")
        
        # Add document filtering if specified
        document_match_info = None
        if document_name:
            match_result = await self.find_document_by_name(dataset_id, document_name)
            if match_result['status'] == 'single_match':
                data["document_ids"] = [match_result['document_id']]
                logger.info(f"Filtering results to document '{match_result['document_name']}' (ID: {match_result['document_id'][:10]}...)")
                document_match_info = match_result
            elif match_result['status'] == 'multiple_matches':
                # Use the best match but inform user about alternatives
                data["document_ids"] = [match_result['document_id']]
                logger.info(f"Multiple documents match '{document_name}'. Using best match: '{match_result['document_name']}' (ID: {match_result['document_id'][:10]}...)")
                document_match_info = match_result
            else:
                logger.warning(f"Document '{document_name}' not found in dataset {dataset_id[:10]}..., proceeding without filtering")
                document_match_info = match_result
        
        # Only use reranking if explicitly enabled
        if use_rerank and self.default_rerank:
            data["rerank_id"] = self.default_rerank
        
        try:
            result = await self._make_request("POST", endpoint, data)
            
            # Validate response structure
            if not isinstance(result, dict):
                raise RAGFlowAPIError("Invalid response format from RAGFlow API")
            
            # Add document match information if document filtering was used
            if document_match_info:
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata']['document_filtering'] = document_match_info
                
            return result
        except (RAGFlowAPIError, RAGFlowConnectionError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in retrieval query: {redact_sensitive_data(str(e))}")
            raise RAGFlowAPIError(f"Retrieval query failed: {str(e)}")
    
    def _configure_dspy_if_needed(self) -> bool:
        """Configure DSPy if not already configured.
        
        Returns:
            True if configuration successful, False otherwise
        """
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
            return True
            
        try:
            # Load configuration
            config_path = os.path.expanduser(os.path.join(os.path.dirname(__file__), '..', '..', 'config.json'))
            config = {}
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Get DSPy model from config or use default
            dspy_model = config.get('DSPY_MODEL', 'openai/gpt-4o-mini')
            
            # Configure DSPy with provider-based configuration
            lm_kwargs = {}
            provider = config.get('PROVIDER', 'openai').lower()
            
            if provider == 'openrouter':
                if 'OPENROUTER_API_KEY' not in config:
                    raise DSPyConfigurationError("OPENROUTER_API_KEY must be set when PROVIDER is 'openrouter'")
                
                if 'OPENAI_API_KEY' not in os.environ:
                    os.environ['OPENAI_API_KEY'] = config['OPENROUTER_API_KEY']
                lm_kwargs['api_base'] = 'https://openrouter.ai/api/v1'
                
                # Optional OpenRouter headers for analytics
                if 'OPENROUTER_SITE_URL' in config:
                    lm_kwargs['default_headers'] = {
                        'HTTP-Referer': config['OPENROUTER_SITE_URL'],
                        'X-Title': config.get('OPENROUTER_SITE_NAME', 'RAGFlow MCP Server')
                    }
                
                logger.info(f"DSPy configured with OpenRouter model: {dspy_model}")
            elif provider == 'openai':
                if 'OPENAI_API_KEY' not in config:
                    raise DSPyConfigurationError("OPENAI_API_KEY must be set when PROVIDER is 'openai'")
                
                if 'OPENAI_API_KEY' not in os.environ:
                    os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
                logger.info(f"DSPy configured with OpenAI model: {dspy_model}")
            else:
                raise DSPyConfigurationError(f"Unknown provider '{provider}'. Supported providers: 'openai', 'openrouter'")
            
            # Configure DSPy
            dspy.configure(lm=dspy.LM(dspy_model, **lm_kwargs))
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure DSPy LM: {e}")
            return False
    
    async def retrieval_with_deepening(
        self, 
        dataset_id: str, 
        query: str, 
        deepening_level: int = 0, 
        document_name: Optional[str] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Enhanced retrieval with optional DSPy query deepening.
        
        Args:
            dataset_id: The dataset ID to search
            query: The search query
            deepening_level: Level of DSPy refinement (0-3)
            document_name: Optional document name filter
            **kwargs: Additional arguments for retrieval
            
        Returns:
            Dictionary containing enhanced search results
        """
        # Validate inputs
        dataset_id = validate_dataset_id(dataset_id)
        query = validate_query(query)
        deepening_level = validate_deepening_level(deepening_level) or 0
        
        if document_name:
            document_name = validate_document_name(document_name)
        
        # If no deepening requested or DSPy not available, use standard retrieval
        if deepening_level <= 0 or not DSPY_AVAILABLE:
            if deepening_level > 0 and not DSPY_AVAILABLE:
                logger.warning("DSPy deepening requested but not available, falling back to standard retrieval")
            return await self.retrieval_query(dataset_id, query, document_name=document_name, **kwargs)
        
        # Use DSPy query deepening
        logger.info(f"Starting DSPy query deepening (level {deepening_level}) for query: '{query[:50]}...'")
        
        try:
            # Configure DSPy if needed
            if not self._configure_dspy_if_needed():
                logger.warning("DSPy configuration failed, falling back to standard retrieval")
                return await self.retrieval_query(dataset_id, query, document_name=document_name, **kwargs)
            
            deepener = get_deepener()
            
            # Perform deepened search
            deepening_result = await deepener.deepen_search(
                ragflow_client=self,
                dataset_id=dataset_id,
                original_query=query,
                deepening_level=deepening_level,
                **kwargs
            )
            
            # Return the enhanced result with additional metadata
            final_results = deepening_result['results']
            
            # Add deepening metadata to the response
            if 'metadata' not in final_results:
                final_results['metadata'] = {}
            
            final_results['metadata']['deepening'] = {
                'original_query': deepening_result['original_query'],
                'final_query': deepening_result['final_query'],
                'queries_used': deepening_result['queries_used'],
                'deepening_level': deepening_result['deepening_level'],
                'refinement_log': deepening_result['refinement_log']
            }
            
            return final_results
            
        except DSPyConfigurationError as e:
            logger.error(f"DSPy configuration error: {e}")
            logger.info("Falling back to standard retrieval")
            return await self.retrieval_query(dataset_id, query, document_name=document_name, **kwargs)
        except Exception as e:
            logger.error(f"DSPy query deepening failed: {redact_sensitive_data(str(e))}")
            logger.info("Falling back to standard retrieval")
            return await self.retrieval_query(dataset_id, query, document_name=document_name, **kwargs)

    async def list_datasets(self) -> Dict[str, Any]:
        """List available datasets/knowledge bases with improved caching."""
        endpoint = "/api/v1/datasets"
        
        try:
            result = await self._make_request("GET", endpoint)
            
            # Validate response structure
            if not isinstance(result, dict):
                raise RAGFlowAPIError("Invalid response format from datasets API")
            
            # Cache datasets for name lookup with proper error handling
            if result.get("code") == 0 and "data" in result:
                datasets = result["data"]
                if datasets and isinstance(datasets, list):
                    dataset_mapping = {
                        dataset["name"]: dataset["id"]
                        for dataset in datasets
                        if isinstance(dataset, dict) and "name" in dataset and "id" in dataset
                    }
                    
                    if dataset_mapping:
                        self.dataset_cache.set_datasets(dataset_mapping)
                        cache_stats = self.dataset_cache.stats()
                        logger.info(f"Dataset cache updated with {cache_stats['size']} datasets")
                    else:
                        logger.warning("No valid datasets found in API response")
                else:
                    logger.warning("Dataset list returned empty or invalid data array")
            else:
                error_msg = result.get('message', 'Unknown error')
                logger.error(f"Failed to list datasets: code={result.get('code')}, message={error_msg}")
            
            return result
            
        except (RAGFlowAPIError, RAGFlowConnectionError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing datasets: {redact_sensitive_data(str(e))}")
            raise RAGFlowAPIError(f"Failed to list datasets: {str(e)}")

    async def find_document_by_name(self, dataset_id: str, document_name: str) -> Dict[str, Any]:
        """Find document ID by name within a dataset with ranking and multiple match handling.
        
        Args:
            dataset_id: The dataset ID to search in
            document_name: The document name to find
            
        Returns:
            Dictionary with match status and document information
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        dataset_id = validate_dataset_id(dataset_id)
        document_name = validate_document_name(document_name)
        try:
            documents_result = await self.list_documents(dataset_id)
            
            if not isinstance(documents_result, dict):
                raise RAGFlowAPIError("Invalid response format from documents API")
                
            if documents_result.get('code') == 0 and 'data' in documents_result:
                documents = documents_result['data'].get('docs', [])
                
                if not isinstance(documents, list):
                    logger.warning(f"Invalid documents data format for dataset {dataset_id[:10]}...")
                    return {'status': 'error', 'matches': []}
                
                # Find all matching documents
                matches = []
                document_name_lower = document_name.lower()
                
                for doc in documents:
                    if not isinstance(doc, dict) or 'name' not in doc:
                        continue
                        
                    doc_name = doc.get('name', '').lower()
                    if document_name_lower in doc_name or doc_name in document_name_lower:
                        # Calculate match score for ranking
                        score = self._calculate_match_score(doc_name, document_name_lower, doc)
                        matches.append({
                            'id': doc.get('id'),
                            'name': doc.get('name'),
                            'score': score,
                            'size': doc.get('size', 0),
                            'create_time': doc.get('create_time', ''),
                            'update_time': doc.get('update_time', '')
                        })
                
                if not matches:
                    logger.warning(f"Document '{document_name}' not found in dataset {dataset_id[:10]}...")
                    return {'status': 'not_found', 'matches': []}
                
                # Sort by score (higher is better) and then by update time (more recent first)
                matches.sort(key=lambda x: (x['score'], x['update_time']), reverse=True)
                
                if len(matches) == 1:
                    return {
                        'status': 'single_match',
                        'document_id': matches[0]['id'],
                        'document_name': matches[0]['name'],
                        'matches': matches
                    }
                else:
                    logger.info(f"Multiple documents match '{document_name}': {[m['name'] for m in matches[:3]]}...")
                    return {
                        'status': 'multiple_matches',
                        'document_id': matches[0]['id'],  # Use best match as default
                        'document_name': matches[0]['name'],
                        'matches': matches
                    }
            else:
                error_msg = documents_result.get('message', 'Unknown error')
                logger.error(f"Failed to list documents for dataset {dataset_id[:10]}...: {error_msg}")
                return {'status': 'error', 'matches': []}
                
        except (RAGFlowAPIError, RAGFlowConnectionError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error finding document '{document_name}': {redact_sensitive_data(str(e))}")
            return {'status': 'error', 'matches': []}
    
    def _calculate_match_score(self, doc_name: str, search_term: str, doc: Dict) -> float:
        """Calculate match score for ranking (higher is better)"""
        score = 0.0
        
        # Exact match gets highest score
        if doc_name == search_term:
            score += 100.0
        
        # Starts with search term
        elif doc_name.startswith(search_term):
            score += 50.0
        
        # Contains search term
        elif search_term in doc_name:
            score += 30.0
        
        # Search term contains document name (partial match)
        elif doc_name in search_term:
            score += 20.0
        
        # Prefer more recent documents (bonus for newer update times)
        update_time = doc.get('update_time', '')
        if update_time:
            try:
                # Assume update_time is a timestamp or ISO string
                # Add small bonus for more recent files
                score += 0.1
            except (ValueError, TypeError):
                # Ignore invalid timestamp formats
                logger.debug(f"Invalid update_time format: {update_time}")
                
        # Prefer documents with certain keywords in name
        if any(keyword in doc_name for keyword in ['2024', '2023', 'latest', 'current', 'new']):
            score += 5.0
            
        return score

    async def find_dataset_by_name(self, name: str) -> Optional[str]:
        """Find dataset ID by name (case-insensitive) with validation."""
        # Validate input
        name = validate_dataset_name(name)
        
        # Try to get from cache first
        dataset_id = self.dataset_cache.get_dataset_id(name)
        if dataset_id:
            logger.debug(f"Found dataset '{name}' in cache")
            return dataset_id
        
        # Cache miss - refresh and try again
        logger.info(f"Dataset '{name}' not found in cache, refreshing...")
        try:
            await self.list_datasets()
            dataset_id = self.dataset_cache.get_dataset_id(name)
            if dataset_id:
                logger.debug(f"Found dataset '{name}' after cache refresh")
                return dataset_id
        except Exception as e:
            logger.error(f"Failed to refresh dataset cache: {redact_sensitive_data(str(e))}")
        
        # Final attempt with fuzzy matching
        available_names = self.dataset_cache.get_all_names()
        name_lower = name.lower()
        
        # Fuzzy match - find datasets containing the name
        matches = [
            available_name for available_name in available_names
            if name_lower in available_name or available_name in name_lower
        ]
        
        if matches:
            matched_name = matches[0]
            dataset_id = self.dataset_cache.get_dataset_id(matched_name)
            logger.info(f"Found fuzzy match for '{name}' -> '{matched_name}'")
            return dataset_id
        
        # Not found
        logger.warning(f"No dataset found for '{name}'. Available datasets: {available_names[:10]}")
        raise DatasetNotFoundError(name, available_names)

    async def upload_documents(self, dataset_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """Upload documents to a dataset"""
        endpoint = f"/api/v1/datasets/{dataset_id}/documents"
        # Note: This requires multipart/form-data, simplified for MCP
        return {"message": "Upload endpoint available but requires file handling"}

    async def list_documents(self, dataset_id: str) -> Dict[str, Any]:
        """List documents in a dataset with validation."""
        dataset_id = validate_dataset_id(dataset_id)
        endpoint = f"/api/v1/datasets/{dataset_id}/documents"
        return await self._make_request("GET", endpoint)

    async def get_document_chunks(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """Get chunks from a specific document with validation."""
        dataset_id = validate_dataset_id(dataset_id)
        document_id = validate_document_id(document_id)
        endpoint = f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
        return await self._make_request("GET", endpoint)

# Initialize the MCP server
server = Server("ragflow-mcp")

# Load configuration from config.json
def load_config() -> Dict[str, Any]:
    """Load configuration from config.json with validation."""
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
    """Get or create the RAGFlow client instance with proper configuration validation."""
    global _ragflow_client
    if _ragflow_client is None:
        try:
            config = load_config()
            
            # RAGFlow configuration with validation
            RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", config.get("RAGFLOW_BASE_URL"))
            RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", config.get("RAGFLOW_API_KEY"))
            RAGFLOW_DEFAULT_RERANK = config.get("RAGFLOW_DEFAULT_RERANK", "rerank-multilingual-v3.0")
            
            # Cloudflare Zero Trust configuration (optional)
            CF_ACCESS_CLIENT_ID = os.getenv("CF_ACCESS_CLIENT_ID", config.get("CF_ACCESS_CLIENT_ID"))
            CF_ACCESS_CLIENT_SECRET = os.getenv("CF_ACCESS_CLIENT_SECRET", config.get("CF_ACCESS_CLIENT_SECRET"))

            if not RAGFLOW_BASE_URL or not RAGFLOW_API_KEY:
                raise ConfigurationError(
                    "RAGFLOW_BASE_URL and RAGFLOW_API_KEY must be set in config.json or environment variables"
                )
            
            # Validate URL format
            if not RAGFLOW_BASE_URL.startswith(('http://', 'https://')):
                raise ConfigurationError("RAGFLOW_BASE_URL must start with http:// or https://")
            
            # Validate API key format (basic check)
            if len(RAGFLOW_API_KEY.strip()) < 10:
                raise ConfigurationError("RAGFLOW_API_KEY appears to be invalid (too short)")

            _ragflow_client = RAGFlowMCPServer(
                RAGFLOW_BASE_URL, 
                RAGFLOW_API_KEY, 
                RAGFLOW_DEFAULT_RERANK,
                CF_ACCESS_CLIENT_ID,
                CF_ACCESS_CLIENT_SECRET
            )
            logger.info("RAGFlow client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGFlow client: {redact_sensitive_data(str(e))}")
            raise
    
    return _ragflow_client

# Global client instance (initialized lazily)
_ragflow_client: Optional[RAGFlowMCPServer] = None

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
                "required": ["dataset_name", "query"]
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
    """Handle tool execution errors with appropriate logging and response."""
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
    dataset_id: str,
    arguments: Dict[str, Any],
    include_dataset_info: bool = False
) -> Dict[str, Any]:
    """Handle retrieval operations with common logic."""
    deepening_level = arguments.get("deepening_level", 0)
    
    retrieval_args = {
        "top_k": arguments.get("top_k", 1024),
        "similarity_threshold": arguments.get("similarity_threshold", 0.2),
        "page": arguments.get("page", 1),
        "page_size": arguments.get("page_size", 10),
        "use_rerank": arguments.get("use_rerank", False),
        "document_name": arguments.get("document_name")
    }
    
    if deepening_level > 0:
        result = await ragflow_client.retrieval_with_deepening(
            dataset_id=dataset_id,
            query=arguments["query"],
            deepening_level=deepening_level,
            **retrieval_args
        )
    else:
        result = await ragflow_client.retrieval_query(
            dataset_id=dataset_id,
            query=arguments["query"],
            **retrieval_args
        )
    
    if include_dataset_info:
        dataset_name = arguments.get("dataset_name", "unknown")
        return {
            "dataset_found": {
                "name": dataset_name,
                "id": dataset_id
            },
            "retrieval_result": result
        }
    
    return result

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle tool calls with improved error handling and organization."""
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
            result = await _handle_retrieval_tool(
                ragflow_client, arguments["dataset_id"], arguments, include_dataset_info=False
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_retrieval_by_name":
            dataset_name = arguments["dataset_name"]
            try:
                dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            except DatasetNotFoundError:
                # Try refreshing cache once
                logger.info(f"Dataset '{dataset_name}' not found, refreshing cache...")
                await ragflow_client.list_datasets()
                dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            
            result = await _handle_retrieval_tool(
                ragflow_client, dataset_id, arguments, include_dataset_info=True
            )
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "ragflow_list_documents_by_name":
            dataset_name = arguments["dataset_name"]
            try:
                dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            except DatasetNotFoundError:
                # Try refreshing cache once
                logger.info(f"Dataset '{dataset_name}' not found, refreshing cache...")
                await ragflow_client.list_datasets()
                dataset_id = await ragflow_client.find_dataset_by_name(dataset_name)
            
            result = await ragflow_client.list_documents(dataset_id)
            
            response_data = {
                "dataset_found": {
                    "name": dataset_name,
                    "id": dataset_id
                },
                "documents": result
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
    """Run the MCP server with proper cleanup."""
    # Log DSPy availability
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
                            logging=types.LoggingCapability()
                        )
                    )
                )
        finally:
            # Cleanup resources
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