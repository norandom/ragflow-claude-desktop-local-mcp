"""HTTP client for RAGFlow's REST API.

The RAGFlowMCPServer class wraps the RAGFlow `/api/v1` endpoints used by the
MCP server: retrieval, listing datasets/documents, fetching chunks. It owns
the shared aiohttp session and the dataset name->ID cache.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

from ..common.exceptions import (
    RAGFlowAPIError, RAGFlowConnectionError,
    ValidationError, DatasetNotFoundError,
    DSPyConfigurationError,
)
from ..common.validation import (
    validate_dataset_id, validate_dataset_ids, validate_document_id,
    validate_query, validate_dataset_name, validate_dataset_names,
    validate_document_name, validate_pagination_params,
    validate_similarity_threshold, validate_top_k, validate_deepening_level,
    redact_sensitive_data,
)
from ..common.cache import DatasetCache

# DSPy is optional. If it's not installed, retrieval_with_deepening falls back
# to standard retrieval. The actual `DSPY_AVAILABLE` flag lives on the server
# module so tests can patch it there; we read it lazily inside the method.
try:
    import dspy
    from ..dspy.deepening import get_deepener
except ImportError:
    dspy = None
    get_deepener = None

logger = logging.getLogger("ragflow-mcp")


class RAGFlowMCPServer:
    def __init__(self, base_url: str, api_key: str, default_rerank: Optional[str] = None,
                 cf_access_client_id: Optional[str] = None, cf_access_client_secret: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.default_rerank = default_rerank
        self.cf_access_client_id = cf_access_client_id
        self.cf_access_client_secret = cf_access_client_secret

        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Cloudflare Zero Trust headers when both service-token values are set.
        if cf_access_client_id and cf_access_client_secret:
            self.headers['CF-Access-Client-Id'] = cf_access_client_id
            self.headers['CF-Access-Client-Secret'] = cf_access_client_secret
            logger.info("Cloudflare Zero Trust authentication enabled")

        self.active_sessions: Dict[str, str] = {}  # dataset_id -> chat_id
        self.dataset_cache = DatasetCache()
        self._session: Optional[aiohttp.ClientSession] = None  # one shared aiohttp session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-create the HTTP session and reuse it across requests."""
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
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a request to the RAGFlow API. Wraps errors as RAGFlow*Error.

        Args:
            method: HTTP method
            endpoint: API path (joined to base_url)
            data: JSON body for POST/PATCH/etc.
            params: query string for GET
        """
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        try:
            async with session.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                params=params
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
        dataset_ids: List[str],
        query: str,
        top_k: int = 1024,
        similarity_threshold: float = 0.2,
        page: int = 1,
        page_size: int = 10,
        use_rerank: bool = False,
        document_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Hit RAGFlow's /retrieval endpoint directly. Returns raw chunks with similarity scores.

        Args:
            dataset_ids: list of dataset IDs to search across
            query: the search query
            top_k: vector candidate pool size
            similarity_threshold: 0.0-1.0; chunks below this score are filtered out
            page: 1-based page number
            page_size: chunks per page
            use_rerank: enable reranking (currently broken upstream — see README)
            document_name: optional document filter (fuzzy match)

        Raises:
            ValidationError: input validation failed
            RAGFlowAPIError: the API call failed
        """
        dataset_ids = validate_dataset_ids(dataset_ids)
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
            "dataset_ids": dataset_ids,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "page": page,
            "page_size": page_size
        }

        dataset_ids_str = ",".join(dataset_ids)
        logger.info(f"Executing retrieval query for datasets [{dataset_ids_str[:50]}...] with {len(query)} char query")

        # Document filter is per-dataset (RAGFlow document IDs are scoped to datasets).
        # When several datasets are requested, we resolve the doc against the first one.
        document_match_info = None
        if document_name:
            primary_dataset_id = dataset_ids[0]
            match_result = await self.find_document_by_name(primary_dataset_id, document_name)
            if match_result['status'] == 'single_match':
                data["document_ids"] = [match_result['document_id']]
                logger.info(f"Filtering results to document '{match_result['document_name']}' (ID: {match_result['document_id'][:10]}...) in dataset {primary_dataset_id[:10]}...")
                document_match_info = match_result
            elif match_result['status'] == 'multiple_matches':
                # Pick the best match; alternatives end up in metadata.
                data["document_ids"] = [match_result['document_id']]
                logger.info(f"Multiple documents match '{document_name}'. Using best match: '{match_result['document_name']}' (ID: {match_result['document_id'][:10]}...)")
                document_match_info = match_result
            else:
                logger.warning(f"Document '{document_name}' not found in dataset {primary_dataset_id[:10]}..., proceeding without filtering")
                document_match_info = match_result

        # Rerank is opt-in.
        if use_rerank and self.default_rerank:
            data["rerank_id"] = self.default_rerank

        try:
            result = await self._make_request("POST", endpoint, data)

            if not isinstance(result, dict):
                raise RAGFlowAPIError("Invalid response format from RAGFlow API")

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
        """Set up DSPy's LM if it isn't already. Returns True on success."""
        if dspy is None:
            return False
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
            return True

        try:
            # config.json sits at the project root, three levels up from this file.
            config_path = os.path.expanduser(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.json')
            )
            config = {}

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

            dspy_model = config.get('DSPY_MODEL', 'openai/gpt-4o-mini')

            lm_kwargs = {}
            provider = config.get('PROVIDER', 'openai').lower()

            if provider == 'openrouter':
                if 'OPENROUTER_API_KEY' not in config:
                    raise DSPyConfigurationError("OPENROUTER_API_KEY must be set when PROVIDER is 'openrouter'")
                if 'OPENAI_API_KEY' not in os.environ:
                    os.environ['OPENAI_API_KEY'] = config['OPENROUTER_API_KEY']
                lm_kwargs['api_base'] = 'https://openrouter.ai/api/v1'
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

            dspy.configure(lm=dspy.LM(dspy_model, **lm_kwargs))
            return True

        except Exception as e:
            logger.error(f"Failed to configure DSPy LM: {e}")
            return False

    async def retrieval_with_deepening(
        self,
        dataset_ids: List[str],
        query: str,
        deepening_level: int = 0,
        document_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieval with optional DSPy refinement loop on top.

        Args:
            dataset_ids: list of dataset IDs to search across
            query: the search query
            deepening_level: 0 = standard retrieval, 1-3 = DSPy refinement passes
            document_name: optional document filter
            **kwargs: forwarded to retrieval_query

        Returns the standard retrieval result, with `metadata.deepening` populated
        when refinement ran.
        """
        # DSPY_AVAILABLE lives on the server module so tests can patch it there.
        # Late import avoids a circular import at module load time.
        from ragflow_claude_mcp import server as _server

        dataset_ids = validate_dataset_ids(dataset_ids)
        query = validate_query(query)
        deepening_level = validate_deepening_level(deepening_level) or 0

        if document_name:
            document_name = validate_document_name(document_name)

        if deepening_level <= 0 or not _server.DSPY_AVAILABLE:
            if deepening_level > 0 and not _server.DSPY_AVAILABLE:
                logger.warning("DSPy deepening requested but not available, falling back to standard retrieval")
            return await self.retrieval_query(dataset_ids, query, document_name=document_name, **kwargs)

        logger.info(f"Starting DSPy query deepening (level {deepening_level}) for query: '{query[:50]}...'")

        try:
            if not self._configure_dspy_if_needed():
                logger.warning("DSPy configuration failed, falling back to standard retrieval")
                return await self.retrieval_query(dataset_ids, query, document_name=document_name, **kwargs)

            deepener = get_deepener()

            deepening_result = await deepener.deepen_search(
                ragflow_client=self,
                dataset_ids=dataset_ids,
                original_query=query,
                deepening_level=deepening_level,
                **kwargs
            )

            final_results = deepening_result['results']

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
            return await self.retrieval_query(dataset_ids, query, document_name=document_name, **kwargs)
        except Exception as e:
            logger.error(f"DSPy query deepening failed: {redact_sensitive_data(str(e))}")
            logger.info("Falling back to standard retrieval")
            return await self.retrieval_query(dataset_ids, query, document_name=document_name, **kwargs)

    async def list_datasets(self) -> Dict[str, Any]:
        """List datasets across all pages and refresh the name → ID cache as a side effect.

        Walks the paginated API (~30 entries/page upstream; we ask for 100 to cut requests).
        """
        endpoint = "/api/v1/datasets"
        all_datasets = []
        page = 1
        page_size = 100
        total_count = None

        try:
            while True:
                params = {"page": page, "page_size": page_size}
                result = await self._make_request("GET", endpoint, params=params)

                if not isinstance(result, dict):
                    raise RAGFlowAPIError("Invalid response format from datasets API")

                if result.get("code") != 0:
                    error_msg = result.get('message', 'Unknown error')
                    logger.error(f"Failed to list datasets: code={result.get('code')}, message={error_msg}")
                    break

                datasets = result.get("data", [])
                if not isinstance(datasets, list):
                    logger.warning(f"Invalid data format on page {page}")
                    break

                if total_count is None:
                    total_count = result.get("total", 0)
                    if total_count:
                        logger.info(f"Fetching {total_count} datasets across multiple pages...")

                if not datasets:
                    break

                all_datasets.extend(datasets)

                if total_count is not None and len(all_datasets) >= total_count:
                    break

                if page > 200:
                    logger.warning("Reached maximum page limit (200) for datasets listing")
                    break

                page += 1

            if all_datasets:
                dataset_mapping = {
                    dataset["name"]: dataset["id"]
                    for dataset in all_datasets
                    if isinstance(dataset, dict) and "name" in dataset and "id" in dataset
                }

                if dataset_mapping:
                    self.dataset_cache.set_datasets(dataset_mapping)
                    cache_stats = self.dataset_cache.stats()
                    logger.info(f"Dataset cache updated with {cache_stats['size']} datasets")
                else:
                    logger.warning("No valid datasets found in API response")
            else:
                logger.info("No datasets found in RAGFlow instance")

            return {
                "code": 0,
                "data": all_datasets,
                "total": len(all_datasets)
            }

        except (RAGFlowAPIError, RAGFlowConnectionError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing datasets: {redact_sensitive_data(str(e))}")
            raise RAGFlowAPIError(f"Failed to list datasets: {str(e)}")

    async def find_datasets_by_names(self, names: List[str]) -> List[str]:
        """Resolve a list of dataset names to IDs (case-insensitive)."""
        dataset_ids = []
        names = validate_dataset_names(names)

        for name in names:
            dataset_id = await self.find_dataset_by_name(name)
            if dataset_id:
                dataset_ids.append(dataset_id)

        return dataset_ids

    async def find_document_by_name(self, dataset_id: str, document_name: str) -> Dict[str, Any]:
        """Look up a document by name in a dataset.

        Returns a dict with `status`: `single_match`, `multiple_matches`, `not_found`,
        or `error`. When several documents match, all matches are included so the
        caller can pick a more specific name.

        Raises:
            ValidationError: input validation failed
        """
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

                matches = []
                document_name_lower = document_name.lower()

                for doc in documents:
                    if not isinstance(doc, dict) or 'name' not in doc:
                        continue

                    doc_name = doc.get('name', '').lower()
                    if document_name_lower in doc_name or doc_name in document_name_lower:
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

                # Score first, then update_time as tiebreaker (most recent wins).
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
                        'document_id': matches[0]['id'],
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
        """Score a candidate document name. Higher = better match."""
        score = 0.0

        # Tiered match strength: exact > prefix > substring > reverse-substring.
        if doc_name == search_term:
            score += 100.0
        elif doc_name.startswith(search_term):
            score += 50.0
        elif search_term in doc_name:
            score += 30.0
        elif doc_name in search_term:
            score += 20.0

        update_time = doc.get('update_time', '')
        if update_time:
            try:
                score += 0.1
            except (ValueError, TypeError):
                logger.debug(f"Invalid update_time format: {update_time}")

        # Bias toward names that look recent.
        if any(keyword in doc_name for keyword in ['2024', '2023', 'latest', 'current', 'new']):
            score += 5.0

        return score

    async def find_dataset_by_name(self, name: str) -> Optional[str]:
        """Resolve a dataset name to its ID. Case-insensitive, with a fuzzy fallback."""
        name = validate_dataset_name(name)

        # Cache hit is the common path.
        dataset_id = self.dataset_cache.get_dataset_id(name)
        if dataset_id:
            logger.debug(f"Found dataset '{name}' in cache")
            return dataset_id

        # Cache miss: refresh once and retry.
        logger.info(f"Dataset '{name}' not found in cache, refreshing...")
        try:
            await self.list_datasets()
            dataset_id = self.dataset_cache.get_dataset_id(name)
            if dataset_id:
                logger.debug(f"Found dataset '{name}' after cache refresh")
                return dataset_id
        except Exception as e:
            logger.error(f"Failed to refresh dataset cache: {redact_sensitive_data(str(e))}")

        # Last try: fuzzy match against whatever names we know about.
        available_names = self.dataset_cache.get_all_names()
        name_lower = name.lower()

        matches = [
            available_name for available_name in available_names
            if name_lower in available_name or available_name in name_lower
        ]

        if matches:
            matched_name = matches[0]
            dataset_id = self.dataset_cache.get_dataset_id(matched_name)
            logger.info(f"Found fuzzy match for '{name}' -> '{matched_name}'")
            return dataset_id

        logger.warning(f"No dataset found for '{name}'. Available datasets: {available_names[:10]}")
        raise DatasetNotFoundError(name, available_names)

    async def upload_documents(self, dataset_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """Stub. Real upload needs multipart/form-data; not wired up over MCP yet."""
        endpoint = f"/api/v1/datasets/{dataset_id}/documents"
        return {"message": "Upload endpoint available but requires file handling"}

    async def list_documents(self, dataset_id: str) -> Dict[str, Any]:
        """List documents in a dataset, walking through all pages.

        Upstream defaults to 10 per page; we ask for 100 to cut request count.
        """
        dataset_id = validate_dataset_id(dataset_id)
        endpoint = f"/api/v1/datasets/{dataset_id}/documents"
        all_documents = []
        page = 1
        page_size = 100
        total_count = None

        try:
            while True:
                params = {"page": page, "page_size": page_size}
                result = await self._make_request("GET", endpoint, params=params)

                if not isinstance(result, dict):
                    raise RAGFlowAPIError("Invalid response format from documents API")

                if result.get("code") != 0:
                    error_msg = result.get('message', 'Unknown error')
                    logger.error(f"Failed to list documents: code={result.get('code')}, message={error_msg}")
                    break

                data = result.get("data", {})
                if not isinstance(data, dict):
                    logger.warning(f"Invalid data format on page {page}")
                    break

                documents = data.get("docs", [])
                if not isinstance(documents, list):
                    logger.warning(f"Invalid docs format on page {page}")
                    break

                if total_count is None:
                    total_count = data.get("total") or result.get("total")

                if not documents:
                    break

                all_documents.extend(documents)

                if total_count is not None and len(all_documents) >= total_count:
                    break

                if page > 1000:
                    logger.warning("Reached maximum page limit (1000) for documents listing")
                    break

                page += 1

            logger.info(f"Retrieved {len(all_documents)} documents for dataset {dataset_id[:10]}...")

            return {
                "code": 0,
                "data": {
                    "docs": all_documents,
                    "total": len(all_documents)
                }
            }

        except (RAGFlowAPIError, RAGFlowConnectionError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing documents: {redact_sensitive_data(str(e))}")
            raise RAGFlowAPIError(f"Failed to list documents: {str(e)}")

    async def get_document_chunks(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """Get chunks for a single document."""
        dataset_id = validate_dataset_id(dataset_id)
        document_id = validate_document_id(document_id)
        endpoint = f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"
        return await self._make_request("GET", endpoint)
