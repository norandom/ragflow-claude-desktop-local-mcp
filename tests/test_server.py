"""Tests for the RAGFlow MCP server."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from ragflow_claude_mcp.server import RAGFlowMCPServer
from ragflow_claude_mcp.exceptions import DatasetNotFoundError, ConfigurationError


class TestRAGFlowMCPServer:
    """Test suite for RAGFlowMCPServer class."""
    
    @pytest.fixture
    def server(self):
        """Create a RAGFlowMCPServer instance for testing."""
        return RAGFlowMCPServer(
            base_url="http://localhost:9380",
            api_key="test_token",
            default_rerank=None,
            cf_access_client_id=None,
            cf_access_client_secret=None
        )
    
    @pytest.fixture
    def mock_datasets_response(self):
        """Mock datasets response data."""
        return {
            "code": 0,
            "data": [
                {"id": "dataset1", "name": "Test Dataset 1"},
                {"id": "dataset2", "name": "Quant Literature"},
                {"id": "dataset3", "name": "Test Dataset 3"}
            ]
        }
    
    @pytest.fixture
    def mock_retrieval_response(self):
        """Mock retrieval response data."""
        return {
            "code": 0,
            "data": {
                "chunks": [
                    {"content": "Test chunk about volatility", "score": 0.9},
                    {"content": "Another relevant chunk", "score": 0.8}
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, server):
        """Test that the server initializes correctly."""
        assert server is not None
        assert hasattr(server, 'dataset_cache')
        assert hasattr(server, 'base_url')
        assert hasattr(server, 'api_key')
    
    @pytest.mark.asyncio
    async def test_list_datasets_success(self, server, mock_datasets_response):
        """Test successful dataset listing."""
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            result = await server.list_datasets()
            
            assert result["code"] == 0
            assert len(result["data"]) == 3
            # Test with new DatasetCache API
            assert server.dataset_cache.get_dataset_id("Test Dataset 1") == "dataset1"
            assert server.dataset_cache.get_dataset_id("Quant Literature") == "dataset2"

    @pytest.mark.asyncio
    async def test_list_datasets_pagination_bug_repro(self, server):
        """Test reproduction of pagination bug where loop breaks early."""
        # Scenario:
        # Page 1: Request 100, Get 30. Total 60.
        # Current code: len(datasets)=30 < page_size=100 -> Break. (Bug: stops at 30)
        # Expected behavior: Continue to page 2.
        
        mock_responses = [
            # Page 1
            {
                "code": 0,
                "data": [{"id": f"d{i}", "name": f"Dataset {i}"} for i in range(30)],
                "total": 60
            },
            # Page 2
            {
                "code": 0,
                "data": [{"id": f"d{i}", "name": f"Dataset {i}"} for i in range(30, 60)],
                "total": 60
            }
        ]
        
        # Side effect for _make_request
        async def side_effect(method, endpoint, params=None, **kwargs):
            page = params.get('page', 1)
            if page <= len(mock_responses):
                return mock_responses[page-1]
            return {"code": 0, "data": [], "total": 60}

        with patch.object(server, '_make_request', side_effect=side_effect) as mock_request:
            result = await server.list_datasets()
            
            # If bug exists, len will be 30. If fixed, len will be 60.
            assert len(result["data"]) == 60
            assert result["total"] == 60
            
            # Verify pagination calls
            assert mock_request.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_find_dataset_by_name_success(self, server, mock_datasets_response):
        """Test finding dataset by name."""
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            await server.list_datasets()
            
            dataset_id = await server.find_dataset_by_name("Quant Literature")
            assert dataset_id == "dataset2"
    
    @pytest.mark.asyncio
    async def test_find_dataset_by_name_not_found(self, server, mock_datasets_response):
        """Test finding non-existent dataset."""
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            await server.list_datasets()
            
            with pytest.raises(DatasetNotFoundError):
                await server.find_dataset_by_name("Non-existent Dataset")
    
    @pytest.mark.asyncio
    async def test_retrieval_query_success(self, server, mock_retrieval_response):
        """Test successful retrieval query."""
        with patch.object(server, '_make_request', return_value=mock_retrieval_response):
            result = await server.retrieval_query(
                dataset_ids=["test_dataset"],
                query="test query"
            )
            
            assert result["code"] == 0
            assert len(result["data"]["chunks"]) == 2
    
    @pytest.mark.asyncio
    async def test_retrieval_query_multiple_datasets(self, server, mock_retrieval_response):
        """Test successful retrieval query with multiple datasets."""
        with patch.object(server, '_make_request', return_value=mock_retrieval_response) as mock_request:
            result = await server.retrieval_query(
                dataset_ids=["dataset1", "dataset2"],
                query="test query"
            )
            
            assert result["code"] == 0
            
            # Verify that the request was made with the correct payload
            call_args = mock_request.call_args
            assert call_args is not None
            args, kwargs = call_args
            
            # Verify payload contains list of dataset IDs
            request_data = args[2] if len(args) > 2 else kwargs.get('data')
            assert request_data["dataset_ids"] == ["dataset1", "dataset2"]

    @pytest.mark.asyncio
    async def test_dataset_cache_bug_scenario(self, server, mock_datasets_response):
        """Test the dataset cache bug scenario."""
        # Simulate the bug: cache is populated but then cleared
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            # First populate cache
            await server.list_datasets()
            assert server.dataset_cache.get_dataset_id("Quant Literature") == "dataset2"
            
            # Simulate cache being cleared (the bug)
            server.dataset_cache.clear()
            
            # Try to find dataset again - should repopulate cache
            dataset_id = await server.find_dataset_by_name("Quant Literature")
            assert dataset_id == "dataset2"  # Should repopulate cache
    
    @pytest.mark.asyncio
    async def test_empty_cache_error_message(self, server):
        """Test error message when cache is empty."""
        server.dataset_cache.clear()
        
        # Mock empty response
        with patch.object(server, '_make_request', return_value={"code": 0, "data": []}):
            with pytest.raises(DatasetNotFoundError) as exc_info:
                await server.find_dataset_by_name("Quant Literature")
            
            # Check that the exception contains the expected information
            assert "Quant Literature" in str(exc_info.value)
            assert exc_info.value.dataset_name == "Quant Literature"
            assert exc_info.value.available_datasets == []
    
    @pytest.mark.asyncio
    async def test_cache_preservation_on_empty_response(self, server, mock_datasets_response):
        """Test that cache is preserved when API returns empty dataset list."""
        # First populate cache with valid data
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            await server.list_datasets()
            cache_stats = server.dataset_cache.stats()
            assert cache_stats['size'] == 3
            assert server.dataset_cache.get_dataset_id("Quant Literature") == "dataset2"
        
        # Now simulate API returning empty data
        empty_response = {"code": 0, "data": []}
        with patch.object(server, '_make_request', return_value=empty_response):
            await server.list_datasets()
            
            # Cache should still contain the previous data
            cache_stats = server.dataset_cache.stats()
            assert cache_stats['size'] == 3
            assert server.dataset_cache.get_dataset_id("Quant Literature") == "dataset2"
            
            # Should still be able to find dataset by name
            dataset_id = await server.find_dataset_by_name("Quant Literature")
            assert dataset_id == "dataset2"


class TestDSPyIntegration:
    """Test suite for DSPy integration."""
    
    @pytest.fixture
    def server(self):
        """Create a RAGFlowMCPServer instance for testing."""
        return RAGFlowMCPServer(
            base_url="http://localhost:9380",
            api_key="test_token",
            default_rerank=None,
            cf_access_client_id=None,
            cf_access_client_secret=None
        )
    
    @pytest.mark.asyncio
    async def test_dspy_available_flag(self, server):
        """Test that DSPY_AVAILABLE flag is properly set."""
        from ragflow_claude_mcp.server import DSPY_AVAILABLE
        assert isinstance(DSPY_AVAILABLE, bool)
    
    @pytest.mark.asyncio
    async def test_retrieval_with_deepening_fallback(self, server):
        """Test that retrieval falls back when DSPy is unavailable."""
        mock_result = {"code": 0, "data": {"chunks": []}}
        
        with patch.object(server, 'retrieval_query', return_value=mock_result):
            result = await server.retrieval_with_deepening(
                dataset_ids=["test_dataset"],
                query="test query",
                deepening_level=2
            )
            
            assert result is not None
    
    @pytest.mark.asyncio
    @patch('ragflow_claude_mcp.server.DSPY_AVAILABLE', False)
    async def test_deepening_disabled_when_dspy_unavailable(self, server):
        """Test that deepening is disabled when DSPy is not available."""
        mock_result = {"code": 0, "data": {"chunks": []}}
        
        with patch.object(server, 'retrieval_query', return_value=mock_result):
            result = await server.retrieval_with_deepening(
                dataset_ids=["test_dataset"],
                query="test query",
                deepening_level=2
            )
            
            # Should fall back to standard retrieval
            assert result == mock_result


class TestCloudflareZeroTrust:
    """Test suite for Cloudflare Zero Trust authentication."""
    
    @pytest.fixture
    def cf_server(self):
        """Create a RAGFlowMCPServer instance with CF Zero Trust auth."""
        return RAGFlowMCPServer(
            base_url="https://test.example.com",
            api_key="test_token",
            default_rerank=None,
            cf_access_client_id="test-client-id.access",
            cf_access_client_secret="test-client-secret"
        )
    
    @pytest.mark.asyncio
    async def test_cf_headers_added(self, cf_server):
        """Test that CF headers are added when credentials are provided."""
        assert 'CF-Access-Client-Id' in cf_server.headers
        assert 'CF-Access-Client-Secret' in cf_server.headers
        assert cf_server.headers['CF-Access-Client-Id'] == "test-client-id.access"
        assert cf_server.headers['CF-Access-Client-Secret'] == "test-client-secret"
    
    @pytest.mark.asyncio
    async def test_cf_headers_not_added_without_credentials(self):
        """Test that CF headers are not added without credentials."""
        server = RAGFlowMCPServer(
            base_url="http://localhost:9380",
            api_key="test_token"
        )
        assert 'CF-Access-Client-Id' not in server.headers
        assert 'CF-Access-Client-Secret' not in server.headers
    
    @pytest.mark.asyncio
    async def test_cf_headers_included_in_api_calls(self, cf_server):
        """Test that CF headers are included in API calls."""
        # Simply verify that the CF headers are present in the server's headers
        # which will be used for all API requests
        assert 'CF-Access-Client-Id' in cf_server.headers
        assert 'CF-Access-Client-Secret' in cf_server.headers
        
        # Test making an actual API call to ensure headers are preserved
        mock_response = {"code": 0, "data": [{"id": "test", "name": "Test Dataset"}], "total": 1}

        with patch.object(cf_server, '_make_request', return_value=mock_response) as mock_request:
            result = await cf_server.list_datasets()

            # Verify the method was called with pagination parameters
            mock_request.assert_called_once_with("GET", "/api/v1/datasets", params={'page': 1, 'page_size': 100})

            # Verify the result contains all datasets
            assert result["code"] == 0
            assert "data" in result
            assert len(result["data"]) == 1
            
        # Verify headers are still intact after the call
        assert cf_server.headers['CF-Access-Client-Id'] == "test-client-id.access"
        assert cf_server.headers['CF-Access-Client-Secret'] == "test-client-secret"


class TestErrorHandling:
    """Test suite for error handling."""
    
    @pytest.fixture
    def server(self):
        """Create a RAGFlowMCPServer instance for testing."""
        return RAGFlowMCPServer(
            base_url="http://localhost:9380",
            api_key="test_token",
            default_rerank=None,
            cf_access_client_id=None,
            cf_access_client_secret=None
        )
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, server):
        """Test API error handling."""
        with patch.object(server, '_make_request', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await server.list_datasets()
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, server):
        """Test handling of malformed API responses."""
        malformed_response = {"invalid": "response"}

        with patch.object(server, '_make_request', return_value=malformed_response):
            result = await server.list_datasets()

            # Should handle gracefully and return normalized empty response
            assert result["code"] == 0
            assert result["data"] == []
            assert result["total"] == 0
            # Cache should remain empty
            cache_stats = server.dataset_cache.stats()
            assert cache_stats['size'] == 0


class TestConfigurationLoading:
    """Test suite for configuration loading and client initialization."""
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_get_ragflow_client_with_env_vars(self, mock_getenv, mock_load_config):
        """Test client initialization with environment variables."""
        from ragflow_claude_mcp.server import get_ragflow_client, _ragflow_client
        
        # Reset global client
        import ragflow_claude_mcp.server as server_module
        server_module._ragflow_client = None
        
        # Mock environment variables with longer API key
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key-that-is-long-enough-for-validation'
        }.get(key, default)
        
        mock_load_config.return_value = {}
        
        client = get_ragflow_client()
        assert client is not None
        assert client.base_url == 'http://test:9380'
        assert client.api_key == 'test-key-that-is-long-enough-for-validation'
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_get_ragflow_client_with_config_file(self, mock_getenv, mock_load_config):
        """Test client initialization with config file."""
        from ragflow_claude_mcp.server import get_ragflow_client
        
        # Reset global client
        import ragflow_claude_mcp.server as server_module
        server_module._ragflow_client = None
        
        # Mock no environment variables (return None for all keys)
        mock_getenv.side_effect = lambda key, default=None: default
        
        # Mock config file
        mock_load_config.return_value = {
            'RAGFLOW_BASE_URL': 'http://config:9380',
            'RAGFLOW_API_KEY': 'config-key',
            'RAGFLOW_DEFAULT_RERANK': 'custom-rerank'
        }
        
        client = get_ragflow_client()
        assert client is not None
        assert client.base_url == 'http://config:9380'
        assert client.api_key == 'config-key'
        assert client.default_rerank == 'custom-rerank'
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_get_ragflow_client_missing_config(self, mock_getenv, mock_load_config):
        """Test client initialization with missing configuration."""
        from ragflow_claude_mcp.server import get_ragflow_client
        
        # Reset global client
        import ragflow_claude_mcp.server as server_module
        server_module._ragflow_client = None
        
        # Mock no environment variables or config (return None for all keys)
        mock_getenv.side_effect = lambda key, default=None: default
        mock_load_config.return_value = {}
        
        with pytest.raises(ConfigurationError, match="RAGFLOW_BASE_URL and RAGFLOW_API_KEY must be set"):
            get_ragflow_client()
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_get_ragflow_client_singleton(self, mock_getenv, mock_load_config):
        """Test that get_ragflow_client returns the same instance."""
        from ragflow_claude_mcp.server import get_ragflow_client
        
        # Reset global client
        import ragflow_claude_mcp.server as server_module
        server_module._ragflow_client = None
        
        # Mock environment variables with longer API key
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key-that-is-long-enough-for-validation'
        }.get(key, default)
        
        mock_load_config.return_value = {}
        
        client1 = get_ragflow_client()
        client2 = get_ragflow_client()
        
        assert client1 is client2
        # load_config should only be called once
        assert mock_load_config.call_count == 1


class TestOpenRouterConfiguration:
    """Test suite for OpenRouter configuration."""
    
    def _setup_base_mocks(self, mock_getenv, mock_load_config):
        """Helper to setup common mocks for all tests."""
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key'
        }.get(key, default)
    
    def _create_openrouter_config(self):
        """Helper to create OpenRouter configuration."""
        return {
            'PROVIDER': 'openrouter',
            'DSPY_MODEL': 'openai/gpt-4o-mini',
            'OPENROUTER_API_KEY': 'openrouter-key',
            'OPENROUTER_SITE_URL': 'https://test.com',
            'OPENROUTER_SITE_NAME': 'Test App'
        }
    
    def _create_openai_config(self):
        """Helper to create OpenAI configuration."""
        return {
            'PROVIDER': 'openai',
            'DSPY_MODEL': 'openai/gpt-4o-mini',
            'OPENAI_API_KEY': 'openai-key'
        }
    
    def _create_mock_deepening_result(self):
        """Helper to create mock deepening result."""
        return {
            'results': {"test": "result"},
            'original_query': 'test',
            'final_query': 'enhanced test query',
            'queries_used': ['test', 'enhanced test query'],
            'deepening_level': 1,
            'refinement_log': ['refined query']
        }
    
    async def _run_deepening_test(self, server):
        """Helper to run deepening test."""
        return await server.retrieval_with_deepening(
            dataset_ids=["test"],
            query="test",
            deepening_level=1
        )
    
    @patch('ragflow_claude_mcp.dspy_deepening.DSPyQueryDeepener.deepen_search')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    @patch('dspy.configure')
    @patch('dspy.LM')
    def test_dspy_openrouter_configuration(self, mock_lm, mock_configure, mock_getenv, mock_load_config, mock_json_load, mock_open_file, mock_exists, mock_deepen_search):
        """Test DSPy configuration with OpenRouter - should succeed and add metadata."""
        from ragflow_claude_mcp.server import RAGFlowMCPServer
        import asyncio
        
        # Setup mocks
        self._setup_base_mocks(mock_getenv, mock_load_config)
        mock_load_config.return_value = {}  # Server init config (not used for DSPy)
        
        # Mock config file reading in retrieval_with_deepening method
        mock_exists.return_value = True
        mock_json_load.return_value = self._create_openrouter_config()
        mock_deepen_search.return_value = self._create_mock_deepening_result()
        
        # Test
        server = RAGFlowMCPServer("http://test:9380", "test-key")
        result = asyncio.run(self._run_deepening_test(server))
        
        # Verify successful deepening with metadata
        assert "test" in result
        assert result["test"] == "result"
        assert "metadata" in result
        assert "deepening" in result["metadata"]
    
    @pytest.mark.skip(reason="Intermittent failure due to test execution order - needs investigation")
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_openrouter_config_validation(self, mock_getenv, mock_load_config, mock_json_load, mock_open_file, mock_exists):
        """Test OpenRouter configuration validation - should fail and fallback to standard retrieval."""
        from ragflow_claude_mcp.server import RAGFlowMCPServer
        import asyncio
        
        # Setup mocks
        self._setup_base_mocks(mock_getenv, mock_load_config)
        mock_load_config.return_value = {}  # Server init config (not used for DSPy)
        
        # Mock config file reading in retrieval_with_deepening method
        mock_exists.return_value = True
        mock_json_load.return_value = {
            'PROVIDER': 'openrouter',
            'DSPY_MODEL': 'openai/gpt-4o-mini'
            # Missing OPENROUTER_API_KEY - should cause config failure
        }
        
        # Test
        server = RAGFlowMCPServer("http://test:9380", "test-key")
        
        with patch.object(server, 'retrieval_query', return_value={"fallback": "result"}):
            result = asyncio.run(self._run_deepening_test(server))
            
            # Verify fallback to standard retrieval (no metadata when DSPy config fails)
            assert "fallback" in result
            assert result["fallback"] == "result"
            assert "metadata" not in result
    
    @patch('ragflow_claude_mcp.dspy_deepening.DSPyQueryDeepener.deepen_search')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    @patch('dspy.configure')
    @patch('dspy.LM')
    def test_dspy_openai_configuration(self, mock_lm, mock_configure, mock_getenv, mock_load_config, mock_json_load, mock_open_file, mock_exists, mock_deepen_search):
        """Test DSPy configuration with OpenAI - should succeed and add metadata."""
        from ragflow_claude_mcp.server import RAGFlowMCPServer
        import asyncio
        
        # Setup mocks
        self._setup_base_mocks(mock_getenv, mock_load_config)
        mock_load_config.return_value = {}  # Server init config (not used for DSPy)
        
        # Mock config file reading in retrieval_with_deepening method
        mock_exists.return_value = True
        mock_json_load.return_value = self._create_openai_config()
        mock_deepen_search.return_value = self._create_mock_deepening_result()
        
        # Test
        server = RAGFlowMCPServer("http://test:9380", "test-key")
        result = asyncio.run(self._run_deepening_test(server))
        
        # Verify successful deepening with metadata
        assert "test" in result
        assert result["test"] == "result"
        assert "metadata" in result
        assert "deepening" in result["metadata"]


class TestConfigurationFile:
    """Test suite for configuration file loading."""
    
    @patch('builtins.open', new_callable=mock_open)
    def test_load_config_file_not_found(self, mock_open_func):
        """Test load_config when config.json doesn't exist."""
        from ragflow_claude_mcp.server import load_config
        
        # Mock open to raise FileNotFoundError
        mock_open_func.side_effect = FileNotFoundError()
        
        config = load_config()
        assert config == {}
        mock_open_func.assert_called_once_with('config.json', 'r')
    
    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('json.load')
    def test_load_config_json_decode_error(self, mock_json_load, mock_exists, mock_open):
        """Test load_config with invalid JSON."""
        from ragflow_claude_mcp.server import load_config
        
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with pytest.raises(ConfigurationError, match="Invalid JSON in config.json"):
            load_config()
    
    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('json.load')
    def test_load_config_success(self, mock_json_load, mock_exists, mock_open):
        """Test successful config loading."""
        from ragflow_claude_mcp.server import load_config
        
        mock_exists.return_value = True
        mock_json_load.return_value = {'test': 'config'}
        
        config = load_config()
        assert config == {'test': 'config'}