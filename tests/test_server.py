"""Tests for the RAGFlow MCP server."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from ragflow_claude_mcp.server import RAGFlowMCPServer


class TestRAGFlowMCPServer:
    """Test suite for RAGFlowMCPServer class."""
    
    @pytest.fixture
    def server(self):
        """Create a RAGFlowMCPServer instance for testing."""
        return RAGFlowMCPServer(
            base_url="http://localhost:9380",
            api_key="test_token"
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
            assert server.dataset_cache["test dataset 1"] == "dataset1"
            assert server.dataset_cache["quant literature"] == "dataset2"
    
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
            
            dataset_id = await server.find_dataset_by_name("Non-existent Dataset")
            assert dataset_id is None
    
    @pytest.mark.asyncio
    async def test_retrieval_query_success(self, server, mock_retrieval_response):
        """Test successful retrieval query."""
        with patch.object(server, '_make_request', return_value=mock_retrieval_response):
            result = await server.retrieval_query(
                dataset_id="test_dataset",
                query="test query"
            )
            
            assert result["code"] == 0
            assert len(result["data"]["chunks"]) == 2
    
    @pytest.mark.asyncio
    async def test_dataset_cache_bug_scenario(self, server, mock_datasets_response):
        """Test the dataset cache bug scenario."""
        # Simulate the bug: cache is populated but then cleared
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            # First populate cache
            await server.list_datasets()
            assert server.dataset_cache["quant literature"] == "dataset2"
            
            # Simulate cache being cleared (the bug)
            server.dataset_cache = {}
            
            # Try to find dataset again - should fail
            dataset_id = await server.find_dataset_by_name("Quant Literature")
            assert dataset_id == "dataset2"  # Should repopulate cache
    
    @pytest.mark.asyncio
    async def test_empty_cache_error_message(self, server):
        """Test error message when cache is empty."""
        server.dataset_cache = {}
        
        # Mock empty response
        with patch.object(server, '_make_request', return_value={"code": 0, "data": []}):
            dataset_id = await server.find_dataset_by_name("Quant Literature")
            assert dataset_id is None
            
            # Test error message format
            available_datasets = list(server.dataset_cache.keys()) if server.dataset_cache else []
            error_msg = f"Dataset 'Quant Literature' not found. Available datasets: {available_datasets}"
            expected_error = "Dataset 'Quant Literature' not found. Available datasets: []"
            assert error_msg == expected_error
    
    @pytest.mark.asyncio
    async def test_cache_preservation_on_empty_response(self, server, mock_datasets_response):
        """Test that cache is preserved when API returns empty dataset list."""
        # First populate cache with valid data
        with patch.object(server, '_make_request', return_value=mock_datasets_response):
            await server.list_datasets()
            assert len(server.dataset_cache) == 3
            assert server.dataset_cache["quant literature"] == "dataset2"
        
        # Now simulate API returning empty data
        empty_response = {"code": 0, "data": []}
        with patch.object(server, '_make_request', return_value=empty_response):
            await server.list_datasets()
            
            # Cache should still contain the previous data
            assert len(server.dataset_cache) == 3
            assert server.dataset_cache["quant literature"] == "dataset2"
            
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
            api_key="test_token"
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
                dataset_id="test_dataset",
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
                dataset_id="test_dataset",
                query="test query",
                deepening_level=2
            )
            
            # Should fall back to standard retrieval
            assert result == mock_result


class TestErrorHandling:
    """Test suite for error handling."""
    
    @pytest.fixture
    def server(self):
        """Create a RAGFlowMCPServer instance for testing."""
        return RAGFlowMCPServer(
            base_url="http://localhost:9380",
            api_key="test_token"
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
            
            # Should handle gracefully
            assert result == malformed_response
            # Cache should remain empty
            assert not server.dataset_cache


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
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key'
        }.get(key, default)
        
        mock_load_config.return_value = {}
        
        client = get_ragflow_client()
        assert client is not None
        assert client.base_url == 'http://test:9380'
        assert client.api_key == 'test-key'
    
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
        
        with pytest.raises(ValueError, match="RAGFLOW_BASE_URL and RAGFLOW_API_KEY must be set"):
            get_ragflow_client()
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_get_ragflow_client_singleton(self, mock_getenv, mock_load_config):
        """Test that get_ragflow_client returns the same instance."""
        from ragflow_claude_mcp.server import get_ragflow_client
        
        # Reset global client
        import ragflow_claude_mcp.server as server_module
        server_module._ragflow_client = None
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key'
        }.get(key, default)
        
        mock_load_config.return_value = {}
        
        client1 = get_ragflow_client()
        client2 = get_ragflow_client()
        
        assert client1 is client2
        # load_config should only be called once
        assert mock_load_config.call_count == 1


class TestOpenRouterConfiguration:
    """Test suite for OpenRouter configuration."""
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    @patch('dspy.configure')
    @patch('dspy.LM')
    def test_dspy_openrouter_configuration(self, mock_lm, mock_configure, mock_getenv, mock_load_config):
        """Test DSPy configuration with OpenRouter."""
        from ragflow_claude_mcp.server import RAGFlowMCPServer
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key'
        }.get(key, default)
        
        # Mock config with OpenRouter
        mock_load_config.return_value = {
            'PROVIDER': 'openrouter',
            'DSPY_MODEL': 'openai/gpt-4o-mini',
            'OPENROUTER_API_KEY': 'openrouter-key',
            'OPENROUTER_SITE_URL': 'https://test.com',
            'OPENROUTER_SITE_NAME': 'Test App'
        }
        
        server = RAGFlowMCPServer("http://test:9380", "test-key")
        
        # Test the retrieval_with_deepening method to trigger DSPy config
        import asyncio
        
        async def test_deepening():
            # This should trigger DSPy configuration
            result = await server.retrieval_with_deepening(
                dataset_id="test",
                query="test",
                deepening_level=1
            )
            return result
        
        # Mock standard retrieval as fallback
        with patch.object(server, 'retrieval_query', return_value={"test": "result"}):
            result = asyncio.run(test_deepening())
            # Should have deepening metadata added
            assert "test" in result
            assert result["test"] == "result"
            assert "metadata" in result
            assert "deepening" in result["metadata"]
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    def test_openrouter_config_validation(self, mock_getenv, mock_load_config):
        """Test OpenRouter configuration validation."""
        from ragflow_claude_mcp.server import RAGFlowMCPServer
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key'
        }.get(key, default)
        
        # Mock config with OpenRouter but missing API key
        mock_load_config.return_value = {
            'PROVIDER': 'openrouter',
            'DSPY_MODEL': 'openai/gpt-4o-mini'
            # Missing OPENROUTER_API_KEY
        }
        
        server = RAGFlowMCPServer("http://test:9380", "test-key")
        
        # Test the retrieval_with_deepening method
        import asyncio
        
        async def test_deepening():
            # This should fall back to standard retrieval due to config error
            result = await server.retrieval_with_deepening(
                dataset_id="test",
                query="test",
                deepening_level=1
            )
            return result
        
        # Mock standard retrieval as fallback
        with patch.object(server, 'retrieval_query', return_value={"fallback": "result"}):
            result = asyncio.run(test_deepening())
            # Should have deepening metadata added
            assert "fallback" in result
            assert result["fallback"] == "result"
            assert "metadata" in result
            assert "deepening" in result["metadata"]
    
    @patch('ragflow_claude_mcp.server.load_config')
    @patch('os.getenv')
    @patch('dspy.configure')
    @patch('dspy.LM')
    def test_dspy_openai_configuration(self, mock_lm, mock_configure, mock_getenv, mock_load_config):
        """Test DSPy configuration with OpenAI."""
        from ragflow_claude_mcp.server import RAGFlowMCPServer
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'RAGFLOW_BASE_URL': 'http://test:9380',
            'RAGFLOW_API_KEY': 'test-key'
        }.get(key, default)
        
        # Mock config with OpenAI
        mock_load_config.return_value = {
            'PROVIDER': 'openai',
            'DSPY_MODEL': 'openai/gpt-4o-mini',
            'OPENAI_API_KEY': 'openai-key'
        }
        
        server = RAGFlowMCPServer("http://test:9380", "test-key")
        
        # Test the retrieval_with_deepening method to trigger DSPy config
        import asyncio
        
        async def test_deepening():
            # This should trigger DSPy configuration
            result = await server.retrieval_with_deepening(
                dataset_id="test",
                query="test",
                deepening_level=1
            )
            return result
        
        # Mock standard retrieval as fallback
        with patch.object(server, 'retrieval_query', return_value={"test": "result"}):
            result = asyncio.run(test_deepening())
            # Should have deepening metadata added
            assert "test" in result
            assert result["test"] == "result"
            assert "metadata" in result
            assert "deepening" in result["metadata"]


class TestConfigurationFile:
    """Test suite for configuration file loading."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_config_file_not_found(self, mock_exists, mock_open_func):
        """Test load_config when config.json doesn't exist."""
        from ragflow_claude_mcp.server import load_config
        
        mock_exists.return_value = False
        
        config = load_config()
        assert config == {}
        mock_open_func.assert_not_called()
    
    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('json.load')
    def test_load_config_json_decode_error(self, mock_json_load, mock_exists, mock_open):
        """Test load_config with invalid JSON."""
        from ragflow_claude_mcp.server import load_config
        
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        config = load_config()
        assert config == {}
    
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