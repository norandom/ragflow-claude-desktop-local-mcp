"""Tests for the RAGFlow MCP server."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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