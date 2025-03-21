"""Tests for the text processor module."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.text_processing import AbstractTextProcessor, AITextProcessor


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    # Create a response that matches the OpenAI API format
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock.embeddings.create.return_value = MagicMock(
        data=[mock_embedding]
    )
    return mock


@pytest.fixture
def text_processor(test_embeddings_dir, mock_openai):
    """Create a TextProcessor instance with mocks."""
    processor = AITextProcessor(
        openai_api_key="test-key",
        cache_dir=str(test_embeddings_dir)
    )
    processor.client = mock_openai
    return processor


def test_process_batch_basic(text_processor):
    """Test basic batch processing functionality."""
    texts = ["Test description"]
    queries = ["Test summary"]
    result = text_processor.process_batch(texts, queries)
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1  # One result
    assert result.shape[1] == 3  # Based on mock response


def test_process_batch_edge_cases(text_processor):
    """Test batch processing edge cases."""
    # Test empty inputs
    with pytest.raises(ValueError):
        text_processor.process_batch([], ["query"])
    
    with pytest.raises(ValueError):
        text_processor.process_batch(["text"], [])
    
    # Test mismatched lengths
    with pytest.raises(ValueError):
        text_processor.process_batch(["text1", "text2"], ["query1"])


def test_get_embeddings_basic(text_processor, mock_openai):
    """Test basic embedding functionality."""
    texts = ["Test text"]
    embeddings = text_processor._get_embeddings(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)  # Should be list not ndarray
    assert len(embeddings[0]) == 3  # Based on mock response


def test_get_embeddings_caching(text_processor, mock_openai):
    """Test embedding caching functionality."""
    texts = ["Cache test"]
    metadata = [{"key": "TEST-1", "summary": "Test"}]
    
    # First call should use API
    embeddings1 = text_processor._get_embeddings(texts, metadata=metadata)
    assert mock_openai.embeddings.create.call_count == 1
    
    # Second call should use cache
    embeddings2 = text_processor._get_embeddings(texts, metadata=metadata)
    assert mock_openai.embeddings.create.call_count == 1  # No additional API calls
    
    assert len(embeddings1) == len(embeddings2)
    # Compare each element since they're lists
    assert all(a == b for a, b in zip(embeddings1[0], embeddings2[0]))


def test_get_embeddings_error_handling(text_processor, mock_openai):
    """Test embedding error handling."""
    # Test API error
    mock_openai.embeddings.create.side_effect = Exception("API Error")
    with pytest.raises(Exception):
        text_processor._get_embeddings(["Error test"])
    
    # Reset mock
    mock_openai.embeddings.create.side_effect = None
    
    # Test empty input - should work but return empty list
    result = text_processor._get_embeddings([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_chunk_text(text_processor):
    """Test text chunking functionality."""
    # Create a text that will exceed the token limit
    text = "This is a test. " * 5000  # Much longer text to force chunking
    chunks = text_processor.chunk_text(text, overlap=10)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 1  # Should be split into multiple chunks

    # Test empty text
    assert text_processor.chunk_text("") == []
    
    # Test short text
    short_text = "Short text"
    assert text_processor.chunk_text(short_text) == [short_text]


def test_combine_embeddings_with_attention(text_processor):
    """Test embedding combination with attention."""
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
    query_embedding = [0.2, 0.2, 0.2]
    
    combined = text_processor._combine_embeddings_with_attention(
        embeddings, 
        query_embedding,
        temperature=1.0
    )
    
    assert isinstance(combined, (list, np.ndarray))
    if isinstance(combined, np.ndarray):
        assert combined.shape == (3,)
    else:
        assert len(combined) == 3
    
    # Test with single embedding
    single = text_processor._combine_embeddings_with_attention(
        [embeddings[0]], 
        query_embedding
    )
    
    if isinstance(single, np.ndarray):
        np.testing.assert_array_equal(single, embeddings[0])
    else:
        assert all(a == b for a, b in zip(single, embeddings[0]))
    
    # Test with empty list
    with pytest.raises(ValueError):
        text_processor._combine_embeddings_with_attention([], query_embedding)
