"""Tests for the data cache module."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data_fetching.cache import DataCache


def test_cache_initialization(test_cache_dir):
    """Test cache initialization creates required directories."""
    cache = DataCache(str(test_cache_dir))
    assert Path(test_cache_dir).exists()
    assert cache.metadata_file.exists()
    assert json.loads(cache.metadata_file.read_text()) == {}


def test_save_and_load_data(test_cache_dir):
    """Test saving and loading data from cache."""
    cache = DataCache(str(test_cache_dir))
    
    # Test data
    data = [
        {"key": "TEST-1", "summary": "Test 1", "time_spent": 3600},
        {"key": "TEST-2", "summary": "Test 2", "time_spent": 7200},
    ]
    query_hash = "test_query"
    query_params = {"project_keys": ["TEST"]}
    
    # Save data
    cache.save_data(data, query_hash, query_params)
    
    # Verify metadata
    metadata = json.loads(cache.metadata_file.read_text())
    assert query_hash in metadata
    assert metadata[query_hash]["num_records"] == 2
    
    # Load data
    df = cache.get_cached_data(query_hash)
    assert len(df) == 2
    assert list(df["key"]) == ["TEST-1", "TEST-2"]


def test_cache_update_strategy(test_cache_dir):
    """Test cache update strategy with different parameters."""
    cache = DataCache(str(test_cache_dir))
    
    # Mock update function
    def update_func(max_results=None, updated_after=None):
        base_data = [
            {"key": "TEST-1", "summary": "Test 1", "time_spent": 3600},
            {"key": "TEST-2", "summary": "Test 2", "time_spent": 7200},
        ]
        return pd.DataFrame(base_data[:max_results if max_results else len(base_data)])
    
    # Test initial load
    df = cache.load("test_query", update_func, {"max_results": 2})
    assert len(df) == 2
    
    # Test force update
    df = cache.load("test_query", update_func, {"max_results": 1}, force_update=True)
    assert len(df) == 1  # Should respect new max_results


def test_cache_with_invalid_data(test_cache_dir):
    """Test cache behavior with invalid data."""
    cache = DataCache(str(test_cache_dir))
    
    # Test with empty data
    empty_df = pd.DataFrame()
    cache.save_data(empty_df, "empty_query", {})
    df = cache.get_cached_data("empty_query")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    
    # Test with None data
    df = cache.load("none_query", lambda: pd.DataFrame(), use_cache=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_cache_max_results_handling(test_cache_dir):
    """Test handling of max_results parameter."""
    cache = DataCache(str(test_cache_dir))
    
    # Create test data
    data = [{"key": f"TEST-{i}", "summary": f"Test {i}"} for i in range(1, 11)]
    df = pd.DataFrame(data)
    
    def update_func(max_results=None):
        if max_results:
            return df.head(max_results)
        return df
    
    # Initial fetch with max_results=5
    df1 = cache.load("test_query", update_func, {"max_results": 5})
    assert len(df1) == 5
    
    # Fetch with larger max_results should still return original cached amount
    df2 = cache.load("test_query", update_func, {"max_results": 10})
    assert len(df2) == 5  # Should match original cached amount
    
    # Force update to get more results
    df3 = cache.load("test_query", update_func, {"max_results": 10}, force_update=True)
    assert len(df3) == 10
