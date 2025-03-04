"""Test configuration and fixtures."""
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def test_data_dir(tmp_path) -> Path:
    """Create and return a temporary test data directory."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def test_cache_dir(test_data_dir) -> Path:
    """Create and return a temporary cache directory."""
    cache_dir = test_data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir
    # Cleanup after tests
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.fixture
def test_embeddings_dir(test_data_dir) -> Path:
    """Create and return a temporary embeddings directory."""
    embeddings_dir = test_data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    yield embeddings_dir
    # Cleanup after tests
    if embeddings_dir.exists():
        shutil.rmtree(embeddings_dir)


class MockJiraClient:
    """Mock JIRA client for testing."""
    
    def __init__(self, issues=None):
        self.issues = issues or []
        
    def search_issues(self, jql_str, maxResults=None, startAt=0, fields=None):
        """Mock search_issues method to match JIRA API."""
        if maxResults:
            return self.issues[startAt:startAt + maxResults]
        return self.issues[startAt:]


class MockIssue:
    """Mock JIRA issue for testing."""
    
    def __init__(self, key, fields):
        """Initialize mock issue."""
        self.key = key
        
        # Process fields and ensure proper types
        processed_fields = {}
        for field_name, value in fields.items():
            if field_name in ['timespent', 'timeoriginalestimate']:
                # Convert time fields to integers or None
                processed_fields[field_name] = int(value) if value is not None else None
            elif field_name in ['summary', 'description']:
                # Convert text fields to strings
                processed_fields[field_name] = str(value) if value is not None else ""
            else:
                processed_fields[field_name] = value
        
        # Create a Fields class with the processed fields as attributes
        class Fields:
            def __init__(self, fields_dict):
                for k, v in fields_dict.items():
                    setattr(self, k, v)
                    
            def __getattr__(self, name):
                # Return None for non-existent fields
                return None
        
        self.fields = Fields(processed_fields)
    
    def strip_formatting(self):
        """Strip formatting from text fields."""
        return str(self.fields.summary)


@pytest.fixture
def mock_jira_client():
    """Create a mock JIRA client."""
    return MockJiraClient()
