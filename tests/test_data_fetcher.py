"""Tests for the JIRA data fetcher module."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data_fetching.data_fetcher import JiraDataFetcher
from src.text_processing import TextProcessor
from tests.conftest import MockIssue


@pytest.fixture
def mock_text_processor():
    """Create a mock text processor."""
    processor = MagicMock(spec=TextProcessor)
    #processor.process_text.return_value = [0.1, 0.2, 0.3]  # Return embedding vector
    processor.strip_formatting.return_value = "Test Summary"  # Return fixed string for summary
    return processor


@pytest.fixture
def mock_issues():
    """Create mock JIRA issues."""
    def create_mock_issue(key, summary, description, time_spent, original_estimate=None):
        return MockIssue(key, {
            'summary': summary,
            'description': description,
            'created': '2025-01-01T00:00:00.000+0000',
            'updated': '2025-01-02T00:00:00.000+0000',
            'timespent': time_spent,
            'timeoriginalestimate': original_estimate
        })
    
    return [
        create_mock_issue('TEST-1', 'Test 1', 'Description 1', 3600),
        create_mock_issue('TEST-2', 'Test 2', 'Description 2', 7200, 3600),
        create_mock_issue('TEST-3', 'Test 3', 'Description 3', 0),  # Invalid time
        create_mock_issue('TEST-4', 'Test 4', None, 3600),  # No description
    ]


@pytest.fixture
def data_fetcher(mock_jira_client, mock_text_processor, test_cache_dir):
    """Create a JiraDataFetcher instance with mocks."""
    fetcher = JiraDataFetcher(
        jira_url="https://test.atlassian.net",
        jira_email="test@example.com",
        jira_token="test-token",
        text_processor=mock_text_processor,
        cache_dir=str(test_cache_dir)
    )
    fetcher.jira = mock_jira_client
    return fetcher


def test_fetch_tickets_basic(data_fetcher, mock_issues, mock_jira_client):
    """Test basic ticket fetching functionality."""
    mock_jira_client.issues = mock_issues
    
    df = data_fetcher.fetch_tickets(
        project_keys=["TEST"],
        max_results=10,
        use_cache=False
    )
    
    assert len(df) == 3  # Should exclude TEST-3 due to 0 time spent
    assert list(df.columns) == [
        'key', 'summary', 'description', 'created', 'updated',
        'original_estimate', 'time_spent'
    ]
    assert df['time_spent'].min() > 0


def test_fetch_tickets_with_cache(data_fetcher, mock_issues, mock_jira_client):
    """Test ticket fetching with cache enabled."""
    mock_jira_client.issues = mock_issues
    
    # Initial fetch
    df1 = data_fetcher.fetch_tickets(
        project_keys=["TEST"],
        max_results=10,
        use_cache=True
    )
    
    # Second fetch should use cache
    df2 = data_fetcher.fetch_tickets(
        project_keys=["TEST"],
        max_results=10,
        use_cache=True,
        update_cache=False
    )
    
    assert len(df1) == len(df2)
    assert df1['key'].tolist() == df2['key'].tolist()


def test_fetch_tickets_with_filters(data_fetcher, mock_issues, mock_jira_client):
    """Test ticket fetching with various filters."""
    mock_jira_client.issues = mock_issues
    
    df = data_fetcher.fetch_tickets(
        project_keys=["TEST"],
        max_results=2,
        exclude_labels=["ignore"],
        include_subtasks=True,
        use_cache=False
    )
    
    assert len(df) == 2  # Should respect max_results
    assert df['time_spent'].min() > 0


def test_fetch_tickets_error_handling(data_fetcher, mock_jira_client):
    """Test error handling in ticket fetching."""
    # Test with no issues
    mock_jira_client.issues = []
    df = data_fetcher.fetch_tickets(use_cache=False)
    assert len(df) == 0
    
    # Test with invalid max_results
    df = data_fetcher.fetch_tickets(max_results=-1, use_cache=False)
    assert len(df) == 0


def test_fetch_tickets_time_validation(data_fetcher, mock_issues, mock_jira_client):
    """Test validation of time values in tickets."""
    # Add issue with invalid time
    mock_issues.append(
        MockIssue('TEST-5', {
            'summary': 'Invalid Time',
            'description': 'Description',
            'created': '2025-01-01T00:00:00.000+0000',
            'updated': '2025-01-02T00:00:00.000+0000',
            'timespent': 1040 * 3600 + 1,  # Exceeds maximum
            'timeoriginalestimate': 0
        })
    )
    
    mock_jira_client.issues = mock_issues
    df = data_fetcher.fetch_tickets(use_cache=False)
    
    # Should exclude issues with invalid time values
    assert len(df) == 3
    assert 'TEST-5' not in df['key'].values
    assert df['time_spent'].max() <= 1040  # Maximum allowed hours
