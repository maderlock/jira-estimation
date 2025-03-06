"""Tests for main script functionality."""
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.main import main

@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    return pd.DataFrame({
        'key': ['TEST-1', 'TEST-2', 'TEST-3'],
        'summary': ['Test 1', 'Test 2', 'Test 3'],
        'description': ['Desc 1', 'Desc 2', 'Desc 3'],
        'time_spent': [1.0, 2.0, 3.0]
    })

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

@pytest.fixture
def mock_args():
    """Create mock arguments."""
    class Args:
        project_keys = ["TEST"]
        log_level = "INFO"
        max_results = None
        force_update = False
        model_type = "random_forest"
        use_cv = False
        cv_splits = 3
        n_estimators = 100
        max_depth = 15
        min_samples_split = 5
        min_samples_leaf = 2
        max_features = 0.8
        bootstrap = False
        epochs = 100
        batch_size = 32
        learning_rate = 0.001
        data_dir = None
        embedding_model = None
        exclude_labels = None
        include_subtasks = False
        no_cache = False
        random_seed = 424
    return Args

def test_random_forest_with_custom_estimators(mock_data, mock_embeddings, mock_args):
    """Test random forest estimation with custom n_estimators."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure args
        args = mock_args()
        args.data_dir = temp_dir
        args.n_estimators = 20
        
        # Setup patches
        with patch('src.main.JiraDataFetcher') as mock_fetcher, \
             patch('src.main.TextProcessor') as mock_processor, \
             patch('src.main.ModelLearner') as mock_learner, \
             patch('src.main.get_openai_api_key', return_value='test-key'), \
             patch('src.main.load_environment'), \
             patch('src.main.get_model_config') as mock_config:
            
            # Configure mocks
            mock_fetcher.return_value.fetch_tickets.return_value = mock_data
            mock_processor.return_value.process_dataframe.return_value = mock_embeddings
            mock_learner.return_value.train.return_value = None
            mock_config.return_value.random_seed = 2
            
            # Run main
            main(args)
            
            # Verify RandomForestRegressor was created with custom parameters
            mock_learner.assert_called_once()
            model = mock_learner.call_args[1]['model']
            assert isinstance(model, RandomForestRegressor)
            assert model.n_estimators == 20
            assert model.random_state == 424
            assert model.max_depth == 15
            assert model.min_samples_split == 5
            assert model.min_samples_leaf == 2
            assert model.max_features == 0.8
            assert model.bootstrap == False
