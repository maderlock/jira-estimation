"""Tests for the linear model learning functionality."""
import logging
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.models.linear_model import LinearEstimator
from src.utils import get_model_config


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    # Create 100 samples with 10 features each
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def mock_times():
    """Create mock time values for testing."""
    np.random.seed(42)
    # Create realistic time values between 1 and 100 hours
    return np.random.uniform(1, 100, 100)


@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def linear_estimator(test_logger):
    """Create a LinearEstimator instance for testing."""
    config = get_model_config()
    return LinearEstimator(logger=test_logger, config=config)


def test_prepare_data_basic(linear_estimator, mock_embeddings, mock_times):
    """Test basic data preparation functionality."""
    X_train, X_test, y_train, y_test = linear_estimator.prepare_data(
        mock_embeddings, mock_times, test_size=0.2
    )
    
    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == mock_embeddings.shape[1]
    assert X_test.shape[1] == mock_embeddings.shape[1]
    
    # Check data split ratio
    assert len(y_train) == int(0.8 * len(mock_times))
    assert len(y_test) == int(0.2 * len(mock_times))
    
    # Check scaling
    assert np.abs(X_train.mean()) < 1e-10  # Should be close to 0
    assert np.abs(X_train.std() - 1) < 1e-10  # Should be close to 1


def test_prepare_data_invalid_times(linear_estimator, mock_embeddings):
    """Test data preparation with invalid time values."""
    # Create times with invalid values
    invalid_times = np.array([-1, 0, 1041, 100, 50])  # -1, 0, and 1041 are invalid
    valid_embeddings = mock_embeddings[:5]  # Match length with times
    
    X_train, X_test, y_train, y_test = linear_estimator.prepare_data(
        valid_embeddings, invalid_times, test_size=0.4
    )
    
    # Only 2 valid times (100 and 50) should remain
    assert len(y_train) + len(y_test) == 2


def test_train_basic(linear_estimator, mock_embeddings, mock_times):
    """Test basic model training functionality."""
    metrics = linear_estimator.train(
        mock_embeddings,
        mock_times,
        test_size=0.2,
        use_cv=False
    )
    
    # Check that we have the expected metrics
    assert 'r2' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    
    # Basic sanity checks on metrics
    assert -1 <= metrics['r2'] <= 1  # R2 should be between -1 and 1
    assert metrics['mae'] >= 0  # MAE should be non-negative
    assert metrics['rmse'] >= 0  # RMSE should be non-negative


def test_train_with_cv(linear_estimator, mock_embeddings, mock_times):
    """Test model training with cross-validation."""
    metrics = linear_estimator.train(
        mock_embeddings,
        mock_times,
        use_cv=True,
        n_splits=5
    )
    
    # Check that we have the expected CV metrics
    expected_metrics = [
        'cv_r2_mean', 'cv_r2_std',
        'cv_mae_mean', 'cv_mae_std',
        'cv_rmse_mean', 'cv_rmse_std'
    ]
    for metric in expected_metrics:
        assert metric in metrics
    
    # Basic sanity checks on metrics
    assert -1 <= metrics['cv_r2_mean'] <= 1
    assert metrics['cv_mae_mean'] >= 0
    assert metrics['cv_rmse_mean'] >= 0
    
    # Standard deviations should be non-negative
    assert metrics['cv_r2_std'] >= 0
    assert metrics['cv_mae_std'] >= 0
    assert metrics['cv_rmse_std'] >= 0


def test_predict(linear_estimator, mock_embeddings, mock_times):
    """Test model prediction functionality."""
    # Train the model first
    linear_estimator.train(mock_embeddings, mock_times, test_size=0.2)
    
    # Create new test data
    np.random.seed(43)  # Different seed for test data
    test_embeddings = np.random.randn(20, 10)
    
    # Make predictions
    predictions = linear_estimator.predict(test_embeddings)
    
    # Check predictions shape and values
    assert len(predictions) == len(test_embeddings)
    assert isinstance(predictions, np.ndarray)
    assert not np.any(np.isnan(predictions))  # No NaN values
    assert not np.any(np.isinf(predictions))  # No infinite values


def test_model_persistence(linear_estimator, mock_embeddings, mock_times, tmp_path):
    """Test model saving and loading functionality."""
    # Train the model
    linear_estimator.train(mock_embeddings, mock_times, test_size=0.2)
    
    # Get predictions before saving
    test_embeddings = np.random.randn(20, 10)
    orig_predictions = linear_estimator.predict(test_embeddings)
    
    # Save the model
    save_path = tmp_path / "model.joblib"
    linear_estimator.save(save_path)
    assert save_path.exists()
    
    # Create a new estimator and load the model
    new_estimator = LinearEstimator()
    # Prepare data to fit the scaler
    new_estimator.prepare_data(mock_embeddings, mock_times, test_size=0.2)
    new_estimator.load(save_path)
    
    # Make predictions with loaded model
    loaded_predictions = new_estimator.predict(test_embeddings)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(orig_predictions, loaded_predictions)


def test_edge_cases(linear_estimator):
    """Test edge cases and error handling."""
    # Test with empty data
    with pytest.raises(ValueError, match="No training examples provided"):
        linear_estimator.prepare_data(np.array([]), np.array([]))
    
    # Test with mismatched dimensions
    with pytest.raises(ValueError, match="No valid training examples after filtering"):
        X = np.random.randn(10, 5)
        y = np.zeros(10)  # All invalid times
        linear_estimator.prepare_data(X, y)
    
    # Test with all invalid times (redundant with above, but kept for clarity)
    X = np.random.randn(10, 5)
    all_invalid_times = np.zeros(10)  # All zeros are invalid
    with pytest.raises(ValueError, match="No valid training examples after filtering"):
        linear_estimator.prepare_data(X, all_invalid_times)
